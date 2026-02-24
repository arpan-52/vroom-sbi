"""
SBI Trainer for VROOM-SBI.

Handles training of neural posterior estimators using the SBI library.
Features:
- Proper checkpointing with torch.save()
- Training progress visualization
- Memory management
- Mixed precision training
"""

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List
from tqdm import tqdm
from datetime import datetime
import logging

from sbi.inference import SNPE

# Handle different SBI versions
try:
    from sbi.utils import posterior_nn
except ImportError:
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        from sbi.utils.get_nn_models import posterior_nn

from ..config import Configuration
from ..core.checkpoint import CheckpointManager, ModelCheckpoint, save_training_plots
from ..core.result import TrainingResult, TrainingMetrics
from ..simulator import RMSimulator, build_prior, sample_prior
from ..simulator.augmentation import augment_weights_combined
from .networks import SpectralEmbedding
from .data_loader import save_simulations

logger = logging.getLogger(__name__)


class SBITrainer:
    """
    Trainer for SBI neural posterior estimators.
    
    Handles the full training pipeline including:
    - Simulation generation
    - Network architecture setup
    - Training with SBI library
    - Checkpointing and monitoring
    
    Parameters
    ----------
    config : Configuration
        Full configuration object
    checkpoint_manager : CheckpointManager, optional
        Manager for saving checkpoints
    """
    
    def __init__(
        self,
        config: Configuration,
        checkpoint_manager: Optional[CheckpointManager] = None,
    ):
        self.config = config
        self.device = config.training.device
        
        # Set device
        if self.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            self.device = "cpu"
        
        # Checkpoint manager
        if checkpoint_manager is None:
            checkpoint_manager = CheckpointManager(
                checkpoint_dir=Path(config.training.save_dir) / "checkpoints"
            )
        self.checkpoint_manager = checkpoint_manager
        
        # Track training state
        self.current_model_type = None
        self.current_n_components = None
        self.training_results: Dict[str, TrainingResult] = {}
    
    def train_model(
        self,
        model_type: str,
        n_components: int,
        n_simulations: Optional[int] = None,
        save_simulations_flag: bool = True,
    ) -> TrainingResult:
        """
        Train a single posterior model.
        
        Parameters
        ----------
        model_type : str
            Physical model type
        n_components : int
            Number of components
        n_simulations : int, optional
            Override number of simulations
        save_simulations_flag : bool
            Whether to save simulations for classifier training
            
        Returns
        -------
        TrainingResult
            Training results and metadata
        """
        self.current_model_type = model_type
        self.current_n_components = n_components
        
        # Get scaled number of simulations
        if n_simulations is None:
            n_simulations = self.config.training.get_scaled_simulations(n_components)
        
        logger.info(f"\n{'='*60}")
        logger.info(f"Training {model_type} N={n_components}")
        logger.info(f"Simulations: {n_simulations:,}")
        logger.info(f"Device: {self.device}")
        logger.info(f"{'='*60}")
        
        # Get all prior bounds from centralized config
        flat_priors = self.config.priors.to_flat_dict()
        base_noise_level = self.config.noise.base_level
        
        # Create simulator
        simulator = RMSimulator(
            freq_file=self.config.freq_file,
            n_components=n_components,
            base_noise_level=base_noise_level,
            model_type=model_type,
        )
        
        # Build prior (all bounds come from flat_priors)
        prior = build_prior(
            n_components,
            flat_priors,
            device=self.device,
            model_type=model_type,
        )
        
        logger.info(f"Simulator: n_params={simulator.n_params}, n_freq={simulator.n_freq}")
        
        # ============================================================
        # PHASE 1: Generate all simulations as chunked .pt files
        # ============================================================
        chunk_dir = self._generate_simulation_chunks(
            simulator, n_simulations, n_components, 
            flat_priors, model_type
        )
        
        # ============================================================
        # PHASE 2: Train on chunks
        # ============================================================
        
        # Build embedding network
        input_dim = 2 * simulator.n_freq
        embedding_dim = self.config.sbi.embedding_dim
        embedding_net = SpectralEmbedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
        ).to(self.device)
        
        # Get architecture config
        arch_config = self.config.sbi.get_architecture(n_components)
        
        logger.info(f"SBI Architecture: {self.config.sbi.model.upper()}")
        logger.info(f"  Hidden: {arch_config.hidden_features}, Transforms: {arch_config.num_transforms}")
        
        # Get training parameters
        learning_rate = self.config.training.learning_rate
        training_batch_size = self.config.training.training_batch_size
        stop_after_epochs = self.config.training.stop_after_epochs
        
        logger.info(f"Training: batch_size={training_batch_size}, lr={learning_rate}, patience={stop_after_epochs}")
        
        # Train on chunks
        start_time = datetime.now()
        
        density_estimator, training_history = self._train_on_chunks(
            chunk_dir=chunk_dir,
            prior=prior,
            embedding_net=embedding_net,
            arch_config=arch_config,
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            stop_after_epochs=stop_after_epochs,
            validation_fraction=self.config.training.validation_fraction,
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Log training summary
        if training_history.get('train_loss'):
            logger.info(f"Training epochs: {len(training_history['train_loss'])}")
            logger.info(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
        if training_history.get('val_loss'):
            val_losses = [v for v in training_history['val_loss'] if v is not None]
            if val_losses:
                best_val = min(val_losses)
                best_epoch = val_losses.index(best_val) + 1
                logger.info(f"Best validation loss: {best_val:.4f} (epoch {best_epoch})")
        
        # Build posterior using DirectPosterior with our wrapped flow
        # density_estimator is already a StandardizedFlowWrapper from streaming training
        from sbi.inference.posteriors import DirectPosterior
        posterior = DirectPosterior(
            posterior_estimator=density_estimator,
            prior=prior,
        )
        
        # Save model with torch.save()
        output_dir = Path(self.config.training.save_dir)
        model_path = output_dir / f"posterior_{model_type}_n{n_components}.pt"
        
        self._save_posterior(
            posterior, prior, embedding_net, simulator,
            model_type, n_components, training_history,
            model_path
        )
        
        # Create training result
        result = TrainingResult(
            model_type=model_type,
            n_components=n_components,
            n_simulations=n_simulations,
            n_params=simulator.n_params,
            final_train_loss=training_history.get('train_loss', [0])[-1] if training_history.get('train_loss') else 0,
            final_val_loss=training_history.get('val_loss', [None])[-1] if training_history.get('val_loss') else None,
            model_path=str(model_path),
            training_time_seconds=training_time,
            config_snapshot=self.config.to_dict(),
        )
        
        # Add metrics history
        if 'train_loss' in training_history:
            for i, train_loss in enumerate(training_history['train_loss']):
                val_loss = training_history.get('val_loss', [None] * len(training_history['train_loss']))[i]
                result.add_metric(TrainingMetrics(
                    epoch=i,
                    train_loss=train_loss,
                    val_loss=val_loss,
                ))
        
        # Save training plots
        if training_history and (training_history.get('train_loss') or training_history.get('val_loss')):
            plot_path = output_dir / f"training_{model_type}_n{n_components}.png"
            save_training_plots(training_history, plot_path, model_type, n_components)
            logger.info(f"Saved training plot to {plot_path}")
        else:
            logger.warning("No training history available for plotting")
        
        # Store result
        key = f"{model_type}_n{n_components}"
        self.training_results[key] = result
        
        logger.info(f"Training complete in {training_time:.1f}s")
        logger.info(f"Model saved to {model_path}")
        
        return result
    

    def _generate_simulation_chunks(
        self,
        simulator: RMSimulator,
        n_simulations: int,
        n_components: int,
        flat_priors: Dict[str, float],
        model_type: str,
    ) -> Path:
        """
        Generate simulations and save as chunked .pt files.
        
        User controls everything:
        - n_simulations: total simulations
        - simulation_batch_size: chunk size (each .pt file)
        - training_batch_size: mini-batch for GPU
        
        Returns path to chunk directory.
        """
        n_freq = simulator.n_freq
        n_params = simulator.n_params
        
        # Chunk size = simulation_batch_size (user-supplied, no auto)
        chunk_size = self.config.training.simulation_batch_size
        
        # Number of chunks
        n_full_chunks = n_simulations // chunk_size
        remainder = n_simulations % chunk_size
        n_chunks = n_full_chunks + (1 if remainder > 0 else 0)
        
        logger.info(f"Generating {n_simulations:,} simulations in {n_chunks} chunk(s)")
        logger.info(f"Chunk size: {chunk_size:,} samples")
        
        # Create chunk directory
        output_dir = Path(self.config.training.save_dir)
        chunk_dir = output_dir / f"chunks_{model_type}_n{n_components}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate and save chunks
        samples_generated = 0
        for chunk_idx in range(n_chunks):
            # Last chunk may be smaller
            this_chunk_size = min(chunk_size, n_simulations - samples_generated)
            
            logger.info(f"Generating chunk {chunk_idx + 1}/{n_chunks} ({this_chunk_size:,} samples)...")
            
            # Sample parameters for this chunk
            chunk_theta = sample_prior(
                this_chunk_size, n_components, flat_priors,
                model_type=model_type
            )
            
            # Generate augmented weights
            chunk_weights = np.array([
                augment_weights_combined(
                    simulator.weights,
                    scattered_prob=self.config.weight_augmentation.scattered_prob,
                    gap_prob=self.config.weight_augmentation.gap_prob,
                    large_block_prob=self.config.weight_augmentation.large_block_prob,
                    noise_variation=self.config.weight_augmentation.noise_variation,
                ) for _ in range(this_chunk_size)
            ], dtype=np.float32)
            
            # Get noise percentage (with optional augmentation)
            base_noise_percent = self.config.noise.base_percent
            if self.config.noise.augmentation_enable:
                # Vary noise percentage randomly for robustness
                noise_percent = np.random.uniform(
                    base_noise_percent * self.config.noise.augmentation_min_factor,
                    base_noise_percent * self.config.noise.augmentation_max_factor,
                )
            else:
                noise_percent = base_noise_percent
            
            # Simulate with percentage-based noise
            chunk_x = simulator.simulate_batch(chunk_theta, chunk_weights, noise_percent)
            
            # Save chunk to disk
            chunk_path = chunk_dir / f"chunk_{chunk_idx:04d}.pt"
            torch.save({
                'theta': torch.tensor(chunk_theta, dtype=torch.float32),
                'x': torch.tensor(chunk_x, dtype=torch.float32),
            }, chunk_path)
            
            logger.info(f"  Saved {chunk_path.name}")
            
            # Free memory
            del chunk_theta, chunk_weights, chunk_x
            
            samples_generated += this_chunk_size
        
        # Save metadata
        meta_path = chunk_dir / "metadata.pt"
        torch.save({
            'n_simulations': n_simulations,
            'n_chunks': n_chunks,
            'chunk_size': chunk_size,
            'n_components': n_components,
            'model_type': model_type,
            'n_freq': n_freq,
            'n_params': n_params,
        }, meta_path)
        
        logger.info(f"Generated {n_simulations:,} simulations in {n_chunks} chunk(s)")
        return chunk_dir
    
    def _train_on_chunks(
        self,
        chunk_dir: Path,
        prior,
        embedding_net: nn.Module,
        arch_config,
        learning_rate: float,
        training_batch_size: int,
        stop_after_epochs: int,
        validation_fraction: float,
    ) -> tuple:
        """
        Train on chunked .pt files using custom streaming training.
        
        This uses a custom training loop that truly streams from disk,
        avoiding the RAM bottleneck of SBI's append_simulations().
        
        Based on official SBI tutorial:
        https://sbi-dev.github.io/sbi/latest/advanced_tutorials/18_training_interface/
        
        For single-round amortized NPE:
        - Uses density_estimator.loss() method
        - No proposal correction needed
        """
        from .streaming_trainer import StreamingNPETrainer
        
        logger.info("Using async streaming NPE training (memory efficient)")
        
        # Create streaming trainer
        trainer = StreamingNPETrainer(device=self.device)
        
        # Train with async chunk streaming
        density_estimator, history = trainer.train(
            chunk_dir=chunk_dir,
            prior=prior,
            embedding_net=embedding_net,
            flow_type=self.config.sbi.model,
            hidden_features=arch_config.hidden_features,
            num_transforms=arch_config.num_transforms,
            num_bins=self.config.sbi.num_bins,
            learning_rate=learning_rate,
            training_batch_size=training_batch_size,
            validation_fraction=validation_fraction,
            max_epochs=500,  # Will early stop
            stop_after_epochs=stop_after_epochs,
            clip_grad_norm=5.0,
            show_progress=True,
        )
        
        # Store the trainer for building posterior later
        self._streaming_trainer = trainer
        
        return density_estimator, history
    
    def _extract_training_history(self, inference) -> Dict[str, List[float]]:
        """
        Extract training history from SBI inference object.
        
        SBI stores training info in inference.summary (dict) with keys that vary by version.
        We need to handle:
        - training_log_probs / validation_log_probs (newer SBI)
        - training_loss / validation_loss (some versions)
        - best_validation_log_prob / epochs_trained
        
        We convert log probs to loss (negative log prob) for plotting.
        """
        history = {}
        
        try:
            # Get summary - attribute name varies by SBI version
            summary = None
            if hasattr(inference, 'summary'):
                summary = inference.summary
            elif hasattr(inference, '_summary'):
                summary = inference._summary
            
            if summary is None or not isinstance(summary, dict):
                logger.warning(f"No valid summary found. Type: {type(summary)}")
                return history
            
            # Log available keys for debugging
            logger.info(f"SBI summary keys available: {list(summary.keys())}")
            
            # Helper to safely convert to float
            def to_float(x):
                # Handle lists/tuples - take first element or last element
                if isinstance(x, (list, tuple)):
                    if len(x) == 0:
                        return 0.0
                    x = x[-1]  # Take last element (usually the final/best value)
                # Handle tensors
                if hasattr(x, 'item'):
                    return x.item()
                elif hasattr(x, 'cpu'):
                    return float(x.cpu().numpy())
                return float(x)
            
            # Extract training log probs → convert to loss (negative log prob)
            for key in ['training_log_probs', 'train_log_probs', 'training_loss']:
                if key in summary and summary[key] is not None:
                    values = summary[key]
                    if isinstance(values, (list, tuple)) and len(values) > 0:
                        if 'log_prob' in key:
                            # Convert log prob to loss (negate)
                            history['train_loss'] = [-to_float(x) for x in values]
                        else:
                            history['train_loss'] = [to_float(x) for x in values]
                        logger.info(f"Found training data in '{key}': {len(history['train_loss'])} epochs")
                        break
            
            # Extract validation log probs
            for key in ['validation_log_probs', 'val_log_probs', 'validation_loss']:
                if key in summary and summary[key] is not None:
                    values = summary[key]
                    if isinstance(values, (list, tuple)) and len(values) > 0:
                        if 'log_prob' in key:
                            history['val_loss'] = [-to_float(x) for x in values]
                        else:
                            history['val_loss'] = [to_float(x) for x in values]
                        logger.info(f"Found validation data in '{key}': {len(history['val_loss'])} epochs")
                        break
            
            # Extract best validation log prob
            for key in ['best_validation_log_prob', 'best_validation_loss']:
                if key in summary and summary[key] is not None:
                    val = summary[key]
                    if 'log_prob' in key:
                        history['best_val_loss'] = -to_float(val)
                    else:
                        history['best_val_loss'] = to_float(val)
                    break
            
            # Extract epochs trained
            if 'epochs_trained' in summary:
                epochs = summary['epochs_trained']
                if isinstance(epochs, (list, tuple)):
                    history['epochs_trained'] = int(epochs[-1]) if epochs else 0
                else:
                    history['epochs_trained'] = int(to_float(epochs))
            
            # Summary
            if history.get('train_loss'):
                logger.info(f"Training history: {len(history['train_loss'])} epochs")
                logger.info(f"  Final train loss: {history['train_loss'][-1]:.4f}")
            if history.get('val_loss'):
                best_val = min(history['val_loss'])
                best_epoch = history['val_loss'].index(best_val) + 1
                logger.info(f"  Best val loss: {best_val:.4f} (epoch {best_epoch})")
            
            if not history:
                logger.warning("Could not extract any training history")
                logger.warning(f"Summary content: {summary}")
                
        except Exception as e:
            logger.warning(f"Error extracting training history: {e}")
            import traceback
            traceback.print_exc()
        
        return history
    
    def _save_posterior(
        self,
        posterior,
        prior,
        embedding_net: nn.Module,
        simulator: RMSimulator,
        model_type: str,
        n_components: int,
        training_history: Dict[str, List[float]],
        save_path: Path,
    ):
        """Save posterior model using torch.save()."""
        
        # Get prior bounds - handle different SBI versions
        try:
            # Try newer SBI API
            if hasattr(prior, 'base_dist'):
                prior_bounds = {
                    'low': prior.base_dist.low.cpu().numpy().tolist(),
                    'high': prior.base_dist.high.cpu().numpy().tolist(),
                }
            elif hasattr(prior, '_prior') and hasattr(prior._prior, 'base_dist'):
                prior_bounds = {
                    'low': prior._prior.base_dist.low.cpu().numpy().tolist(),
                    'high': prior._prior.base_dist.high.cpu().numpy().tolist(),
                }
            elif hasattr(prior, 'low') and hasattr(prior, 'high'):
                # Direct BoxUniform attributes
                prior_bounds = {
                    'low': prior.low.cpu().numpy().tolist(),
                    'high': prior.high.cpu().numpy().tolist(),
                }
            else:
                # Fallback: reconstruct from config
                low, high = self.config.priors.get_bounds_for_model(model_type, n_components)
                prior_bounds = {
                    'low': low.tolist(),
                    'high': high.tolist(),
                }
                logger.warning("Could not extract prior bounds from SBI object, using config")
        except Exception as e:
            logger.warning(f"Error extracting prior bounds: {e}, using config")
            low, high = self.config.priors.get_bounds_for_model(model_type, n_components)
            prior_bounds = {
                'low': low.tolist(),
                'high': high.tolist(),
            }
        
        save_dict = {
            # Model info
            'model_type': model_type,
            'n_components': n_components,
            'n_params': simulator.n_params,
            'n_freq': simulator.n_freq,
            'params_per_comp': simulator.params_per_comp,
            'param_names': simulator.get_param_names(),  # Save parameter names
            
            # Network states
            'posterior_state': posterior.state_dict() if hasattr(posterior, 'state_dict') else None,
            'embedding_net_state': embedding_net.state_dict(),
            
            # For SBI posterior reconstruction - save the full object
            'posterior': posterior,
            
            # Prior info
            'prior_bounds': prior_bounds,
            
            # Training info
            'training_history': training_history,
            'lambda_sq': simulator.lambda_sq.tolist(),
            
            # Config
            'config_used': {
                'sbi_model': self.config.sbi.model,
                'embedding_dim': self.config.sbi.embedding_dim,
                'hidden_features': self.config.sbi.get_architecture(n_components).hidden_features,
                'num_transforms': self.config.sbi.get_architecture(n_components).num_transforms,
            },
            
            # Metadata
            'save_timestamp': datetime.now().isoformat(),
            'torch_version': torch.__version__,
            'training_type': 'streaming_npe',  # Mark as streaming trained
        }
        
        torch.save(save_dict, save_path)
        logger.info(f"Saved posterior model to {save_path}")


def train_model(
    config: Configuration,
    model_type: str,
    n_components: int,
    **kwargs
) -> TrainingResult:
    """
    Convenience function to train a single model.
    """
    trainer = SBITrainer(config)
    return trainer.train_model(model_type, n_components, **kwargs)


def train_all_models(
    config: Configuration,
    classifier_only: bool = False,
) -> Dict[str, Any]:
    """
    Train all models specified in configuration.
    
    Parameters
    ----------
    config : Configuration
        Full configuration
    classifier_only : bool
        If True, only train classifier (requires existing simulations)
        
    Returns
    -------
    dict
        Results for all trained models
    """
    results = {}
    
    trainer = SBITrainer(config)
    min_components = config.model_selection.min_components
    max_components = config.model_selection.max_components
    model_types = config.physics.model_types
    
    n_models = len(model_types) * (max_components - min_components + 1)
    
    logger.info(f"\n{'='*60}")
    logger.info("VROOM-SBI TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Components: {min_components} to {max_components}")
    logger.info(f"Total models: {n_models}")
    
    if not classifier_only:
        # Train all posterior models
        for model_type in model_types:
            for n_comp in range(min_components, max_components + 1):
                result = trainer.train_model(model_type, n_comp)
                key = f"{model_type}_n{n_comp}"
                results[key] = result
    
    # Train classifier if enabled
    if config.model_selection.use_classifier:
        from .classifier_trainer import train_classifier
        
        logger.info(f"\n{'='*60}")
        logger.info("Training Model Selection Classifier")
        logger.info(f"{'='*60}")
        
        classifier_result = train_classifier(
            config=config,
            output_dir=Path(config.training.save_dir),
            min_components=min_components,
            max_components=max_components,
            model_types=model_types,
            cross_model_training=len(model_types) > 1,
        )
        results['classifier'] = classifier_result
    
    # Save training summary
    _save_training_summary(results, Path(config.training.save_dir))
    
    return results


def _save_training_summary(results: Dict[str, Any], output_dir: Path):
    """Save training summary to file."""
    summary_path = output_dir / "training_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VROOM-SBI TRAINING SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n\n")
        
        for key, result in results.items():
            if key == 'classifier':
                f.write(f"\nClassifier:\n")
                f.write(f"  Accuracy: {result.get('final_val_accuracy', 'N/A'):.2f}%\n")
            elif isinstance(result, TrainingResult):
                f.write(f"\n{result.model_type} N={result.n_components}:\n")
                f.write(f"  Simulations: {result.n_simulations:,}\n")
                f.write(f"  Training time: {result.training_time_seconds:.1f}s\n")
                if result.final_val_loss is not None:
                    f.write(f"  Final val loss: {result.final_val_loss:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
    
    logger.info(f"Saved training summary to {summary_path}")
