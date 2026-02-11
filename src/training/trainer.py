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
from ..simulator.augmentation import augment_weights_combined, augment_base_noise_level
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
        
        # Generate simulations
        theta, x, all_weights = self._generate_simulations(
            simulator, n_simulations, n_components, 
            flat_priors, model_type
        )
        
        # Save simulations for classifier
        if save_simulations_flag:
            output_dir = Path(self.config.training.save_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            sim_path = output_dir / f"simulations_{model_type}_n{n_components}.pt"
            save_simulations(
                sim_path, x, all_weights, theta,
                n_components, model_type, simulator.n_freq,
                metadata={'config': self.config.to_dict()}
            )
        
        # Convert to torch
        theta_t = torch.tensor(theta, dtype=torch.float32, device=self.device)
        x_t = torch.tensor(x, dtype=torch.float32, device=self.device)
        
        # Build embedding network
        input_dim = 2 * simulator.n_freq
        embedding_dim = self.config.sbi.embedding_dim
        embedding_net = SpectralEmbedding(
            input_dim=input_dim,
            output_dim=embedding_dim,
        ).to(self.device)
        
        # Get architecture config
        arch_config = self.config.sbi.get_architecture(n_components)
        
        # Build density estimator
        density_estimator_builder = posterior_nn(
            model=self.config.sbi.model,
            hidden_features=arch_config.hidden_features,
            num_transforms=arch_config.num_transforms,
            num_bins=self.config.sbi.num_bins,
            embedding_net=embedding_net,
        )
        
        logger.info(f"SBI Architecture: {self.config.sbi.model.upper()}")
        logger.info(f"  Hidden: {arch_config.hidden_features}, Transforms: {arch_config.num_transforms}")
        
        # Create SNPE inference
        inference = SNPE(
            prior=prior,
            density_estimator=density_estimator_builder,
            device=self.device,
        )
        
        # Append simulations
        inference.append_simulations(theta_t, x_t)
        
        # Train with proper settings from config
        start_time = datetime.now()
        
        # Get training parameters from config
        learning_rate = self.config.training.learning_rate
        training_batch_size = self.config.training.training_batch_size
        stop_after_epochs = self.config.training.stop_after_epochs
        
        logger.info(f"Training: batch_size={training_batch_size}, lr={learning_rate}, patience={stop_after_epochs}")
        
        density_estimator = inference.train(
            training_batch_size=training_batch_size,
            learning_rate=learning_rate,
            validation_fraction=self.config.training.validation_fraction,
            stop_after_epochs=stop_after_epochs,
            show_train_summary=True,
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract training history from SBI summary
        training_history = self._extract_training_history(inference)
        
        # Log training summary
        if training_history.get('train_loss'):
            logger.info(f"Training epochs: {len(training_history['train_loss'])}")
            logger.info(f"Final train loss: {training_history['train_loss'][-1]:.4f}")
        if training_history.get('val_loss'):
            best_val = min(training_history['val_loss'])
            best_epoch = training_history['val_loss'].index(best_val) + 1
            logger.info(f"Best validation loss: {best_val:.4f} (epoch {best_epoch})")
        
        # Build posterior
        posterior = inference.build_posterior(density_estimator)
        
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
    
    def _generate_simulations(
        self,
        simulator: RMSimulator,
        n_simulations: int,
        n_components: int,
        flat_priors: Dict[str, float],
        model_type: str,
    ) -> tuple:
        """Generate simulations with augmentation - VECTORIZED version."""
        
        # Sample parameters (flat_priors contains ALL bounds)
        theta = sample_prior(
            n_simulations, n_components, flat_priors,
            model_type=model_type
        )
        
        base_noise_level = self.config.noise.base_level
        
        logger.info(f"Generating {n_simulations} simulations...")
        
        # Process in batches for memory efficiency
        batch_size = min(1000, n_simulations)  # Process 1000 at a time
        
        xs = []
        all_weights = []
        
        for i in tqdm(range(0, n_simulations, batch_size), desc="Simulating"):
            batch_theta = theta[i:i + batch_size]
            actual_batch = len(batch_theta)
            
            # Generate augmented weights for batch
            batch_weights = np.array([
                augment_weights_combined(
                    simulator.weights,
                    scattered_prob=self.config.weight_augmentation.scattered_prob,
                    gap_prob=self.config.weight_augmentation.gap_prob,
                    large_block_prob=self.config.weight_augmentation.large_block_prob,
                    noise_variation=self.config.weight_augmentation.noise_variation,
                ) for _ in range(actual_batch)
            ])
            
            # Augment noise levels for batch
            if self.config.noise.augmentation_enable:
                noise_levels = np.array([
                    augment_base_noise_level(
                        base_noise_level,
                        self.config.noise.augmentation_min_factor,
                        self.config.noise.augmentation_max_factor,
                    ) for _ in range(actual_batch)
                ])
            else:
                noise_levels = np.full(actual_batch, base_noise_level)
            
            # Simulate entire batch at once (vectorized)
            batch_x = simulator.simulate_batch(batch_theta, batch_weights, noise_levels)
            
            xs.append(batch_x)
            all_weights.append(batch_weights)
        
        x = np.vstack(xs)
        all_weights = np.vstack(all_weights)
        
        logger.info(f"Generated {len(x)} simulations")
        
        return theta, x, all_weights
    
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
    max_components = config.model_selection.max_components
    model_types = config.physics.model_types
    
    logger.info(f"\n{'='*60}")
    logger.info("VROOM-SBI TRAINING")
    logger.info(f"{'='*60}")
    logger.info(f"Model types: {model_types}")
    logger.info(f"Max components: {max_components}")
    logger.info(f"Total models: {len(model_types) * max_components}")
    
    if not classifier_only:
        # Train all posterior models
        for model_type in model_types:
            for n_comp in range(1, max_components + 1):
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
