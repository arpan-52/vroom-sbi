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
        
        # Train
        start_time = datetime.now()
        
        density_estimator = inference.train(
            training_batch_size=min(4096, max(1, self.config.training.batch_size)),
            learning_rate=5e-4,
            show_train_summary=True,
            validation_fraction=self.config.training.validation_fraction,
        )
        
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Extract training history
        training_history = self._extract_training_history(inference)
        
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
        if training_history:
            plot_path = output_dir / f"training_{model_type}_n{n_components}.png"
            save_training_plots(training_history, plot_path, model_type, n_components)
        
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
        """Generate simulations with augmentation."""
        
        # Sample parameters (flat_priors contains ALL bounds)
        theta = sample_prior(
            n_simulations, n_components, flat_priors,
            model_type=model_type
        )
        
        batch_size = self.config.training.batch_size
        base_noise_level = self.config.noise.base_level
        
        xs = []
        all_weights = []
        
        logger.info("Generating simulations...")
        
        for i in tqdm(range(0, n_simulations, batch_size), desc="Simulating"):
            batch_theta = theta[i:i + batch_size]
            batch_size_actual = len(batch_theta)
            
            # Generate augmented weights
            augmented_weights = np.zeros((batch_size_actual, simulator.n_freq))
            for j in range(batch_size_actual):
                augmented_weights[j] = augment_weights_combined(
                    simulator.weights,
                    scattered_prob=self.config.weight_augmentation.scattered_prob,
                    gap_prob=self.config.weight_augmentation.gap_prob,
                    large_block_prob=self.config.weight_augmentation.large_block_prob,
                    noise_variation=self.config.weight_augmentation.noise_variation,
                )
            
            # Simulate with augmented noise levels
            batch_xs = []
            for j in range(batch_size_actual):
                # Augment noise level
                if self.config.noise.augmentation_enable:
                    aug_noise = augment_base_noise_level(
                        base_noise_level,
                        self.config.noise.augmentation_min_factor,
                        self.config.noise.augmentation_max_factor,
                    )
                    simulator.base_noise_level = aug_noise
                
                x_sample = simulator(batch_theta[j:j+1], weights=augmented_weights[j])
                batch_xs.append(x_sample)
            
            xs.append(np.vstack(batch_xs))
            all_weights.append(augmented_weights)
        
        # Restore original noise level
        simulator.base_noise_level = base_noise_level
        
        x = np.vstack(xs)
        all_weights = np.vstack(all_weights)
        
        return theta, x, all_weights
    
    def _extract_training_history(self, inference) -> Dict[str, List[float]]:
        """Extract training history from SBI inference object."""
        history = {}
        
        try:
            # Try different SBI versions' APIs
            summary = None
            
            # Newer SBI (0.22+)
            if hasattr(inference, 'summary'):
                summary = inference.summary
            # Older SBI
            elif hasattr(inference, '_summary'):
                summary = inference._summary
            
            if summary is not None:
                # Extract training log probs (negative = loss)
                if 'training_log_probs' in summary:
                    history['train_loss'] = [-x for x in summary['training_log_probs']]
                elif 'train_log_probs' in summary:
                    history['train_loss'] = [-x for x in summary['train_log_probs']]
                
                # Extract validation log probs
                if 'validation_log_probs' in summary:
                    history['val_loss'] = [-x for x in summary['validation_log_probs']]
                
                # Extract epochs
                if 'epochs_trained' in summary:
                    history['epochs'] = list(range(1, summary['epochs_trained'] + 1))
                elif 'epochs' in summary:
                    history['epochs'] = summary['epochs']
                
                # Extract best validation
                if 'best_validation_log_prob' in summary:
                    history['best_val_loss'] = -summary['best_validation_log_prob']
                    
            # If no history found, log it
            if not history:
                logger.warning("Could not extract training history from SBI inference object")
                
        except Exception as e:
            logger.warning(f"Could not extract training history: {e}")
        
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
