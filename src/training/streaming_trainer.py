"""
Custom Streaming NPE Trainer for VROOM-SBI.

This implements a custom training loop for Neural Posterior Estimation (NPE)
that truly streams data from disk, avoiding the RAM bottleneck of SBI's
native append_simulations() approach.

Based on official SBI tutorial:
https://sbi-dev.github.io/sbi/latest/advanced_tutorials/18_training_interface/

Key Features:
- True streaming from chunked .pt files (no full dataset in RAM)
- Custom PyTorch DataLoader with lazy chunk loading
- Uses SBI's density_estimator.loss() method (handles standardization internally)
- Early stopping with validation
- Learning rate scheduling
- Gradient clipping
- Compatible with SBI's DirectPosterior for inference

For single-round NPE:
- Loss is simply density_estimator.loss(theta, condition=x) 
- Returns negative log-likelihood per sample
- No proposal correction needed (unlike multi-round SNPE)
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from datetime import datetime
from copy import deepcopy
import logging
import bisect

# SBI imports for building the flow
from sbi.neural_nets.net_builders import build_nsf, build_maf
from sbi.inference.posteriors import DirectPosterior

logger = logging.getLogger(__name__)


class ChunkedSimulationDataset(Dataset):
    """
    PyTorch Dataset that streams from chunked .pt files.
    
    Instead of loading all data into RAM, this loads chunks on-demand.
    Uses a simple caching strategy to avoid re-loading the same chunk
    for consecutive accesses.
    
    Parameters
    ----------
    chunk_dir : Path
        Directory containing chunk_XXXX.pt files
    """
    
    def __init__(self, chunk_dir: Path):
        self.chunk_dir = Path(chunk_dir)
        
        # Find all chunk files
        self.chunk_files = sorted(self.chunk_dir.glob("chunk_*.pt"))
        if not self.chunk_files:
            raise ValueError(f"No chunk files found in {chunk_dir}")
        
        # Pre-compute cumulative lengths for index mapping
        self.chunk_lengths = []
        self.cumulative_lengths = [0]
        
        logger.info(f"Scanning {len(self.chunk_files)} chunk files...")
        for f in self.chunk_files:
            # Load just to get length
            data = torch.load(f, weights_only=True)
            length = len(data['theta'])
            self.chunk_lengths.append(length)
            self.cumulative_lengths.append(self.cumulative_lengths[-1] + length)
            del data
        
        self.total_length = self.cumulative_lengths[-1]
        logger.info(f"Total samples: {self.total_length:,} across {len(self.chunk_files)} chunks")
        
        # Simple cache for current chunk (per-worker)
        self._current_chunk_idx = None
        self._current_chunk_data = None
        
        # Get dimensions from first chunk
        first_chunk = torch.load(self.chunk_files[0], weights_only=True)
        self.theta_dim = first_chunk['theta'].shape[1]
        self.x_dim = first_chunk['x'].shape[1]
        del first_chunk
    
    def __len__(self):
        return self.total_length
    
    def __getitem__(self, idx):
        # Find which chunk contains this index
        chunk_idx = bisect.bisect_right(self.cumulative_lengths, idx) - 1
        local_idx = idx - self.cumulative_lengths[chunk_idx]
        
        # Load chunk if not cached
        if self._current_chunk_idx != chunk_idx:
            self._current_chunk_data = torch.load(
                self.chunk_files[chunk_idx], 
                weights_only=True
            )
            self._current_chunk_idx = chunk_idx
        
        theta = self._current_chunk_data['theta'][local_idx]
        x = self._current_chunk_data['x'][local_idx]
        
        return theta, x
    
    def get_sample_batch(self, n_samples: int = 1000) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get a sample batch for initializing the density estimator.
        
        SBI's build_nsf/build_maf need sample data to infer shapes
        and compute internal standardization statistics.
        """
        all_theta = []
        all_x = []
        samples_collected = 0
        
        for chunk_file in self.chunk_files:
            if samples_collected >= n_samples:
                break
            data = torch.load(chunk_file, weights_only=True)
            needed = min(n_samples - samples_collected, len(data['theta']))
            all_theta.append(data['theta'][:needed])
            all_x.append(data['x'][:needed])
            samples_collected += needed
            del data
        
        return torch.cat(all_theta, dim=0), torch.cat(all_x, dim=0)


class StreamingNPETrainer:
    """
    Custom streaming trainer for Neural Posterior Estimation.
    
    This bypasses SBI's append_simulations() to enable true streaming
    from disk, avoiding RAM bottlenecks for large datasets.
    
    Based on official SBI custom training tutorial.
    
    Parameters
    ----------
    device : str
        Training device ('cuda' or 'cpu')
    """
    
    def __init__(self, device: str = "cuda"):
        self.device = device if torch.cuda.is_available() else "cpu"
        self.density_estimator = None
        self.prior = None
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'learning_rates': [],
        }
    
    def build_density_estimator(
        self,
        theta_sample: torch.Tensor,
        x_sample: torch.Tensor,
        embedding_net: Optional[nn.Module] = None,
        flow_type: str = "nsf",
        hidden_features: int = 256,
        num_transforms: int = 15,
        num_bins: int = 16,
    ) -> nn.Module:
        """
        Build the normalizing flow density estimator.
        
        SBI's build functions handle standardization internally based on
        the sample data provided.
        
        Parameters
        ----------
        theta_sample : torch.Tensor
            Sample of theta for shape inference and standardization
        x_sample : torch.Tensor
            Sample of x for shape inference and standardization
        embedding_net : nn.Module, optional
            Network to embed x before conditioning
        flow_type : str
            Type of flow ('nsf' or 'maf')
        hidden_features : int
            Hidden layer size
        num_transforms : int
            Number of flow transforms
        num_bins : int
            Number of bins for spline flows
            
        Returns
        -------
        nn.Module
            The flow network
        """
        logger.info(f"Building {flow_type.upper()} density estimator...")
        logger.info(f"  theta shape: {theta_sample.shape}, x shape: {x_sample.shape}")
        
        # Build kwargs - only include embedding_net if it's not None
        # Passing embedding_net=None explicitly causes issues in some SBI versions
        build_kwargs = {
            'hidden_features': hidden_features,
            'num_transforms': num_transforms,
        }
        
        if embedding_net is not None:
            build_kwargs['embedding_net'] = embedding_net
        
        if flow_type.lower() == "nsf":
            build_kwargs['num_bins'] = num_bins
            self.density_estimator = build_nsf(
                theta_sample,
                x_sample,
                **build_kwargs
            )
        elif flow_type.lower() == "maf":
            self.density_estimator = build_maf(
                theta_sample,
                x_sample,
                **build_kwargs
            )
        else:
            raise ValueError(f"Unknown flow type: {flow_type}")
        
        self.density_estimator = self.density_estimator.to(self.device)
        
        # Count parameters
        n_params = sum(p.numel() for p in self.density_estimator.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {n_params:,}")
        
        return self.density_estimator
    
    def train(
        self,
        chunk_dir: Path,
        prior,
        embedding_net: Optional[nn.Module] = None,
        flow_type: str = "nsf",
        hidden_features: int = 256,
        num_transforms: int = 15,
        num_bins: int = 16,
        learning_rate: float = 5e-4,
        training_batch_size: int = 256,
        validation_fraction: float = 0.1,
        max_epochs: int = 500,
        stop_after_epochs: int = 20,
        clip_grad_norm: Optional[float] = 5.0,
        num_workers: int = 0,
        show_progress: bool = True,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train the NPE with streaming from disk.
        
        Following the official SBI tutorial pattern:
        1. Build density estimator with sample data
        2. Create train/val split
        3. Training loop with early stopping
        
        Parameters
        ----------
        chunk_dir : Path
            Directory containing chunk_XXXX.pt files
        prior : Distribution
            Prior distribution (stored for posterior building)
        embedding_net : nn.Module, optional
            Embedding network for x
        flow_type : str
            Type of flow ('nsf' or 'maf')
        hidden_features : int
            Hidden layer size
        num_transforms : int
            Number of flow transforms
        num_bins : int
            Number of bins for NSF
        learning_rate : float
            Initial learning rate
        training_batch_size : int
            Batch size for training
        validation_fraction : float
            Fraction of data for validation
        max_epochs : int
            Maximum number of epochs
        stop_after_epochs : int
            Early stopping patience
        clip_grad_norm : float, optional
            Gradient clipping threshold (None to disable)
        num_workers : int
            DataLoader workers (0 for main process)
        show_progress : bool
            Show progress bars
            
        Returns
        -------
        density_estimator : nn.Module
            Trained density estimator
        history : Dict[str, List[float]]
            Training history
        """
        chunk_dir = Path(chunk_dir)
        self.prior = prior
        
        # ================================================================
        # Step 1: Create dataset and get sample for density estimator init
        # ================================================================
        logger.info("Creating streaming dataset...")
        dataset = ChunkedSimulationDataset(chunk_dir)
        
        # Get sample for building the density estimator
        # SBI uses this for shape inference AND internal standardization
        n_sample = min(5000, len(dataset))
        theta_sample, x_sample = dataset.get_sample_batch(n_samples=n_sample)
        
        logger.info(f"Using {n_sample} samples for density estimator initialization")
        
        # ================================================================
        # Step 2: Build the density estimator
        # ================================================================
        self.build_density_estimator(
            theta_sample=theta_sample,
            x_sample=x_sample,
            embedding_net=embedding_net,
            flow_type=flow_type,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_bins=num_bins,
        )
        
        # Clean up sample data
        del theta_sample, x_sample
        
        # ================================================================
        # Step 3: Create train/val split using SBI's pattern
        # ================================================================
        num_examples = len(dataset)
        num_training_examples = int((1 - validation_fraction) * num_examples)
        num_validation_examples = num_examples - num_training_examples
        
        logger.info(f"Train samples: {num_training_examples:,}, Val samples: {num_validation_examples:,}")
        
        # Random permutation for train/val split
        permuted_indices = torch.randperm(num_examples)
        train_indices = permuted_indices[:num_training_examples]
        val_indices = permuted_indices[num_training_examples:]
        
        # Create data loaders following SBI pattern
        train_loader = DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_training_examples),
            drop_last=True,
            sampler=SubsetRandomSampler(train_indices.tolist()),
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        
        val_loader = DataLoader(
            dataset,
            batch_size=min(training_batch_size, num_validation_examples),
            shuffle=False,
            drop_last=True,
            sampler=SubsetRandomSampler(val_indices.tolist()),
            num_workers=num_workers,
            pin_memory=True if self.device == "cuda" else False,
        )
        
        # ================================================================
        # Step 4: Training loop (following official SBI tutorial exactly)
        # ================================================================
        optimizer = Adam(list(self.density_estimator.parameters()), lr=learning_rate)
        
        # Optional: learning rate scheduler
        scheduler = ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10, verbose=True
        )
        
        best_val_loss = float('inf')
        best_model_state_dict = None
        epochs_since_last_improvement = 0
        
        logger.info(f"Starting training: max_epochs={max_epochs}, patience={stop_after_epochs}")
        logger.info(f"Batch size: {training_batch_size}, LR: {learning_rate}")
        
        epoch = 0
        converged = False
        
        while epoch < max_epochs and not converged:
            # ============================================================
            # Training phase
            # ============================================================
            self.density_estimator.train()
            train_loss_sum = 0
            num_train_batches = 0
            
            pbar = tqdm(
                train_loader,
                desc=f"Epoch {epoch+1}/{max_epochs} [Train]",
                disable=not show_progress,
                leave=False,
            )
            
            for theta_batch, x_batch in pbar:
                theta_batch = theta_batch.to(self.device)
                x_batch = x_batch.to(self.device)
                
                optimizer.zero_grad()
                
                # Key: Use density_estimator.loss() as per SBI tutorial
                # This returns per-sample losses (negative log-likelihood)
                train_losses = self.density_estimator.loss(theta_batch, condition=x_batch)
                train_loss = torch.mean(train_losses)
                
                train_loss.backward()
                
                # Optional gradient clipping
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.density_estimator.parameters(), 
                        clip_grad_norm
                    )
                
                optimizer.step()
                
                train_loss_sum += train_losses.sum().item()
                num_train_batches += len(train_losses)
                
                pbar.set_postfix({'loss': f'{train_loss.item():.4f}'})
            
            train_loss_average = train_loss_sum / num_train_batches
            
            # ============================================================
            # Validation phase
            # ============================================================
            self.density_estimator.eval()
            val_loss_sum = 0
            num_val_batches = 0
            
            with torch.no_grad():
                for theta_batch, x_batch in tqdm(
                    val_loader,
                    desc=f"Epoch {epoch+1}/{max_epochs} [Val]",
                    disable=not show_progress,
                    leave=False,
                ):
                    theta_batch = theta_batch.to(self.device)
                    x_batch = x_batch.to(self.device)
                    
                    val_losses = self.density_estimator.loss(theta_batch, condition=x_batch)
                    val_loss_sum += val_losses.sum().item()
                    num_val_batches += len(val_losses)
            
            val_loss_average = val_loss_sum / num_val_batches
            
            epoch += 1
            
            # ============================================================
            # Record history and check for improvement
            # ============================================================
            self.training_history['train_loss'].append(train_loss_average)
            self.training_history['val_loss'].append(val_loss_average)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # Learning rate scheduling
            scheduler.step(val_loss_average)
            
            # Early stopping check
            if val_loss_average < best_val_loss:
                best_val_loss = val_loss_average
                epochs_since_last_improvement = 0
                # Save best model state
                best_model_state_dict = deepcopy(self.density_estimator.state_dict())
            else:
                epochs_since_last_improvement += 1
            
            # Log progress
            logger.info(
                f"Epoch {epoch}: train_loss={train_loss_average:.4f}, "
                f"val_loss={val_loss_average:.4f}, "
                f"best_val={best_val_loss:.4f}, "
                f"patience={epochs_since_last_improvement}/{stop_after_epochs}"
            )
            
            # Check for convergence
            if epochs_since_last_improvement > stop_after_epochs - 1:
                self.density_estimator.load_state_dict(best_model_state_dict)
                converged = True
                logger.info(f"Neural network converged after {epoch} epochs")
        
        if not converged:
            logger.info(f"Reached max epochs ({max_epochs})")
            if best_model_state_dict is not None:
                self.density_estimator.load_state_dict(best_model_state_dict)
        
        logger.info(f"Training complete. Best val loss: {best_val_loss:.4f}")
        
        return self.density_estimator, self.training_history
    
    def build_posterior(self, prior=None) -> DirectPosterior:
        """
        Build SBI DirectPosterior from trained density estimator.
        
        Following the official SBI tutorial:
        posterior = DirectPosterior(density_estimator, prior)
        
        The DirectPosterior adds:
        - Automatic rejection of samples outside prior bounds
        - MAP estimation capability
        
        Parameters
        ----------
        prior : Distribution, optional
            The prior distribution (uses stored prior if not provided)
            
        Returns
        -------
        DirectPosterior
            Posterior ready for sampling
        """
        if self.density_estimator is None:
            raise RuntimeError("No density estimator trained yet")
        
        if prior is None:
            prior = self.prior
        
        posterior = DirectPosterior(
            posterior_estimator=self.density_estimator,
            prior=prior,
        )
        
        return posterior
    
    def save(self, path: Path, extra_state: Dict[str, Any] = None):
        """
        Save trained model.
        
        Parameters
        ----------
        path : Path
            Save path
        extra_state : Dict, optional
            Additional state to save
        """
        save_dict = {
            'density_estimator_state': self.density_estimator.state_dict() 
                if self.density_estimator else None,
            'training_history': self.training_history,
            'device': self.device,
        }
        
        if extra_state:
            save_dict.update(extra_state)
        
        torch.save(save_dict, path)
        logger.info(f"Saved model to {path}")
    
    def load(self, path: Path):
        """
        Load trained model state.
        
        Note: You must rebuild the density estimator first with
        the same architecture before loading.
        
        Parameters
        ----------
        path : Path
            Load path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        if checkpoint.get('density_estimator_state') and self.density_estimator:
            self.density_estimator.load_state_dict(checkpoint['density_estimator_state'])
        
        self.training_history = checkpoint.get('training_history', {})
        
        logger.info(f"Loaded model from {path}")
        return checkpoint


def train_streaming_npe(
    chunk_dir: Path,
    prior,
    embedding_net: Optional[nn.Module] = None,
    device: str = "cuda",
    **train_kwargs
) -> Tuple[nn.Module, "DirectPosterior", Dict]:
    """
    Convenience function to train streaming NPE.
    
    Parameters
    ----------
    chunk_dir : Path
        Directory with chunk files
    prior : Distribution
        Prior distribution
    embedding_net : nn.Module, optional
        Embedding network
    device : str
        Training device
    **train_kwargs
        Additional arguments to StreamingNPETrainer.train()
        
    Returns
    -------
    density_estimator : nn.Module
        Trained density estimator
    posterior : DirectPosterior
        Ready-to-use posterior
    history : Dict
        Training history
    """
    trainer = StreamingNPETrainer(device=device)
    
    density_estimator, history = trainer.train(
        chunk_dir=chunk_dir,
        prior=prior,
        embedding_net=embedding_net,
        **train_kwargs
    )
    
    posterior = trainer.build_posterior(prior)
    
    return density_estimator, posterior, history
