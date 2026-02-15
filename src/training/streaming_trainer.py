"""
Async Streaming NPE Trainer for VROOM-SBI.

Implements Neural Posterior Estimation with async chunk streaming:
- GPU trains on current chunk while CPU prefetches next chunk
- Memory bounded: only 2 chunks in RAM at any time
- User controls chunk size via config (simulation_batch_size)

Based on official SBI tutorial:
https://sbi-dev.github.io/sbi/latest/advanced_tutorials/18_training_interface/
"""

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from tqdm import tqdm
from copy import deepcopy
from concurrent.futures import ThreadPoolExecutor
import logging
import random

from sbi.neural_nets.net_builders import build_nsf, build_maf
from sbi.inference.posteriors import DirectPosterior

logger = logging.getLogger(__name__)


def load_chunk(chunk_path: Path) -> Dict[str, torch.Tensor]:
    """Load a chunk from disk. Runs in background thread."""
    return torch.load(chunk_path, weights_only=True)


class AsyncChunkStreamer:
    """
    Async streaming iterator over chunked simulation files.
    
    While GPU trains on current chunk, CPU loads next chunk in background.
    
    Memory usage: ~2 chunks at peak (current + prefetched)
    
    Parameters
    ----------
    chunk_files : List[Path]
        List of chunk file paths
    batch_size : int
        Batch size for training
    device : str
        Device to move batches to
    shuffle_chunks : bool
        Whether to shuffle chunk order each iteration
    shuffle_within_chunk : bool
        Whether to shuffle samples within each chunk
    drop_last : bool
        Whether to drop last incomplete batch
    """
    
    def __init__(
        self,
        chunk_files: List[Path],
        batch_size: int,
        device: str = "cuda",
        shuffle_chunks: bool = True,
        shuffle_within_chunk: bool = True,
        drop_last: bool = True,
    ):
        self.chunk_files = list(chunk_files)
        self.batch_size = batch_size
        self.device = device
        self.shuffle_chunks = shuffle_chunks
        self.shuffle_within_chunk = shuffle_within_chunk
        self.drop_last = drop_last
        
        # Get total samples for progress tracking
        self.total_samples = 0
        self.chunk_sizes = []
        for f in self.chunk_files:
            data = torch.load(f, weights_only=True)
            self.chunk_sizes.append(len(data['theta']))
            self.total_samples += len(data['theta'])
            del data
    
    def __len__(self):
        """Approximate number of batches."""
        if self.drop_last:
            return sum(size // self.batch_size for size in self.chunk_sizes)
        else:
            return sum((size + self.batch_size - 1) // self.batch_size for size in self.chunk_sizes)
    
    def __iter__(self):
        """
        Iterate over batches with async chunk prefetching.
        
        Pattern:
            1. Load first chunk
            2. While training on chunk N, prefetch chunk N+1
            3. Yield batches from current chunk
            4. Swap: current = prefetched, start loading next
        """
        # Determine chunk order
        if self.shuffle_chunks:
            chunk_order = random.sample(range(len(self.chunk_files)), len(self.chunk_files))
        else:
            chunk_order = list(range(len(self.chunk_files)))
        
        # Thread pool for async loading
        executor = ThreadPoolExecutor(max_workers=1)
        
        # Load first chunk synchronously
        current_chunk = load_chunk(self.chunk_files[chunk_order[0]])
        
        # Start prefetching second chunk if exists
        prefetch_future = None
        if len(chunk_order) > 1:
            prefetch_future = executor.submit(load_chunk, self.chunk_files[chunk_order[1]])
        
        try:
            for i, chunk_idx in enumerate(chunk_order):
                # Get current chunk data
                theta = current_chunk['theta']
                x = current_chunk['x']
                
                # Shuffle within chunk
                if self.shuffle_within_chunk:
                    perm = torch.randperm(len(theta))
                    theta = theta[perm]
                    x = x[perm]
                
                # Yield batches from current chunk
                n_samples = len(theta)
                n_batches = n_samples // self.batch_size
                
                for batch_idx in range(n_batches):
                    start = batch_idx * self.batch_size
                    end = start + self.batch_size
                    
                    theta_batch = theta[start:end].to(self.device)
                    x_batch = x[start:end].to(self.device)
                    
                    yield theta_batch, x_batch
                
                # Handle last incomplete batch
                if not self.drop_last and n_samples % self.batch_size > 0:
                    start = n_batches * self.batch_size
                    theta_batch = theta[start:].to(self.device)
                    x_batch = x[start:].to(self.device)
                    yield theta_batch, x_batch
                
                # Free current chunk memory
                del current_chunk, theta, x
                
                # Get prefetched chunk (blocks if not ready)
                if prefetch_future is not None:
                    current_chunk = prefetch_future.result()
                    
                    # Start prefetching next chunk
                    if i + 2 < len(chunk_order):
                        prefetch_future = executor.submit(
                            load_chunk, self.chunk_files[chunk_order[i + 2]]
                        )
                    else:
                        prefetch_future = None
        
        finally:
            executor.shutdown(wait=False)


class StreamingNPETrainer:
    """
    Async streaming trainer for Neural Posterior Estimation.
    
    Trains on chunked simulation files with async prefetching:
    - GPU trains on current chunk while CPU loads next chunk
    - Memory: only 2 chunks in RAM at peak
    - User controls chunk size via simulation_batch_size in config
    
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
        """
        logger.info(f"Building {flow_type.upper()} density estimator...")
        logger.info(f"  theta shape: {theta_sample.shape}, x shape: {x_sample.shape}")
        
        # Build kwargs - only include embedding_net if provided
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
        
        n_params = sum(p.numel() for p in self.density_estimator.parameters() if p.requires_grad)
        logger.info(f"  Parameters: {n_params:,}")
        
        return self.density_estimator
    
    def _get_sample_from_chunks(
        self, 
        chunk_files: List[Path], 
        n_samples: int = 5000
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get sample data from chunks for density estimator initialization."""
        all_theta = []
        all_x = []
        collected = 0
        
        for chunk_file in chunk_files:
            if collected >= n_samples:
                break
            data = torch.load(chunk_file, weights_only=True)
            needed = min(n_samples - collected, len(data['theta']))
            all_theta.append(data['theta'][:needed])
            all_x.append(data['x'][:needed])
            collected += needed
            del data
        
        return torch.cat(all_theta, dim=0), torch.cat(all_x, dim=0)
    
    def _compute_validation_loss(
        self,
        val_chunks: List[Path],
        batch_size: int,
    ) -> float:
        """Compute validation loss over validation chunks."""
        self.density_estimator.eval()
        
        val_loss_sum = 0.0
        n_samples = 0
        
        with torch.no_grad():
            for chunk_file in val_chunks:
                data = torch.load(chunk_file, weights_only=True)
                theta = data['theta']
                x = data['x']
                
                # Process in batches
                for i in range(0, len(theta), batch_size):
                    theta_batch = theta[i:i+batch_size].to(self.device)
                    x_batch = x[i:i+batch_size].to(self.device)
                    
                    losses = self.density_estimator.loss(theta_batch, condition=x_batch)
                    val_loss_sum += losses.sum().item()
                    n_samples += len(losses)
                
                del data, theta, x
        
        return val_loss_sum / n_samples if n_samples > 0 else float('inf')
    
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
        show_progress: bool = True,
    ) -> Tuple[nn.Module, Dict[str, List[float]]]:
        """
        Train NPE with async chunk streaming.
        
        Each epoch:
            1. Shuffle chunk order
            2. Stream through all training chunks (async prefetch)
            3. Compute validation loss
            4. Check early stopping
        
        Parameters
        ----------
        chunk_dir : Path
            Directory containing chunk_XXXX.pt files
        prior : Distribution
            Prior distribution
        embedding_net : nn.Module, optional
            Embedding network for observations
        flow_type : str
            'nsf' or 'maf'
        hidden_features : int
            Hidden layer size in flow
        num_transforms : int
            Number of flow transforms
        num_bins : int
            Spline bins (NSF only)
        learning_rate : float
            Learning rate
        training_batch_size : int
            Batch size for GPU training
        validation_fraction : float
            Fraction of chunks for validation
        max_epochs : int
            Maximum epochs
        stop_after_epochs : int
            Early stopping patience
        clip_grad_norm : float, optional
            Gradient clipping (None to disable)
        show_progress : bool
            Show progress bars
        
        Returns
        -------
        density_estimator : nn.Module
            Trained model
        history : dict
            Training history
        """
        chunk_dir = Path(chunk_dir)
        self.prior = prior
        
        # ================================================================
        # Step 1: Discover chunks and split train/val
        # ================================================================
        all_chunks = sorted(chunk_dir.glob("chunk_*.pt"))
        if not all_chunks:
            raise ValueError(f"No chunk files found in {chunk_dir}")
        
        # Split at chunk level (simpler, consistent)
        n_val_chunks = max(1, int(len(all_chunks) * validation_fraction))
        
        # Shuffle before split to ensure randomness
        shuffled_chunks = random.sample(all_chunks, len(all_chunks))
        val_chunks = shuffled_chunks[:n_val_chunks]
        train_chunks = shuffled_chunks[n_val_chunks:]
        
        # Count samples
        train_samples = sum(
            len(torch.load(f, weights_only=True)['theta']) 
            for f in train_chunks
        )
        val_samples = sum(
            len(torch.load(f, weights_only=True)['theta']) 
            for f in val_chunks
        )
        
        logger.info(f"Chunks: {len(train_chunks)} train, {len(val_chunks)} val")
        logger.info(f"Samples: {train_samples:,} train, {val_samples:,} val")
        
        # ================================================================
        # Step 2: Build density estimator with sample data
        # ================================================================
        n_init_samples = min(5000, train_samples)
        theta_sample, x_sample = self._get_sample_from_chunks(train_chunks, n_init_samples)
        
        self.build_density_estimator(
            theta_sample=theta_sample,
            x_sample=x_sample,
            embedding_net=embedding_net,
            flow_type=flow_type,
            hidden_features=hidden_features,
            num_transforms=num_transforms,
            num_bins=num_bins,
        )
        
        del theta_sample, x_sample
        
        # ================================================================
        # Step 3: Setup optimizer and scheduler
        # ================================================================
        optimizer = Adam(self.density_estimator.parameters(), lr=learning_rate)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)
        
        best_val_loss = float('inf')
        best_state = None
        epochs_without_improvement = 0
        
        logger.info(f"Starting async streaming training")
        logger.info(f"  Max epochs: {max_epochs}, Patience: {stop_after_epochs}")
        logger.info(f"  Batch size: {training_batch_size}, LR: {learning_rate}")
        logger.info(f"  Device: {self.device}")
        
        # ================================================================
        # Step 4: Training loop with async chunk streaming
        # ================================================================
        for epoch in range(max_epochs):
            # ----------------------------------------------------------
            # Training phase
            # ----------------------------------------------------------
            self.density_estimator.train()
            train_loss_sum = 0.0
            n_train_samples = 0
            
            # Create async streamer for this epoch
            train_streamer = AsyncChunkStreamer(
                chunk_files=train_chunks,
                batch_size=training_batch_size,
                device=self.device,
                shuffle_chunks=True,
                shuffle_within_chunk=True,
                drop_last=True,
            )
            
            pbar = tqdm(
                train_streamer,
                total=len(train_streamer),
                desc=f"Epoch {epoch+1}/{max_epochs}",
                disable=not show_progress,
                leave=False,
            )
            
            for theta_batch, x_batch in pbar:
                optimizer.zero_grad()
                
                # Forward pass
                losses = self.density_estimator.loss(theta_batch, condition=x_batch)
                loss = losses.mean()
                
                # Backward pass
                loss.backward()
                
                if clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(
                        self.density_estimator.parameters(),
                        clip_grad_norm
                    )
                
                optimizer.step()
                
                train_loss_sum += losses.sum().item()
                n_train_samples += len(losses)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            train_loss = train_loss_sum / n_train_samples
            
            # ----------------------------------------------------------
            # Validation phase
            # ----------------------------------------------------------
            val_loss = self._compute_validation_loss(val_chunks, training_batch_size)
            
            # ----------------------------------------------------------
            # Record history
            # ----------------------------------------------------------
            self.training_history['train_loss'].append(train_loss)
            self.training_history['val_loss'].append(val_loss)
            self.training_history['learning_rates'].append(optimizer.param_groups[0]['lr'])
            
            # LR scheduling
            scheduler.step(val_loss)
            
            # ----------------------------------------------------------
            # Early stopping check
            # ----------------------------------------------------------
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = deepcopy(self.density_estimator.state_dict())
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            logger.info(
                f"Epoch {epoch+1}: train={train_loss:.4f}, val={val_loss:.4f}, "
                f"best={best_val_loss:.4f}, patience={epochs_without_improvement}/{stop_after_epochs}"
            )
            
            if epochs_without_improvement >= stop_after_epochs:
                logger.info(f"Early stopping at epoch {epoch+1}")
                break
        
        # Restore best model
        if best_state is not None:
            self.density_estimator.load_state_dict(best_state)
            logger.info(f"Restored best model (val_loss={best_val_loss:.4f})")
        
        return self.density_estimator, self.training_history
    
    def build_posterior(self, prior=None) -> DirectPosterior:
        """
        Build SBI DirectPosterior from trained density estimator.
        """
        if self.density_estimator is None:
            raise RuntimeError("No density estimator trained yet")
        
        if prior is None:
            prior = self.prior
        
        return DirectPosterior(
            posterior_estimator=self.density_estimator,
            prior=prior,
        )
    
    def save(self, path: Path, extra_state: Dict[str, Any] = None):
        """Save trained model."""
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
        """Load trained model state."""
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
) -> Tuple[nn.Module, DirectPosterior, Dict]:
    """
    Convenience function to train streaming NPE.
    
    Returns
    -------
    density_estimator : nn.Module
    posterior : DirectPosterior
    history : dict
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
