"""
Checkpoint management for VROOM-SBI training.

Handles saving/loading of model checkpoints using torch.save().
Supports both full checkpoints and lightweight state dicts.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional, List
from datetime import datetime
import torch
import torch.nn as nn
import json
import logging

logger = logging.getLogger(__name__)


@dataclass
class ModelCheckpoint:
    """
    Container for a model checkpoint.
    
    Stores all necessary information to resume training.
    """
    # Model info
    model_type: str
    n_components: int
    n_params: int
    n_freq: int
    
    # State
    epoch: int
    state_dict: Dict[str, torch.Tensor]
    optimizer_state: Optional[Dict[str, Any]] = None
    scheduler_state: Optional[Dict[str, Any]] = None
    
    # Metrics
    train_loss: float = 0.0
    val_loss: Optional[float] = None
    best_val_loss: float = float('inf')
    
    # Training history
    loss_history: Dict[str, List[float]] = field(default_factory=dict)
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    
    # SBI-specific
    prior_bounds: Optional[Dict[str, List[float]]] = None
    embedding_net_state: Optional[Dict[str, torch.Tensor]] = None


class CheckpointManager:
    """
    Manages model checkpoints for training.
    
    Features:
    - Save checkpoints with torch.save()
    - Load checkpoints for resuming training
    - Keep track of best model
    - Automatic cleanup of old checkpoints
    """
    
    def __init__(
        self,
        checkpoint_dir: str,
        max_checkpoints: int = 5,
        save_best_only: bool = False,
    ):
        """
        Initialize checkpoint manager.
        
        Parameters
        ----------
        checkpoint_dir : str
            Directory for checkpoints
        max_checkpoints : int
            Maximum number of checkpoints to keep (per model)
        save_best_only : bool
            If True, only save when validation improves
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        
        # Track best models
        self.best_val_loss: Dict[str, float] = {}
        
        # Track all checkpoints for cleanup
        self.checkpoint_files: Dict[str, List[Path]] = {}
    
    def _get_checkpoint_path(
        self, 
        model_type: str, 
        n_components: int, 
        epoch: int,
        is_best: bool = False
    ) -> Path:
        """Generate checkpoint filename."""
        if is_best:
            return self.checkpoint_dir / f"best_{model_type}_n{n_components}.pt"
        return self.checkpoint_dir / f"checkpoint_{model_type}_n{n_components}_epoch{epoch:04d}.pt"
    
    def _get_model_key(self, model_type: str, n_components: int) -> str:
        """Get unique key for model type + n_components."""
        return f"{model_type}_n{n_components}"
    
    def save_checkpoint(
        self,
        checkpoint: ModelCheckpoint,
        is_best: bool = False,
    ) -> Path:
        """
        Save a checkpoint to disk.
        
        Parameters
        ----------
        checkpoint : ModelCheckpoint
            Checkpoint to save
        is_best : bool
            Whether this is the best model so far
            
        Returns
        -------
        Path
            Path to saved checkpoint
        """
        model_key = self._get_model_key(checkpoint.model_type, checkpoint.n_components)
        
        # Prepare data for torch.save
        save_dict = {
            'model_type': checkpoint.model_type,
            'n_components': checkpoint.n_components,
            'n_params': checkpoint.n_params,
            'n_freq': checkpoint.n_freq,
            'epoch': checkpoint.epoch,
            'state_dict': checkpoint.state_dict,
            'optimizer_state': checkpoint.optimizer_state,
            'scheduler_state': checkpoint.scheduler_state,
            'train_loss': checkpoint.train_loss,
            'val_loss': checkpoint.val_loss,
            'best_val_loss': checkpoint.best_val_loss,
            'loss_history': checkpoint.loss_history,
            'timestamp': checkpoint.timestamp.isoformat(),
            'config_snapshot': checkpoint.config_snapshot,
            'prior_bounds': checkpoint.prior_bounds,
            'embedding_net_state': checkpoint.embedding_net_state,
        }
        
        # Determine path
        checkpoint_path = self._get_checkpoint_path(
            checkpoint.model_type,
            checkpoint.n_components,
            checkpoint.epoch,
            is_best=is_best
        )
        
        # Save with torch
        torch.save(save_dict, checkpoint_path)
        logger.info(f"Saved checkpoint: {checkpoint_path}")
        
        # Track for cleanup (only regular checkpoints, not best)
        if not is_best:
            if model_key not in self.checkpoint_files:
                self.checkpoint_files[model_key] = []
            self.checkpoint_files[model_key].append(checkpoint_path)
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints(model_key)
        
        # Save best separately
        if is_best:
            best_path = self._get_checkpoint_path(
                checkpoint.model_type,
                checkpoint.n_components,
                checkpoint.epoch,
                is_best=True
            )
            torch.save(save_dict, best_path)
            logger.info(f"Saved best model: {best_path}")
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model_type: str,
        n_components: int,
        epoch: Optional[int] = None,
        load_best: bool = False,
    ) -> Optional[ModelCheckpoint]:
        """
        Load a checkpoint from disk.
        
        Parameters
        ----------
        model_type : str
            Physical model type
        n_components : int
            Number of components
        epoch : int, optional
            Specific epoch to load. If None, loads latest.
        load_best : bool
            If True, load the best model instead
            
        Returns
        -------
        ModelCheckpoint or None
            Loaded checkpoint, or None if not found
        """
        if load_best:
            checkpoint_path = self._get_checkpoint_path(model_type, n_components, 0, is_best=True)
        elif epoch is not None:
            checkpoint_path = self._get_checkpoint_path(model_type, n_components, epoch)
        else:
            # Find latest
            model_key = self._get_model_key(model_type, n_components)
            if model_key in self.checkpoint_files and self.checkpoint_files[model_key]:
                checkpoint_path = max(self.checkpoint_files[model_key])
            else:
                # Search directory
                pattern = f"checkpoint_{model_type}_n{n_components}_epoch*.pt"
                matches = list(self.checkpoint_dir.glob(pattern))
                if not matches:
                    return None
                checkpoint_path = max(matches)
        
        if not checkpoint_path.exists():
            logger.warning(f"Checkpoint not found: {checkpoint_path}")
            return None
        
        # Load with torch
        save_dict = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        
        # Reconstruct checkpoint
        checkpoint = ModelCheckpoint(
            model_type=save_dict['model_type'],
            n_components=save_dict['n_components'],
            n_params=save_dict['n_params'],
            n_freq=save_dict['n_freq'],
            epoch=save_dict['epoch'],
            state_dict=save_dict['state_dict'],
            optimizer_state=save_dict.get('optimizer_state'),
            scheduler_state=save_dict.get('scheduler_state'),
            train_loss=save_dict.get('train_loss', 0.0),
            val_loss=save_dict.get('val_loss'),
            best_val_loss=save_dict.get('best_val_loss', float('inf')),
            loss_history=save_dict.get('loss_history', {}),
            timestamp=datetime.fromisoformat(save_dict.get('timestamp', datetime.now().isoformat())),
            config_snapshot=save_dict.get('config_snapshot', {}),
            prior_bounds=save_dict.get('prior_bounds'),
            embedding_net_state=save_dict.get('embedding_net_state'),
        )
        
        logger.info(f"Loaded checkpoint: {checkpoint_path} (epoch {checkpoint.epoch})")
        return checkpoint
    
    def _cleanup_old_checkpoints(self, model_key: str):
        """Remove old checkpoints beyond max_checkpoints."""
        if model_key not in self.checkpoint_files:
            return
            
        files = self.checkpoint_files[model_key]
        if len(files) <= self.max_checkpoints:
            return
        
        # Sort by modification time
        files.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0)
        
        # Remove oldest
        while len(files) > self.max_checkpoints:
            old_file = files.pop(0)
            if old_file.exists():
                old_file.unlink()
                logger.debug(f"Removed old checkpoint: {old_file}")
    
    def has_checkpoint(self, model_type: str, n_components: int) -> bool:
        """Check if any checkpoint exists for this model."""
        best_path = self._get_checkpoint_path(model_type, n_components, 0, is_best=True)
        if best_path.exists():
            return True
        
        pattern = f"checkpoint_{model_type}_n{n_components}_epoch*.pt"
        matches = list(self.checkpoint_dir.glob(pattern))
        return len(matches) > 0
    
    def get_latest_epoch(self, model_type: str, n_components: int) -> int:
        """Get the latest checkpoint epoch number."""
        pattern = f"checkpoint_{model_type}_n{n_components}_epoch*.pt"
        matches = list(self.checkpoint_dir.glob(pattern))
        
        if not matches:
            return 0
        
        # Extract epoch from filename
        latest = max(matches)
        try:
            epoch_str = latest.stem.split('_epoch')[-1]
            return int(epoch_str)
        except (ValueError, IndexError):
            return 0
    
    def list_checkpoints(self, model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        checkpoints = []
        
        pattern = "checkpoint_*.pt" if model_type is None else f"checkpoint_{model_type}_*.pt"
        
        for path in self.checkpoint_dir.glob(pattern):
            # Load minimal info
            try:
                save_dict = torch.load(path, map_location='cpu', weights_only=False)
                checkpoints.append({
                    'path': str(path),
                    'model_type': save_dict.get('model_type'),
                    'n_components': save_dict.get('n_components'),
                    'epoch': save_dict.get('epoch'),
                    'val_loss': save_dict.get('val_loss'),
                    'timestamp': save_dict.get('timestamp'),
                })
            except Exception as e:
                logger.warning(f"Could not read checkpoint {path}: {e}")
        
        return sorted(checkpoints, key=lambda x: (x['model_type'], x['n_components'], x['epoch']))


def save_training_plots(
    loss_history: Dict[str, List[float]],
    output_path: Path,
    model_type: str,
    n_components: int,
):
    """
    Save training loss plots.
    
    Parameters
    ----------
    loss_history : dict
        Dictionary with 'train_loss' and optionally 'val_loss' lists
    output_path : Path
        Where to save the plot
    model_type : str
        Physical model type
    n_components : int
        Number of components
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if we have any data
    has_train = 'train_loss' in loss_history and loss_history['train_loss']
    has_val = 'val_loss' in loss_history and loss_history['val_loss']
    
    if not has_train and not has_val:
        logger.warning(f"No training history to plot for {model_type} N={n_components}")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # --- Left plot: Loss curves ---
    ax1 = axes[0]
    
    if has_train:
        train_loss = loss_history['train_loss']
        epochs = range(1, len(train_loss) + 1)
        ax1.plot(epochs, train_loss, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    
    if has_val:
        val_loss = loss_history['val_loss']
        epochs_val = range(1, len(val_loss) + 1)
        ax1.plot(epochs_val, val_loss, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        
        # Mark best validation
        best_idx = np.argmin(val_loss)
        ax1.axvline(x=best_idx + 1, color='g', linestyle='--', alpha=0.5, label=f'Best (epoch {best_idx + 1})')
        ax1.scatter([best_idx + 1], [val_loss[best_idx]], color='g', s=100, zorder=5, marker='*')
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Negative Log Probability (Loss)', fontsize=12)
    ax1.set_title(f'Training Progress: {model_type} (N={n_components})', fontsize=14)
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # --- Right plot: Convergence analysis ---
    ax2 = axes[1]
    
    if has_val and len(val_loss) > 1:
        # Plot smoothed validation loss
        window = min(5, len(val_loss) // 3) if len(val_loss) > 5 else 1
        if window > 1:
            smoothed = np.convolve(val_loss, np.ones(window)/window, mode='valid')
            epochs_smooth = range(window, len(val_loss) + 1)
            ax2.plot(epochs_smooth, smoothed, 'r-', linewidth=2, label=f'Smoothed Val (window={window})')
        
        # Calculate improvement rate
        improvements = np.diff(val_loss)
        ax2.bar(range(2, len(val_loss) + 1), -improvements, alpha=0.5, color='blue', label='Epoch Improvement')
        ax2.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Improvement (decrease in loss)', fontsize=12)
        ax2.set_title('Convergence Analysis', fontsize=14)
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'Not enough data\nfor convergence analysis',
                 ha='center', va='center', transform=ax2.transAxes, fontsize=14)
        ax2.set_title('Convergence Analysis', fontsize=14)
    
    # Add summary text box
    summary_text = []
    if has_train:
        summary_text.append(f"Final Train Loss: {train_loss[-1]:.4f}")
    if has_val:
        summary_text.append(f"Final Val Loss: {val_loss[-1]:.4f}")
        summary_text.append(f"Best Val Loss: {min(val_loss):.4f}")
        summary_text.append(f"Best Epoch: {np.argmin(val_loss) + 1}")
        summary_text.append(f"Total Epochs: {len(val_loss)}")
    
    if summary_text:
        fig.text(0.02, 0.02, '\n'.join(summary_text), fontsize=9,
                 verticalalignment='bottom', fontfamily='monospace',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2)  # Make room for summary text
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    logger.info(f"Saved training plot: {output_path}")
