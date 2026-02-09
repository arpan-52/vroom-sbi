"""
Classifier Trainer for VROOM-SBI model selection.

Trains a CNN classifier to predict the number of components
and model type directly from observed spectra.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import logging

from ..config import Configuration, ClassifierConfig
from .networks import SpectralClassifier
from .data_loader import prepare_classifier_data

logger = logging.getLogger(__name__)


class ClassifierTrainer:
    """
    Trainer for the model selection classifier.
    
    Parameters
    ----------
    n_freq : int
        Number of frequency channels
    n_classes : int
        Number of model classes
    config : ClassifierConfig
        Classifier configuration
    device : str
        Training device
    """
    
    def __init__(
        self,
        n_freq: int,
        n_classes: int,
        config: Optional[ClassifierConfig] = None,
        device: str = 'cpu',
    ):
        self.n_freq = n_freq
        self.n_classes = n_classes
        self.device = device
        
        if config is None:
            config = ClassifierConfig()
        self.config = config
        
        # Build model
        self.model = SpectralClassifier(
            n_freq=n_freq,
            n_classes=n_classes,
            conv_channels=config.conv_channels,
            kernel_sizes=config.kernel_sizes,
            dropout=config.dropout,
        ).to(device)
        
        # Optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate,
        )
        
        # Loss function
        self.criterion = nn.NLLLoss()
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
        }
        
        # Best model state
        self.best_state = None
        self.best_val_acc = 0.0
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        n_epochs: Optional[int] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """
        Train the classifier.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data
        val_loader : DataLoader
            Validation data
        n_epochs : int, optional
            Number of epochs (overrides config)
        verbose : bool
            Print progress
            
        Returns
        -------
        dict
            Training history
        """
        if n_epochs is None:
            n_epochs = self.config.n_epochs
        
        for epoch in range(n_epochs):
            # Training phase
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch in train_loader:
                x = batch['x'].to(self.device)
                y = batch['label'].to(self.device)
                
                self.optimizer.zero_grad()
                log_probs = self.model(x)
                loss = self.criterion(log_probs, y)
                loss.backward()
                self.optimizer.step()
                
                train_loss += loss.item() * len(y)
                pred = log_probs.argmax(dim=1)
                train_correct += (pred == y).sum().item()
                train_total += len(y)
            
            train_loss /= train_total
            train_acc = 100.0 * train_correct / train_total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for batch in val_loader:
                    x = batch['x'].to(self.device)
                    y = batch['label'].to(self.device)
                    
                    log_probs = self.model(x)
                    loss = self.criterion(log_probs, y)
                    
                    val_loss += loss.item() * len(y)
                    pred = log_probs.argmax(dim=1)
                    val_correct += (pred == y).sum().item()
                    val_total += len(y)
            
            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total
            
            # Record history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if verbose:
                logger.info(
                    f"Epoch {epoch+1}/{n_epochs} - "
                    f"Train: {train_loss:.4f} ({train_acc:.1f}%) | "
                    f"Val: {val_loss:.4f} ({val_acc:.1f}%)"
                )
        
        # Restore best model
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            if verbose:
                logger.info(f"Restored best model with {self.best_val_acc:.1f}% accuracy")
        
        return self.history
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate classifier on a dataset.
        
        Returns
        -------
        dict
            Evaluation metrics including overall and per-class accuracy
        """
        self.model.eval()
        
        correct = 0
        total = 0
        class_correct = {i: 0 for i in range(self.n_classes)}
        class_total = {i: 0 for i in range(self.n_classes)}
        
        with torch.no_grad():
            for batch in data_loader:
                x = batch['x'].to(self.device)
                y = batch['label'].to(self.device)
                
                pred, _ = self.model.predict(x)
                
                correct += (pred == y).sum().item()
                total += len(y)
                
                for i in range(self.n_classes):
                    mask = (y == i)
                    class_correct[i] += (pred[mask] == y[mask]).sum().item()
                    class_total[i] += mask.sum().item()
        
        results = {'accuracy': 100.0 * correct / total}
        
        for i in range(self.n_classes):
            if class_total[i] > 0:
                results[f'accuracy_{i+1}comp'] = 100.0 * class_correct[i] / class_total[i]
        
        return results
    
    def predict(self, x: torch.Tensor) -> Tuple[int, Dict[int, float]]:
        """
        Predict for a single spectrum.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (3 * n_freq,) or (1, 3 * n_freq)
            
        Returns
        -------
        tuple
            (n_components, probability_dict)
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        pred, probs = self.model.predict(x)
        
        n_comp = pred.item() + 1  # Convert to 1-indexed
        prob_dict = {i + 1: probs[0, i].item() for i in range(self.n_classes)}
        
        return n_comp, prob_dict
    
    def save(self, path: str):
        """Save classifier to file using torch.save()."""
        save_dict = {
            'model_state': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'n_freq': self.n_freq,
            'n_classes': self.n_classes,
            'config': self.config,
            'history': self.history,
            'best_val_acc': self.best_val_acc,
            'conv_channels': self.model.conv_channels,
            'kernel_sizes': self.model.kernel_sizes,
            'dropout': self.model.dropout_rate,
            'save_timestamp': datetime.now().isoformat(),
        }
        torch.save(save_dict, path)
        logger.info(f"Saved classifier to {path}")
    
    def load(self, path: str):
        """Load classifier from file."""
        save_dict = torch.load(path, map_location='cpu', weights_only=False)
        
        self.n_freq = save_dict['n_freq']
        self.n_classes = save_dict['n_classes']
        self.config = save_dict.get('config', self.config)
        self.history = save_dict.get('history', {})
        self.best_val_acc = save_dict.get('best_val_acc', 0.0)
        
        # Rebuild model
        self.model = SpectralClassifier(
            n_freq=self.n_freq,
            n_classes=self.n_classes,
            conv_channels=save_dict['conv_channels'],
            kernel_sizes=save_dict['kernel_sizes'],
            dropout=save_dict['dropout'],
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state'])
        self.model.eval()
        
        logger.info(f"Loaded classifier from {path}")


def train_classifier(
    config: Configuration,
    output_dir: Path,
    max_components: int = 5,
    model_types: Optional[List[str]] = None,
    cross_model_training: bool = False,
) -> Dict[str, Any]:
    """
    Train the model selection classifier.
    
    Parameters
    ----------
    config : Configuration
        Full configuration
    output_dir : Path
        Output directory
    max_components : int
        Maximum number of components
    model_types : List[str], optional
        Model types to train on
    cross_model_training : bool
        Whether to train cross-model classifier
        
    Returns
    -------
    dict
        Training results
    """
    if model_types is None:
        model_types = config.physics.model_types
    
    # Calculate number of classes
    if cross_model_training:
        n_classes = len(model_types) * max_components
        logger.info(f"Cross-model classifier: {n_classes} classes")
    else:
        n_classes = max_components
        logger.info(f"Single-model classifier: {n_classes} classes")
    
    # Load data
    train_loader, val_loader, n_freq, class_to_label = prepare_classifier_data(
        simulations_dir=output_dir,
        max_components=max_components,
        model_types=model_types,
        cross_model_training=cross_model_training,
        validation_fraction=config.classifier.validation_fraction,
        batch_size=config.classifier.batch_size,
    )
    
    # Create trainer
    device = config.training.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    
    trainer = ClassifierTrainer(
        n_freq=n_freq,
        n_classes=n_classes,
        config=config.classifier,
        device=device,
    )
    
    # Train
    history = trainer.train(train_loader, val_loader)
    
    # Evaluate
    eval_results = trainer.evaluate(val_loader)
    
    # Save
    save_path = output_dir / "classifier.pt"
    trainer.save(str(save_path))
    
    # Save training plot
    _save_classifier_training_plot(history, output_dir)
    
    # Build results
    result = {
        'model_path': str(save_path),
        'n_freq': n_freq,
        'n_classes': n_classes,
        'max_components': max_components,
        'model_types': model_types,
        'cross_model_training': cross_model_training,
        'class_to_label': class_to_label,
        'history': history,
        'final_val_accuracy': eval_results['accuracy'],
    }
    
    for key, value in eval_results.items():
        if key != 'accuracy':
            result[key] = value
    
    return result


def _save_classifier_training_plot(history: Dict[str, List[float]], output_dir: Path):
    """Save classifier training plots."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss plot
    ax1.plot(epochs, history['train_loss'], 'b-', label='Train')
    ax1.plot(epochs, history['val_loss'], 'r-', label='Validation')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Classifier Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy plot
    ax2.plot(epochs, history['train_acc'], 'b-', label='Train')
    ax2.plot(epochs, history['val_acc'], 'r-', label='Validation')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Classifier Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'classifier_training.png', dpi=150)
    plt.close()
    
    logger.info(f"Saved classifier training plot")
