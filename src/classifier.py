#!/usr/bin/env python3
"""
Model Selection Classifier for VROOM-SBI

A neural network classifier that learns to predict the number of RM components
directly from the observed spectrum. No AIC/BIC computation needed.

The classifier learns patterns that distinguish:
- 1-component spectra (simple Faraday rotation)
- 2-component spectra (beating patterns, complex structure)
- (Extensible to 3-comp, 4-comp, etc.)

Input: [Q, U, weights] - spectrum with channel weights
Output: probabilities for each model class

Training uses the same simulated data as posterior training,
with all augmentations (missing channels, RFI gaps, noise variation).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class ClassifierConfig:
    """Configuration for the model selection classifier."""
    hidden_dims: List[int]
    dropout: float
    n_epochs: int
    batch_size: int
    learning_rate: float
    validation_fraction: float
    use_posterior_simulations: bool
    
    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> 'ClassifierConfig':
        """Create from config dictionary."""
        cfg = config.get('classifier', {})
        return cls(
            hidden_dims=cfg.get('hidden_dims', [256, 128, 64]),
            dropout=cfg.get('dropout', 0.1),
            n_epochs=cfg.get('n_epochs', 50),
            batch_size=cfg.get('batch_size', 128),
            learning_rate=cfg.get('learning_rate', 0.001),
            validation_fraction=cfg.get('validation_fraction', 0.2),
            use_posterior_simulations=cfg.get('use_posterior_simulations', True),
        )


class SpectralClassifier(nn.Module):
    """
    Neural network classifier for model selection.
    
    Architecture:
    - Input: [Q, U, weights] concatenated (3 * n_freq dimensions)
    - Hidden layers with LayerNorm, ReLU, Dropout
    - Output: log probabilities for each model class
    
    Parameters
    ----------
    n_freq : int
        Number of frequency channels
    n_classes : int
        Number of model classes (e.g., 2 for 1-comp vs 2-comp)
    hidden_dims : List[int]
        Dimensions of hidden layers
    dropout : float
        Dropout probability
    """
    
    def __init__(
        self,
        n_freq: int,
        n_classes: int = 2,
        hidden_dims: List[int] = [256, 128, 64],
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.n_freq = n_freq
        self.n_classes = n_classes
        self.hidden_dims = hidden_dims
        self.dropout_rate = dropout
        
        # Input: Q (n_freq) + U (n_freq) + weights (n_freq) = 3 * n_freq
        input_dim = 3 * n_freq
        
        # Build layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, n_classes))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq)
            Contains [Q, U, weights] concatenated
        
        Returns
        -------
        torch.Tensor
            Log probabilities of shape (batch, n_classes)
        """
        logits = self.network(x)
        return F.log_softmax(logits, dim=-1)
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict class and probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        
        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            (predicted_class, probabilities)
            predicted_class is 0-indexed, add 1 for n_components
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)
            probs = torch.exp(log_probs)
            predicted = torch.argmax(log_probs, dim=-1)
        return predicted, probs
    
    def predict_n_components(self, x: torch.Tensor) -> Tuple[int, Dict[int, float]]:
        """
        Predict number of components for a single spectrum.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (3 * n_freq,) or (1, 3 * n_freq)
        
        Returns
        -------
        Tuple[int, Dict[int, float]]
            (n_components, probability_dict)
            e.g., (1, {1: 0.95, 2: 0.05})
        """
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        predicted, probs = self.predict(x)
        
        # Convert to n_components (1-indexed)
        n_comp = predicted.item() + 1
        
        # Build probability dict
        prob_dict = {i + 1: probs[0, i].item() for i in range(self.n_classes)}
        
        return n_comp, prob_dict


class SimulationDataset(Dataset):
    """
    Dataset for classifier training using saved simulations.
    
    Parameters
    ----------
    spectra : np.ndarray
        Array of shape (n_samples, 2 * n_freq) containing [Q, U]
    weights : np.ndarray
        Array of shape (n_samples, n_freq) containing channel weights
    labels : np.ndarray
        Array of shape (n_samples,) containing class labels (0-indexed)
    """
    
    def __init__(
        self,
        spectra: np.ndarray,
        weights: np.ndarray,
        labels: np.ndarray
    ):
        self.spectra = torch.tensor(spectra, dtype=torch.float32)
        self.weights = torch.tensor(weights, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)
        
    def __len__(self) -> int:
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Concatenate [Q, U, weights]
        x = torch.cat([self.spectra[idx], self.weights[idx]], dim=0)
        return x, self.labels[idx]


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
        Training configuration
    device : str
        Device to train on ('cpu' or 'cuda')
    """
    
    def __init__(
        self,
        n_freq: int,
        n_classes: int = 2,
        config: Optional[ClassifierConfig] = None,
        device: str = 'cpu'
    ):
        self.n_freq = n_freq
        self.n_classes = n_classes
        self.device = device
        
        # Default config if not provided
        if config is None:
            config = ClassifierConfig(
                hidden_dims=[256, 128, 64],
                dropout=0.1,
                n_epochs=50,
                batch_size=128,
                learning_rate=0.001,
                validation_fraction=0.2,
                use_posterior_simulations=True,
            )
        self.config = config
        
        # Build model
        self.model = SpectralClassifier(
            n_freq=n_freq,
            n_classes=n_classes,
            hidden_dims=config.hidden_dims,
            dropout=config.dropout,
        ).to(device)
        
        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config.learning_rate
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
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        verbose: bool = True
    ) -> Dict[str, List[float]]:
        """
        Train the classifier.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader
            Validation data loader
        verbose : bool
            Print progress
        
        Returns
        -------
        Dict[str, List[float]]
            Training history
        """
        best_val_acc = 0.0
        best_state = None
        
        for epoch in range(self.config.n_epochs):
            # Training
            self.model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (x, y) in enumerate(train_loader):
                x = x.to(self.device)
                y = y.to(self.device)
                
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
            
            # Validation
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for x, y in val_loader:
                    x = x.to(self.device)
                    y = y.to(self.device)
                    
                    log_probs = self.model(x)
                    loss = self.criterion(log_probs, y)
                    
                    val_loss += loss.item() * len(y)
                    pred = log_probs.argmax(dim=1)
                    val_correct += (pred == y).sum().item()
                    val_total += len(y)
            
            val_loss /= val_total
            val_acc = 100.0 * val_correct / val_total
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
            
            if verbose:
                print(f"Epoch {epoch+1}/{self.config.n_epochs} - "
                      f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                      f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Restore best model
        if best_state is not None:
            self.model.load_state_dict(best_state)
            if verbose:
                print(f"\nRestored best model with validation accuracy: {best_val_acc:.2f}%")
        
        return self.history
    
    def predict(self, x: torch.Tensor) -> Tuple[int, Dict[int, float]]:
        """
        Predict number of components for a spectrum.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (3 * n_freq,) or (1, 3 * n_freq)
        
        Returns
        -------
        Tuple[int, Dict[int, float]]
            (n_components, probability_dict)
        """
        x = x.to(self.device)
        return self.model.predict_n_components(x)
    
    def evaluate(self, data_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate classifier on a dataset.
        
        Parameters
        ----------
        data_loader : DataLoader
            Data to evaluate on
        
        Returns
        -------
        Dict[str, float]
            Evaluation metrics
        """
        self.model.eval()
        correct = 0
        total = 0
        
        # Per-class accuracy
        class_correct = {i: 0 for i in range(self.n_classes)}
        class_total = {i: 0 for i in range(self.n_classes)}
        
        with torch.no_grad():
            for x, y in data_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                pred, _ = self.model.predict(x)
                
                correct += (pred == y).sum().item()
                total += len(y)
                
                for i in range(self.n_classes):
                    mask = (y == i)
                    class_correct[i] += (pred[mask] == y[mask]).sum().item()
                    class_total[i] += mask.sum().item()
        
        results = {
            'accuracy': 100.0 * correct / total,
        }
        
        for i in range(self.n_classes):
            if class_total[i] > 0:
                results[f'accuracy_{i+1}comp'] = 100.0 * class_correct[i] / class_total[i]
        
        return results
    
    def save(self, path: str):
        """Save classifier to file."""
        save_dict = {
            'model_state': {k: v.cpu() for k, v in self.model.state_dict().items()},
            'n_freq': self.n_freq,
            'n_classes': self.n_classes,
            'config': self.config,
            'history': self.history,
            'hidden_dims': self.model.hidden_dims,
            'dropout': self.model.dropout_rate,
        }
        with open(path, 'wb') as f:
            pickle.dump(save_dict, f)
    
    def load(self, path: str):
        """Load classifier from file."""
        with open(path, 'rb') as f:
            save_dict = pickle.load(f)
        
        self.n_freq = save_dict['n_freq']
        self.n_classes = save_dict['n_classes']
        self.config = save_dict['config']
        self.history = save_dict['history']
        
        # Rebuild model with correct architecture
        self.model = SpectralClassifier(
            n_freq=self.n_freq,
            n_classes=self.n_classes,
            hidden_dims=save_dict['hidden_dims'],
            dropout=save_dict['dropout'],
        ).to(self.device)
        
        self.model.load_state_dict(save_dict['model_state'])
        self.model.eval()


def prepare_classifier_data(
    simulations_dir: Path,
    max_components: int = 2,
    validation_fraction: float = 0.2,
    batch_size: int = 128,
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Load saved simulations and prepare DataLoaders for classifier training.
    
    Parameters
    ----------
    simulations_dir : Path
        Directory containing saved simulations (simulations_n1.pkl, etc.)
    max_components : int
        Maximum number of components to include
    validation_fraction : float
        Fraction of data to use for validation
    batch_size : int
        Batch size for DataLoaders
    
    Returns
    -------
    Tuple[DataLoader, DataLoader, int]
        (train_loader, val_loader, n_freq)
    """
    all_spectra = []
    all_weights = []
    all_labels = []
    n_freq = None
    
    for n_comp in range(1, max_components + 1):
        sim_path = simulations_dir / f"simulations_n{n_comp}.pkl"
        
        if not sim_path.exists():
            raise FileNotFoundError(f"Simulations not found: {sim_path}")
        
        with open(sim_path, 'rb') as f:
            sim_data = pickle.load(f)
        
        spectra = sim_data['spectra']  # (n_samples, 2 * n_freq)
        weights = sim_data['weights']  # (n_samples, n_freq)
        
        if n_freq is None:
            n_freq = weights.shape[1]
        
        n_samples = len(spectra)
        labels = np.full(n_samples, n_comp - 1, dtype=np.int64)  # 0-indexed
        
        all_spectra.append(spectra)
        all_weights.append(weights)
        all_labels.append(labels)
        
        print(f"  Loaded {n_samples} samples for {n_comp}-component model")
    
    # Combine
    all_spectra = np.vstack(all_spectra)
    all_weights = np.vstack(all_weights)
    all_labels = np.concatenate(all_labels)
    
    print(f"  Total: {len(all_labels)} samples")
    
    # Shuffle
    indices = np.random.permutation(len(all_labels))
    all_spectra = all_spectra[indices]
    all_weights = all_weights[indices]
    all_labels = all_labels[indices]
    
    # Split
    n_val = int(len(all_labels) * validation_fraction)
    
    train_dataset = SimulationDataset(
        all_spectra[n_val:], all_weights[n_val:], all_labels[n_val:]
    )
    val_dataset = SimulationDataset(
        all_spectra[:n_val], all_weights[:n_val], all_labels[:n_val]
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    return train_loader, val_loader, n_freq