"""
Decision Layer: Quality prediction for model selection.

The decision layer is a neural network that takes RM spectra (Q, U) with weights
and predicts quality metrics (log evidence, AIC, BIC) for both 1-comp and 2-comp models.
This allows interpretable model selection based on predicted model quality.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Dict


class DecisionNetwork(nn.Module):
    """
    Neural network classifier for model selection.
    
    Input: [Q_1, ..., Q_M, U_1, ..., U_M, w_1, ..., w_M]
           where M is the number of frequency channels
    Output: [prob_1comp, prob_2comp] (softmax probabilities)
    """
    
    def __init__(self, n_freq: int, hidden_dims: list = None):
        """
        Initialize the decision network.
        
        Parameters
        ----------
        n_freq : int
            Number of frequency channels
        hidden_dims : list, optional
            Hidden layer dimensions. Default: [256, 128, 64]
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        
        self.n_freq = n_freq
        # Input: Q (n_freq) + U (n_freq) + weights (n_freq) = 3 * n_freq
        input_dim = 3 * n_freq
        
        # Build network
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer: 2 classes (1-comp, 2-comp)
        layers.append(nn.Linear(prev_dim, 2))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq)
            [Q_1, ..., Q_M, U_1, ..., U_M, w_1, ..., w_M]
        
        Returns
        -------
        torch.Tensor
            Logits of shape (batch, 2)
        """
        return self.network(x)


class DecisionLayerTrainer:
    """
    Trainer for the decision layer classifier.
    """
    
    def __init__(self, n_freq: int, device: str = "cpu"):
        """
        Initialize the trainer.
        
        Parameters
        ----------
        n_freq : int
            Number of frequency channels
        device : str
            Device to train on ('cuda' or 'cpu')
        """
        self.device = device
        self.model = DecisionNetwork(n_freq).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, 
                   train_loader: torch.utils.data.DataLoader,
                   verbose: bool = True) -> float:
        """
        Train for one epoch.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        float
            Average loss for the epoch
        """
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (x, y) in enumerate(train_loader):
            x = x.to(self.device)
            y = y.to(self.device)
            
            self.optimizer.zero_grad()
            logits = self.model(x)
            loss = self.criterion(logits, y)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Calculate accuracy
            _, predicted = torch.max(logits.data, 1)
            total += y.size(0)
            correct += (predicted == y).sum().item()
            
            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f}, "
                      f"Acc: {100 * correct / total:.2f}%")
        
        avg_loss = total_loss / len(train_loader)
        accuracy = 100 * correct / total
        
        if verbose:
            print(f"Epoch complete - Avg Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return avg_loss
    
    def validate(self, val_loader: torch.utils.data.DataLoader) -> Tuple[float, float]:
        """
        Validate the model.
        
        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader
            
        Returns
        -------
        avg_loss : float
            Average validation loss
        accuracy : float
            Validation accuracy (0-100)
        """
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(self.device)
                y = y.to(self.device)
                
                logits = self.model(x)
                loss = self.criterion(logits, y)
                
                total_loss += loss.item()
                
                _, predicted = torch.max(logits.data, 1)
                total += y.size(0)
                correct += (predicted == y).sum().item()
        
        avg_loss = total_loss / len(val_loader)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader = None,
             n_epochs: int = 20,
             verbose: bool = True) -> dict:
        """
        Train the decision layer.
        
        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        n_epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress
            
        Returns
        -------
        dict
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy': []
        }
        
        for epoch in range(n_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")
            
            train_loss = self.train_epoch(train_loader, verbose=verbose)
            history['train_loss'].append(train_loss)
            
            if val_loader is not None:
                val_loss, val_acc = self.validate(val_loader)
                history['val_loss'].append(val_loss)
                history['val_accuracy'].append(val_acc)
                
                if verbose:
                    print(f"Validation - Loss: {val_loss:.4f}, Accuracy: {val_acc:.2f}%")
        
        return history
    
    def predict(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict the number of components.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq) or (3 * n_freq,)
            
        Returns
        -------
        np.ndarray
            Predicted number of components (1 or 2) for each sample
        """
        self.model.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)
            # Convert to 1 or 2 (not 0 or 1)
            predictions = predictions + 1
        
        return predictions.cpu().numpy()
    
    def predict_proba(self, x: torch.Tensor) -> np.ndarray:
        """
        Predict class probabilities.
        
        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq) or (3 * n_freq,)
            
        Returns
        -------
        np.ndarray
            Class probabilities of shape (batch, 2)
            [:, 0] = probability of 1 component
            [:, 1] = probability of 2 components
        """
        self.model.eval()
        
        if x.dim() == 1:
            x = x.unsqueeze(0)
        
        x = x.to(self.device)
        
        with torch.no_grad():
            logits = self.model(x)
            probabilities = torch.softmax(logits, dim=1)
        
        return probabilities.cpu().numpy()
    
    def save(self, path: str):
        """
        Save the model.
        
        Parameters
        ----------
        path : str
            Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_freq': self.model.n_freq,
        }, path)
    
    def load(self, path: str):
        """
        Load the model.
        
        Parameters
        ----------
        path : str
            Path to load the model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.model = DecisionNetwork(checkpoint['n_freq']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])


# ============================================================================
# Quality Prediction Network (New Architecture)
# ============================================================================

class QualityPredictionNetwork(nn.Module):
    """
    Neural network for predicting model quality metrics.

    Input: [Q_1, ..., Q_M, U_1, ..., U_M, w_1, ..., w_M]
    Output: {
        'log_evidence': [log_ev_1, log_ev_2],
        'aic': [AIC_1, AIC_2],
        'bic': [BIC_1, BIC_2]
    }
    """

    def __init__(self, n_freq: int, hidden_dims: list = None):
        """
        Initialize the quality prediction network.

        Parameters
        ----------
        n_freq : int
            Number of frequency channels
        hidden_dims : list, optional
            Hidden layer dimensions. Default: [256, 128, 64]
        """
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        self.n_freq = n_freq
        input_dim = 3 * n_freq  # Q + U + weights

        # Shared encoder
        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim

        self.encoder = nn.Sequential(*layers)

        # Separate heads for different metrics
        # Each head outputs 2 values: [metric_1comp, metric_2comp]
        self.log_evidence_head = nn.Linear(prev_dim, 2)
        self.aic_head = nn.Linear(prev_dim, 2)
        self.bic_head = nn.Linear(prev_dim, 2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq)

        Returns
        -------
        Dict[str, torch.Tensor]
            Dictionary with keys 'log_evidence', 'aic', 'bic'
            Each value has shape (batch, 2) for [N=1, N=2]
        """
        features = self.encoder(x)

        return {
            'log_evidence': self.log_evidence_head(features),
            'aic': self.aic_head(features),
            'bic': self.bic_head(features)
        }


class QualityPredictionTrainer:
    """
    Trainer for the quality prediction network.
    """

    def __init__(self, n_freq: int, device: str = "cpu", hidden_dims: list = None):
        """
        Initialize the trainer.

        Parameters
        ----------
        n_freq : int
            Number of frequency channels
        device : str
            Device to train on ('cuda' or 'cpu')
        hidden_dims : list, optional
            Hidden layer dimensions
        """
        self.device = device
        self.model = QualityPredictionNetwork(n_freq, hidden_dims).to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=5e-4)

        # Use MSE loss for regression
        self.criterion = nn.MSELoss()

    def train_epoch(self,
                   train_loader: torch.utils.data.DataLoader,
                   verbose: bool = True) -> Dict[str, float]:
        """
        Train for one epoch.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader yielding (x, targets) where
            targets is a dict with 'log_evidence', 'aic', 'bic'
        verbose : bool
            Whether to print progress

        Returns
        -------
        Dict[str, float]
            Average losses for each metric
        """
        self.model.train()

        total_loss_log_ev = 0.0
        total_loss_aic = 0.0
        total_loss_bic = 0.0
        total_loss = 0.0
        n_batches = 0

        for batch_idx, (x, targets) in enumerate(train_loader):
            x = x.to(self.device)

            # Move targets to device
            targets_log_ev = targets['log_evidence'].to(self.device)
            targets_aic = targets['aic'].to(self.device)
            targets_bic = targets['bic'].to(self.device)

            self.optimizer.zero_grad()
            predictions = self.model(x)

            # Multi-task loss
            loss_log_ev = self.criterion(predictions['log_evidence'], targets_log_ev)
            loss_aic = self.criterion(predictions['aic'], targets_aic)
            loss_bic = self.criterion(predictions['bic'], targets_bic)

            # Weighted combination (log evidence is most important)
            loss = loss_log_ev + 0.3 * loss_aic + 0.3 * loss_bic

            loss.backward()
            self.optimizer.step()

            total_loss_log_ev += loss_log_ev.item()
            total_loss_aic += loss_aic.item()
            total_loss_bic += loss_bic.item()
            total_loss += loss.item()
            n_batches += 1

            if verbose and (batch_idx + 1) % 10 == 0:
                print(f"  Batch {batch_idx + 1}/{len(train_loader)}, "
                      f"Loss: {loss.item():.4f} "
                      f"(LogEv: {loss_log_ev.item():.4f}, "
                      f"AIC: {loss_aic.item():.4f}, "
                      f"BIC: {loss_bic.item():.4f})")

        return {
            'total': total_loss / n_batches,
            'log_evidence': total_loss_log_ev / n_batches,
            'aic': total_loss_aic / n_batches,
            'bic': total_loss_bic / n_batches
        }

    def validate(self, val_loader: torch.utils.data.DataLoader) -> Dict[str, float]:
        """
        Validate the model.

        Parameters
        ----------
        val_loader : DataLoader
            Validation data loader

        Returns
        -------
        Dict[str, float]
            Validation losses and accuracy metrics
        """
        self.model.eval()

        total_loss_log_ev = 0.0
        total_loss_aic = 0.0
        total_loss_bic = 0.0
        total_loss = 0.0
        n_batches = 0

        # For computing selection accuracy
        correct_log_ev = 0
        correct_aic = 0
        correct_bic = 0
        correct_ensemble = 0
        total_samples = 0

        with torch.no_grad():
            for x, targets in val_loader:
                x = x.to(self.device)

                targets_log_ev = targets['log_evidence'].to(self.device)
                targets_aic = targets['aic'].to(self.device)
                targets_bic = targets['bic'].to(self.device)
                true_n = targets['true_n'].to(self.device)  # Ground truth N

                predictions = self.model(x)

                # Compute losses
                loss_log_ev = self.criterion(predictions['log_evidence'], targets_log_ev)
                loss_aic = self.criterion(predictions['aic'], targets_aic)
                loss_bic = self.criterion(predictions['bic'], targets_bic)
                loss = loss_log_ev + 0.3 * loss_aic + 0.3 * loss_bic

                total_loss_log_ev += loss_log_ev.item()
                total_loss_aic += loss_aic.item()
                total_loss_bic += loss_bic.item()
                total_loss += loss.item()
                n_batches += 1

                # Compute selection accuracy
                # Select based on each metric
                pred_n_log_ev = torch.argmax(predictions['log_evidence'], dim=1) + 1  # 1 or 2
                pred_n_aic = torch.argmin(predictions['aic'], dim=1) + 1  # Lower is better
                pred_n_bic = torch.argmin(predictions['bic'], dim=1) + 1  # Lower is better

                # Ensemble: majority vote
                votes = torch.stack([pred_n_log_ev, pred_n_aic, pred_n_bic], dim=1)
                pred_n_ensemble = torch.mode(votes, dim=1)[0]

                correct_log_ev += (pred_n_log_ev == true_n).sum().item()
                correct_aic += (pred_n_aic == true_n).sum().item()
                correct_bic += (pred_n_bic == true_n).sum().item()
                correct_ensemble += (pred_n_ensemble == true_n).sum().item()
                total_samples += x.size(0)

        return {
            'loss_total': total_loss / n_batches,
            'loss_log_evidence': total_loss_log_ev / n_batches,
            'loss_aic': total_loss_aic / n_batches,
            'loss_bic': total_loss_bic / n_batches,
            'accuracy_log_ev': 100.0 * correct_log_ev / total_samples,
            'accuracy_aic': 100.0 * correct_aic / total_samples,
            'accuracy_bic': 100.0 * correct_bic / total_samples,
            'accuracy_ensemble': 100.0 * correct_ensemble / total_samples
        }

    def train(self,
             train_loader: torch.utils.data.DataLoader,
             val_loader: torch.utils.data.DataLoader = None,
             n_epochs: int = 50,
             verbose: bool = True) -> dict:
        """
        Train the quality prediction network.

        Parameters
        ----------
        train_loader : DataLoader
            Training data loader
        val_loader : DataLoader, optional
            Validation data loader
        n_epochs : int
            Number of training epochs
        verbose : bool
            Whether to print progress

        Returns
        -------
        dict
            Training history
        """
        history = {
            'train_loss': [],
            'val_loss': [],
            'val_accuracy_log_ev': [],
            'val_accuracy_aic': [],
            'val_accuracy_bic': [],
            'val_accuracy_ensemble': []
        }

        for epoch in range(n_epochs):
            if verbose:
                print(f"\nEpoch {epoch + 1}/{n_epochs}")

            train_losses = self.train_epoch(train_loader, verbose=verbose)
            history['train_loss'].append(train_losses['total'])

            if verbose:
                print(f"Epoch complete - Avg Loss: {train_losses['total']:.4f}")

            if val_loader is not None:
                val_metrics = self.validate(val_loader)
                history['val_loss'].append(val_metrics['loss_total'])
                history['val_accuracy_log_ev'].append(val_metrics['accuracy_log_ev'])
                history['val_accuracy_aic'].append(val_metrics['accuracy_aic'])
                history['val_accuracy_bic'].append(val_metrics['accuracy_bic'])
                history['val_accuracy_ensemble'].append(val_metrics['accuracy_ensemble'])

                if verbose:
                    print(f"Validation - Loss: {val_metrics['loss_total']:.4f}")
                    print(f"  Selection Accuracy:")
                    print(f"    By Log Evidence: {val_metrics['accuracy_log_ev']:.2f}%")
                    print(f"    By AIC: {val_metrics['accuracy_aic']:.2f}%")
                    print(f"    By BIC: {val_metrics['accuracy_bic']:.2f}%")
                    print(f"    By Ensemble: {val_metrics['accuracy_ensemble']:.2f}%")

        return history

    def predict_qualities(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """
        Predict quality metrics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor of shape (batch, 3 * n_freq) or (3 * n_freq,)

        Returns
        -------
        Dict[str, np.ndarray]
            Predicted quality metrics
        """
        self.model.eval()

        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.to(self.device)

        with torch.no_grad():
            predictions = self.model(x)

        return {
            'log_evidence': predictions['log_evidence'].cpu().numpy(),
            'aic': predictions['aic'].cpu().numpy(),
            'bic': predictions['bic'].cpu().numpy()
        }

    def select_model(self, x: torch.Tensor, strategy: str = 'ensemble') -> Tuple[int, Dict]:
        """
        Select best model based on predicted quality metrics.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor
        strategy : str
            Selection strategy: 'log_evidence', 'aic', 'bic', or 'ensemble'

        Returns
        -------
        best_n : int
            Selected number of components (1 or 2)
        info : Dict
            Detailed prediction information
        """
        qualities = self.predict_qualities(x)

        # Select based on strategy
        log_ev = qualities['log_evidence'][0]  # [log_ev_1, log_ev_2]
        aic = qualities['aic'][0]
        bic = qualities['bic'][0]

        n_by_log_ev = 1 if log_ev[0] > log_ev[1] else 2
        n_by_aic = 1 if aic[0] < aic[1] else 2
        n_by_bic = 1 if bic[0] < bic[1] else 2

        if strategy == 'log_evidence':
            best_n = n_by_log_ev
        elif strategy == 'aic':
            best_n = n_by_aic
        elif strategy == 'bic':
            best_n = n_by_bic
        elif strategy == 'ensemble':
            # Majority vote
            votes = [n_by_log_ev, n_by_aic, n_by_bic]
            best_n = max(set(votes), key=votes.count)
        else:
            raise ValueError(f"Unknown strategy: {strategy}")

        info = {
            'log_evidence': log_ev,
            'aic': aic,
            'bic': bic,
            'selection_by_log_ev': n_by_log_ev,
            'selection_by_aic': n_by_aic,
            'selection_by_bic': n_by_bic,
            'confidence': {
                'log_ev_diff': abs(log_ev[1] - log_ev[0]),
                'aic_diff': abs(aic[1] - aic[0]),
                'bic_diff': abs(bic[1] - bic[0]),
                'agreement': len(set([n_by_log_ev, n_by_aic, n_by_bic])) == 1
            }
        }

        return best_n, info

    def save(self, path: str):
        """Save the model."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'n_freq': self.model.n_freq,
        }, path)

    def load(self, path: str):
        """Load the model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.model = QualityPredictionNetwork(checkpoint['n_freq']).to(self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])