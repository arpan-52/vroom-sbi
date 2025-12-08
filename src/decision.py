"""
Decision Layer: Classifier for selecting between 1-component and 2-component models.

The decision layer is a neural network that takes RM spectra (Q, U) with weights
and decides whether to use the 1-component or 2-component model.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Tuple


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
