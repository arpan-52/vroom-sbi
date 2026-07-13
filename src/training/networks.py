"""
Neural network architectures for VROOM-SBI.

Contains:
- SpectralEmbedding: Embedding network for SBI
- SpectralClassifier: CNN classifier for model selection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SpectralEmbedding(nn.Module):
    """
    Custom embedding network for high-dimensional spectral data.

    Processes Q and U spectra into a lower-dimensional representation
    suitable for the flow-based density estimator.

    Parameters
    ----------
    input_dim : int
        Dimension of input spectra (2 * n_freq for Q and U)
    output_dim : int
        Dimension of output embedding
    hidden_dims : List[int]
        Hidden layer dimensions
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int = 64,
        hidden_dims: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if hidden_dims is None:
            hidden_dims = [256, 128]

        layers = []
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(prev_dim, hidden_dim),
                    nn.LayerNorm(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                ]
            )
            prev_dim = hidden_dim

        # Final projection
        layers.extend([nn.Linear(prev_dim, output_dim), nn.ReLU()])

        self.net = nn.Sequential(*layers)
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectralEmbeddingCNN(nn.Module):
    """
    1D CNN embedding network for SBI, for spectral data whose identifying
    signature is local/oscillatory (e.g. a Faraday-thin phase term whose
    period scales with RM) rather than a smooth global shape. Convolutional
    weight sharing across frequency lets one learned filter generalize across
    the whole parameter range, instead of MLP's independent per-position
    weights forcing it to relearn the same pattern at every shift/scale.

    Same conv-block structure as ``SpectralClassifier``, with a linear
    projection to an embedding vector instead of class logits.

    Parameters
    ----------
    n_freq : int
        Number of frequency channels
    output_dim : int
        Dimension of output embedding
    in_channels : int
        Number of input channels (3 for [Q, U, weights], 2 if weights are
        not concatenated — must match ``input_dim / n_freq`` upstream)
    conv_channels : List[int]
        Number of channels in each conv layer
    kernel_sizes : List[int]
        Kernel size for each conv layer
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        n_freq: int,
        output_dim: int = 64,
        in_channels: int = 3,
        conv_channels: list[int] = None,
        kernel_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        self.n_freq = n_freq
        self.output_dim = output_dim
        self.in_channels = in_channels
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        conv_layers = []
        prev_channels = in_channels
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(prev_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            prev_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)
        self.final_channels = conv_channels[-1]

        # Global average + max pooling -> 2 * final_channels features
        self.fc = nn.Sequential(
            nn.Linear(self.final_channels * 2, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : torch.Tensor
            Shape (batch, 3 * n_freq); [Q, U, weights] concatenated.

        Returns
        -------
        torch.Tensor
            Embedding of shape (batch, output_dim)
        """
        batch_size = x.shape[0]
        x = x.view(batch_size, self.in_channels, self.n_freq)
        x = self.conv_net(x)
        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2)[0]
        x = torch.cat([avg_pool, max_pool], dim=1)
        return self.fc(x)


class SpectralClassifier(nn.Module):
    """
    1D CNN classifier for model selection.

    Architecture designed for spectral data:
    - Input: [Q, U, weights] as 3 channels × n_freq
    - Convolutional layers to detect local patterns
    - Global pooling for translation invariance
    - Fully connected output layer

    Parameters
    ----------
    n_freq : int
        Number of frequency channels
    n_classes : int
        Number of model classes
    conv_channels : List[int]
        Number of channels in each conv layer
    kernel_sizes : List[int]
        Kernel size for each conv layer
    dropout : float
        Dropout probability
    """

    def __init__(
        self,
        n_freq: int,
        n_classes: int = 2,
        conv_channels: list[int] = None,
        kernel_sizes: list[int] = None,
        dropout: float = 0.1,
    ):
        super().__init__()

        if conv_channels is None:
            conv_channels = [32, 64, 128]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]

        self.n_freq = n_freq
        self.n_classes = n_classes
        self.conv_channels = conv_channels
        self.kernel_sizes = kernel_sizes
        self.dropout_rate = dropout

        # Input: 3 channels (Q, U, weights)
        in_channels = 3

        # Build convolutional layers
        conv_layers = []
        for out_channels, kernel_size in zip(conv_channels, kernel_sizes):
            padding = kernel_size // 2
            conv_layers.extend(
                [
                    nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding),
                    nn.BatchNorm1d(out_channels),
                    nn.ReLU(),
                    nn.MaxPool1d(2),
                    nn.Dropout(dropout),
                ]
            )
            in_channels = out_channels

        self.conv_net = nn.Sequential(*conv_layers)
        self.final_channels = conv_channels[-1]

        # Global average pooling + max pooling
        # Gives 2 * final_channels features
        self.fc = nn.Sequential(
            nn.Linear(self.final_channels * 2, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, n_classes),
        )

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
        batch_size = x.shape[0]

        # Reshape from (batch, 3*n_freq) to (batch, 3, n_freq)
        x = x.view(batch_size, 3, self.n_freq)

        # Apply convolutions
        x = self.conv_net(x)

        # Global pooling: combine avg and max
        avg_pool = x.mean(dim=2)
        max_pool = x.max(dim=2)[0]
        x = torch.cat([avg_pool, max_pool], dim=1)

        # Fully connected
        logits = self.fc(x)

        return F.log_softmax(logits, dim=-1)

    def predict(self, x: torch.Tensor) -> tuple:
        """
        Predict class and probabilities.

        Parameters
        ----------
        x : torch.Tensor
            Input tensor

        Returns
        -------
        tuple
            (predicted_class, probabilities)
        """
        self.eval()
        with torch.no_grad():
            log_probs = self.forward(x)
            probs = torch.exp(log_probs)
            predicted = torch.argmax(log_probs, dim=-1)
        return predicted, probs


