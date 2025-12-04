"""
Forward model simulator for RM Synthesis
"""

import numpy as np
import torch
from sbi.utils import BoxUniform

from .physics import load_frequencies, freq_to_lambda_sq


class RMSimulator:
    """
    Simulator for Faraday rotation of polarized radio emission.
    
    Simulates observations of Q and U Stokes parameters for a given
    set of RM components.
    """
    
    def __init__(self, freq_file, n_components=1):
        """
        Initialize the RM simulator.
        
        Parameters
        ----------
        freq_file : str
            Path to file containing frequency channels
        n_components : int
            Number of RM components to simulate
        """
        self.frequencies = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.frequencies)
        self.n_components = n_components
        self.n_channels = len(self.frequencies)
        
    def forward_model(self, theta):
        """
        Forward model: parameters -> Q, U observations.
        
        For N components, theta contains:
        - RM values (N values)
        - q amplitudes (N values)
        - u amplitudes (N values)
        - noise level (1 value)
        
        Total: 3N + 1 parameters
        
        Parameters
        ----------
        theta : np.ndarray or torch.Tensor
            Parameters [RM_1, ..., RM_N, q_1, ..., q_N, u_1, ..., u_N, noise]
            
        Returns
        -------
        qu : np.ndarray or torch.Tensor
            Stokes Q and U values [Q_1, ..., Q_M, U_1, ..., U_M]
            where M is the number of channels
        """
        is_torch = torch.is_tensor(theta)
        
        if is_torch:
            # Extract parameters
            rm_values = theta[:self.n_components]
            q_amps = theta[self.n_components:2*self.n_components]
            u_amps = theta[2*self.n_components:3*self.n_components]
            noise = theta[3*self.n_components]
            
            # Convert lambda_sq to torch
            lambda_sq = torch.tensor(self.lambda_sq, dtype=torch.float32, device=theta.device)
            
            # Initialize Q and U
            Q = torch.zeros(self.n_channels, dtype=torch.float32, device=theta.device)
            U = torch.zeros(self.n_channels, dtype=torch.float32, device=theta.device)
            
            # Add contribution from each component
            for i in range(self.n_components):
                rm = rm_values[i]
                q_amp = q_amps[i]
                u_amp = u_amps[i]
                
                # Faraday rotation
                angle = 2 * rm * lambda_sq
                Q += q_amp * torch.cos(angle) - u_amp * torch.sin(angle)
                U += q_amp * torch.sin(angle) + u_amp * torch.cos(angle)
            
            # Add noise
            Q_noise = torch.randn_like(Q) * noise
            U_noise = torch.randn_like(U) * noise
            
            Q += Q_noise
            U += U_noise
            
            # Concatenate Q and U
            qu = torch.cat([Q, U])
            
        else:
            # NumPy version
            rm_values = theta[:self.n_components]
            q_amps = theta[self.n_components:2*self.n_components]
            u_amps = theta[2*self.n_components:3*self.n_components]
            noise = theta[3*self.n_components]
            
            # Initialize Q and U
            Q = np.zeros(self.n_channels)
            U = np.zeros(self.n_channels)
            
            # Add contribution from each component
            for i in range(self.n_components):
                rm = rm_values[i]
                q_amp = q_amps[i]
                u_amp = u_amps[i]
                
                # Faraday rotation
                angle = 2 * rm * self.lambda_sq
                Q += q_amp * np.cos(angle) - u_amp * np.sin(angle)
                U += q_amp * np.sin(angle) + u_amp * np.cos(angle)
            
            # Add noise
            Q += np.random.randn(self.n_channels) * noise
            U += np.random.randn(self.n_channels) * noise
            
            # Concatenate Q and U
            qu = np.concatenate([Q, U])
        
        return qu
    
    def __call__(self, theta):
        """Allow simulator to be called as a function."""
        return self.forward_model(theta)


def build_prior(n_components, rm_range=(-500, 500), amp_range=(0, 1), 
                noise_range=(0.001, 0.1)):
    """
    Build a prior distribution for the parameters.
    
    Parameters
    ----------
    n_components : int
        Number of RM components
    rm_range : tuple
        (min, max) range for RM values in rad/m^2
    amp_range : tuple
        (min, max) range for q and u amplitudes
    noise_range : tuple
        (min, max) range for noise level
        
    Returns
    -------
    prior : sbi.utils.BoxUniform
        Prior distribution
    """
    # Total number of parameters: 3N + 1
    n_params = 3 * n_components + 1
    
    # Build lower and upper bounds
    lower = []
    upper = []
    
    # RM bounds
    for _ in range(n_components):
        lower.append(rm_range[0])
        upper.append(rm_range[1])
    
    # q amplitude bounds
    for _ in range(n_components):
        lower.append(amp_range[0])
        upper.append(amp_range[1])
    
    # u amplitude bounds
    for _ in range(n_components):
        lower.append(amp_range[0])
        upper.append(amp_range[1])
    
    # Noise bounds
    lower.append(noise_range[0])
    upper.append(noise_range[1])
    
    # Create BoxUniform prior
    lower = torch.tensor(lower, dtype=torch.float32)
    upper = torch.tensor(upper, dtype=torch.float32)
    
    prior = BoxUniform(low=lower, high=upper)
    
    return prior


def sample_prior(prior, n_samples=1):
    """
    Sample from the prior distribution.
    
    Parameters
    ----------
    prior : sbi.utils.BoxUniform
        Prior distribution
    n_samples : int
        Number of samples to draw
        
    Returns
    -------
    samples : torch.Tensor
        Samples from the prior, shape (n_samples, n_params)
    """
    return prior.sample((n_samples,))
