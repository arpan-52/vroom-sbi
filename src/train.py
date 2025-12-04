"""
Training functions for neural posterior estimation
"""

import os
import torch
from sbi.inference import SNPE
from tqdm import tqdm

from .simulator import RMSimulator, build_prior


def train_model(freq_file, n_components, n_simulations=10000, 
                config=None, device="cuda"):
    """
    Train a neural posterior estimator for a given number of components.
    
    Parameters
    ----------
    freq_file : str
        Path to frequency file
    n_components : int
        Number of RM components
    n_simulations : int
        Number of simulations for training
    config : dict, optional
        Configuration dictionary with prior ranges and training settings
    device : str
        Device to use ('cuda' or 'cpu')
        
    Returns
    -------
    posterior : sbi posterior
        Trained posterior estimator
    """
    # Set default config
    if config is None:
        config = {
            'priors': {
                'rm': {'min': -500.0, 'max': 500.0},
                'amp': {'min': 0.0, 'max': 1.0},
                'noise': {'min': 0.001, 'max': 0.1}
            },
            'training': {
                'batch_size': 50,
                'validation_fraction': 0.1
            }
        }
    
    # Extract config
    rm_range = (config['priors']['rm']['min'], config['priors']['rm']['max'])
    amp_range = (config['priors']['amp']['min'], config['priors']['amp']['max'])
    noise_range = (config['priors']['noise']['min'], config['priors']['noise']['max'])
    
    # Create simulator and prior
    simulator = RMSimulator(freq_file, n_components=n_components)
    prior = build_prior(n_components, rm_range=rm_range, 
                       amp_range=amp_range, noise_range=noise_range)
    
    # Initialize SNPE
    inference = SNPE(prior=prior, device=device)
    
    print(f"Training model for {n_components} component(s)...")
    print(f"  Number of parameters: {3 * n_components + 1}")
    print(f"  Number of simulations: {n_simulations}")
    print(f"  Device: {device}")
    
    # Generate training data
    print("Generating training data...")
    theta = prior.sample((n_simulations,))
    
    # Simulate observations
    x = []
    for i in tqdm(range(n_simulations), desc="Simulating"):
        x_i = simulator(theta[i])
        x.append(x_i)
    
    x = torch.stack(x)
    
    # Train the density estimator
    print("Training neural density estimator...")
    inference.append_simulations(theta, x)
    
    density_estimator = inference.train(
        training_batch_size=config['training'].get('batch_size', 50),
        validation_fraction=config['training'].get('validation_fraction', 0.1)
    )
    
    # Build posterior
    posterior = inference.build_posterior(density_estimator)
    
    print(f"Training complete for {n_components} component(s)!\n")
    
    return posterior


def train_all_models(freq_file, max_components=5, n_simulations=10000,
                     config=None, device="cuda", save_dir="models"):
    """
    Train neural posterior estimators for all component numbers from 1 to max_components.
    
    Parameters
    ----------
    freq_file : str
        Path to frequency file
    max_components : int
        Maximum number of components
    n_simulations : int
        Number of simulations for each model
    config : dict, optional
        Configuration dictionary
    device : str
        Device to use ('cuda' or 'cpu')
    save_dir : str
        Directory to save trained models
        
    Returns
    -------
    posteriors : dict
        Dictionary mapping n_components -> posterior
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    posteriors = {}
    
    for n_components in range(1, max_components + 1):
        # Train model
        posterior = train_model(
            freq_file=freq_file,
            n_components=n_components,
            n_simulations=n_simulations,
            config=config,
            device=device
        )
        
        # Save model
        model_path = os.path.join(save_dir, f"model_n{n_components}.pkl")
        torch.save(posterior, model_path)
        print(f"Model saved to {model_path}")
        
        posteriors[n_components] = posterior
    
    print(f"\nAll {max_components} models trained and saved!")
    
    return posteriors
