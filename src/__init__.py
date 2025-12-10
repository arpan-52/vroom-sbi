"""
VROOM-SBI: Simulation-Based Inference for RM Synthesis
"""

from .simulator import RMSimulator, build_prior, sample_prior, sort_posterior_samples
from .physics import load_frequencies, freq_to_lambda_sq
from .train import train_model, train_all_models

__all__ = [
    'RMSimulator',
    'build_prior',
    'sample_prior',
    'sort_posterior_samples',
    'load_frequencies',
    'freq_to_lambda_sq',
    'train_model',
    'train_all_models',
]