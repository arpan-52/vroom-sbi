"""
Simulator module for VROOM-SBI.

Contains physical models and simulation utilities.
"""

from .physics import (
    load_frequencies,
    freq_to_lambda_sq,
    compute_rmsf,
    get_rmsf_properties,
)
from .base_simulator import RMSimulator
from .prior import build_prior, sample_prior, sort_components_by_rm, sort_posterior_samples
from .augmentation import (
    augment_weights_combined,
    augment_base_noise_level,
    augment_weights_scattered,
    augment_weights_contiguous_gap,
)

__all__ = [
    'load_frequencies',
    'freq_to_lambda_sq',
    'compute_rmsf',
    'get_rmsf_properties',
    'RMSimulator',
    'build_prior',
    'sample_prior',
    'sort_components_by_rm',
    'sort_posterior_samples',
    'augment_weights_combined',
    'augment_base_noise_level',
    'augment_weights_scattered',
    'augment_weights_contiguous_gap',
]
