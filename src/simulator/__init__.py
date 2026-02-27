"""
Simulator module for VROOM-SBI.

Contains physical models and simulation utilities.
"""

from .augmentation import (
    augment_base_noise_level,
    augment_weights_combined,
    augment_weights_contiguous_gap,
    augment_weights_scattered,
)
from .base_simulator import RMSimulator
from .physics import (
    compute_rmsf,
    freq_to_lambda_sq,
    get_rmsf_properties,
    load_frequencies,
)
from .prior import (
    build_prior,
    sample_prior,
    sort_components_by_rm,
    sort_posterior_samples,
)

__all__ = [
    "load_frequencies",
    "freq_to_lambda_sq",
    "compute_rmsf",
    "get_rmsf_properties",
    "RMSimulator",
    "build_prior",
    "sample_prior",
    "sort_components_by_rm",
    "sort_posterior_samples",
    "augment_weights_combined",
    "augment_base_noise_level",
    "augment_weights_scattered",
    "augment_weights_contiguous_gap",
]
