"""
Inference module for VROOM-SBI.

Contains inference engine and model loading utilities.
"""

from .engine import (
    InferenceEngine,
    load_posterior,
    load_classifier,
)

__all__ = [
    'InferenceEngine',
    'load_posterior',
    'load_classifier',
]
