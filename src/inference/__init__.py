"""
Inference module for VROOM-SBI.

Contains inference engine and model loading utilities.
"""

from .engine import (
    InferenceEngine,
    load_classifier,
    load_posterior,
)

__all__ = [
    "InferenceEngine",
    "load_posterior",
    "load_classifier",
]
