"""
VROOM-SBI: Simulation-Based Inference for RM Synthesis

A Python package for inferring Rotation Measure (RM) components from
polarized radio observations using neural posterior estimation.
"""

__version__ = "2.0.0"

from .config import Configuration, validate_config
from .simulator import RMSimulator, load_frequencies, freq_to_lambda_sq
from .training import train_model, train_all_models
from .inference import InferenceEngine
from .utils import push_to_huggingface, download_from_huggingface

__all__ = [
    'Configuration',
    'validate_config',
    'RMSimulator',
    'load_frequencies',
    'freq_to_lambda_sq',
    'train_model',
    'train_all_models',
    'InferenceEngine',
    'push_to_huggingface',
    'download_from_huggingface',
]
