"""VROOM-SBI: Simulation-Based Inference for RM Synthesis"""

__version__ = "2.0.0"

from .config import Configuration, validate_config
from .simulator import RMSimulator, load_frequencies, freq_to_lambda_sq
from .training import train_model, train_all_models
from .inference import InferenceEngine
from .validation import Validator, run_validation
from .utils import push_to_huggingface, download_from_huggingface

__all__ = [
    'Configuration', 'validate_config',
    'RMSimulator', 'load_frequencies', 'freq_to_lambda_sq',
    'train_model', 'train_all_models',
    'InferenceEngine',
    'Validator', 'run_validation',
    'push_to_huggingface', 'download_from_huggingface',
]

# io is an optional submodule (requires spectral_cube).
# Import it lazily so the package still loads without spectral_cube installed.
try:
    from . import io
    __all__.append('io')
except ImportError:
    pass
