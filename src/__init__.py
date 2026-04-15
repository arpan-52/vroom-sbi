"""VROOM-SBI: Simulation-Based Inference for RM Synthesis"""

__version__ = "2.0.0"

from .config import Configuration, validate_config
from .inference import InferenceEngine
from .simulator import RMSimulator, freq_to_lambda_sq, load_frequencies
from .training import train_all_models, train_model
from .utils import download_from_huggingface, push_to_huggingface
from .validation import Validator, run_validation

__all__ = [
    "Configuration",
    "validate_config",
    "RMSimulator",
    "load_frequencies",
    "freq_to_lambda_sq",
    "train_model",
    "train_all_models",
    "InferenceEngine",
    "Validator",
    "run_validation",
    "push_to_huggingface",
    "download_from_huggingface",
]

# io is an optional submodule (requires spectral_cube).
# Import it lazily so the package still loads without spectral_cube installed.
try:
    from . import io  # noqa: F401

    __all__.append("io")
except ImportError:
    pass
