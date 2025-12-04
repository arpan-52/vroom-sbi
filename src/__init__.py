"""
VROOM-SBI: Simulation-Based Inference for RM Synthesis
"""

__version__ = "0.1.0"

from . import physics
from . import simulator
from . import train
from . import inference
from . import plots

__all__ = ['physics', 'simulator', 'train', 'inference', 'plots']
