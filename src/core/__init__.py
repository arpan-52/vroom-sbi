"""
Core components for VROOM-SBI.
"""

from .base_classes import (
    BaseSimulator,
    PosteriorInterface,
    InferenceEngineInterface,
)
from .result import (
    ComponentResult,
    InferenceResult,
    TrainingResult,
    TrainingMetrics,
)
from .checkpoint import (
    CheckpointManager,
    ModelCheckpoint,
)

__all__ = [
    'BaseSimulator',
    'PosteriorInterface', 
    'InferenceEngineInterface',
    'ComponentResult',
    'InferenceResult',
    'TrainingResult',
    'TrainingMetrics',
    'CheckpointManager',
    'ModelCheckpoint',
]
