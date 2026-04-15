"""
Core components for VROOM-SBI.
"""

from .base_classes import (
    BaseSimulator,
    InferenceEngineInterface,
    PosteriorInterface,
)
from .checkpoint import (
    CheckpointManager,
    ModelCheckpoint,
)
from .result import (
    ComponentResult,
    InferenceResult,
    TrainingMetrics,
    TrainingResult,
)

__all__ = [
    "BaseSimulator",
    "PosteriorInterface",
    "InferenceEngineInterface",
    "ComponentResult",
    "InferenceResult",
    "TrainingResult",
    "TrainingMetrics",
    "CheckpointManager",
    "ModelCheckpoint",
]
