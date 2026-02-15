"""
Training module for VROOM-SBI.

Contains trainers for SBI posteriors and classifiers.
"""

from .trainer import (
    SBITrainer,
    train_model,
    train_all_models,
)
from .classifier_trainer import (
    ClassifierTrainer,
    train_classifier,
)
from .data_loader import (
    SimulationDataset,
    create_simulation_dataloader,
    save_simulations,
    load_simulations,
)
from .networks import (
    SpectralEmbedding,
    SpectralClassifier,
)
from .streaming_trainer import (
    StreamingNPETrainer,
    AsyncChunkStreamer,
    train_streaming_npe,
)

__all__ = [
    'SBITrainer',
    'train_model',
    'train_all_models',
    'ClassifierTrainer',
    'train_classifier',
    'SimulationDataset',
    'create_simulation_dataloader',
    'save_simulations',
    'load_simulations',
    'SpectralEmbedding',
    'SpectralClassifier',
    # Streaming NPE
    'StreamingNPETrainer',
    'AsyncChunkStreamer',
    'train_streaming_npe',
]
