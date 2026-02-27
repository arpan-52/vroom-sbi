"""
Training module for VROOM-SBI.

Contains trainers for SBI posteriors and classifiers.
"""

from .classifier_trainer import (
    ClassifierTrainer,
    train_classifier,
)
from .data_loader import (
    SimulationDataset,
    create_simulation_dataloader,
    load_simulations,
    save_simulations,
)
from .networks import (
    SpectralClassifier,
    SpectralEmbedding,
)
from .streaming_trainer import (
    AsyncChunkStreamer,
    StreamingNPETrainer,
    train_streaming_npe,
)
from .trainer import (
    SBITrainer,
    train_all_models,
    train_model,
)

__all__ = [
    "SBITrainer",
    "train_model",
    "train_all_models",
    "ClassifierTrainer",
    "train_classifier",
    "SimulationDataset",
    "create_simulation_dataloader",
    "save_simulations",
    "load_simulations",
    "SpectralEmbedding",
    "SpectralClassifier",
    # Streaming NPE
    "StreamingNPETrainer",
    "AsyncChunkStreamer",
    "train_streaming_npe",
]
