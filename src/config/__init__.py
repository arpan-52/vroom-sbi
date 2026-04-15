"""
Configuration management for VROOM-SBI.
"""

from .configuration import (
    ClassifierConfig,
    Configuration,
    HardwareConfig,
    MemoryConfig,
    ModelSelectionConfig,
    NoiseConfig,
    PhysicsConfig,
    PriorConfig,
    SBIConfig,
    TrainingConfig,
    WeightAugmentationConfig,
)
from .hardware import (
    HardwareInfo,
    OptimizedSettings,
    auto_configure,
    detect_hardware,
    optimize_for_hardware,
)
from .validators import ConfigurationError, print_config_summary, validate_config

__all__ = [
    "Configuration",
    "PriorConfig",
    "NoiseConfig",
    "TrainingConfig",
    "HardwareConfig",
    "ModelSelectionConfig",
    "PhysicsConfig",
    "SBIConfig",
    "ClassifierConfig",
    "MemoryConfig",
    "WeightAugmentationConfig",
    "validate_config",
    "print_config_summary",
    "ConfigurationError",
    # Hardware auto-optimization
    "detect_hardware",
    "optimize_for_hardware",
    "auto_configure",
    "HardwareInfo",
    "OptimizedSettings",
]
