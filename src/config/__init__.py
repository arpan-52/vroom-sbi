"""
Configuration management for VROOM-SBI.
"""

from .configuration import (
    Configuration,
    PriorConfig,
    NoiseConfig,
    TrainingConfig,
    HardwareConfig,
    ModelSelectionConfig,
    PhysicsConfig,
    SBIConfig,
    ClassifierConfig,
    MemoryConfig,
    WeightAugmentationConfig,
)
from .validators import validate_config, print_config_summary, ConfigurationError
from .hardware import (
    detect_hardware,
    optimize_for_hardware,
    auto_configure,
    HardwareInfo,
    OptimizedSettings,
)

__all__ = [
    'Configuration',
    'PriorConfig',
    'NoiseConfig', 
    'TrainingConfig',
    'HardwareConfig',
    'ModelSelectionConfig',
    'PhysicsConfig',
    'SBIConfig',
    'ClassifierConfig',
    'MemoryConfig',
    'WeightAugmentationConfig',
    'validate_config',
    'print_config_summary',
    'ConfigurationError',
    # Hardware auto-optimization
    'detect_hardware',
    'optimize_for_hardware',
    'auto_configure',
    'HardwareInfo',
    'OptimizedSettings',
]
