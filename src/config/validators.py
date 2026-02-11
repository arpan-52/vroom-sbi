"""
Configuration validation for VROOM-SBI.

Ensures configuration is valid and provides helpful error messages.
"""

from pathlib import Path
from typing import List
import torch

from .configuration import Configuration


class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass


def validate_config(config: Configuration) -> List[str]:
    """
    Validate configuration and return list of warnings.
    
    Raises ConfigurationError for critical issues.
    Returns list of warning messages for non-critical issues.
    """
    warnings = []
    
    # Validate freq_file exists
    if not Path(config.freq_file).exists():
        raise ConfigurationError(
            f"Frequency file not found: {config.freq_file}\n"
            f"Please create or specify a valid frequency file."
        )
    
    # Validate prior ranges
    if config.priors.rm_min >= config.priors.rm_max:
        raise ConfigurationError(
            f"Invalid RM prior range: [{config.priors.rm_min}, {config.priors.rm_max}]\n"
            f"rm_min must be less than rm_max"
        )
    
    if config.priors.amp_min >= config.priors.amp_max:
        raise ConfigurationError(
            f"Invalid amplitude prior range: [{config.priors.amp_min}, {config.priors.amp_max}]\n"
            f"amp_min must be less than amp_max"
        )
    
    if config.priors.amp_min <= 0:
        warnings.append(
            f"amp_min={config.priors.amp_min} <= 0 may cause numerical issues. "
            f"Consider using amp_min >= 0.001"
        )
    
    # Validate noise config
    if config.noise.base_level <= 0:
        raise ConfigurationError(
            f"Invalid noise base_level: {config.noise.base_level}\n"
            f"base_level must be positive"
        )
    
    if config.noise.augmentation_min_factor >= config.noise.augmentation_max_factor:
        warnings.append(
            f"Noise augmentation min_factor ({config.noise.augmentation_min_factor}) >= "
            f"max_factor ({config.noise.augmentation_max_factor}). "
            f"No variation will be applied."
        )
    
    # Validate training config
    if config.training.n_simulations < 1000:
        warnings.append(
            f"n_simulations={config.training.n_simulations} is quite low. "
            f"Consider using at least 10000 for reliable posteriors."
        )
    
    # Validate device
    if config.training.device == "cuda" and not torch.cuda.is_available():
        warnings.append(
            f"CUDA requested but not available. Falling back to CPU. "
            f"Training will be significantly slower."
        )
    
    # Validate memory config
    if config.memory.max_vram_gb > 0:
        if torch.cuda.is_available():
            total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            if config.memory.max_vram_gb > total_vram:
                warnings.append(
                    f"Requested max_vram_gb ({config.memory.max_vram_gb}) > "
                    f"available GPU memory ({total_vram:.1f}GB). "
                    f"May cause OOM errors."
                )
    
    # Validate model selection
    if config.model_selection.max_components < 1:
        raise ConfigurationError(
            f"Invalid max_components: {config.model_selection.max_components}\n"
            f"Must be at least 1"
        )
    
    if config.model_selection.max_components > 10:
        warnings.append(
            f"max_components={config.model_selection.max_components} is quite high. "
            f"Training may be very slow and memory-intensive."
        )
    
    # Validate physics config
    valid_model_types = [
        "faraday_thin", 
        "burn_slab", 
        "external_dispersion", 
        "internal_dispersion"
    ]
    
    for model_type in config.physics.model_types:
        if model_type not in valid_model_types:
            raise ConfigurationError(
                f"Invalid model type: {model_type}\n"
                f"Valid types: {valid_model_types}"
            )
    
    # Validate SBI config
    valid_sbi_models = ["nsf", "maf", "mdn", "made"]
    if config.sbi.model.lower() not in valid_sbi_models:
        warnings.append(
            f"SBI model '{config.sbi.model}' may not be supported. "
            f"Recommended models: {valid_sbi_models}"
        )
    
    # Validate classifier config
    if len(config.classifier.conv_channels) != len(config.classifier.kernel_sizes):
        raise ConfigurationError(
            f"Classifier conv_channels ({len(config.classifier.conv_channels)}) and "
            f"kernel_sizes ({len(config.classifier.kernel_sizes)}) must have same length"
        )
    
    for ks in config.classifier.kernel_sizes:
        if ks % 2 == 0:
            warnings.append(
                f"Classifier kernel_size={ks} is even. "
                f"Odd kernel sizes are recommended for symmetric padding."
            )
    
    # Validate save directory is writable
    save_dir = Path(config.training.save_dir)
    try:
        save_dir.mkdir(parents=True, exist_ok=True)
        test_file = save_dir / ".write_test"
        test_file.touch()
        test_file.unlink()
    except Exception as e:
        raise ConfigurationError(
            f"Cannot write to save directory: {save_dir}\n"
            f"Error: {e}"
        )
    
    return warnings


def print_config_summary(config: Configuration):
    """Print a human-readable configuration summary."""
    print("\n" + "=" * 70)
    print("VROOM-SBI CONFIGURATION SUMMARY")
    print("=" * 70)
    
    print(f"\nFrequency file: {config.freq_file}")
    
    print(f"\nPriors:")
    print(f"  RM/phi: [{config.priors.rm_min}, {config.priors.rm_max}] rad/m²")
    print(f"  Amplitude: [{config.priors.amp_min}, {config.priors.amp_max}]")
    print(f"  chi0: [{config.priors.chi0_min:.3f}, {config.priors.chi0_max:.3f}] rad")
    print(f"  sigma_phi: [{config.priors.sigma_phi_min}, {config.priors.sigma_phi_max}] rad/m²")
    print(f"  delta_phi: [{config.priors.delta_phi_min}, {config.priors.delta_phi_max}] rad/m²")
    
    print(f"\nNoise:")
    print(f"  Base level: {config.noise.base_level}")
    print(f"  Augmentation: {config.noise.augmentation_enable}")
    if config.noise.augmentation_enable:
        print(f"  Noise range: [{config.noise.augmentation_min_factor}x, {config.noise.augmentation_max_factor}x]")
    
    print(f"\nWeight Augmentation:")
    print(f"  Enabled: {config.weight_augmentation.enable}")
    print(f"  Scattered prob: {config.weight_augmentation.scattered_prob}")
    print(f"  Gap prob: {config.weight_augmentation.gap_prob}")
    print(f"  Large block prob: {config.weight_augmentation.large_block_prob}")
    
    print(f"\nTraining:")
    print(f"  Device: {config.training.device}")
    print(f"  Base simulations: {config.training.n_simulations:,}")
    print(f"  Scaling mode: {config.training.simulation_scaling_mode}")
    batch_str = "auto" if config.training.training_batch_size == 0 else config.training.training_batch_size
    print(f"  Training batch size: {batch_str}")
    print(f"  Learning rate: {config.training.learning_rate}")
    print(f"  Early stopping patience: {config.training.stop_after_epochs}")
    
    print(f"\nHardware:")
    hw_vram = "auto" if config.hardware.vram_gb == 0 else f"{config.hardware.vram_gb}GB"
    hw_ram = "auto" if config.hardware.ram_gb == 0 else f"{config.hardware.ram_gb}GB"
    print(f"  VRAM: {hw_vram}")
    print(f"  RAM: {hw_ram}")
    
    print(f"\nModel Selection:")
    print(f"  Max components: {config.model_selection.max_components}")
    print(f"  Use classifier: {config.model_selection.use_classifier}")
    
    print(f"\nPhysics:")
    print(f"  Model types: {config.physics.model_types}")
    
    print(f"\nSBI Architecture:")
    print(f"  Model: {config.sbi.model.upper()}")
    print(f"  Embedding dim: {config.sbi.embedding_dim}")
    
    print("\n" + "=" * 70)
