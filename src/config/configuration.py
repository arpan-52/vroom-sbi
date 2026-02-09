"""
Configuration dataclasses for VROOM-SBI.

Provides structured, type-safe configuration management.
All configuration is loaded from YAML and validated.

ALL PRIORS ARE DEFINED IN ONE PLACE (PriorConfig).
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Any, Optional
import yaml
import numpy as np


@dataclass
class PriorConfig:
    """
    Prior ranges for ALL physical parameters.
    
    This is the SINGLE SOURCE OF TRUTH for all parameter bounds.
    
    Parameters by model type:
    - faraday_thin: [RM, amp, chi0] per component
    - burn_slab: [phi_c, delta_phi, amp, chi0] per component  
    - external_dispersion: [phi, sigma_phi, amp, chi0] per component
    - internal_dispersion: [phi, sigma_phi, amp, chi0] per component
    """
    # Faraday depth / Rotation Measure (rad/m²)
    # Used for: RM (faraday_thin), phi/phi_c (all other models)
    rm_min: float = -500.0
    rm_max: float = 500.0
    
    # Fractional polarization amplitude (dimensionless, 0-1)
    amp_min: float = 0.01
    amp_max: float = 1.0
    
    # Intrinsic polarization angle (radians)
    # Physical constraint: [0, π]
    chi0_min: float = 0.0
    chi0_max: float = np.pi
    
    # RM dispersion for external/internal dispersion models (rad/m²)
    # Used for: sigma_phi in external_dispersion, internal_dispersion
    sigma_phi_min: float = 0.0
    sigma_phi_max: float = 200.0
    
    # Slab half-width for burn_slab model (rad/m²)
    # Used for: delta_phi in burn_slab (full width = 2*delta_phi)
    delta_phi_min: float = 0.0
    delta_phi_max: float = 200.0
    
    def to_flat_dict(self) -> Dict[str, float]:
        """Convert to flat dictionary for simulator functions."""
        return {
            "rm_min": self.rm_min,
            "rm_max": self.rm_max,
            "amp_min": max(self.amp_min, 1e-6),  # Safety guard
            "amp_max": self.amp_max,
            "chi0_min": self.chi0_min,
            "chi0_max": self.chi0_max,
            "sigma_phi_min": self.sigma_phi_min,
            "sigma_phi_max": self.sigma_phi_max,
            "delta_phi_min": self.delta_phi_min,
            "delta_phi_max": self.delta_phi_max,
        }
    
    def get_bounds_for_model(self, model_type: str, n_components: int) -> tuple:
        """
        Get (low, high) bounds arrays for a specific model configuration.
        
        Returns arrays suitable for SBI BoxUniform prior.
        """
        low = []
        high = []
        
        if model_type == "faraday_thin":
            # 3 params per component: [RM, amp, chi0]
            for _ in range(n_components):
                low.extend([self.rm_min, self.amp_min, self.chi0_min])
                high.extend([self.rm_max, self.amp_max, self.chi0_max])
        
        elif model_type == "burn_slab":
            # 4 params per component: [phi_c, delta_phi, amp, chi0]
            for _ in range(n_components):
                low.extend([self.rm_min, self.delta_phi_min, self.amp_min, self.chi0_min])
                high.extend([self.rm_max, self.delta_phi_max, self.amp_max, self.chi0_max])
        
        else:  # external_dispersion, internal_dispersion
            # 4 params per component: [phi, sigma_phi, amp, chi0]
            for _ in range(n_components):
                low.extend([self.rm_min, self.sigma_phi_min, self.amp_min, self.chi0_min])
                high.extend([self.rm_max, self.sigma_phi_max, self.amp_max, self.chi0_max])
        
        return np.array(low), np.array(high)


@dataclass
class NoiseConfig:
    """Noise configuration for simulations."""
    base_level: float = 0.01
    augmentation_enable: bool = True
    augmentation_min_factor: float = 0.5
    augmentation_max_factor: float = 2.0


@dataclass
class TrainingConfig:
    """Training hyperparameters."""
    n_simulations: int = 30000
    simulation_scaling: bool = True
    simulation_scaling_mode: str = "power"  # "linear", "quadratic", "subquadratic", "power"
    scaling_power: float = 2.5
    batch_size: int = 50
    n_rounds: int = 1
    device: str = "cuda"
    validation_fraction: float = 0.1
    save_dir: str = "models"
    
    # Checkpointing and monitoring
    checkpoint_interval: int = 5  # Save checkpoint every N epochs
    log_interval: int = 1  # Log metrics every N epochs
    early_stopping_patience: int = 10  # Stop if no improvement for N epochs
    
    def get_scaled_simulations(self, n_components: int) -> int:
        """Calculate scaled number of simulations based on model complexity."""
        if not self.simulation_scaling:
            return self.n_simulations
            
        if self.simulation_scaling_mode == "power":
            factor = n_components ** self.scaling_power
        elif self.simulation_scaling_mode == "quadratic":
            factor = n_components ** 2
        elif self.simulation_scaling_mode == "subquadratic":
            factor = n_components ** 1.5
        else:  # "linear" or default
            scaling_factors = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
            factor = scaling_factors.get(n_components, n_components * 2)
            
        return int(self.n_simulations * factor)


@dataclass
class MemoryConfig:
    """Memory management configuration."""
    max_ram_gb: float = 16.0  # Maximum RAM to use
    max_vram_gb: float = 8.0  # Maximum VRAM to use
    stage_models_to_ram: bool = True  # Keep models in RAM when not on GPU
    prefetch_models: bool = True  # Load all models at startup
    gradient_checkpointing: bool = False  # Trade compute for memory
    mixed_precision: bool = True  # Use FP16 where possible


@dataclass
class ModelSelectionConfig:
    """Model selection configuration."""
    max_components: int = 5
    use_classifier: bool = True
    classifier_only: bool = False


@dataclass
class PhysicsConfig:
    """Physical model configuration."""
    model_types: List[str] = field(default_factory=lambda: ["faraday_thin"])


@dataclass
class SBIArchitectureConfig:
    """Architecture configuration for a specific component count."""
    hidden_features: int = 256
    num_transforms: int = 15


@dataclass
class SBIConfig:
    """SBI (Neural Posterior Estimation) configuration."""
    model: str = "nsf"  # Neural Spline Flow
    num_bins: int = 16
    embedding_dim: int = 64
    architecture_scaling: Dict[int, SBIArchitectureConfig] = field(
        default_factory=lambda: {
            1: SBIArchitectureConfig(256, 15),
            2: SBIArchitectureConfig(256, 15),
            3: SBIArchitectureConfig(256, 15),
            4: SBIArchitectureConfig(256, 15),
            5: SBIArchitectureConfig(256, 15),
        }
    )
    
    def get_architecture(self, n_components: int) -> SBIArchitectureConfig:
        """Get architecture for given component count."""
        if n_components in self.architecture_scaling:
            return self.architecture_scaling[n_components]
        # Fallback to closest larger
        available = sorted([k for k in self.architecture_scaling.keys() if k >= n_components])
        if available:
            return self.architecture_scaling[available[0]]
        return self.architecture_scaling[max(self.architecture_scaling.keys())]


@dataclass
class ClassifierConfig:
    """Model selection classifier configuration."""
    conv_channels: List[int] = field(default_factory=lambda: [32, 64, 128])
    kernel_sizes: List[int] = field(default_factory=lambda: [7, 5, 3])
    dropout: float = 0.1
    n_epochs: int = 50
    batch_size: int = 1024
    learning_rate: float = 0.0001
    validation_fraction: float = 0.2
    use_posterior_simulations: bool = True


@dataclass  
class WeightAugmentationConfig:
    """Weight augmentation settings."""
    enable: bool = True
    scattered_prob: float = 0.3
    gap_prob: float = 0.3
    large_block_prob: float = 0.1
    noise_variation: bool = True


@dataclass
class Configuration:
    """
    Main configuration container for VROOM-SBI.
    
    Loads from YAML and provides structured access to all settings.
    """
    freq_file: str = "freq.txt"
    
    priors: PriorConfig = field(default_factory=PriorConfig)
    noise: NoiseConfig = field(default_factory=NoiseConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    model_selection: ModelSelectionConfig = field(default_factory=ModelSelectionConfig)
    physics: PhysicsConfig = field(default_factory=PhysicsConfig)
    sbi: SBIConfig = field(default_factory=SBIConfig)
    classifier: ClassifierConfig = field(default_factory=ClassifierConfig)
    weight_augmentation: WeightAugmentationConfig = field(default_factory=WeightAugmentationConfig)
    
    @classmethod
    def from_yaml(cls, config_path: str) -> 'Configuration':
        """Load configuration from YAML file."""
        with open(config_path, 'r') as f:
            raw = yaml.safe_load(f)
        return cls.from_dict(raw)
    
    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> 'Configuration':
        """Create Configuration from dictionary."""
        # Extract ALL priors from ONE place
        priors_raw = raw.get("priors", {})
        rm_raw = priors_raw.get("rm", {})
        amp_raw = priors_raw.get("amp", {})
        chi0_raw = priors_raw.get("chi0", {})
        sigma_phi_raw = priors_raw.get("sigma_phi", {})
        delta_phi_raw = priors_raw.get("delta_phi", {})
        
        priors = PriorConfig(
            rm_min=float(rm_raw.get("min", -500.0)),
            rm_max=float(rm_raw.get("max", 500.0)),
            amp_min=float(amp_raw.get("min", 0.01)),
            amp_max=float(amp_raw.get("max", 1.0)),
            chi0_min=float(chi0_raw.get("min", 0.0)),
            chi0_max=float(chi0_raw.get("max", np.pi)),
            sigma_phi_min=float(sigma_phi_raw.get("min", 0.0)),
            sigma_phi_max=float(sigma_phi_raw.get("max", 200.0)),
            delta_phi_min=float(delta_phi_raw.get("min", 0.0)),
            delta_phi_max=float(delta_phi_raw.get("max", 200.0)),
        )
        
        # Extract noise config
        noise_raw = raw.get("noise", {})
        aug_raw = noise_raw.get("augmentation", {})
        noise = NoiseConfig(
            base_level=float(noise_raw.get("base_level", 0.01)),
            augmentation_enable=bool(aug_raw.get("enable", True)),
            augmentation_min_factor=float(aug_raw.get("min_factor", 0.5)),
            augmentation_max_factor=float(aug_raw.get("max_factor", 2.0)),
        )
        
        # Extract training config
        train_raw = raw.get("training", {})
        training = TrainingConfig(
            n_simulations=int(train_raw.get("n_simulations", 30000)),
            simulation_scaling=bool(train_raw.get("simulation_scaling", True)),
            simulation_scaling_mode=str(train_raw.get("simulation_scaling_mode", "power")),
            scaling_power=float(train_raw.get("scaling_power", 2.5)),
            batch_size=int(train_raw.get("batch_size", 50)),
            n_rounds=int(train_raw.get("n_rounds", 1)),
            device=str(train_raw.get("device", "cuda")),
            validation_fraction=float(train_raw.get("validation_fraction", 0.1)),
            save_dir=str(train_raw.get("save_dir", "models")),
            checkpoint_interval=int(train_raw.get("checkpoint_interval", 5)),
            log_interval=int(train_raw.get("log_interval", 1)),
            early_stopping_patience=int(train_raw.get("early_stopping_patience", 10)),
        )
        
        # Extract memory config
        mem_raw = raw.get("memory", {})
        memory = MemoryConfig(
            max_ram_gb=float(mem_raw.get("max_ram_gb", 16.0)),
            max_vram_gb=float(mem_raw.get("max_vram_gb", 8.0)),
            stage_models_to_ram=bool(mem_raw.get("stage_models_to_ram", True)),
            prefetch_models=bool(mem_raw.get("prefetch_models", True)),
            gradient_checkpointing=bool(mem_raw.get("gradient_checkpointing", False)),
            mixed_precision=bool(mem_raw.get("mixed_precision", True)),
        )
        
        # Extract model selection
        ms_raw = raw.get("model_selection", {})
        model_selection = ModelSelectionConfig(
            max_components=int(ms_raw.get("max_components", 5)),
            use_classifier=bool(ms_raw.get("use_classifier", True)),
            classifier_only=bool(ms_raw.get("classifier_only", False)),
        )
        
        # Extract physics config (just model types - priors are in PriorConfig!)
        phys_raw = raw.get("physics", {})
        model_types = phys_raw.get("model_types", phys_raw.get("model_type", ["faraday_thin"]))
        if isinstance(model_types, str):
            model_types = [model_types]
        
        physics = PhysicsConfig(model_types=model_types)
        
        # Extract SBI config
        sbi_raw = raw.get("sbi", {})
        arch_raw = sbi_raw.get("architecture_scaling", {})
        arch_scaling = {}
        for k, v in arch_raw.items():
            arch_scaling[int(k)] = SBIArchitectureConfig(
                hidden_features=int(v.get("hidden_features", 256)),
                num_transforms=int(v.get("num_transforms", 15)),
            )
        if not arch_scaling:
            # Default architecture
            for i in range(1, 6):
                arch_scaling[i] = SBIArchitectureConfig(256, 15)
                
        sbi = SBIConfig(
            model=str(sbi_raw.get("model", "nsf")),
            num_bins=int(sbi_raw.get("num_bins", 16)),
            embedding_dim=int(sbi_raw.get("embedding_dim", 64)),
            architecture_scaling=arch_scaling,
        )
        
        # Extract classifier config
        cls_raw = raw.get("classifier", {})
        classifier = ClassifierConfig(
            conv_channels=cls_raw.get("conv_channels", [32, 64, 128]),
            kernel_sizes=cls_raw.get("kernel_sizes", [7, 5, 3]),
            dropout=float(cls_raw.get("dropout", 0.1)),
            n_epochs=int(cls_raw.get("n_epochs", 50)),
            batch_size=int(cls_raw.get("batch_size", 1024)),
            learning_rate=float(cls_raw.get("learning_rate", 0.0001)),
            validation_fraction=float(cls_raw.get("validation_fraction", 0.2)),
            use_posterior_simulations=bool(cls_raw.get("use_posterior_simulations", True)),
        )
        
        # Extract weight augmentation
        wa_raw = raw.get("weight_augmentation", {})
        weight_aug = WeightAugmentationConfig(
            enable=bool(wa_raw.get("enable", True)),
            scattered_prob=float(wa_raw.get("scattered_prob", 0.3)),
            gap_prob=float(wa_raw.get("gap_prob", 0.3)),
            large_block_prob=float(wa_raw.get("large_block_prob", 0.1)),
            noise_variation=bool(wa_raw.get("noise_variation", True)),
        )
        
        return cls(
            freq_file=str(raw.get("freq_file", "freq.txt")),
            priors=priors,
            noise=noise,
            training=training,
            memory=memory,
            model_selection=model_selection,
            physics=physics,
            sbi=sbi,
            classifier=classifier,
            weight_augmentation=weight_aug,
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "freq_file": self.freq_file,
            "priors": {
                "rm": {"min": self.priors.rm_min, "max": self.priors.rm_max},
                "amp": {"min": self.priors.amp_min, "max": self.priors.amp_max},
                "chi0": {"min": self.priors.chi0_min, "max": self.priors.chi0_max},
                "sigma_phi": {"min": self.priors.sigma_phi_min, "max": self.priors.sigma_phi_max},
                "delta_phi": {"min": self.priors.delta_phi_min, "max": self.priors.delta_phi_max},
            },
            "noise": {
                "base_level": self.noise.base_level,
                "augmentation": {
                    "enable": self.noise.augmentation_enable,
                    "min_factor": self.noise.augmentation_min_factor,
                    "max_factor": self.noise.augmentation_max_factor,
                },
            },
            "training": {
                "n_simulations": self.training.n_simulations,
                "simulation_scaling": self.training.simulation_scaling,
                "simulation_scaling_mode": self.training.simulation_scaling_mode,
                "scaling_power": self.training.scaling_power,
                "batch_size": self.training.batch_size,
                "device": self.training.device,
                "validation_fraction": self.training.validation_fraction,
                "save_dir": self.training.save_dir,
            },
            "model_selection": {
                "max_components": self.model_selection.max_components,
                "use_classifier": self.model_selection.use_classifier,
            },
            "physics": {
                "model_types": self.physics.model_types,
            },
        }
    
    def save(self, path: str):
        """Save configuration to YAML file."""
        with open(path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)
