"""
Hardware detection and automatic optimization for VROOM-SBI.

Detects available GPU/CPU resources and configures optimal training parameters
to achieve ~95-100% GPU utilization.
"""

import os
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class HardwareInfo:
    """Detected hardware specifications."""
    # GPU info
    gpu_available: bool = False
    gpu_name: str = ""
    gpu_vram_gb: float = 0.0
    gpu_vram_free_gb: float = 0.0
    num_gpus: int = 0
    cuda_version: str = ""
    
    # CPU/RAM info
    ram_total_gb: float = 0.0
    ram_available_gb: float = 0.0
    cpu_cores: int = 1
    
    def __str__(self):
        lines = ["=" * 60, "HARDWARE DETECTED", "=" * 60]
        if self.gpu_available:
            lines.append(f"GPU: {self.gpu_name}")
            lines.append(f"VRAM: {self.gpu_vram_gb:.1f} GB total, {self.gpu_vram_free_gb:.1f} GB free")
            lines.append(f"CUDA: {self.cuda_version}")
        else:
            lines.append("GPU: None (CPU mode)")
        lines.append(f"RAM: {self.ram_total_gb:.1f} GB total, {self.ram_available_gb:.1f} GB available")
        lines.append(f"CPU cores: {self.cpu_cores}")
        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass  
class OptimizedSettings:
    """Auto-optimized training settings based on hardware."""
    # Training batch size (sized to maximize VRAM usage)
    training_batch_size: int = 256
    
    # Simulation batch size (sized based on RAM)
    simulation_batch_size: int = 5000
    
    # Data loading (sized to prefetch into RAM)
    num_workers: int = 4
    pin_memory: bool = True
    prefetch_factor: int = 2
    
    # Device
    device: str = "cuda"
    
    def __str__(self):
        lines = [
            "=" * 60,
            "OPTIMIZED SETTINGS",
            "=" * 60,
            f"Device: {self.device}",
            f"Training batch size: {self.training_batch_size}",
            f"Simulation batch size: {self.simulation_batch_size}",
            f"Data workers: {self.num_workers}",
            f"Prefetch factor: {self.prefetch_factor}",
            f"Pin memory: {self.pin_memory}",
            "=" * 60,
        ]
        return "\n".join(lines)


def detect_hardware(config_hardware=None) -> HardwareInfo:
    """
    Detect available hardware (GPU, RAM, CPU).
    
    If config_hardware is provided, use those values instead of auto-detecting.
    
    Parameters
    ----------
    config_hardware : HardwareConfig, optional
        Hardware settings from config (0 = auto-detect)
    
    Returns
    -------
    HardwareInfo
        Detected hardware specifications
    """
    info = HardwareInfo()
    
    # Check if config provides explicit values
    config_vram = getattr(config_hardware, 'vram_gb', 0) if config_hardware else 0
    config_ram = getattr(config_hardware, 'ram_gb', 0) if config_hardware else 0
    config_workers = getattr(config_hardware, 'num_workers', 0) if config_hardware else 0
    
    # Detect CPU cores
    try:
        info.cpu_cores = os.cpu_count() or 1
    except:
        info.cpu_cores = 1
    
    # Detect or use config RAM
    if config_ram > 0:
        info.ram_total_gb = config_ram
        info.ram_available_gb = config_ram * 0.8  # Assume 80% available
        logger.info(f"Using config RAM: {config_ram} GB")
    else:
        # Auto-detect RAM
        try:
            import psutil
            mem = psutil.virtual_memory()
            info.ram_total_gb = mem.total / (1024**3)
            info.ram_available_gb = mem.available / (1024**3)
        except ImportError:
            # Fallback: read from /proc/meminfo on Linux
            try:
                with open('/proc/meminfo', 'r') as f:
                    for line in f:
                        if line.startswith('MemTotal:'):
                            info.ram_total_gb = int(line.split()[1]) / (1024**2)
                        elif line.startswith('MemAvailable:'):
                            info.ram_available_gb = int(line.split()[1]) / (1024**2)
                if info.ram_total_gb == 0:
                    raise ValueError("Could not parse /proc/meminfo")
            except Exception as e:
                logger.warning(f"RAM detection failed: {e}. Set hardware.ram_gb in config.yaml")
                info.ram_total_gb = 16.0
                info.ram_available_gb = 12.0
    
    # Detect or use config VRAM
    try:
        import torch
        if torch.cuda.is_available():
            info.gpu_available = True
            info.num_gpus = torch.cuda.device_count()
            info.gpu_name = torch.cuda.get_device_name(0)
            info.cuda_version = torch.version.cuda or "unknown"
            
            if config_vram > 0:
                info.gpu_vram_gb = config_vram
                info.gpu_vram_free_gb = config_vram * 0.9  # Assume 90% available
                logger.info(f"Using config VRAM: {config_vram} GB")
            else:
                # Auto-detect VRAM
                props = torch.cuda.get_device_properties(0)
                info.gpu_vram_gb = props.total_memory / (1024**3)
                torch.cuda.empty_cache()
                free_mem, total_mem = torch.cuda.mem_get_info(0)
                info.gpu_vram_free_gb = free_mem / (1024**3)
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
        info.gpu_available = False
    
    # Override num_workers if config specifies
    if config_workers > 0:
        info.cpu_cores = config_workers * 2  # Will be divided by 2 later
    
    return info


def optimize_for_hardware(
    hardware: Optional[HardwareInfo] = None,
    config=None,
    verbose: bool = True
) -> OptimizedSettings:
    """
    Calculate optimal training settings based on hardware.
    
    Goal: Maximize GPU utilization (~95-100%) by:
    - Setting batch size based on VRAM
    - Setting prefetch/workers based on RAM
    - Setting parallel jobs based on CPU cores
    
    Parameters
    ----------
    hardware : HardwareInfo, optional
        Hardware info (auto-detected if not provided)
    config : Configuration, optional
        Configuration object (to get hardware settings)
    verbose : bool
        Print hardware and settings info
        
    Returns
    -------
    OptimizedSettings
        Optimized training configuration
    """
    # Get hardware config if available
    config_hardware = getattr(config, 'hardware', None) if config else None
    
    if hardware is None:
        hardware = detect_hardware(config_hardware)
    
    if verbose:
        print(hardware)
    
    settings = OptimizedSettings()
    
    # === Device Selection ===
    if hardware.gpu_available:
        settings.device = "cuda"
    else:
        settings.device = "cpu"
        logger.warning("No GPU detected - training will be slow on CPU")
    
    # === VRAM-based Batch Size ===
    # Calculate based on actual memory requirements
    #
    # Memory per sample during training (approximate):
    # - Input: n_freq * 2 (Q,U) * 4 bytes = ~4KB for 500 channels
    # - Forward pass activations: ~10x input = ~40KB
    # - Backward pass gradients: ~10x input = ~40KB  
    # - Optimizer states (Adam): ~2x parameters
    # - Total per sample: ~100KB with float32
    #
    # Model memory (NSF with embedding):
    # - Embedding net: ~1-2M params = ~8MB
    # - NSF flow: ~5-10M params = ~40MB
    # - Total model: ~50-100MB
    #
    # Safe formula: batch_size = (VRAM_available - model_overhead) / mem_per_sample
    
    if hardware.gpu_available:
        vram_gb = hardware.gpu_vram_free_gb
        
        # Reserve 2GB for model, CUDA overhead, fragmentation
        usable_vram_gb = max(0.5, vram_gb - 2.0)
        usable_vram_mb = usable_vram_gb * 1024
        
        # ~0.5 MB per sample (conservative estimate including all activations)
        mem_per_sample_mb = 0.5
        
        # Calculate max batch size
        max_batch = int(usable_vram_mb / mem_per_sample_mb)
        
        # Round down to nearest power of 2 for efficiency
        batch_size = 1
        while batch_size * 2 <= max_batch:
            batch_size *= 2
        
        # Clamp to reasonable range
        settings.training_batch_size = max(64, min(4096, batch_size))
        
        logger.info(f"VRAM: {vram_gb:.1f}GB free → batch_size={settings.training_batch_size}")
    else:
        settings.training_batch_size = 128
    
    # === RAM-based Prefetching ===
    ram = hardware.ram_available_gb
    
    if ram >= 64:
        settings.num_workers = min(16, hardware.cpu_cores)
        settings.prefetch_factor = 8
        settings.simulation_batch_size = 50000  # 60GB+ RAM: huge batches
    elif ram >= 32:
        settings.num_workers = min(8, hardware.cpu_cores)
        settings.prefetch_factor = 4
        settings.simulation_batch_size = 20000  # 32GB RAM
    elif ram >= 16:
        settings.num_workers = min(4, hardware.cpu_cores)
        settings.prefetch_factor = 2
        settings.simulation_batch_size = 10000  # 16GB RAM
    elif ram >= 8:
        settings.num_workers = min(2, hardware.cpu_cores)
        settings.prefetch_factor = 2
        settings.simulation_batch_size = 5000   # 8GB RAM
    else:
        settings.num_workers = 1
        settings.prefetch_factor = 2
        settings.simulation_batch_size = 1000   # Low RAM
    
    # Pin memory only if we have GPU and enough RAM
    settings.pin_memory = hardware.gpu_available and ram >= 8
    
    if verbose:
        print(settings)
    
    return settings


def apply_settings_to_config(config, settings: OptimizedSettings):
    """
    Apply optimized settings to a Configuration object.
    """
    # Training settings
    config.training.device = settings.device
    
    # Only override if set to auto (0)
    if config.training.training_batch_size == 0:
        config.training.training_batch_size = settings.training_batch_size
    
    if config.training.simulation_batch_size == 0:
        config.training.simulation_batch_size = settings.simulation_batch_size
    
    # Store data loading settings
    if not hasattr(config, '_optimized'):
        config._optimized = {}
    config._optimized['num_workers'] = settings.num_workers
    config._optimized['pin_memory'] = settings.pin_memory
    config._optimized['prefetch_factor'] = settings.prefetch_factor
    
    return config


def auto_configure(config, verbose: bool = True):
    """
    Convenience function: detect hardware and apply optimal settings to config.
    """
    hardware = detect_hardware(getattr(config, 'hardware', None))
    settings = optimize_for_hardware(hardware, config=config, verbose=verbose)
    return apply_settings_to_config(config, settings)
