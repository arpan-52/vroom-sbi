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
            f"Data workers: {self.num_workers}",
            f"Prefetch factor: {self.prefetch_factor}",
            f"Pin memory: {self.pin_memory}",
            "=" * 60,
        ]
        return "\n".join(lines)


def detect_hardware() -> HardwareInfo:
    """
    Detect available hardware (GPU, RAM, CPU).
    
    Returns
    -------
    HardwareInfo
        Detected hardware specifications
    """
    info = HardwareInfo()
    
    # Detect CPU cores
    try:
        info.cpu_cores = os.cpu_count() or 1
    except:
        info.cpu_cores = 1
    
    # Detect RAM
    try:
        import psutil
        mem = psutil.virtual_memory()
        info.ram_total_gb = mem.total / (1024**3)
        info.ram_available_gb = mem.available / (1024**3)
    except ImportError:
        # Fallback: try to read from /proc/meminfo on Linux
        try:
            with open('/proc/meminfo', 'r') as f:
                for line in f:
                    if line.startswith('MemTotal:'):
                        info.ram_total_gb = int(line.split()[1]) / (1024**2)
                    elif line.startswith('MemAvailable:'):
                        info.ram_available_gb = int(line.split()[1]) / (1024**2)
        except:
            info.ram_total_gb = 16.0  # Assume 16GB
            info.ram_available_gb = 8.0
    
    # Detect GPU
    try:
        import torch
        if torch.cuda.is_available():
            info.gpu_available = True
            info.num_gpus = torch.cuda.device_count()
            info.gpu_name = torch.cuda.get_device_name(0)
            
            # Get VRAM info
            props = torch.cuda.get_device_properties(0)
            info.gpu_vram_gb = props.total_memory / (1024**3)
            
            # Get free VRAM
            torch.cuda.empty_cache()
            free_mem, total_mem = torch.cuda.mem_get_info(0)
            info.gpu_vram_free_gb = free_mem / (1024**3)
            
            # CUDA version
            info.cuda_version = torch.version.cuda or "unknown"
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")
        info.gpu_available = False
    
    return info


def optimize_for_hardware(
    hardware: Optional[HardwareInfo] = None,
    verbose: bool = True
) -> OptimizedSettings:
    """
    Calculate optimal training settings based on detected hardware.
    
    Goal: Maximize GPU utilization (~95-100%) by:
    - Setting batch size based on VRAM
    - Setting prefetch/workers based on RAM
    
    Parameters
    ----------
    hardware : HardwareInfo, optional
        Hardware info (auto-detected if not provided)
    verbose : bool
        Print hardware and settings info
        
    Returns
    -------
    OptimizedSettings
        Optimized training configuration
    """
    if hardware is None:
        hardware = detect_hardware()
    
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
    # Goal: Use as much VRAM as possible without OOM
    # Reserve ~2-3GB for model + overhead, use rest for batches
    if hardware.gpu_available:
        vram = hardware.gpu_vram_free_gb
        
        if vram >= 22:      # 24GB+ GPU (A100, RTX 4090)
            settings.training_batch_size = 2048
        elif vram >= 14:    # 16GB GPU (V100, RTX 4080)
            settings.training_batch_size = 1024
        elif vram >= 10:    # 12GB GPU (RTX 3080)
            settings.training_batch_size = 512
        elif vram >= 6:     # 8GB GPU (RTX 3070)
            settings.training_batch_size = 256
        elif vram >= 4:     # 6GB GPU
            settings.training_batch_size = 128
        else:               # <4GB GPU
            settings.training_batch_size = 64
            logger.warning("Low VRAM - training may be slow")
    else:
        # CPU mode
        settings.training_batch_size = 128
    
    # === RAM-based Prefetching ===
    # More RAM = more aggressive prefetching = GPU never waits for data
    ram = hardware.ram_available_gb
    
    if ram >= 64:
        settings.num_workers = min(16, hardware.cpu_cores)
        settings.prefetch_factor = 8
    elif ram >= 32:
        settings.num_workers = min(8, hardware.cpu_cores)
        settings.prefetch_factor = 4
    elif ram >= 16:
        settings.num_workers = min(4, hardware.cpu_cores)
        settings.prefetch_factor = 2
    elif ram >= 8:
        settings.num_workers = min(2, hardware.cpu_cores)
        settings.prefetch_factor = 2
    else:
        settings.num_workers = 1
        settings.prefetch_factor = 2
    
    # Pin memory only if we have GPU and enough RAM
    settings.pin_memory = hardware.gpu_available and ram >= 8
    
    if verbose:
        print(settings)
    
    return settings


def apply_settings_to_config(config, settings: OptimizedSettings):
    """
    Apply optimized settings to a Configuration object.
    
    Parameters
    ----------
    config : Configuration
        Configuration object to modify
    settings : OptimizedSettings
        Optimized settings to apply
    """
    # Training settings
    config.training.device = settings.device
    config.training.training_batch_size = settings.training_batch_size
    
    # Store data loading settings for use during training
    if not hasattr(config, '_optimized'):
        config._optimized = {}
    config._optimized['num_workers'] = settings.num_workers
    config._optimized['pin_memory'] = settings.pin_memory
    config._optimized['prefetch_factor'] = settings.prefetch_factor
    
    return config


def auto_configure(config, verbose: bool = True):
    """
    Convenience function: detect hardware and apply optimal settings to config.
    
    Parameters
    ----------
    config : Configuration
        Configuration to optimize
    verbose : bool
        Print hardware and settings info
        
    Returns
    -------
    Configuration
        Modified configuration with optimal settings
    """
    hardware = detect_hardware()
    settings = optimize_for_hardware(hardware, verbose=verbose)
    return apply_settings_to_config(config, settings)
