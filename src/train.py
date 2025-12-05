#!/usr/bin/env python3
"""
Training utilities for VROOM-SBI tailored to the provided config layout. 

This module expects config. yaml with the exact structure you provided:
- freq_file
- phi: min, max, n_samples
- priors: rm {min,max}, amp {min,max}, noise {min,max}
- training: n_simulations, batch_size, n_rounds, device, validation_fraction
- model_selection: max_components, use_log_evidence

Public API:
- train_all_models(config: dict) -> dict[int, dict]
"""
from pathlib import Path
import pickle
from typing import Dict, Any

import numpy as np
import torch
from tqdm import tqdm

from sbi.inference import SNPE

from . simulator import RMSimulator, build_prior, sample_prior


def _flatten_priors(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Turn the config['priors'] block into the flat prior dict used by simulator functions."""
    pri = cfg. get("priors", {})
    rm = pri.get("rm", {})
    amp = pri.get("amp", {})
    noise = pri.get("noise", {})

    rm_min = float(rm.get("min", -800.0))
    rm_max = float(rm.get("max", 800.0))

    amp_min = float(amp.get("min", 1e-6))
    amp_max = float(amp.get("max", 1.0))

    noise_min = float(noise.get("min", 1e-9))
    noise_max = float(noise.get("max", 0.1))

    # Safety guards
    if amp_min <= 0:
        amp_min = 1e-6
    if noise_min <= 0:
        noise_min = 1e-9

    return {
        "rm_min": rm_min,
        "rm_max": rm_max,
        "amp_min": amp_min,
        "amp_max": amp_max,
        "noise_min": noise_min,
        "noise_max": noise_max,
    }


def _extract_training_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract training hyperparameters from the exact config layout you provided."""
    training = cfg.get("training", {}) or {}
    model_selection = cfg. get("model_selection", {}) or {}

    max_components = int(model_selection.get("max_components", 5))
    n_simulations = int(training.get("n_simulations", 10000))
    batch_size = int(training.get("batch_size", 10000))
    n_rounds = int(training.get("n_rounds", 1))
    device = training.get("device", "cpu")
    validation_fraction = float(training.get("validation_fraction", 0.1))
    
    # NEW: scaling factor for complex models
    simulation_scaling = training.get("simulation_scaling", True)

    output_dir = Path(training.get("save_dir", "models"))
    output_dir.mkdir(parents=True, exist_ok=True)

    return {
        "max_components": max_components,
        "n_simulations": n_simulations,
        "batch_size": batch_size,
        "n_rounds": n_rounds,
        "device": device,
        "validation_fraction": validation_fraction,
        "output_dir": output_dir,
        "simulation_scaling": simulation_scaling,
    }


def get_scaled_simulations(n_components: int, base_simulations: int, scaling: bool = True) -> int:
    """
    Scale the number of simulations based on model complexity.
    
    Higher-dimensional models need more training data for good posterior coverage.
    
    Parameters
    ----------
    n_components : int
        Number of RM components
    base_simulations : int
        Base number of simulations (for N=1)
    scaling : bool
        Whether to apply scaling (if False, returns base_simulations)
    
    Returns
    -------
    int
        Scaled number of simulations
    """
    if not scaling:
        return base_simulations
    
    # Scaling factors: N=1 -> 1x, N=2 -> 2x, N=3 -> 4x, N=4 -> 6x, N=5 -> 8x
    scaling_factors = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
    factor = scaling_factors. get(n_components, n_components * 2)
    
    return base_simulations * factor


def train_model(
    n_components: int,
    freq_file: str,
    flat_priors: Dict[str, float],
    n_simulations: int = 10000,
    batch_size: int = 10000,
    device: str = "cpu",
    validation_fraction: float = 0.1,
    output_dir: Path = Path("models"),
) -> Dict[str, Any]:
    """
    Train a single model with n_components, using the provided flat_priors
    (keys: rm_min, rm_max, amp_min, amp_max, noise_min, noise_max). 

    Saves the posterior to output_dir/posterior_n{n_components}.pkl and returns metadata dict.
    """
    print(f"\n{'='*60}")
    print(f"Training model N={n_components} on device={device}")
    print(f"{'='*60}")

    simulator = RMSimulator(freq_file, n_components)

    # Build prior on requested device
    try:
        prior = build_prior(n_components, flat_priors, device=device)
    except TypeError:
        prior = build_prior(n_components, flat_priors)

    print(f"Simulator params: n_params={simulator.n_params}, n_freq={simulator.n_freq}")
    print(f"Simulations: {n_simulations:,}, batch_size: {batch_size}, validation_fraction: {validation_fraction}")

    # Generate simulations (numpy)
    theta = sample_prior(n_simulations, n_components, flat_priors)

    # Simulate in batches to avoid excessive memory usage
    xs = []
    for i in tqdm(range(0, n_simulations, batch_size), desc="Simulating"):
        batch_theta = theta[i:i + batch_size]
        xs.append(simulator(batch_theta))
    x = np.vstack(xs)

    # Convert to torch and move to device
    try:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    except Exception:
        theta_t = torch.tensor(theta, dtype=torch.float32, device="cpu")
        x_t = torch. tensor(x, dtype=torch.float32, device="cpu")
        device = "cpu"
        print("Warning: falling back to CPU tensors (device change).")

    # Create SNPE inference object
    inference = SNPE(prior=prior, device=device)

    # Append simulations and train
    inference.append_simulations(theta_t, x_t)

    density_estimator = inference.train(
        training_batch_size=min(4096, max(1, batch_size)),
        learning_rate=5e-4,
        show_train_summary=True,
        validation_fraction=validation_fraction,
    )

    posterior = inference.build_posterior(density_estimator)

    # Save posterior and metadata
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / f"posterior_n{n_components}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "posterior": posterior,
            "n_components": n_components,
            "n_freq": simulator.n_freq,
            "lambda_sq": simulator.lambda_sq,
            "flat_priors": flat_priors,
            "config_used": {
                "freq_file": freq_file,
                "n_simulations": n_simulations,
                "batch_size": batch_size,
                "device": device,
            },
        }, f)

    print(f"Saved posterior -> {save_path}")

    return {
        "posterior": posterior,
        "n_components": n_components,
        "n_freq": simulator. n_freq,
        "lambda_sq": simulator.lambda_sq,
        "path": str(save_path),
        "flat_priors": flat_priors,
    }


def train_all_models(config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Train models for N = 1 ..  max_components using the exact config structure
    the repository is using.  Returns dict mapping n_components -> saved data dict.
    
    Now with automatic scaling of simulations for complex models!
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict loaded from YAML.")

    freq_file = config. get("freq_file", "freq.txt")
    training_cfg = _extract_training_cfg(config)
    flat_priors = _flatten_priors(config)

    results: Dict[int, Dict[str, Any]] = {}

    for n in range(1, training_cfg["max_components"] + 1):
        # Scale simulations for complex models
        n_sims = get_scaled_simulations(
            n_components=n,
            base_simulations=training_cfg["n_simulations"],
            scaling=training_cfg["simulation_scaling"],
        )
        
        print(f"\n>>> Model N={n}: Using {n_sims:,} simulations (base: {training_cfg['n_simulations']:,})")
        
        data = train_model(
            n_components=n,
            freq_file=freq_file,
            flat_priors=flat_priors,
            n_simulations=n_sims,
            batch_size=training_cfg["batch_size"],
            device=training_cfg["device"],
            validation_fraction=training_cfg["validation_fraction"],
            output_dir=training_cfg["output_dir"],
        )
        results[n] = data

    print("\nAll done.  Trained models for N =", list(results.keys()))
    return results