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
    # Apply weight augmentation if enabled
    from .augmentation import augment_weights_combined
    
    xs = []
    for i in tqdm(range(0, n_simulations, batch_size), desc="Simulating"):
        batch_theta = theta[i:i + batch_size]
        batch_size_actual = len(batch_theta)
        
        # Generate augmented weights for this batch
        augmented_weights = np.zeros((batch_size_actual, simulator.n_freq))
        for j in range(batch_size_actual):
            augmented_weights[j] = augment_weights_combined(simulator.weights)
        
        # Simulate with augmented weights
        batch_xs = []
        for j in range(batch_size_actual):
            x_sample = simulator(batch_theta[j:j+1], weights=augmented_weights[j])
            batch_xs.append(x_sample)
        xs.append(np.vstack(batch_xs))
    
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
        "n_freq": simulator.n_freq,
        "lambda_sq": simulator.lambda_sq,
        "path": str(save_path),
        "flat_priors": flat_priors,
    }


def train_decision_layer(
    freq_file: str,
    flat_priors: Dict[str, float],
    config: Dict[str, Any],
    output_dir: Path = Path("models"),
) -> Dict[str, Any]:
    """
    Train the decision layer classifier for 1-comp vs 2-comp selection.
    
    Parameters
    ----------
    freq_file : str
        Path to frequency file
    flat_priors : Dict[str, float]
        Flattened prior configuration
    config : Dict[str, Any]
        Full configuration dictionary
    output_dir : Path
        Directory to save the trained model
        
    Returns
    -------
    Dict[str, Any]
        Training results and metadata
    """
    from .simulator import RMSimulator, sample_prior
    from .decision import DecisionLayerTrainer
    from .augmentation import augment_weights_combined
    
    print(f"\n{'='*60}")
    print("Training Decision Layer (1-comp vs 2-comp classifier)")
    print(f"{'='*60}")
    
    # Get decision layer config
    decision_cfg = config.get("decision_layer", {})
    n_samples = decision_cfg.get("n_training_samples", 5000)
    n_epochs = decision_cfg.get("n_epochs", 20)
    batch_size = decision_cfg.get("batch_size", 64)
    val_fraction = decision_cfg.get("validation_fraction", 0.2)
    device = config.get("training", {}).get("device", "cpu")
    
    # Create simulators for 1-comp and 2-comp
    sim_1comp = RMSimulator(freq_file, 1)
    sim_2comp = RMSimulator(freq_file, 2)
    n_freq = sim_1comp.n_freq
    
    print(f"Generating {n_samples} samples per class...")
    
    # Generate training data
    X_train = []
    y_train = []
    
    # Generate 1-component samples (label = 0)
    theta_1comp = sample_prior(n_samples, 1, flat_priors)
    for i in tqdm(range(n_samples), desc="Simulating 1-comp"):
        aug_weights = augment_weights_combined(sim_1comp.weights)
        qu = sim_1comp(theta_1comp[i:i+1], weights=aug_weights).flatten()
        # Input: [Q, U, weights]
        x = np.concatenate([qu, aug_weights])
        X_train.append(x)
        y_train.append(0)  # 1-component = class 0
    
    # Generate 2-component samples (label = 1)
    theta_2comp = sample_prior(n_samples, 2, flat_priors)
    for i in tqdm(range(n_samples), desc="Simulating 2-comp"):
        aug_weights = augment_weights_combined(sim_2comp.weights)
        qu = sim_2comp(theta_2comp[i:i+1], weights=aug_weights).flatten()
        # Input: [Q, U, weights]
        x = np.concatenate([qu, aug_weights])
        X_train.append(x)
        y_train.append(1)  # 2-component = class 1
    
    X_train = np.array(X_train)
    y_train = np.array(y_train)
    
    # Shuffle and split
    indices = np.random.permutation(len(X_train))
    X_train = X_train[indices]
    y_train = y_train[indices]
    
    n_val = int(len(X_train) * val_fraction)
    X_val = X_train[:n_val]
    y_val = y_train[:n_val]
    X_train = X_train[n_val:]
    y_train = y_train[n_val:]
    
    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
    
    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.long)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)
    y_val_t = torch.tensor(y_val, dtype=torch.long)
    
    # Create data loaders
    train_dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
    val_dataset = torch.utils.data.TensorDataset(X_val_t, y_val_t)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    # Train
    trainer = DecisionLayerTrainer(n_freq=n_freq, device=device)
    history = trainer.train(train_loader, val_loader, n_epochs=n_epochs, verbose=True)
    
    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "decision_layer.pkl"
    trainer.save(str(save_path))
    
    print(f"\nSaved decision layer -> {save_path}")
    print(f"Final validation accuracy: {history['val_accuracy'][-1]:.2f}%")
    
    return {
        "model_path": str(save_path),
        "n_freq": n_freq,
        "history": history,
        "final_val_accuracy": history['val_accuracy'][-1],
    }


def train_all_models(config: Dict[str, Any]) -> Dict[int, Dict[str, Any]]:
    """
    Train models for N = 1 ..  max_components using the exact config structure
    the repository is using.  Returns dict mapping n_components -> saved data dict.
    
    Now with two-layer system:
    - Train worker models (1-comp and 2-comp)
    - Train decision layer classifier
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict loaded from YAML.")

    freq_file = config.get("freq_file", "freq.txt")
    training_cfg = _extract_training_cfg(config)
    flat_priors = _flatten_priors(config)

    results: Dict[int, Dict[str, Any]] = {}

    # Train worker models (limit to 1 and 2 components for two-layer system)
    max_comp = min(training_cfg["max_components"], 2)
    
    print(f"\n{'='*60}")
    print("TRAINING TWO-LAYER SYSTEM")
    print(f"{'='*60}")
    print("Phase 1: Training Worker Models (1-comp and 2-comp)")
    
    for n in range(1, max_comp + 1):
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

    print("\n" + "="*60)
    print("Phase 2: Training Decision Layer")
    print("="*60)
    
    # Train decision layer if enabled
    model_selection = config.get("model_selection", {})
    if model_selection.get("use_decision_layer", True):
        decision_result = train_decision_layer(
            freq_file=freq_file,
            flat_priors=flat_priors,
            config=config,
            output_dir=training_cfg["output_dir"],
        )
        results["decision_layer"] = decision_result
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Worker models trained: N = {[n for n in results.keys() if isinstance(n, int)]}")
    if "decision_layer" in results:
        print(f"Decision layer accuracy: {results['decision_layer']['final_val_accuracy']:.2f}%")
    
    return results