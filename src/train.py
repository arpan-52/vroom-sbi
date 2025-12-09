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
import torch.nn as nn
from tqdm import tqdm

from sbi.inference import SNPE
from sbi.utils import posterior_nn

from . simulator import RMSimulator, build_prior, sample_prior


# ============================================================================
# Custom Embedding Network for SBI
# ============================================================================

class SpectralEmbedding(nn.Module):
    """
    Custom embedding network for high-dimensional spectral data.

    Processes Q and U spectra (typical dims: 200-400) into a lower-dimensional
    representation suitable for the flow-based density estimator.
    """

    def __init__(self, input_dim: int, output_dim: int = 64):
        """
        Parameters
        ----------
        input_dim : int
            Dimension of input spectra (2 * n_freq for Q and U)
        output_dim : int
            Dimension of output embedding (default: 64)
        """
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),

            nn.Linear(128, output_dim),
            nn.ReLU()
        )

    def forward(self, x):
        return self.net(x)


def _flatten_priors(cfg: Dict[str, Any]) -> Dict[str, float]:
    """Turn the config['priors'] block into the flat prior dict used by simulator functions."""
    pri = cfg.get("priors", {})
    rm = pri.get("rm", {})
    amp = pri.get("amp", {})

    rm_min = float(rm.get("min", -800.0))
    rm_max = float(rm.get("max", 800.0))

    amp_min = float(amp.get("min", 1e-6))
    amp_max = float(amp.get("max", 1.0))

    # Safety guards
    if amp_min <= 0:
        amp_min = 1e-6

    return {
        "rm_min": rm_min,
        "rm_max": rm_max,
        "amp_min": amp_min,
        "amp_max": amp_max,
    }


def _get_base_noise_level(cfg: Dict[str, Any]) -> float:
    """Extract base noise level from config."""
    noise_cfg = cfg.get("noise", {})
    base_level = float(noise_cfg.get("base_level", 0.01))
    
    # Safety guard
    if base_level <= 0:
        base_level = 0.01
    
    return base_level


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
    base_noise_level: float = 0.01,
    n_simulations: int = 10000,
    batch_size: int = 10000,
    device: str = "cpu",
    validation_fraction: float = 0.1,
    output_dir: Path = Path("models"),
) -> Dict[str, Any]:
    """
    Train a single model with n_components, using the provided flat_priors
    (keys: rm_min, rm_max, amp_min, amp_max). 

    Saves the posterior to output_dir/posterior_n{n_components}.pkl and returns metadata dict.
    """
    print(f"\n{'='*60}")
    print(f"Training model N={n_components} on device={device}")
    print(f"{'='*60}")

    simulator = RMSimulator(freq_file, n_components, base_noise_level=base_noise_level)

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
    # Apply weight and noise augmentation if enabled
    from .augmentation import augment_weights_combined, augment_base_noise_level

    xs = []
    for i in tqdm(range(0, n_simulations, batch_size), desc="Simulating"):
        batch_theta = theta[i:i + batch_size]
        batch_size_actual = len(batch_theta)

        # Generate augmented weights for this batch
        augmented_weights = np.zeros((batch_size_actual, simulator.n_freq))
        for j in range(batch_size_actual):
            augmented_weights[j] = augment_weights_combined(simulator.weights)

        # Simulate with augmented weights and noise levels
        batch_xs = []
        for j in range(batch_size_actual):
            # Augment base noise level for this sample
            aug_noise = augment_base_noise_level(base_noise_level, min_factor=0.5, max_factor=2.0)
            simulator.base_noise_level = aug_noise

            x_sample = simulator(batch_theta[j:j+1], weights=augmented_weights[j])
            batch_xs.append(x_sample)
        xs.append(np.vstack(batch_xs))

    # Restore original base noise level
    simulator.base_noise_level = base_noise_level
    
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

    # Create SNPE inference object with custom density estimator
    # Build custom embedding for spectral data
    input_dim = 2 * simulator.n_freq  # Q and U
    embedding_net = SpectralEmbedding(input_dim=input_dim, output_dim=64).to(device)

    # Build neural posterior with NSF (Neural Spline Flow) and custom embedding
    density_estimator_builder = posterior_nn(
        model="nsf",              # Neural Spline Flow (better than MAF)
        hidden_features=128,      # Larger hidden layers for complex posteriors
        num_transforms=10,        # More transforms for better expressivity
        num_bins=16,              # Spline resolution
        embedding_net=embedding_net
    )

    print(f"Using Neural Spline Flow with custom spectral embedding:")
    print(f"  Input dim: {input_dim}, Embedding dim: 64")
    print(f"  Hidden features: 128, Num transforms: 10")

    inference = SNPE(prior=prior, density_estimator=density_estimator_builder, device=device)

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
    base_noise_level: float,
    config: Dict[str, Any],
    output_dir: Path = Path("models"),
) -> Dict[str, Any]:
    """
    Train the quality prediction decision layer.

    This function trains a neural network to predict AIC, BIC, and log evidence
    for both 1-comp and 2-comp models given an observed spectrum.

    NOTE: This is expensive during training because we must fit both SBI models
    to each simulated spectrum to get the true quality metrics.

    Parameters
    ----------
    freq_file : str
        Path to frequency file
    flat_priors : Dict[str, float]
        Flattened prior configuration
    base_noise_level : float
        Fixed base noise level
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
    from .decision import QualityPredictionTrainer
    from .augmentation import augment_weights_combined, augment_base_noise_level

    print(f"\n{'='*60}")
    print("Training Quality Prediction Decision Layer")
    print(f"{'='*60}")
    
    # Get decision layer config
    decision_cfg = config.get("decision_layer", {})
    n_samples = decision_cfg.get("n_training_samples", 1000)  # Fewer samples (expensive!)
    n_epochs = decision_cfg.get("n_epochs", 50)
    batch_size = decision_cfg.get("batch_size", 32)
    val_fraction = decision_cfg.get("validation_fraction", 0.2)
    device = config.get("training", {}).get("device", "cpu")
    hidden_dims = decision_cfg.get("hidden_dims", [256, 128, 64])

    # Create simulators
    sim_1comp = RMSimulator(freq_file, 1, base_noise_level=base_noise_level)
    sim_2comp = RMSimulator(freq_file, 2, base_noise_level=base_noise_level)
    n_freq = sim_1comp.n_freq

    # Build priors for SBI
    prior_1comp = build_prior(1, flat_priors, device=device)
    prior_2comp = build_prior(2, flat_priors, device=device)

    print(f"Generating {n_samples} training samples (this will take a while)...")
    print(f"For each sample, we fit BOTH 1-comp and 2-comp models to compute true quality metrics.")

    # Data containers
    X_all = []  # Spectra + weights
    targets_log_ev = []  # [log_ev_1, log_ev_2]
    targets_aic = []  # [AIC_1, AIC_2]
    targets_bic = []  # [BIC_1, BIC_2]
    true_n_list = []  # Ground truth N (1 or 2)

    # Helper function to compute AIC/BIC
    def compute_quality_metrics(posterior, x_obs, n_params, n_data_points=100, n_samples=1000):
        """Compute log evidence, AIC, and BIC for a posterior."""
        x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device)

        # Sample from posterior
        samples = posterior.sample((n_samples,), x=x_obs_t)

        # Compute log evidence via importance sampling
        log_probs = posterior.log_prob(samples, x=x_obs_t)
        log_evidence = torch.logsumexp(log_probs, dim=0).item() - np.log(n_samples)

        # Compute log likelihood (average of log probs)
        log_likelihood = torch.mean(log_probs).item()

        # AIC = 2k - 2 * log(L)
        aic = 2 * n_params - 2 * log_likelihood

        # BIC = k * log(n) - 2 * log(L)
        bic = n_params * np.log(n_data_points) - 2 * log_likelihood

        return log_evidence, aic, bic

    # Generate samples from both 1-comp and 2-comp
    for class_label, n_comp in enumerate([1, 2]):
        print(f"\nGenerating {n_samples} samples from {n_comp}-component model...")

        theta_samples = sample_prior(n_samples, n_comp, flat_priors)
        simulator = sim_1comp if n_comp == 1 else sim_2comp

        for i in tqdm(range(n_samples), desc=f"Processing {n_comp}-comp"):
            # Simulate spectrum with augmented weights and noise
            aug_weights = augment_weights_combined(simulator.weights)
            aug_noise = augment_base_noise_level(base_noise_level, min_factor=0.5, max_factor=2.0)
            simulator.base_noise_level = aug_noise
            qu_obs = simulator(theta_samples[i:i+1], weights=aug_weights).flatten()

            # Input: [Q, U, weights]
            x_input = np.concatenate([qu_obs, aug_weights])

            # Now fit BOTH models to this spectrum (expensive!)
            # For computational efficiency, use small number of simulations

            # Fit 1-comp model
            theta_1 = sample_prior(500, 1, flat_priors)  # Reduced for speed
            x_1 = []
            for j in range(len(theta_1)):
                x_1.append(sim_1comp(theta_1[j:j+1], weights=aug_weights).flatten())
            x_1 = np.array(x_1)

            theta_1_t = torch.tensor(theta_1, dtype=torch.float32, device=device)
            x_1_t = torch.tensor(x_1, dtype=torch.float32, device=device)

            inf_1 = SNPE(prior=prior_1comp, device=device, show_progress_bars=False)
            inf_1.append_simulations(theta_1_t, x_1_t)
            dens_1 = inf_1.train(training_batch_size=128, show_train_summary=False)
            post_1 = inf_1.build_posterior(dens_1)

            # Fit 2-comp model
            theta_2 = sample_prior(500, 2, flat_priors)
            x_2 = []
            for j in range(len(theta_2)):
                x_2.append(sim_2comp(theta_2[j:j+1], weights=aug_weights).flatten())
            x_2 = np.array(x_2)

            theta_2_t = torch.tensor(theta_2, dtype=torch.float32, device=device)
            x_2_t = torch.tensor(x_2, dtype=torch.float32, device=device)

            inf_2 = SNPE(prior=prior_2comp, device=device, show_progress_bars=False)
            inf_2.append_simulations(theta_2_t, x_2_t)
            dens_2 = inf_2.train(training_batch_size=128, show_train_summary=False)
            post_2 = inf_2.build_posterior(dens_2)

            # Compute quality metrics
            n_params_1 = 3  # RM, q, u
            n_params_2 = 6  # 2*RM, 2*q, 2*u

            log_ev_1, aic_1, bic_1 = compute_quality_metrics(post_1, qu_obs, n_params_1)
            log_ev_2, aic_2, bic_2 = compute_quality_metrics(post_2, qu_obs, n_params_2)

            # Store
            X_all.append(x_input)
            targets_log_ev.append([log_ev_1, log_ev_2])
            targets_aic.append([aic_1, aic_2])
            targets_bic.append([bic_1, bic_2])
            true_n_list.append(n_comp)

    # Convert to arrays
    X_all = np.array(X_all)
    targets_log_ev = np.array(targets_log_ev)
    targets_aic = np.array(targets_aic)
    targets_bic = np.array(targets_bic)
    true_n = np.array(true_n_list)

    print(f"\nGenerated {len(X_all)} total samples")

    # Shuffle
    indices = np.random.permutation(len(X_all))
    X_all = X_all[indices]
    targets_log_ev = targets_log_ev[indices]
    targets_aic = targets_aic[indices]
    targets_bic = targets_bic[indices]
    true_n = true_n[indices]

    # Split train/val
    n_val = int(len(X_all) * val_fraction)
    X_train = X_all[n_val:]
    X_val = X_all[:n_val]
    targets_train = {
        'log_evidence': targets_log_ev[n_val:],
        'aic': targets_aic[n_val:],
        'bic': targets_bic[n_val:],
        'true_n': true_n[n_val:]
    }
    targets_val = {
        'log_evidence': targets_log_ev[:n_val],
        'aic': targets_aic[:n_val],
        'bic': targets_bic[:n_val],
        'true_n': true_n[:n_val]
    }

    print(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")

    # Convert to torch
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    X_val_t = torch.tensor(X_val, dtype=torch.float32)

    targets_train_t = {
        'log_evidence': torch.tensor(targets_train['log_evidence'], dtype=torch.float32),
        'aic': torch.tensor(targets_train['aic'], dtype=torch.float32),
        'bic': torch.tensor(targets_train['bic'], dtype=torch.float32),
        'true_n': torch.tensor(targets_train['true_n'], dtype=torch.long)
    }
    targets_val_t = {
        'log_evidence': torch.tensor(targets_val['log_evidence'], dtype=torch.float32),
        'aic': torch.tensor(targets_val['aic'], dtype=torch.float32),
        'bic': torch.tensor(targets_val['bic'], dtype=torch.float32),
        'true_n': torch.tensor(targets_val['true_n'], dtype=torch.long)
    }

    # Custom dataset class
    class QualityDataset(torch.utils.data.Dataset):
        def __init__(self, X, targets):
            self.X = X
            self.targets = targets

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], {
                'log_evidence': self.targets['log_evidence'][idx],
                'aic': self.targets['aic'][idx],
                'bic': self.targets['bic'][idx],
                'true_n': self.targets['true_n'][idx]
            }

    train_dataset = QualityDataset(X_train_t, targets_train_t)
    val_dataset = QualityDataset(X_val_t, targets_val_t)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )

    # Train
    trainer = QualityPredictionTrainer(n_freq=n_freq, device=device, hidden_dims=hidden_dims)
    history = trainer.train(train_loader, val_loader, n_epochs=n_epochs, verbose=True)

    # Save model
    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "decision_layer.pkl"
    trainer.save(str(save_path))

    print(f"\nSaved quality prediction decision layer -> {save_path}")
    print(f"Final ensemble accuracy: {history['val_accuracy_ensemble'][-1]:.2f}%")

    return {
        "model_path": str(save_path),
        "n_freq": n_freq,
        "history": history,
        "final_val_accuracy_ensemble": history['val_accuracy_ensemble'][-1],
        "final_val_accuracy_log_ev": history['val_accuracy_log_ev'][-1],
        "final_val_accuracy_aic": history['val_accuracy_aic'][-1],
        "final_val_accuracy_bic": history['val_accuracy_bic'][-1],
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
    base_noise_level = _get_base_noise_level(config)

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
            base_noise_level=base_noise_level,
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
            base_noise_level=base_noise_level,
            config=config,
            output_dir=training_cfg["output_dir"],
        )
        results["decision_layer"] = decision_result
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Worker models trained: N = {[n for n in results.keys() if isinstance(n, int)]}")
    if "decision_layer" in results:
        dl_result = results['decision_layer']
        print(f"Quality Prediction Decision Layer:")
        print(f"  Ensemble accuracy: {dl_result['final_val_accuracy_ensemble']:.2f}%")
        print(f"  Log evidence accuracy: {dl_result['final_val_accuracy_log_ev']:.2f}%")
        print(f"  AIC accuracy: {dl_result['final_val_accuracy_aic']:.2f}%")
        print(f"  BIC accuracy: {dl_result['final_val_accuracy_bic']:.2f}%")

    return results