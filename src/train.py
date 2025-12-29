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
from typing import Dict, Any, List

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server
import matplotlib.pyplot as plt

from sbi.inference import SNPE

# Handle different SBI versions - posterior_nn location varies
try:
    from sbi.utils import posterior_nn
except ImportError:
    try:
        from sbi.neural_nets import posterior_nn
    except ImportError:
        # Fallback for older versions
        from sbi.utils.get_nn_models import posterior_nn

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
    simulation_scaling_mode = training.get("simulation_scaling_mode", "linear")
    scaling_power = float(training.get("scaling_power", 2.0))

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
        "simulation_scaling_mode": simulation_scaling_mode,
        "scaling_power": scaling_power,
    }


def _extract_sbi_cfg(cfg: Dict[str, Any]) -> Dict[str, Any]:
    """Extract SBI architecture settings from config."""
    sbi_cfg = cfg.get("sbi", {}) or {}

    return {
        "model": sbi_cfg.get("model", "nsf"),
        "num_bins": int(sbi_cfg.get("num_bins", 16)),
        "embedding_dim": int(sbi_cfg.get("embedding_dim", 64)),
        "architecture_scaling": sbi_cfg.get("architecture_scaling", {
            1: {"hidden_features": 128, "num_transforms": 10},
            2: {"hidden_features": 128, "num_transforms": 10},
            3: {"hidden_features": 192, "num_transforms": 12},
            4: {"hidden_features": 256, "num_transforms": 15},
            5: {"hidden_features": 384, "num_transforms": 20},
        }),
    }


def get_scaled_simulations(n_components: int, base_simulations: int, scaling: bool = True, scaling_mode: str = "linear", scaling_power: float = 2.0) -> int:
    """
    Scale the number of simulations based on model complexity.

    Higher-dimensional models need more training data for good posterior coverage.
    Addresses the curse of dimensionality with aggressive scaling options.

    Parameters
    ----------
    n_components : int
        Number of RM components
    base_simulations : int
        Base number of simulations (for N=1)
    scaling : bool
        Whether to apply scaling (if False, returns base_simulations)
    scaling_mode : str
        Scaling strategy:
        - "linear": Custom factors (1, 2, 4, 6, 8) - old default
        - "quadratic": N^2 scaling (1, 4, 9, 16, 25) - recommended for high-D
        - "subquadratic": N^1.5 scaling (1, 2.8, 5.2, 8, 11.2) - balanced
        - "power": N^scaling_power - fully tunable exponent!
    scaling_power : float
        Exponent for "power" mode (e.g., 2.0 = N^2, 2.5 = N^2.5, 3.0 = N^3)
        Higher values for very large SBI architectures

    Returns
    -------
    int
        Scaled number of simulations

    Examples
    --------
    With base_simulations=20000:
    - linear:       20k, 40k, 80k, 120k, 160k
    - quadratic:    20k, 80k, 180k, 320k, 500k (N^2)
    - subquadratic: 20k, 57k, 104k, 160k, 224k (N^1.5)
    - power (2.5):  20k, 113k, 324k, 640k, 1.12M (N^2.5)
    - power (3.0):  20k, 160k, 540k, 1.28M, 2.5M (N^3)
    """
    if not scaling:
        return base_simulations

    if scaling_mode == "power":
        # Fully tunable power scaling - adjust exponent as needed!
        factor = n_components ** scaling_power
    elif scaling_mode == "quadratic":
        # N^2 scaling - aggressive for curse of dimensionality
        factor = n_components ** 2
    elif scaling_mode == "subquadratic":
        # N^1.5 scaling - balanced approach
        factor = n_components ** 1.5
    else:  # "linear" or default
        # Custom linear-ish factors (legacy)
        scaling_factors = {1: 1, 2: 2, 3: 4, 4: 6, 5: 8}
        factor = scaling_factors.get(n_components, n_components * 2)

    return int(base_simulations * factor)


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
    sbi_cfg: Dict[str, Any] = None,
    save_simulations: bool = True,
    model_type: str = "faraday_thin",
    model_params: dict = None,
) -> Dict[str, Any]:
    """
    Train a single model with n_components, using the provided flat_priors
    (keys: rm_min, rm_max, amp_min, amp_max). 

    Saves the posterior to output_dir/posterior_n{n_components}.pkl and returns metadata dict.
    
    If save_simulations=True, also saves the simulated spectra and weights
    to output_dir/simulations_n{n_components}.pkl for classifier training.
    """
    # Default SBI config if not provided
    if sbi_cfg is None:
        sbi_cfg = {
            "model": "nsf",
            "hidden_features": 128,
            "num_transforms": 10,
            "num_bins": 16,
            "embedding_dim": 64,
        }
    
    print(f"\n{'='*60}")
    print(f"Training model N={n_components} ({model_type}) on device={device}")
    print(f"{'='*60}")

    simulator = RMSimulator(
        freq_file,
        n_components,
        base_noise_level=base_noise_level,
        model_type=model_type,
        model_params=model_params
    )

    # Build prior on requested device
    try:
        prior = build_prior(n_components, flat_priors, device=device, model_type=model_type)
    except TypeError:
        prior = build_prior(n_components, flat_priors, model_type=model_type)

    print(f"Model type: {model_type}")
    print(f"Simulator params: n_params={simulator.n_params}, n_freq={simulator.n_freq}, params_per_comp={simulator.params_per_comp}")
    print(f"Simulations: {n_simulations:,}, batch_size: {batch_size}, validation_fraction: {validation_fraction}")

    # Generate simulations (numpy)
    theta = sample_prior(n_simulations, n_components, flat_priors, model_type=model_type)

    # Simulate in batches to avoid excessive memory usage
    # Apply weight and noise augmentation if enabled
    from .augmentation import augment_weights_combined, augment_base_noise_level

    xs = []
    all_weights = []  # Save weights for classifier training
    
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
        all_weights.append(augmented_weights)

    # Restore original base noise level
    simulator.base_noise_level = base_noise_level
    
    x = np.vstack(xs)
    all_weights = np.vstack(all_weights)
    
    # Save simulations for classifier training
    if save_simulations:
        sim_save_path = output_dir / f"simulations_{model_type}_n{n_components}.pkl"
        with open(sim_save_path, 'wb') as f:
            pickle.dump({
                'spectra': x,  # (n_samples, 2 * n_freq) - Q and U
                'weights': all_weights,  # (n_samples, n_freq)
                'theta': theta,  # (n_samples, n_params) - parameters
                'n_components': n_components,
                'model_type': model_type,  # NEW: Include model type
                'n_freq': simulator.n_freq,
            }, f)
        print(f"Saved simulations for classifier -> {sim_save_path}")

    # Convert to torch and move to device
    try:
        theta_t = torch.tensor(theta, dtype=torch.float32, device=device)
        x_t = torch.tensor(x, dtype=torch.float32, device=device)
    except Exception:
        theta_t = torch.tensor(theta, dtype=torch.float32, device="cpu")
        x_t = torch.tensor(x, dtype=torch.float32, device="cpu")
        device = "cpu"
        print("Warning: falling back to CPU tensors (device change).")

    # Create SNPE inference object with custom density estimator
    # Build custom embedding for spectral data
    input_dim = 2 * simulator.n_freq  # Q and U
    embedding_dim = sbi_cfg["embedding_dim"]
    embedding_net = SpectralEmbedding(input_dim=input_dim, output_dim=embedding_dim).to(device)

    # Get architecture for this component count (adaptive scaling)
    arch_scaling = sbi_cfg["architecture_scaling"]
    # If exact match exists, use it; otherwise use closest larger one
    if n_components in arch_scaling:
        arch_config = arch_scaling[n_components]
    else:
        # Find the next larger component count that has a config
        available_counts = sorted([k for k in arch_scaling.keys() if k >= n_components])
        if available_counts:
            arch_config = arch_scaling[available_counts[0]]
        else:
            # Fallback to largest defined architecture
            arch_config = arch_scaling[max(arch_scaling.keys())]

    # Build neural posterior with adaptive architecture
    density_estimator_builder = posterior_nn(
        model=sbi_cfg["model"],
        hidden_features=arch_config["hidden_features"],
        num_transforms=arch_config["num_transforms"],
        num_bins=sbi_cfg["num_bins"],
        embedding_net=embedding_net
    )

    print(f"Using {sbi_cfg['model'].upper()} with custom spectral embedding:")
    print(f"  Input dim: {input_dim}, Embedding dim: {embedding_dim}")
    print(f"  Architecture for N={n_components}: Hidden={arch_config['hidden_features']}, Transforms={arch_config['num_transforms']}")

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
    save_path = output_dir / f"posterior_{model_type}_n{n_components}.pkl"
    with open(save_path, "wb") as f:
        pickle.dump({
            "posterior": posterior,
            "prior": prior,  # Save prior for AIC/BIC computation in decision layer
            "n_components": n_components,
            "n_freq": simulator.n_freq,
            "lambda_sq": simulator.lambda_sq,
            "flat_priors": flat_priors,
            "config_used": {
                "freq_file": freq_file,
                "n_simulations": n_simulations,
                "batch_size": batch_size,
                "device": device,
                "sbi_cfg": sbi_cfg,
            },
        }, f)

    print(f"Saved posterior -> {save_path}")

    return {
        "posterior": posterior,
        "prior": prior,
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
    Train the quality prediction decision layer using PRE-TRAINED models.

    This function trains a neural network to predict AIC, BIC, and log evidence
    for both 1-comp and 2-comp models given an observed spectrum.

    Uses the already-trained worker models (posterior_n1.pkl, posterior_n2.pkl)
    to compute quality metrics, rather than fitting new models from scratch.

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
    import pickle

    print(f"\n{'='*60}")
    print("Training Quality Prediction Decision Layer")
    print(f"{'='*60}")

    # Get decision layer config
    decision_cfg = config.get("decision_layer", {})
    n_samples = decision_cfg.get("n_training_samples", 1000)
    n_epochs = decision_cfg.get("n_epochs", 50)
    batch_size = decision_cfg.get("batch_size", 32)
    val_fraction = decision_cfg.get("validation_fraction", 0.2)
    device = config.get("training", {}).get("device", "cpu")
    hidden_dims = decision_cfg.get("hidden_dims", [256, 128, 64])
    n_posterior_samples = decision_cfg.get("n_posterior_samples", 1000)  # NEW: configurable

    # Load PRE-TRAINED posteriors
    print("\nLoading pre-trained worker models...")
    with open(output_dir / "posterior_n1.pkl", "rb") as f:
        data_1 = pickle.load(f)
        posterior_1 = data_1["posterior"]
    print("  ✓ Loaded 1-comp posterior")

    with open(output_dir / "posterior_n2.pkl", "rb") as f:
        data_2 = pickle.load(f)
        posterior_2 = data_2["posterior"]
    print("  ✓ Loaded 2-comp posterior")

    # Create simulators
    sim_1comp = RMSimulator(freq_file, 1, base_noise_level=base_noise_level)
    sim_2comp = RMSimulator(freq_file, 2, base_noise_level=base_noise_level)
    n_freq = sim_1comp.n_freq

    print(f"\nGenerating {n_samples} training samples per model...")
    print(f"Using PRE-TRAINED posteriors to compute quality metrics")
    print(f"Posterior samples for metric computation: {n_posterior_samples}")

    # Data containers
    X_all = []  # Spectra + weights
    targets_log_ev = []  # [log_ev_1, log_ev_2]
    targets_aic = []  # [AIC_1, AIC_2]
    targets_bic = []  # [BIC_1, BIC_2]
    true_n_list = []  # Ground truth N (1 or 2)

    # Load priors for likelihood computation
    prior_1 = data_1.get("prior", None)
    prior_2 = data_2.get("prior", None)

    # Helper function to compute AIC/BIC - FIXED VERSION
    def compute_quality_metrics(posterior, prior, x_obs, n_params, n_data_points, n_posterior_samples=1000):
        """
        Compute log evidence, AIC, and BIC for a posterior.
        
        FIXED: Uses correct computation:
        - Find MAP estimate (point with highest posterior density)
        - Compute log_likelihood = log_posterior - log_prior at MAP
        - Use this for AIC/BIC
        
        Note: The log_likelihood is only defined up to a constant (the evidence),
        but this constant cancels when comparing models.
        """
        x_obs_t = torch.tensor(x_obs, dtype=torch.float32, device=device)
        if x_obs_t.dim() == 1:
            x_obs_t = x_obs_t.unsqueeze(0)

        # Sample from posterior
        with torch.no_grad():
            samples = posterior.sample((n_posterior_samples,), x=x_obs_t)
            
            # Find MAP estimate (sample with highest log posterior)
            log_probs = posterior.log_prob(samples, x=x_obs_t)
            map_idx = torch.argmax(log_probs)
            theta_map = samples[map_idx]
            
            # Log posterior at MAP
            log_posterior_map = log_probs[map_idx].item()
            
            # Log prior at MAP
            if prior is not None:
                log_prior_map = prior.log_prob(theta_map.unsqueeze(0)).item()
            else:
                # If prior not available, assume uniform (log_prior = constant)
                log_prior_map = 0.0
            
            # Log likelihood at MAP (up to constant)
            # log P(x|θ) = log P(θ|x) - log P(θ) + log P(x)
            # The log P(x) term is constant across θ, so for AIC/BIC comparison it cancels
            log_likelihood_map = log_posterior_map - log_prior_map
            
            # Compute log evidence estimate via importance sampling
            # This is an approximation: E[p(x|θ)] where θ ~ p(θ|x)
            log_evidence = torch.logsumexp(log_probs, dim=0).item() - np.log(n_posterior_samples)

        # AIC = 2k - 2 * log(L_max)
        aic = 2 * n_params - 2 * log_likelihood_map

        # BIC = k * log(n) - 2 * log(L_max)
        bic = n_params * np.log(n_data_points) - 2 * log_likelihood_map

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

            # Use PRE-TRAINED models to compute quality metrics
            n_params_1 = 3  # RM, amp, chi0
            n_params_2 = 6  # 2 * (RM, amp, chi0)
            n_data_points = 2 * n_freq  # Q and U observations

            log_ev_1, aic_1, bic_1 = compute_quality_metrics(posterior_1, prior_1, qu_obs, n_params_1, n_data_points, n_posterior_samples)
            log_ev_2, aic_2, bic_2 = compute_quality_metrics(posterior_2, prior_2, qu_obs, n_params_2, n_data_points, n_posterior_samples)

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


def _plot_training_summary(results: Dict[str, Any], output_dir: Path, model_types: List[str], max_components: int):
    """
    Generate and save training summary plots for all trained SBI posteriors.

    Creates a visual summary showing validation performance across all models.
    """
    # Extract validation performance data
    # Note: SBI doesn't return detailed training history by default,
    # but we can extract final validation metrics from saved data

    model_names = []
    val_losses = []
    n_samples = []

    for model_type in model_types:
        for n in range(1, max_components + 1):
            key = f"{model_type}_n{n}"
            if key in results:
                model_names.append(f"{model_type}\nN={n}")
                # Load the saved posterior to get metadata
                posterior_path = output_dir / f"posterior_{model_type}_n{n}.pkl"
                try:
                    with open(posterior_path, 'rb') as f:
                        data = pickle.load(f)
                        # Extract metrics if available
                        val_loss = data.get('val_loss', np.nan)
                        n_sims = data.get('n_simulations', 0)
                        val_losses.append(val_loss)
                        n_samples.append(n_sims)
                except Exception as e:
                    print(f"  Warning: Could not load metrics for {key}: {e}")
                    val_losses.append(np.nan)
                    n_samples.append(0)

    # Create summary plot
    if len(model_names) > 0:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))

        x_pos = np.arange(len(model_names))
        colors = plt.cm.tab10(np.linspace(0, 1, len(model_types)))

        # Plot 1: Number of training samples
        bar_colors = []
        for name in model_names:
            for i, mtype in enumerate(model_types):
                if mtype in name:
                    bar_colors.append(colors[i])
                    break

        ax1.bar(x_pos, [s/1000 for s in n_samples], color=bar_colors, alpha=0.7, edgecolor='black')
        ax1.set_ylabel('Training Samples (thousands)', fontsize=12, fontweight='bold')
        ax1.set_title('SBI Training Summary: Sample Counts per Model', fontsize=14, fontweight='bold')
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
        ax1.grid(axis='y', alpha=0.3)

        # Add legend for model types
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor=colors[i], alpha=0.7, label=mtype)
                          for i, mtype in enumerate(model_types)]
        ax1.legend(handles=legend_elements, loc='upper left', fontsize=10)

        # Plot 2: Validation loss (if available)
        if not all(np.isnan(val_losses)):
            ax2.bar(x_pos, val_losses, color=bar_colors, alpha=0.7, edgecolor='black')
            ax2.set_ylabel('Validation Loss', fontsize=12, fontweight='bold')
            ax2.set_title('SBI Training Summary: Final Validation Performance', fontsize=14, fontweight='bold')
            ax2.set_xticks(x_pos)
            ax2.set_xticklabels(model_names, rotation=45, ha='right', fontsize=9)
            ax2.grid(axis='y', alpha=0.3)
        else:
            ax2.text(0.5, 0.5, 'Validation metrics not available\n(SBI library does not return training history by default)',
                    ha='center', va='center', fontsize=12, transform=ax2.transAxes)
            ax2.set_xticks([])
            ax2.set_yticks([])

        plt.tight_layout()
        plot_path = output_dir / "training_summary.png"
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  ✓ Saved training summary plot -> {plot_path}")

        # Also create a text summary
        summary_path = output_dir / "training_summary.txt"
        with open(summary_path, 'w') as f:
            f.write("="*70 + "\n")
            f.write("SBI POSTERIOR TRAINING SUMMARY\n")
            f.write("="*70 + "\n\n")
            for i, name in enumerate(model_names):
                f.write(f"{name.replace(chr(10), ' ')}: {n_samples[i]:,} samples\n")
            f.write(f"\nTotal simulations: {sum(n_samples):,}\n")
            f.write("="*70 + "\n")
        print(f"  ✓ Saved training summary text -> {summary_path}")


def train_all_models(config: Dict[str, Any], decision_layer_only: bool = False) -> Dict[int, Dict[str, Any]]:
    """
    Train models for N = 1 ..  max_components using the exact config structure
    the repository is using.  Returns dict mapping n_components -> saved data dict.
    
    Now with two-layer system:
    - Train worker models (1-comp and 2-comp)
    - Train classifier for model selection
    
    Parameters
    ----------
    config : Dict[str, Any]
        Configuration dictionary loaded from YAML
    decision_layer_only : bool
        If True, skip training worker models and only train the classifier.
        Requires posterior_n1.pkl, posterior_n2.pkl, and simulation files to exist.
    classifier_only : bool
        If True, skip training worker models and only train the classifier.
        Alias for decision_layer_only for backward compatibility.
    """
    if not isinstance(config, dict):
        raise ValueError("config must be a dict loaded from YAML.")

    freq_file = config.get("freq_file", "freq.txt")
    training_cfg = _extract_training_cfg(config)
    flat_priors = _flatten_priors(config)
    base_noise_level = _get_base_noise_level(config)
    sbi_cfg = _extract_sbi_cfg(config)

    results: Dict[str, Dict[str, Any]] = {}  # Changed to string keys for model types

    # Train worker models (1 to max_components, full scale!)
    max_comp = training_cfg["max_components"]

    # Extract physical model types from config
    physics_cfg = config.get("physics", {})

    # Support both single model_type (string) and multiple model_types (list)
    if "model_types" in physics_cfg:
        model_types = physics_cfg["model_types"]
        cross_model_training = True
    elif "model_type" in physics_cfg:
        model_types = [physics_cfg["model_type"]]
        cross_model_training = False
    else:
        model_types = ["faraday_thin"]  # Default
        cross_model_training = False

    print(f"\n{'='*60}")
    if cross_model_training:
        print(f"CROSS-MODEL TRAINING: {len(model_types)} models × {max_comp} components = {len(model_types) * max_comp} total!")
        print(f"Models: {', '.join(model_types)}")
    else:
        print(f"SINGLE-MODEL TRAINING: {model_types[0]} (1-{max_comp} components)")
    print(f"{'='*60}")
    
    if decision_layer_only:
        print("Mode: CLASSIFIER ONLY (skipping worker model training)")
        print(f"Using existing simulations from: {training_cfg['output_dir']}")

        # Verify that simulations exist for all model types
        for model_type in model_types:
            for n in range(1, max_comp + 1):
                sim_path = training_cfg["output_dir"] / f"simulations_{model_type}_n{n}.pkl"
                if not sim_path.exists():
                    raise FileNotFoundError(
                        f"Simulations file not found: {sim_path}\n"
                        f"Cannot use classifier_only mode without saved simulations.\n"
                        f"Run full training first to generate simulations."
                    )
                print(f"  ✓ Found simulations_{model_type}_n{n}.pkl")
    else:
        print(f"\nPhase 1: Training Worker Models")
        print(f"SBI Architecture: {sbi_cfg['model'].upper()} with adaptive scaling, "
              f"embedding_dim={sbi_cfg['embedding_dim']}")
        print(f"  Architecture scaling:")
        for n_comp, arch_cfg in sorted(sbi_cfg['architecture_scaling'].items()):
            print(f"    N={n_comp}: hidden={arch_cfg['hidden_features']}, transforms={arch_cfg['num_transforms']}")

        # Loop over all model types
        for model_type in model_types:
            model_params = physics_cfg.get(model_type, {})

            print(f"\n{'='*60}")
            print(f"Training {model_type.upper()} models (1-{max_comp} components)")
            print(f"{'='*60}")

            for n in range(1, max_comp + 1):
                # Scale simulations for complex models with tunable power index
                n_sims = get_scaled_simulations(
                    n_components=n,
                    base_simulations=training_cfg["n_simulations"],
                    scaling=training_cfg["simulation_scaling"],
                    scaling_mode=training_cfg["simulation_scaling_mode"],
                    scaling_power=training_cfg["scaling_power"],
                )

                print(f"\n>>> {model_type} N={n}: Using {n_sims:,} simulations")

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
                    sbi_cfg=sbi_cfg,
                    save_simulations=True,  # Save for classifier training
                    model_type=model_type,
                    model_params=model_params,
                )

                # Store with composite key
                results[f"{model_type}_n{n}"] = data

    print("\n" + "="*60)
    print("Phase 2: Training Model Selection Classifier")
    if cross_model_training:
        print(f"CROSS-MODEL CLASSIFIER: {len(model_types)} models × {max_comp} components = {len(model_types) * max_comp} classes")
    else:
        print(f"SINGLE-MODEL CLASSIFIER: {max_comp} classes (component count only)")
    print("="*60)

    # Train classifier if enabled
    model_selection = config.get("model_selection", {})
    if model_selection.get("use_classifier", True):
        classifier_result = train_classifier(
            config=config,
            output_dir=training_cfg["output_dir"],
            max_components=max_comp,
            model_types=model_types,
            cross_model_training=cross_model_training,
        )
        results["classifier"] = classifier_result
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    if not decision_layer_only:
        print(f"Worker models trained: N = {[n for n in results.keys() if isinstance(n, int)]}")
    if "classifier" in results:
        cl_result = results['classifier']
        print(f"Model Selection Classifier:")
        print(f"  Validation accuracy: {cl_result['final_val_accuracy']:.2f}%")
        for n in range(1, max_comp + 1):
            acc_key = f'accuracy_{n}comp'
            if acc_key in cl_result:
                print(f"  {n}-comp accuracy: {cl_result[acc_key]:.2f}%")

    # Generate training summary plots
    if not decision_layer_only and len(results) > 0:
        print("\n" + "="*60)
        print("Generating Training Summary Plots...")
        print("="*60)
        _plot_training_summary(results, training_cfg["output_dir"], model_types, max_comp)

    return results


def train_classifier(
    config: Dict[str, Any],
    output_dir: Path = Path("models"),
    max_components: int = 2,
    model_types: List[str] = None,
    cross_model_training: bool = False,
) -> Dict[str, Any]:
    """
    Train the model selection classifier using saved simulations.

    This classifier learns to predict BOTH the physical model type AND
    the number of components directly from the observed spectrum.

    Parameters
    ----------
    config : Dict[str, Any]
        Full configuration dictionary
    output_dir : Path
        Directory containing saved simulations
    max_components : int
        Maximum number of components
    model_types : List[str], optional
        List of model types to train on (for cross-model training)
    cross_model_training : bool
        If True, trains cross-model classifier (model_type × n_components classes)

    Returns
    -------
    Dict[str, Any]
        Training results and metadata
    """
    from .classifier import (
        ClassifierConfig,
        ClassifierTrainer,
        prepare_classifier_data
    )

    if model_types is None:
        model_types = ["faraday_thin"]
    
    # Get classifier config
    classifier_cfg = ClassifierConfig.from_config(config)
    device = config.get("training", {}).get("device", "cpu")

    # Calculate number of classes
    if cross_model_training:
        n_classes = len(model_types) * max_components
        print(f"\nCROSS-MODEL Classifier configuration:")
        print(f"  Model types: {model_types}")
        print(f"  Components per model: 1-{max_components}")
        print(f"  Total classes: {n_classes} ({len(model_types)} models × {max_components} components)")
    else:
        n_classes = max_components
        print(f"\nSINGLE-MODEL Classifier configuration:")
        print(f"  Model type: {model_types[0]}")
        print(f"  Classes: {n_classes} (component count only)")

    print(f"\n  Architecture (1D CNN):")
    print(f"    Conv channels: {classifier_cfg.conv_channels}")
    print(f"    Kernel sizes: {classifier_cfg.kernel_sizes}")
    print(f"    Dropout: {classifier_cfg.dropout}")
    print(f"  Training:")
    print(f"    Epochs: {classifier_cfg.n_epochs}")
    print(f"    Batch size: {classifier_cfg.batch_size}")
    print(f"    Learning rate: {classifier_cfg.learning_rate}")
    print(f"    Device: {device}")

    # Load simulations
    print(f"\nLoading simulations from {output_dir}...")
    train_loader, val_loader, n_freq, class_to_label = prepare_classifier_data(
        simulations_dir=output_dir,
        max_components=max_components,
        model_types=model_types,
        cross_model_training=cross_model_training,
        validation_fraction=classifier_cfg.validation_fraction,
        batch_size=classifier_cfg.batch_size,
    )

    # Create trainer
    trainer = ClassifierTrainer(
        n_freq=n_freq,
        n_classes=n_classes,
        config=classifier_cfg,
        device=device,
    )
    
    # Train
    print(f"\nTraining classifier...")
    history = trainer.train(train_loader, val_loader, verbose=True)
    
    # Evaluate
    eval_results = trainer.evaluate(val_loader)
    
    # Save
    save_path = output_dir / "classifier.pkl"
    trainer.save(str(save_path))
    print(f"\nSaved classifier -> {save_path}")
    
    # Build results dict
    result = {
        "model_path": str(save_path),
        "n_freq": n_freq,
        "n_classes": n_classes,
        "max_components": max_components,
        "model_types": model_types,
        "cross_model_training": cross_model_training,
        "class_to_label": class_to_label,  # Mapping from class index to (model_type, n_comp)
        "history": history,
        "final_val_accuracy": eval_results['accuracy'],
    }

    # Add per-class accuracies
    for key, value in eval_results.items():
        if key != 'accuracy':
            result[key] = value

    return result