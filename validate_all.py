#!/usr/bin/env python3
"""
Comprehensive Validation for ALL Trained SBI Posteriors

Reads config.yaml and validates ALL combinations of:
- model_types (e.g., faraday_thin, burn_slab, external_dispersion, internal_dispersion)
- n_components (1 to max_components)

For each posterior:
- Parameter recovery on test cases
- Recovery plots (true vs recovered)
- Posterior contour plots
- Summary statistics

Usage:
    python validate_all.py
    python validate_all.py --config config.yaml --models-dir models/
    python validate_all.py --n-tests 50  # More test cases
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any, Tuple
import sys

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from src.simulator import RMSimulator, sample_prior
from src.physics import load_frequencies


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_flat_priors(config: Dict) -> Dict[str, float]:
    """Extract flat priors from config."""
    pri = config.get("priors", {})
    rm = pri.get("rm", {})
    amp = pri.get("amp", {})

    return {
        "rm_min": float(rm.get("min", 0.0)),
        "rm_max": float(rm.get("max", 500.0)),
        "amp_min": max(float(amp.get("min", 0.0)), 1e-6),
        "amp_max": float(amp.get("max", 1.0)),
    }


def get_base_noise_level(config: Dict) -> float:
    """Extract base noise level from config."""
    noise_cfg = config.get("noise", {})
    return float(noise_cfg.get("base_level", 0.01))


def get_model_params(config: Dict, model_type: str) -> Dict[str, float]:
    """Extract model-specific parameters from config."""
    physics_cfg = config.get("physics", {})
    model_cfg = physics_cfg.get(model_type, {})
    return model_cfg


def load_posterior(model_path: Path) -> Tuple[Any, Dict]:
    """Load a trained posterior from disk."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["posterior"], data


def validate_one_model(
    model_type: str,
    n_components: int,
    models_dir: Path,
    freq_file: str,
    flat_priors: Dict[str, float],
    model_params: Dict[str, float],
    base_noise_level: float,
    n_tests: int,
    n_samples: int,
    device: str,
    output_dir: Path,
    seed: int = 42
) -> Dict[str, Any]:
    """
    Validate a single posterior model.

    Returns summary statistics.
    """
    # Load posterior
    model_path = models_dir / f"posterior_{model_type}_n{n_components}.pkl"

    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return None

    print(f"\n{'='*70}")
    print(f"Validating: {model_type}, N={n_components}")
    print(f"{'='*70}")

    posterior, model_data = load_posterior(model_path)

    # Verify model matches expected configuration
    saved_n_components = model_data.get("n_components")
    if saved_n_components != n_components:
        print(f"  ⚠️  WARNING: Model file says N={saved_n_components} but expected N={n_components}")
        print(f"  Skipping this model.")
        return None

    # Detect the device the posterior is on and use that device
    # This avoids device mismatch errors
    actual_device = device
    try:
        if hasattr(posterior, 'net') and hasattr(posterior.net, 'parameters'):
            # Get the device from the first parameter
            first_param = next(posterior.net.parameters())
            actual_device = str(first_param.device)
            if actual_device != device:
                print(f"  Note: Posterior is on {actual_device}, using that instead of {device}")
                device = actual_device
    except Exception:
        pass  # Use specified device

    # Create simulator with same settings
    simulator = RMSimulator(
        freq_file=freq_file,
        n_components=n_components,
        base_noise_level=base_noise_level,
        model_type=model_type,
        model_params=model_params
    )

    # Get params_per_comp
    params_per_comp = simulator.params_per_comp

    # Generate test cases
    np.random.seed(seed)
    torch.manual_seed(seed)

    theta_all = sample_prior(
        n_tests, n_components, flat_priors,
        model_type=model_type, model_params=model_params
    )

    # Simulate observations
    x_list = []
    for i in range(n_tests):
        x = simulator(theta_all[i])
        x_list.append(x)
    x_all = np.array(x_list)

    # Run inference on each test case
    theta_recovered_all = []
    theta_std_all = []

    print(f"Running inference on {n_tests} test cases...")
    for i in tqdm(range(n_tests)):
        x_obs = torch.tensor(x_all[i], dtype=torch.float32, device=device)

        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=x_obs)

        samples_np = samples.cpu().numpy()

        # Compute mean and std
        theta_mean = samples_np.mean(axis=0)
        theta_std = samples_np.std(axis=0)

        theta_recovered_all.append(theta_mean)
        theta_std_all.append(theta_std)

    theta_recovered_all = np.array(theta_recovered_all)
    theta_std_all = np.array(theta_std_all)

    # Compute errors
    errors = theta_recovered_all - theta_all
    relative_errors = np.abs(errors) / (np.abs(theta_all) + 1e-10)

    # Summary statistics
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt((errors**2).mean(axis=0))

    # Print summary
    print(f"\nParameter Recovery Summary:")
    param_names = []
    if model_type == "faraday_thin":
        for j in range(n_components):
            param_names.extend([f"RM_{j+1}", f"amp_{j+1}", f"chi0_{j+1}"])
    else:
        for j in range(n_components):
            if model_type == "burn_slab":
                param_names.extend([f"phi_c_{j+1}", f"delta_phi_{j+1}", f"amp_{j+1}", f"chi0_{j+1}"])
            else:
                param_names.extend([f"phi_{j+1}", f"sigma_phi_{j+1}", f"amp_{j+1}", f"chi0_{j+1}"])

    for i, name in enumerate(param_names):
        print(f"  {name:12s}: MAE = {mae[i]:.4f}, RMSE = {rmse[i]:.4f}, σ_post = {theta_std_all.mean(axis=0)[i]:.4f}")

    # Create plots
    model_output_dir = output_dir / f"{model_type}_n{n_components}"
    model_output_dir.mkdir(parents=True, exist_ok=True)

    # Plot 1: Recovery plot (true vs recovered)
    plot_recovery(
        theta_all, theta_recovered_all, theta_std_all,
        param_names, model_type, n_components,
        model_output_dir / "recovery.png"
    )

    print(f"  ✓ Saved plots to {model_output_dir}")

    return {
        "model_type": model_type,
        "n_components": n_components,
        "n_tests": n_tests,
        "mae": mae,
        "rmse": rmse,
        "mean_posterior_std": theta_std_all.mean(axis=0),
        "param_names": param_names,
    }


def plot_recovery(
    theta_true_all: np.ndarray,
    theta_recovered_all: np.ndarray,
    theta_std_all: np.ndarray,
    param_names: List[str],
    model_type: str,
    n_components: int,
    output_path: Path
):
    """Plot true vs recovered for all parameters."""
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        true_vals = theta_true_all[:, i]
        rec_vals = theta_recovered_all[:, i]
        std_vals = theta_std_all[:, i]

        # Scatter plot with error bars
        ax.errorbar(true_vals, rec_vals, yerr=std_vals,
                   fmt='o', alpha=0.6, markersize=4, capsize=3)

        # Perfect recovery line
        lims = [
            min(true_vals.min(), rec_vals.min()),
            max(true_vals.max(), rec_vals.max())
        ]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect recovery')

        # Labels
        ax.set_xlabel(f'True {name}', fontsize=10)
        ax.set_ylabel(f'Recovered {name}', fontsize=10)
        ax.set_title(name, fontsize=11, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='upper left')

        # Compute R²
        ss_res = ((rec_vals - true_vals)**2).sum()
        ss_tot = ((true_vals - true_vals.mean())**2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
               fontsize=9, verticalalignment='top',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Hide unused subplots
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)

    fig.suptitle(f'Parameter Recovery: {model_type}, N={n_components}',
                fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description="Validate ALL trained SBI posteriors"
    )
    parser.add_argument("--config", type=str, default="config.yaml",
                       help="Path to config file")
    parser.add_argument("--models-dir", type=str, default="models",
                       help="Directory containing trained posteriors")
    parser.add_argument("--output-dir", type=str, default="validation_results",
                       help="Output directory for validation results")
    parser.add_argument("--n-tests", type=int, default=20,
                       help="Number of test cases per model")
    parser.add_argument("--n-samples", type=int, default=5000,
                       help="Posterior samples per test case")
    parser.add_argument("--device", type=str, default="cpu",
                       help="Device (cpu/cuda)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed")

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    freq_file = config.get("freq_file", "freq.txt")
    flat_priors = get_flat_priors(config)
    base_noise_level = get_base_noise_level(config)

    # Get model types and max components from config
    physics_cfg = config.get("physics", {})
    if "model_types" in physics_cfg and physics_cfg["model_types"] is not None:
        model_types = physics_cfg["model_types"]
    elif "model_type" in physics_cfg:
        model_types = [physics_cfg["model_type"]]
    else:
        model_types = ["faraday_thin"]  # Fallback

    model_selection_cfg = config.get("model_selection", {})
    max_components = model_selection_cfg.get("max_components", 5)

    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*70)
    print("VROOM-SBI POSTERIOR VALIDATION")
    print("="*70)
    print(f"Config: {args.config}")
    print(f"Models directory: {models_dir}")
    print(f"Model types: {model_types}")
    print(f"Components: 1-{max_components}")
    print(f"Total posteriors to validate: {len(model_types) * max_components}")
    print(f"Test cases per model: {args.n_tests}")
    print(f"Posterior samples per test: {args.n_samples}")
    print("="*70)

    # Validate all models
    all_results = []

    for model_type in model_types:
        model_params = get_model_params(config, model_type)

        for n in range(1, max_components + 1):
            result = validate_one_model(
                model_type=model_type,
                n_components=n,
                models_dir=models_dir,
                freq_file=freq_file,
                flat_priors=flat_priors,
                model_params=model_params,
                base_noise_level=base_noise_level,
                n_tests=args.n_tests,
                n_samples=args.n_samples,
                device=args.device,
                output_dir=output_dir,
                seed=args.seed + n  # Different seed for each N
            )

            if result is not None:
                all_results.append(result)

    # Create summary report
    print("\n" + "="*70)
    print("VALIDATION SUMMARY")
    print("="*70)

    summary_path = output_dir / "validation_summary.txt"
    with open(summary_path, 'w') as f:
        f.write("="*70 + "\n")
        f.write("VROOM-SBI POSTERIOR VALIDATION SUMMARY\n")
        f.write("="*70 + "\n\n")

        for result in all_results:
            model_type = result["model_type"]
            n_comp = result["n_components"]
            f.write(f"\n{model_type}, N={n_comp}:\n")
            f.write("-" * 50 + "\n")

            for i, name in enumerate(result["param_names"]):
                f.write(f"  {name:12s}: MAE = {result['mae'][i]:.4f}, "
                       f"RMSE = {result['rmse'][i]:.4f}, "
                       f"σ_post = {result['mean_posterior_std'][i]:.4f}\n")

        f.write("\n" + "="*70 + "\n")
        f.write(f"Total models validated: {len(all_results)}\n")
        f.write("="*70 + "\n")

    print(f"\n✓ Validation complete!")
    print(f"✓ Results saved to: {output_dir}")
    print(f"✓ Summary saved to: {summary_path}")


if __name__ == "__main__":
    main()
