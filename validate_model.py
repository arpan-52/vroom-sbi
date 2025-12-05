#!/usr/bin/env python3
"""
Simple validation script for VROOM-SBI.
- 20 test cases
- Recovery plots (true vs recovered for each parameter)
- Residual plots for first 5 tests
- Summary statistics
"""

import argparse
import pickle
from pathlib import Path
from typing import Dict, List, Any

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
import yaml

from src.simulator import RMSimulator, sample_prior
from src.physics import load_frequencies, freq_to_lambda_sq


def load_config(config_path: str = "config.yaml") -> Dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def get_flat_priors(config: Dict) -> Dict[str, float]:
    """Extract flat priors from config."""
    pri = config.get("priors", {})
    rm = pri.get("rm", {})
    amp = pri.get("amp", {})
    noise = pri.get("noise", {})
    
    return {
        "rm_min": float(rm.get("min", -500.0)),
        "rm_max": float(rm.get("max", 500.0)),
        "amp_min": max(float(amp.get("min", 0.0)), 1e-6),
        "amp_max": float(amp.get("max", 1.0)),
        "noise_min": max(float(noise.get("min", 0.001)), 1e-9),
        "noise_max": float(noise.get("max", 0.1)),
    }


def load_posterior(model_path: str):
    """Load a trained posterior from disk."""
    with open(model_path, "rb") as f:
        data = pickle.load(f)
    return data["posterior"], data


def generate_test_cases(
    n_components: int,
    n_tests: int,
    freq_file: str,
    flat_priors: Dict[str, float],
    seed: int = 42
):
    """Generate synthetic test cases with known ground truth."""
    np.random.seed(seed)
    
    simulator = RMSimulator(freq_file, n_components)
    theta_true = sample_prior(n_tests, n_components, flat_priors)
    
    x_obs = simulator(theta_true)
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    
    return theta_true, x_obs, simulator


def run_inference(posterior, x_obs: np.ndarray, n_samples: int = 5000, device: str = "cpu"):
    """Run inference on observed data."""
    x_tensor = torch.tensor(x_obs, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor)
    
    return samples.cpu().numpy()


def plot_recovery(theta_true_all, theta_recovered_all, theta_std_all, n_components, output_path):
    """
    Plot true vs recovered for all parameters.
    One subplot per parameter type.
    """
    n_tests = len(theta_true_all)
    param_names = []
    for i in range(n_components):
        param_names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    param_names.append("noise")
    
    n_params = len(param_names)
    n_cols = 2
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4*n_rows))
    axes = axes.flatten()
    
    for i, name in enumerate(param_names):
        ax = axes[i]
        
        true_vals = theta_true_all[:, i]
        rec_vals = theta_recovered_all[:, i]
        std_vals = theta_std_all[:, i]
        
        # Plot with error bars
        ax.errorbar(true_vals, rec_vals, yerr=std_vals, 
                    fmt="o", capsize=3, markersize=8, alpha=0.7)
        
        # 1:1 line
        all_vals = np.concatenate([true_vals, rec_vals])
        vmin, vmax = all_vals.min(), all_vals.max()
        margin = 0.1 * (vmax - vmin)
        lims = [vmin - margin, vmax + margin]
        ax.plot(lims, lims, "k--", alpha=0.5, linewidth=2, label="1:1")
        
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Recovered {name}")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # Calculate and show MAE
        mae = np.mean(np.abs(true_vals - rec_vals))
        ax.text(0.05, 0.95, f"MAE: {mae:.4f}", transform=ax.transAxes, 
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused subplots
    for i in range(n_params, len(axes)):
        axes[i].axis("off")
    
    fig.suptitle(f"Parameter Recovery: N={n_components} components, {n_tests} tests", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_residuals(theta_true, theta_recovered, x_obs, n_components, freq_file, output_path, test_idx):
    """
    Plot Q, U data with model fit and residuals.
    """
    frequencies = load_frequencies(freq_file)
    lambda_sq = freq_to_lambda_sq(frequencies)
    n_freq = len(frequencies)
    freq_ghz = frequencies / 1e9
    
    Q_obs = x_obs[:n_freq]
    U_obs = x_obs[n_freq:]
    
    # Reconstruct from posterior mean
    P_model = np.zeros(n_freq, dtype=complex)
    for i in range(n_components):
        rm = theta_recovered[3*i]
        amp = theta_recovered[3*i + 1]
        chi0 = theta_recovered[3*i + 2]
        phase = 2 * (chi0 + rm * lambda_sq)
        P_model += amp * np.exp(1j * phase)
    Q_model = P_model.real
    U_model = P_model.imag
    
    # Reconstruct from true parameters
    P_true = np.zeros(n_freq, dtype=complex)
    for i in range(n_components):
        rm = theta_true[3*i]
        amp = theta_true[3*i + 1]
        chi0 = theta_true[3*i + 2]
        phase = 2 * (chi0 + rm * lambda_sq)
        P_true += amp * np.exp(1j * phase)
    Q_true = P_true.real
    U_true = P_true.imag
    
    # Noise values
    true_noise = theta_true[-1]
    inferred_noise = theta_recovered[-1]
    
    # Residuals
    residuals_Q = Q_obs - Q_model
    residuals_U = U_obs - U_model
    rms_Q = np.std(residuals_Q)
    rms_U = np.std(residuals_U)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Q data + fit
    ax = axes[0, 0]
    ax.plot(freq_ghz, Q_obs, "ko", markersize=6, label="Observed")
    ax.plot(freq_ghz, Q_model, "b-", linewidth=2, label="Recovered model")
    ax.plot(freq_ghz, Q_true, "r--", linewidth=2, alpha=0.7, label="True model")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes Q")
    ax.set_title("Stokes Q")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # U data + fit
    ax = axes[0, 1]
    ax.plot(freq_ghz, U_obs, "ko", markersize=6, label="Observed")
    ax.plot(freq_ghz, U_model, "b-", linewidth=2, label="Recovered model")
    ax.plot(freq_ghz, U_true, "r--", linewidth=2, alpha=0.7, label="True model")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes U")
    ax.set_title("Stokes U")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q residuals
    ax = axes[1, 0]
    ax.plot(freq_ghz, residuals_Q, "bo-", markersize=6)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.fill_between(freq_ghz, -true_noise, true_noise, alpha=0.3, color="red", label=f"True noise: {true_noise:.4f}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual Q")
    ax.set_title(f"Q Residuals (RMS: {rms_Q:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # U residuals
    ax = axes[1, 1]
    ax.plot(freq_ghz, residuals_U, "bo-", markersize=6)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.fill_between(freq_ghz, -true_noise, true_noise, alpha=0.3, color="red", label=f"True noise: {true_noise:.4f}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual U")
    ax.set_title(f"U Residuals (RMS: {rms_U:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Main title with parameter comparison
    title = f"Test #{test_idx+1} | N={n_components}\n"
    title += f"RM: True={theta_true[0]:.1f}, Rec={theta_recovered[0]:.1f} | "
    title += f"Amp: True={theta_true[1]:.3f}, Rec={theta_recovered[1]:.3f} | "
    title += f"Noise: True={true_noise:.4f}, Rec={inferred_noise:.4f}"
    fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def print_summary(theta_true_all, theta_recovered_all, n_components):
    """Print summary statistics."""
    param_names = []
    for i in range(n_components):
        param_names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    param_names.append("noise")
    
    print("\n" + "="*60)
    print("SUMMARY STATISTICS")
    print("="*60)
    print(f"{'Parameter':<12} {'MAE':>12} {'Mean Error':>12} {'Std Error':>12}")
    print("-"*60)
    
    for i, name in enumerate(param_names):
        errors = theta_recovered_all[:, i] - theta_true_all[:, i]
        mae = np.mean(np.abs(errors))
        mean_err = np.mean(errors)
        std_err = np.std(errors)
        print(f"{name:<12} {mae:>12.4f} {mean_err:>12.4f} {std_err:>12.4f}")
    
    print("="*60)


def main():
    parser = argparse.ArgumentParser(description="Simple validation for VROOM-SBI")
    parser.add_argument("--n-tests", type=int, default=20, help="Number of test cases")
    parser.add_argument("--n-samples", type=int, default=5000, help="Posterior samples per test")
    parser.add_argument("--n-residual-plots", type=int, default=5, help="Number of residual plots to make")
    parser.add_argument("--output-dir", type=str, default="validation_results", help="Output directory")
    parser.add_argument("--model-path", type=str, default="models/posterior_n1.pkl", help="Path to posterior model")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load config and posterior
    config = load_config()
    flat_priors = get_flat_priors(config)
    freq_file = config.get("freq_file", "freq.txt")
    
    print(f"Loading posterior from {args.model_path}...")
    posterior, model_data = load_posterior(args.model_path)
    n_components = model_data["n_components"]
    print(f"Model has {n_components} component(s)")
    
    # Generate test cases
    print(f"\nGenerating {args.n_tests} test cases...")
    theta_true_all, x_obs_all, simulator = generate_test_cases(
        n_components=n_components,
        n_tests=args.n_tests,
        freq_file=freq_file,
        flat_priors=flat_priors,
        seed=args.seed
    )
    
    # Run inference on all test cases
    print(f"\nRunning inference...")
    theta_recovered_all = []
    theta_std_all = []
    all_samples = []
    
    for i in tqdm(range(args.n_tests), desc="Inferring"):
        samples = run_inference(posterior, x_obs_all[i], n_samples=args.n_samples, device=args.device)
        theta_recovered_all.append(np.mean(samples, axis=0))
        theta_std_all.append(np.std(samples, axis=0))
        all_samples.append(samples)
    
    theta_recovered_all = np.array(theta_recovered_all)
    theta_std_all = np.array(theta_std_all)
    
    # Plot 1: Recovery plot
    print("\nGenerating recovery plot...")
    plot_recovery(
        theta_true_all, theta_recovered_all, theta_std_all,
        n_components, output_dir / "recovery.png"
    )
    
    # Plot 2: Residual plots for first N tests
    print(f"\nGenerating {args.n_residual_plots} residual plots...")
    for i in range(min(args.n_residual_plots, args.n_tests)):
        plot_residuals(
            theta_true_all[i], theta_recovered_all[i], x_obs_all[i],
            n_components, freq_file,
            output_dir / f"residuals_test{i+1}.png",
            test_idx=i
        )
    
    # Print summary
    print_summary(theta_true_all, theta_recovered_all, n_components)
    
    print(f"\nDone! Results saved to {output_dir}/")


if __name__ == "__main__":
    main()