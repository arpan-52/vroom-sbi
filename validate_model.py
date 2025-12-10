#!/usr/bin/env python3
"""
Simple validation script for VROOM-SBI.
- 20 test cases
- Recovery plots (true vs recovered for each parameter)
- Residual plots for first 5 tests
- Posterior contour plots
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
import corner

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

    return {
        "rm_min": float(rm.get("min", -500.0)),
        "rm_max": float(rm.get("max", 500.0)),
        "amp_min": max(float(amp.get("min", 0.0)), 1e-6),
        "amp_max": float(amp.get("max", 1.0)),
    }


def get_base_noise_level(config: Dict) -> float:
    """Extract base noise level from config."""
    noise_cfg = config.get("noise", {})
    return float(noise_cfg.get("base_level", 0.01))


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
    base_noise_level: float = 0.01,
    seed: int = 42
):
    """Generate test cases with known parameters."""
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Create simulator with fixed base_noise_level
    simulator = RMSimulator(
        freq_file=freq_file,
        n_components=n_components,
        base_noise_level=base_noise_level
    )

    # Sample n_tests parameter sets from prior
    # Note: theta has shape (n_tests, 3*n_components) - no noise parameter
    theta_all = sample_prior(n_tests, n_components, flat_priors)

    x_list = []
    for i in range(n_tests):
        x = simulator(theta_all[i])
        x_list.append(x)

    return theta_all, np.array(x_list)

def plot_recovery(
    theta_true_all: np.ndarray,
    theta_recovered_all: np.ndarray,
    theta_std_all: np.ndarray,
    n_components: int,
    output_path: Path
):
    """
    Plot true vs recovered for all parameters.
    One subplot per parameter type.
    """
    n_tests = len(theta_true_all)
    param_names = []
    for i in range(n_components):
        param_names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    
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


def plot_residuals(
    theta_true: np.ndarray,
    theta_recovered: np.ndarray,
    x_obs: np.ndarray,
    n_components: int,
    freq_file: str,
    base_noise_level: float,
    output_path: Path,
    test_idx: int = 0
):
    """Plot residuals between observed and model predictions."""
    frequencies, weights = load_frequencies(freq_file)
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
    
    # Residuals
    residuals_Q = Q_obs - Q_model
    residuals_U = U_obs - U_model
    rms_Q = np.sqrt(np.mean(residuals_Q**2))
    rms_U = np.sqrt(np.mean(residuals_U**2))
    
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
    ax.fill_between(freq_ghz, -base_noise_level, base_noise_level, alpha=0.3, color="red",
                    label=f"Expected noise: {base_noise_level:.4f}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual Q")
    ax.set_title(f"Q Residuals (RMS: {rms_Q:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # U residuals
    ax = axes[1, 1]
    ax.plot(freq_ghz, residuals_U, "bo-", markersize=6)
    ax.axhline(0, color="k", linestyle="--", linewidth=1)
    ax.fill_between(freq_ghz, -base_noise_level, base_noise_level, alpha=0.3, color="red",
                    label=f"Expected noise: {base_noise_level:.4f}")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual U")
    ax.set_title(f"U Residuals (RMS: {rms_U:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Main title with parameter comparison
    title = f"Test #{test_idx+1} | N={n_components} | Noise Level={base_noise_level:.4f}\n"
    title += f"RM: True={theta_true[0]:.1f}, Rec={theta_recovered[0]:.1f} | "
    title += f"Amp: True={theta_true[1]:.3f}, Rec={theta_recovered[1]:.3f}"
    fig.suptitle(title, fontsize=12)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_posterior_contours(
    samples: np.ndarray,
    theta_true: np.ndarray,
    n_components: int,
    output_path: Path,
    test_idx: int = 0
):
    """
    Plot posterior contours with true values marked.
    Vibrant pink/cyan/red style.
    """
    # Build parameter labels
    labels = []
    for i in range(n_components):
        labels.extend([f"RM$_{{{i+1}}}$", f"A$_{{{i+1}}}$", f"$\\chi_{{0,{i+1}}}$"])
    
    n_params = len(labels)
    
    # Set up the style
    plt.style.use('default')
    
    # Color scheme - vibrant pink/magenta
    color = "#FF1493"  # Deep pink / magenta
    truth_color = "#00FFFF"  # Cyan
    
    # Create corner plot
    fig = corner.corner(
        samples,
        labels=labels,
        truths=theta_true,
        truth_color=truth_color,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".3f",
        title_kwargs={"fontsize": 11},
        label_kwargs={"fontsize": 12},
        levels=[0.393, 0.865, 0.989],  # 1, 2, 3 sigma
        smooth=1.2,
        smooth1d=1.0,
        plot_contours=True,
        fill_contours=True,
        plot_datapoints=False,
        plot_density=False,
        color=color,
    )
    
    # Style the axes
    axes = np.array(fig.axes).reshape((n_params, n_params))
    
    for i in range(n_params):
        for j in range(n_params):
            if j > i:
                continue
            ax = axes[i, j]
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.spines['left'].set_linewidth(0.8)
            ax.spines['bottom'].set_linewidth(0.8)
            
            # Tick styling
            ax.tick_params(axis='both', which='major', labelsize=9, 
                          direction='out', length=4, width=0.8)
            
            # Add subtle grid to 2D panels
            if j < i:
                ax.grid(True, alpha=0.15, linestyle='-', linewidth=0.5)
    
    # Add title
    fig.suptitle(
        f"Posterior Distribution — Test {test_idx + 1}",
        fontsize=14,
        fontweight='medium',
        y=1.01
    )
    
    # Add legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], color=truth_color, linewidth=2, linestyle='--', label='True value'),
        Line2D([0], [0], color=color, linewidth=8, alpha=0.7, label='Posterior'),
    ]
    fig.legend(handles=legend_elements, loc='upper right', fontsize=10, 
               frameon=True, fancybox=False, edgecolor='none', 
               bbox_to_anchor=(0.98, 0.98))
    
    plt.savefig(output_path, dpi=200, bbox_inches="tight", facecolor='white')
    plt.close(fig)
    print(f"Saved: {output_path}")

def print_summary(
    theta_true_all: np.ndarray,
    theta_recovered_all: np.ndarray,
    n_components: int
):
    """Print summary statistics."""
    param_names = []
    for i in range(n_components):
        param_names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    
    print("\n" + "="*60)
    print("VALIDATION SUMMARY")
    print("="*60)
    
    for i, name in enumerate(param_names):
        true_vals = theta_true_all[:, i]
        rec_vals = theta_recovered_all[:, i]
        
        mae = np.mean(np.abs(true_vals - rec_vals))
        rmse = np.sqrt(np.mean((true_vals - rec_vals)**2))
        bias = np.mean(rec_vals - true_vals)
        
        print(f"\n{name}:")
        print(f"  MAE:  {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  Bias: {bias:.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Simple validation for VROOM-SBI")
    parser.add_argument("--n-tests", type=int, default=20, help="Number of test cases")
    parser.add_argument("--n-samples", type=int, default=5000, help="Posterior samples per test")
    parser.add_argument("--n-residual-plots", type=int, default=5, help="Number of residual plots to make")
    parser.add_argument("--n-contour-plots", type=int, default=5, help="Number of contour plots to make")
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
    base_noise_level = get_base_noise_level(config)
    freq_file = config.get("freq_file", "freq.txt")

    posterior, model_data = load_posterior(args.model_path)
    n_components = model_data.get("n_components", 1)

    print(f"Loaded model for N={n_components} components")
    print(f"Base noise level: {base_noise_level:.4f}")
    print(f"Running {args.n_tests} test cases...")
    
    # Set device
    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        print("CUDA not available, using CPU")
    
    # Generate test cases
    theta_true_all, x_obs_all = generate_test_cases(
        n_components, args.n_tests, freq_file, flat_priors, base_noise_level, args.seed
    )
    
    # Run inference on each test case
    theta_recovered_all = []
    theta_std_all = []
    all_samples = []
    
    for i in tqdm(range(args.n_tests), desc="Running inference"):
        x_obs = torch.tensor(x_obs_all[i], dtype=torch.float32).to(device)
        
        samples = posterior.sample((args.n_samples,), x=x_obs).cpu().numpy()
        
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
            n_components, freq_file, base_noise_level,
            output_dir / f"residuals_test{i+1}.png",
            test_idx=i
        )
    
    # Plot 3: Posterior contour plots for first N tests
    print(f"\nGenerating {args.n_contour_plots} contour plots...")
    for i in range(min(args.n_contour_plots, args.n_tests)):
        plot_posterior_contours(
            all_samples[i],
            theta_true_all[i],
            n_components,
            output_dir / f"contours_test{i+1}.png",
            test_idx=i
        )
    
    # Print summary
    print_summary(theta_true_all, theta_recovered_all, n_components)