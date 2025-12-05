#!/usr/bin/env python3
"""
Validation script for VROOM-SBI trained posteriors.
"""

import argparse
import pickle
import os
from pathlib import Path
from typing import Dict, List, Tuple, Any
import warnings

import numpy as np
import torch
import matplotlib.pyplot as plt
from tqdm import tqdm

try:
    import corner
    HAS_CORNER = True
except ImportError:
    HAS_CORNER = False
    print("Warning: 'corner' package not installed.Corner plots will be skipped.")

import yaml

from src.simulator import RMSimulator, sample_prior
from src.physics import load_frequencies, freq_to_lambda_sq

warnings.filterwarnings("ignore", message=".*device.*cuda.*")


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


def load_posteriors(model_dir: str = "models", max_components: int = 5, device: str = "cpu") -> Dict[int, Any]:
    """Load all trained posteriors from disk."""
    posteriors = {}
    
    print(f"Loading posteriors from {model_dir}/...")
    
    for n in range(1, max_components + 1):
        path = Path(model_dir) / f"posterior_n{n}.pkl"
        
        if path.exists():
            with open(path, "rb") as f:
                data = pickle.load(f)
            
            posterior = data["posterior"]
            posteriors[n] = {
                "posterior": posterior,
                "n_components": n,
                "n_freq": data.get("n_freq", 12),
                "lambda_sq": data.get("lambda_sq"),
                "flat_priors": data.get("flat_priors"),
            }
            print(f"  Loaded N={n} posterior")
        else:
            print(f"  N={n} posterior not found at {path}")
    
    return posteriors


def generate_test_cases(
    n_components: int,
    n_tests: int,
    freq_file: str,
    flat_priors: Dict[str, float],
    seed: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic test cases with known ground truth."""
    np.random.seed(seed)
    
    simulator = RMSimulator(freq_file, n_components)
    theta_true = sample_prior(n_tests, n_components, flat_priors)
    
    x_obs = simulator(theta_true)
    if x_obs.ndim == 1:
        x_obs = x_obs.reshape(1, -1)
    
    return theta_true, x_obs


def run_inference(
    posterior,
    x_obs: np.ndarray,
    n_samples: int = 5000,
    device: str = "cpu"
) -> np.ndarray:
    """Run inference on observed data."""
    x_tensor = torch.tensor(x_obs, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=x_tensor)
    
    return samples.cpu().numpy()


def compute_recovery_metrics(
    theta_true: np.ndarray,
    samples: np.ndarray,
    n_components: int
) -> Dict[str, Any]:
    """Compute parameter recovery metrics."""
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)
    
    error = mean - theta_true
    
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_error = np.abs(error) / (np.abs(theta_true) + 1e-10)
    
    ci_68_low = np.percentile(samples, 16, axis=0)
    ci_68_high = np.percentile(samples, 84, axis=0)
    ci_95_low = np.percentile(samples, 2.5, axis=0)
    ci_95_high = np.percentile(samples, 97.5, axis=0)
    
    in_68ci = (theta_true >= ci_68_low) & (theta_true <= ci_68_high)
    in_95ci = (theta_true >= ci_95_low) & (theta_true <= ci_95_high)
    
    return {
        "true": theta_true,
        "mean": mean,
        "std": std,
        "error": error,
        "abs_error": np.abs(error),
        "relative_error": relative_error,
        "ci_68_low": ci_68_low,
        "ci_68_high": ci_68_high,
        "ci_95_low": ci_95_low,
        "ci_95_high": ci_95_high,
        "in_68ci": in_68ci,
        "in_95ci": in_95ci,
    }


def get_param_names(n_components: int) -> List[str]:
    """Get parameter names for a given number of components."""
    names = []
    for i in range(n_components):
        names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    names.append("noise")
    return names


def plot_corner(
    samples: np.ndarray,
    theta_true: np.ndarray,
    n_components: int,
    output_path: str,
    title: str = ""
):
    """Generate corner plot with true values marked."""
    if not HAS_CORNER:
        return
    
    param_names = get_param_names(n_components)
    
    fig = corner.corner(
        samples,
        labels=param_names,
        truths=theta_true,
        truth_color="red",
        show_titles=True,
        title_kwargs={"fontsize": 10},
        quantiles=[0.16, 0.5, 0.84],
    )
    
    if title:
        fig.suptitle(title, fontsize=14, y=1.02)
    
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_recovery(
    all_metrics: List[Dict],
    n_components: int,
    output_path: str
):
    """Plot true vs recovered parameters for all test cases."""
    param_names = get_param_names(n_components)
    n_params = len(param_names)
    
    true_vals = np.array([m["true"] for m in all_metrics])
    mean_vals = np.array([m["mean"] for m in all_metrics])
    std_vals = np.array([m["std"] for m in all_metrics])
    
    n_cols = min(4, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4*n_cols, 4*n_rows))
    axes = np.atleast_2d(axes).flatten()
    
    for i, (name, ax) in enumerate(zip(param_names, axes)):
        if i >= n_params:
            ax.axis("off")
            continue
        
        ax.errorbar(
            true_vals[:, i], mean_vals[:, i],
            yerr=std_vals[:, i],
            fmt="o", alpha=0.7, capsize=3, markersize=6
        )
        
        lims = [
            min(true_vals[:, i].min(), mean_vals[:, i].min()),
            max(true_vals[:, i].max(), mean_vals[:, i].max())
        ]
        margin = 0.1 * (lims[1] - lims[0])
        lims = [lims[0] - margin, lims[1] + margin]
        
        ax.plot(lims, lims, "k--", alpha=0.5, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        
        ax.set_xlabel(f"True {name}")
        ax.set_ylabel(f"Recovered {name}")
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    
    for i in range(n_params, len(axes)):
        axes[i].axis("off")
    
    fig.suptitle(f"Parameter Recovery: N={n_components} components", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_calibration(
    all_metrics: List[Dict],
    n_components: int,
    output_path: str
):
    """Plot calibration: fraction of true values within credible intervals."""
    param_names = get_param_names(n_components)
    n_params = len(param_names)
    
    in_68ci = np.array([m["in_68ci"] for m in all_metrics])
    in_95ci = np.array([m["in_95ci"] for m in all_metrics])
    
    coverage_68 = np.mean(in_68ci, axis=0)
    coverage_95 = np.mean(in_95ci, axis=0)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(n_params)
    width = 0.35
    
    bars1 = ax.bar(x - width/2, coverage_68, width, label="68% CI", color="steelblue", alpha=0.8)
    bars2 = ax.bar(x + width/2, coverage_95, width, label="95% CI", color="darkorange", alpha=0.8)
    
    ax.axhline(0.68, color="steelblue", linestyle="--", alpha=0.7, label="Expected 68%")
    ax.axhline(0.95, color="darkorange", linestyle="--", alpha=0.7, label="Expected 95%")
    
    ax.set_ylabel("Coverage Fraction")
    ax.set_xlabel("Parameter")
    ax.set_title(f"Posterior Calibration: N={n_components} components")
    ax.set_xticks(x)
    ax.set_xticklabels(param_names, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3, axis="y")
    
    for bar, val in zip(bars1, coverage_68):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    for bar, val in zip(bars2, coverage_95):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_residuals(
    theta_true: np.ndarray,
    x_obs: np.ndarray,
    samples: np.ndarray,
    n_components: int,
    freq_file: str,
    output_path: str,
    test_idx: int = 0
):
    """Plot observed vs reconstructed Q, U and residuals."""
    frequencies = load_frequencies(freq_file)
    lambda_sq = freq_to_lambda_sq(frequencies)
    n_freq = len(frequencies)
    
    Q_obs = x_obs[:n_freq]
    U_obs = x_obs[n_freq:]
    
    theta_mean = np.mean(samples, axis=0)
    
    P = np.zeros(n_freq, dtype=complex)
    for i in range(n_components):
        rm = theta_mean[3*i]
        amp = theta_mean[3*i + 1]
        chi0 = theta_mean[3*i + 2]
        phase = 2 * (chi0 + rm * lambda_sq)
        P += amp * np.exp(1j * phase)
    
    Q_model = P.real
    U_model = P.imag
    
    P_true = np.zeros(n_freq, dtype=complex)
    for i in range(n_components):
        rm = theta_true[3*i]
        amp = theta_true[3*i + 1]
        chi0 = theta_true[3*i + 2]
        phase = 2 * (chi0 + rm * lambda_sq)
        P_true += amp * np.exp(1j * phase)
    
    Q_true = P_true.real
    U_true = P_true.imag
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    freq_ghz = frequencies / 1e9
    
    ax = axes[0, 0]
    ax.plot(freq_ghz, Q_obs, "ko", label="Observed", markersize=8)
    ax.plot(freq_ghz, Q_model, "b-", label="Posterior mean", linewidth=2)
    ax.plot(freq_ghz, Q_true, "r--", label="True model", linewidth=2, alpha=0.7)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes Q")
    ax.set_title("Stokes Q")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(freq_ghz, U_obs, "ko", label="Observed", markersize=8)
    ax.plot(freq_ghz, U_model, "b-", label="Posterior mean", linewidth=2)
    ax.plot(freq_ghz, U_true, "r--", label="True model", linewidth=2, alpha=0.7)
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Stokes U")
    ax.set_title("Stokes U")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 0]
    residuals_Q = Q_obs - Q_model
    ax.plot(freq_ghz, residuals_Q, "bo-", markersize=6)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.fill_between(freq_ghz, -theta_mean[-1], theta_mean[-1], alpha=0.3, color="gray", label="noise")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual Q")
    ax.set_title(f"Q Residuals (RMS: {np.std(residuals_Q):.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    ax = axes[1, 1]
    residuals_U = U_obs - U_model
    ax.plot(freq_ghz, residuals_U, "bo-", markersize=6)
    ax.axhline(0, color="k", linestyle="--", alpha=0.5)
    ax.fill_between(freq_ghz, -theta_mean[-1], theta_mean[-1], alpha=0.3, color="gray", label="noise")
    ax.set_xlabel("Frequency (GHz)")
    ax.set_ylabel("Residual U")
    ax.set_title(f"U Residuals (RMS: {np.std(residuals_U):.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f"Model Fit: N={n_components} components (Test #{test_idx+1})", fontsize=14)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_error_summary(results: Dict[int, List[Dict]], output_path: str):
    """Plot summary of errors across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    models = sorted(results.keys())
    
    ax = axes[0]
    rm_errors = []
    for n in models:
        errors = []
        for m in results[n]:
            for i in range(n):
                errors.append(np.abs(m["error"][3*i]))
        rm_errors.append(errors)
    
    parts = ax.violinplot(rm_errors, positions=models, showmeans=True, showmedians=True)
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Absolute RM Error (rad/m^2)")
    ax.set_title("RM Recovery Error by Model Complexity")
    ax.set_xticks(models)
    ax.grid(True, alpha=0.3, axis="y")
    
    for i, (n, errs) in enumerate(zip(models, rm_errors)):
        median = np.median(errs)
        ax.text(n, median, f"{median:.1f}", ha="center", va="bottom", fontsize=9)
    
    ax = axes[1]
    
    coverage_68_all = []
    coverage_95_all = []
    
    for n in models:
        in_68 = np.concatenate([m["in_68ci"] for m in results[n]])
        in_95 = np.concatenate([m["in_95ci"] for m in results[n]])
        coverage_68_all.append(np.mean(in_68))
        coverage_95_all.append(np.mean(in_95))
    
    x = np.arange(len(models))
    width = 0.35
    
    ax.bar(x - width/2, coverage_68_all, width, label="68% CI", color="steelblue", alpha=0.8)
    ax.bar(x + width/2, coverage_95_all, width, label="95% CI", color="darkorange", alpha=0.8)
    
    ax.axhline(0.68, color="steelblue", linestyle="--", alpha=0.7)
    ax.axhline(0.95, color="darkorange", linestyle="--", alpha=0.7)
    
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Coverage Fraction")
    ax.set_title("Posterior Calibration by Model Complexity")
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_metrics_summary(results: Dict[int, List[Dict]], output_path: str):
    """Save summary metrics to a text file."""
    with open(output_path, "w") as f:
        f.write("=" * 70 + "\n")
        f.write("VROOM-SBI Validation Summary\n")
        f.write("=" * 70 + "\n\n")
        
        for n in sorted(results.keys()):
            metrics = results[n]
            param_names = get_param_names(n)
            
            f.write(f"\n{'='*50}\n")
            f.write(f"Model: N = {n} components\n")
            f.write(f"{'='*50}\n")
            f.write(f"Number of test cases: {len(metrics)}\n\n")
            
            all_errors = np.array([m["abs_error"] for m in metrics])
            all_in_68 = np.array([m["in_68ci"] for m in metrics])
            all_in_95 = np.array([m["in_95ci"] for m in metrics])
            
            f.write(f"{'Parameter':<12} {'MAE':>10} {'Coverage68':>12} {'Coverage95':>12}\n")
            f.write("-" * 50 + "\n")
            
            for i, name in enumerate(param_names):
                mae = np.mean(all_errors[:, i])
                cov68 = np.mean(all_in_68[:, i])
                cov95 = np.mean(all_in_95[:, i])
                f.write(f"{name:<12} {mae:>10.4f} {cov68:>12.2%} {cov95:>12.2%}\n")
            
            f.write("\n")
            
            overall_mae = np.mean(all_errors)
            overall_68 = np.mean(all_in_68)
            overall_95 = np.mean(all_in_95)
            
            f.write(f"{'OVERALL':<12} {overall_mae:>10.4f} {overall_68:>12.2%} {overall_95:>12.2%}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write("Validation complete.\n")


def main():
    parser = argparse.ArgumentParser(description="Validate VROOM-SBI posteriors")
    parser.add_argument("--n-tests", type=int, default=10, help="Number of test cases per model")
    parser.add_argument("--n-samples", type=int, default=5000, help="Posterior samples per test")
    parser.add_argument("--output-dir", type=str, default="validation_results", help="Output directory")
    parser.add_argument("--model-dir", type=str, default="models", help="Directory with trained models")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda or cpu)")
    parser.add_argument("--max-components", type=int, default=5, help="Max components to validate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    config = load_config()
    flat_priors = get_flat_priors(config)
    freq_file = config.get("freq_file", "freq.txt")
    
    posteriors = load_posteriors(args.model_dir, args.max_components, args.device)
    
    if not posteriors:
        print("No posteriors found!  Run training first.")
        return
    
    all_results: Dict[int, List[Dict]] = {}
    
    for n_components, post_data in posteriors.items():
        print(f"\n{'='*60}")
        print(f"Validating N={n_components} component model")
        print(f"{'='*60}")
        
        posterior = post_data["posterior"]
        
        theta_true, x_obs = generate_test_cases(
            n_components=n_components,
            n_tests=args.n_tests,
            freq_file=freq_file,
            flat_priors=flat_priors,
            seed=args.seed + n_components
        )
        
        metrics_list = []
        
        model_dir = output_dir / f"n{n_components}"
        model_dir.mkdir(exist_ok=True)
        
        for i in tqdm(range(args.n_tests), desc=f"Testing N={n_components}"):
            samples = run_inference(
                posterior,
                x_obs[i],
                n_samples=args.n_samples,
                device=args.device
            )
            
            metrics = compute_recovery_metrics(theta_true[i], samples, n_components)
            metrics_list.append(metrics)
            
            if i < 3 and HAS_CORNER:
                plot_corner(
                    samples, theta_true[i], n_components,
                    str(model_dir / f"corner_test{i+1}.png"),
                    title=f"N={n_components}, Test #{i+1}"
                )
            
            if i == 0:
                plot_residuals(
                    theta_true[i], x_obs[i], samples, n_components,
                    freq_file, str(model_dir / f"residuals_test{i+1}.png"),
                    test_idx=i
                )
        
        all_results[n_components] = metrics_list
        
        plot_recovery(
            metrics_list, n_components,
            str(model_dir / "recovery.png")
        )
        
        plot_calibration(
            metrics_list, n_components,
            str(model_dir /"calibration.png")
        )
        
        param_names = get_param_names(n_components)
        print(f"\n  Results for N={n_components}:")
        
        all_errors = np.array([m["abs_error"] for m in metrics_list])
        all_in_68 = np.array([m["in_68ci"] for m in metrics_list])
        all_in_95 = np.array([m["in_95ci"] for m in metrics_list])
        
        for j, name in enumerate(param_names):
            mae = np.mean(all_errors[:, j])
            cov68 = np.mean(all_in_68[:, j])
            cov95 = np.mean(all_in_95[:, j])
            print(f"    {name}: MAE={mae:.4f}, 68%CI={cov68:.0%}, 95%CI={cov95:.0%}")
    
    print(f"\n{'='*60}")
    print("Generating summary plots...")
    print(f"{'='*60}")
    
    plot_error_summary(all_results, str(output_dir/"error_summary.png"))
    save_metrics_summary(all_results, str(output_dir/"metrics_summary.txt"))
    
    print(f"\nValidation complete!")
    print(f"  Results saved to: {output_dir}/")
    print(f"  - Per-model results in n1/, n2/, ...subdirectories")
    print(f"  - Overall summary: error_summary.png, metrics_summary.txt")


if __name__ == "__main__":
    main()