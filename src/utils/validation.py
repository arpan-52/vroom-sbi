#!/usr/bin/env python3
"""
Comprehensive Validation for VROOM-SBI Trained Posteriors.

Validates ALL combinations of:
- model_types (faraday_thin, burn_slab, external_dispersion, internal_dispersion)
- n_components (1 to max_components)

For each posterior:
- Parameter recovery on test cases
- Recovery plots (true vs recovered)
- Posterior contour plots
- Q/U residual plots
- Summary statistics
"""

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def validate_all_models(
    config,
    models_dir: Path,
    output_dir: Path,
    n_tests: int = 20,
    n_samples: int = 5000,
    n_residual_plots: int = 5,
    n_contour_plots: int = 5,
    device: str = "cuda",
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Validate all trained posteriors.
    
    Parameters
    ----------
    config : Configuration
        Configuration object
    models_dir : Path
        Directory containing trained posteriors
    output_dir : Path
        Output directory for validation results
    n_tests : int
        Number of test cases per model
    n_samples : int
        Posterior samples per test case
    n_residual_plots : int
        Number of residual plots per model
    n_contour_plots : int
        Number of contour plots per model
    device : str
        Device (cuda/cpu)
    seed : int
        Random seed
        
    Returns
    -------
    Dict[str, Any]
        Summary of all validation results
    """
    # Import here to avoid circular imports
    from ..simulator import RMSimulator
    from ..simulator.prior import sample_prior, sort_posterior_samples, get_param_names, get_params_per_component
    from ..simulator.physics import load_frequencies
    
    # Check device
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
        logger.warning("CUDA not available, using CPU")
    
    models_dir = Path(models_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get configuration
    flat_priors = {
        "rm_min": config.priors.rm_min,
        "rm_max": config.priors.rm_max,
        "amp_min": config.priors.amp_min,
        "amp_max": config.priors.amp_max,
    }
    base_noise_level = config.noise.base_level
    freq_file = config.freq_file
    model_types = config.physics.model_types
    max_components = config.model_selection.max_components
    
    print("\n" + "=" * 70)
    print("VROOM-SBI POSTERIOR VALIDATION")
    print("=" * 70)
    print(f"Models directory: {models_dir}")
    print(f"Model types: {model_types}")
    print(f"Components: 1-{max_components}")
    print(f"Total posteriors to validate: {len(model_types) * max_components}")
    print(f"Test cases per model: {n_tests}")
    print(f"Posterior samples per test: {n_samples}")
    print(f"Device: {device}")
    print("=" * 70)
    
    all_results = []
    
    for model_type in model_types:
        for n_components in range(1, max_components + 1):
            result = _validate_one_model(
                model_type=model_type,
                n_components=n_components,
                models_dir=models_dir,
                freq_file=freq_file,
                flat_priors=flat_priors,
                base_noise_level=base_noise_level,
                n_tests=n_tests,
                n_samples=n_samples,
                n_residual_plots=n_residual_plots,
                n_contour_plots=n_contour_plots,
                device=device,
                output_dir=output_dir,
                seed=seed + n_components,  # Different seed for each
            )
            
            if result is not None:
                all_results.append(result)
    
    # Create summary report
    _create_summary_report(all_results, output_dir)
    
    print(f"\n✓ Validation complete!")
    print(f"✓ Results saved to: {output_dir}")
    
    return {"results": all_results}


def _validate_one_model(
    model_type: str,
    n_components: int,
    models_dir: Path,
    freq_file: str,
    flat_priors: Dict[str, float],
    base_noise_level: float,
    n_tests: int,
    n_samples: int,
    n_residual_plots: int,
    n_contour_plots: int,
    device: str,
    output_dir: Path,
    seed: int = 42,
) -> Optional[Dict[str, Any]]:
    """Validate a single posterior model."""
    from ..simulator import RMSimulator
    from ..simulator.prior import sample_prior, sort_posterior_samples, get_param_names, get_params_per_component
    from ..simulator.physics import load_frequencies
    
    # Load posterior
    model_path = models_dir / f"posterior_{model_type}_n{n_components}.pt"
    
    # Try .pt first, then .pkl for backwards compatibility
    if not model_path.exists():
        model_path = models_dir / f"posterior_{model_type}_n{n_components}.pkl"
    
    if not model_path.exists():
        print(f"  ⚠️  Model not found: {model_path}")
        return None
    
    print(f"\n{'=' * 70}")
    print(f"Validating: {model_type}, N={n_components}")
    print(f"{'=' * 70}")
    
    # Load model and move to device
    posterior, model_data, actual_device = _load_posterior(model_path, device=device)
    device = actual_device  # Use actual device (may have fallen back to CPU)
    
    # Sanity check
    if posterior is None:
        print(f"  ⚠️  ERROR: Posterior is None after loading!")
        print(f"  Model data keys: {list(model_data.keys())}")
        return None
    
    # Verify model configuration
    saved_n_components = model_data.get("n_components")
    if saved_n_components != n_components:
        print(f"  ⚠️  WARNING: Model has N={saved_n_components} but expected N={n_components}")
        return None
    
    # Set random seeds
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # Create simulator (no model_params needed - bounds are in flat_priors)
    simulator = RMSimulator(
        freq_file=freq_file,
        n_components=n_components,
        base_noise_level=base_noise_level,
        model_type=model_type,
    )
    
    params_per_comp = get_params_per_component(model_type)
    param_names = get_param_names(model_type, n_components)
    
    # Generate test cases with known parameters (sorted by RM)
    theta_true_all = sample_prior(
        n_tests, n_components, flat_priors,
        model_type=model_type
    )
    
    # Simulate observations
    x_obs_all = np.array([simulator(theta) for theta in theta_true_all])
    
    # Run inference on each test case
    theta_recovered_all = []
    theta_std_all = []
    all_samples = []
    
    print(f"Running inference on {n_tests} test cases...")
    for i in tqdm(range(n_tests), desc="Inference"):
        x_obs = torch.tensor(x_obs_all[i], dtype=torch.float32).to(device)
        
        # Sample from posterior
        samples = posterior.sample((n_samples,), x=x_obs).cpu().numpy()
        
        # Sort samples to ensure RM1 > RM2 > ... (break label switching)
        if n_components >= 2:
            samples = sort_posterior_samples(samples, n_components, params_per_comp)
        
        theta_recovered_all.append(np.mean(samples, axis=0))
        theta_std_all.append(np.std(samples, axis=0))
        all_samples.append(samples)
    
    theta_recovered_all = np.array(theta_recovered_all)
    theta_std_all = np.array(theta_std_all)
    
    # Compute error metrics
    errors = theta_recovered_all - theta_true_all
    mae = np.abs(errors).mean(axis=0)
    rmse = np.sqrt((errors ** 2).mean(axis=0))
    
    # Print summary
    print(f"\nParameter Recovery Summary:")
    for i, name in enumerate(param_names):
        print(f"  {name:15s}: MAE = {mae[i]:.4f}, RMSE = {rmse[i]:.4f}, "
              f"σ_post = {theta_std_all.mean(axis=0)[i]:.4f}")
    
    # Create output directory for this model
    model_output_dir = output_dir / f"{model_type}_n{n_components}"
    model_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate plots
    _plot_recovery(
        theta_true_all, theta_recovered_all, theta_std_all,
        param_names, model_type, n_components,
        model_output_dir / "recovery.png"
    )
    
    # Residual plots
    print(f"  Generating {min(n_residual_plots, n_tests)} residual plots...")
    for i in range(min(n_residual_plots, n_tests)):
        _plot_residuals(
            theta_true_all[i], theta_recovered_all[i], x_obs_all[i],
            simulator, freq_file, base_noise_level, model_type, n_components,
            model_output_dir / f"residuals_test{i+1}.png", test_idx=i
        )
    
    # Contour plots
    print(f"  Generating {min(n_contour_plots, n_tests)} contour plots...")
    for i in range(min(n_contour_plots, n_tests)):
        _plot_contours(
            all_samples[i], theta_true_all[i], param_names,
            model_type, n_components,
            model_output_dir / f"contours_test{i+1}.png", test_idx=i
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


def _load_posterior(model_path: Path, device: str = "cpu") -> Tuple[Any, Dict, str]:
    """
    Load posterior from file and move to device.
    
    CRITICAL: For SBI posteriors with rejection sampling, we need to move:
    1. The posterior's neural network  
    2. The prior's bounds (used for rejection sampling support check)
    
    Returns (posterior, data_dict, actual_device)
    """
    import pickle
    
    print(f"  Loading model from {model_path}")
    
    # Load to CPU first
    if model_path.suffix == ".pt":
        data = torch.load(model_path, map_location="cpu")
    else:
        with open(model_path, "rb") as f:
            data = pickle.load(f)
    
    # Debug: show available keys
    print(f"  Model file keys: {list(data.keys())}")
    
    # Try different keys for the posterior
    posterior = None
    for key in ['posterior', 'posterior_object']:
        if key in data and data[key] is not None:
            posterior = data[key]
            print(f"  Found posterior under key: '{key}'")
            break
    
    if posterior is None:
        raise ValueError(f"No posterior found in {model_path}. Available keys: {list(data.keys())}")
    
    actual_device = device
    
    # Move everything to device if needed
    if device != "cpu":
        try:
            # 1. Move the posterior's neural network
            if hasattr(posterior, 'posterior_estimator'):
                posterior.posterior_estimator = posterior.posterior_estimator.to(device)
                print(f"  Moved posterior_estimator to {device}")
            if hasattr(posterior, '_neural_net'):
                posterior._neural_net = posterior._neural_net.to(device)
                print(f"  Moved _neural_net to {device}")
            if hasattr(posterior, 'net'):
                posterior.net = posterior.net.to(device)
                print(f"  Moved net to {device}")
            
            # 2. CRITICAL: Move the prior bounds (needed for rejection sampling)
            def move_prior_to_device(prior, dev):
                """Move prior bounds to device."""
                if prior is None:
                    return False
                moved = False
                # Handle BoxUniform wrapped in Independent
                if hasattr(prior, 'base_dist'):
                    bd = prior.base_dist
                    if hasattr(bd, 'low') and hasattr(bd, 'high'):
                        bd.low = bd.low.to(dev)
                        bd.high = bd.high.to(dev)
                        moved = True
                # Handle direct BoxUniform
                elif hasattr(prior, 'low') and hasattr(prior, 'high'):
                    prior.low = prior.low.to(dev)
                    prior.high = prior.high.to(dev)
                    moved = True
                return moved
            
            # Check different attribute names for prior
            prior_moved = False
            if hasattr(posterior, '_prior'):
                prior_moved = move_prior_to_device(posterior._prior, device) or prior_moved
            if hasattr(posterior, 'prior'):
                prior_moved = move_prior_to_device(posterior.prior, device) or prior_moved
            
            if prior_moved:
                print(f"  Moved prior bounds to {device}")
            
            # 3. Set device attribute if it exists
            if hasattr(posterior, '_device'):
                posterior._device = device
            
            # 4. Try the generic .to() method - BUT DON'T reassign!
            # SBI's .to() modifies in place and may return None
            if hasattr(posterior, 'to'):
                try:
                    result = posterior.to(device)
                    # Only use result if it's not None (some .to() methods return self, some return None)
                    if result is not None:
                        posterior = result
                except Exception:
                    pass
                
            print(f"  Successfully moved posterior to {device}")
        except Exception as e:
            print(f"  Warning: Could not fully move to {device}: {e}")
            import traceback
            traceback.print_exc()
            print(f"  Falling back to CPU")
            actual_device = "cpu"
    
    return posterior, data, actual_device


def _plot_recovery(
    theta_true_all: np.ndarray,
    theta_recovered_all: np.ndarray,
    theta_std_all: np.ndarray,
    param_names: List[str],
    model_type: str,
    n_components: int,
    output_path: Path,
):
    """Plot true vs recovered for all parameters."""
    n_params = len(param_names)
    n_cols = min(3, n_params)
    n_rows = int(np.ceil(n_params / n_cols))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_params == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (ax, name) in enumerate(zip(axes[:n_params], param_names)):
        true_vals = theta_true_all[:, i]
        rec_vals = theta_recovered_all[:, i]
        std_vals = theta_std_all[:, i]
        
        ax.errorbar(true_vals, rec_vals, yerr=std_vals,
                    fmt='o', alpha=0.6, markersize=4, capsize=3)
        
        # Perfect recovery line
        lims = [min(true_vals.min(), rec_vals.min()),
                max(true_vals.max(), rec_vals.max())]
        ax.plot(lims, lims, 'k--', alpha=0.5, linewidth=1, label='Perfect')
        
        ax.set_xlabel(f'True {name}')
        ax.set_ylabel(f'Recovered {name}')
        ax.set_title(name, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # R² score
        ss_res = ((rec_vals - true_vals) ** 2).sum()
        ss_tot = ((true_vals - true_vals.mean()) ** 2).sum()
        r2 = 1 - ss_res / (ss_tot + 1e-10)
        ax.text(0.05, 0.95, f'R² = {r2:.3f}', transform=ax.transAxes,
                fontsize=9, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Hide unused axes
    for j in range(n_params, len(axes)):
        axes[j].set_visible(False)
    
    fig.suptitle(f'Parameter Recovery: {model_type}, N={n_components}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_residuals(
    theta_true: np.ndarray,
    theta_recovered: np.ndarray,
    x_obs: np.ndarray,
    simulator,
    freq_file: str,
    base_noise_level: float,
    model_type: str,
    n_components: int,
    output_path: Path,
    test_idx: int = 0,
):
    """Plot Q/U data with model fits and residuals."""
    from ..simulator.physics import load_frequencies
    
    frequencies, _ = load_frequencies(freq_file)
    n_freq = len(frequencies)
    freq_ghz = frequencies / 1e9
    
    Q_obs = x_obs[:n_freq]
    U_obs = x_obs[n_freq:]
    
    # Reconstruct from recovered parameters
    x_recovered = simulator(theta_recovered)
    Q_model = x_recovered[:n_freq]
    U_model = x_recovered[n_freq:]
    
    # Reconstruct from true parameters
    x_true = simulator(theta_true)
    Q_true = x_true[:n_freq]
    U_true = x_true[n_freq:]
    
    # Residuals
    residuals_Q = Q_obs - Q_model
    residuals_U = U_obs - U_model
    rms_Q = np.sqrt(np.mean(residuals_Q ** 2))
    rms_U = np.sqrt(np.mean(residuals_U ** 2))
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Q data + fits
    ax = axes[0, 0]
    ax.plot(freq_ghz, Q_obs, 'ko', markersize=4, label='Observed')
    ax.plot(freq_ghz, Q_model, 'b-', linewidth=2, label='Recovered')
    ax.plot(freq_ghz, Q_true, 'r--', linewidth=2, alpha=0.7, label='True')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Stokes Q')
    ax.set_title('Stokes Q')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # U data + fits
    ax = axes[0, 1]
    ax.plot(freq_ghz, U_obs, 'ko', markersize=4, label='Observed')
    ax.plot(freq_ghz, U_model, 'b-', linewidth=2, label='Recovered')
    ax.plot(freq_ghz, U_true, 'r--', linewidth=2, alpha=0.7, label='True')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Stokes U')
    ax.set_title('Stokes U')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Q residuals
    ax = axes[1, 0]
    ax.plot(freq_ghz, residuals_Q, 'bo-', markersize=4)
    ax.axhline(0, color='k', linestyle='--')
    ax.fill_between(freq_ghz, -base_noise_level, base_noise_level,
                    alpha=0.3, color='red', label=f'±σ = {base_noise_level:.4f}')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Residual Q')
    ax.set_title(f'Q Residuals (RMS: {rms_Q:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # U residuals
    ax = axes[1, 1]
    ax.plot(freq_ghz, residuals_U, 'bo-', markersize=4)
    ax.axhline(0, color='k', linestyle='--')
    ax.fill_between(freq_ghz, -base_noise_level, base_noise_level,
                    alpha=0.3, color='red', label=f'±σ = {base_noise_level:.4f}')
    ax.set_xlabel('Frequency (GHz)')
    ax.set_ylabel('Residual U')
    ax.set_title(f'U Residuals (RMS: {rms_U:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    fig.suptitle(f'Test #{test_idx+1} | {model_type}, N={n_components}', fontsize=12)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _plot_contours(
    samples: np.ndarray,
    theta_true: np.ndarray,
    param_names: List[str],
    model_type: str,
    n_components: int,
    output_path: Path,
    test_idx: int = 0,
):
    """Plot posterior contours with true values marked."""
    try:
        import corner
    except ImportError:
        logger.warning("corner package not installed, skipping contour plots")
        return
    
    fig = corner.corner(
        samples,
        labels=param_names,
        truths=theta_true,
        truth_color='#00FFFF',
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt='.3f',
        title_kwargs={'fontsize': 10},
        label_kwargs={'fontsize': 10},
        levels=[0.393, 0.865, 0.989],  # 1, 2, 3 sigma
        smooth=1.2,
        smooth1d=1.0,
        plot_contours=True,
        fill_contours=True,
        color='#FF1493',
    )
    
    fig.suptitle(
        f'Posterior Distribution — {model_type}, N={n_components}, Test {test_idx+1}',
        fontsize=12, y=1.01
    )
    
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def _create_summary_report(results: List[Dict], output_dir: Path):
    """Create summary text report."""
    summary_path = output_dir / "validation_summary.txt"
    
    with open(summary_path, 'w') as f:
        f.write("=" * 70 + "\n")
        f.write("VROOM-SBI POSTERIOR VALIDATION SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        for result in results:
            model_type = result["model_type"]
            n_comp = result["n_components"]
            f.write(f"\n{model_type}, N={n_comp}:\n")
            f.write("-" * 50 + "\n")
            
            for i, name in enumerate(result["param_names"]):
                f.write(f"  {name:15s}: MAE = {result['mae'][i]:.4f}, "
                        f"RMSE = {result['rmse'][i]:.4f}, "
                        f"σ_post = {result['mean_posterior_std'][i]:.4f}\n")
        
        f.write("\n" + "=" * 70 + "\n")
        f.write(f"Total models validated: {len(results)}\n")
        f.write("=" * 70 + "\n")
    
    print(f"✓ Summary saved to: {summary_path}")
