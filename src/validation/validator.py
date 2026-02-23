"""
Comprehensive Validation for VROOM-SBI.

Publication-quality validation with:
- Ground truth test case generation
- Posterior inference evaluation
- RM-Tools QUfit comparison (optional)
- Calibration and coverage metrics
- Publication-ready plots
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import subprocess
import tempfile
import logging
from tqdm import tqdm
import warnings

# Suppress matplotlib warnings
warnings.filterwarnings('ignore', category=UserWarning)

logger = logging.getLogger(__name__)

# Publication-quality plot settings
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 11,
    'figure.titlesize': 16,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3,
})


@dataclass
class TestCase:
    """A single validation test case."""
    theta_true: np.ndarray          # True parameters
    Q_obs: np.ndarray               # Observed Stokes Q
    U_obs: np.ndarray               # Observed Stokes U
    Q_true: np.ndarray              # Noise-free Stokes Q
    U_true: np.ndarray              # Noise-free Stokes U
    weights: np.ndarray             # Channel weights (0 = flagged)
    noise_level: float              # Applied noise level
    missing_fraction: float         # Fraction of flagged channels
    lambda_sq: np.ndarray           # Lambda squared values
    
    # Results (filled after inference)
    posterior_samples: Optional[np.ndarray] = None
    posterior_mean: Optional[np.ndarray] = None
    posterior_std: Optional[np.ndarray] = None
    credible_intervals: Optional[Dict[str, np.ndarray]] = None
    inference_time: Optional[float] = None
    
    # RM-Tools results (if compared)
    rmtools_estimate: Optional[np.ndarray] = None
    rmtools_uncertainty: Optional[np.ndarray] = None
    rmtools_time: Optional[float] = None


@dataclass
class ValidationMetrics:
    """Aggregated validation metrics."""
    # Per-parameter metrics
    bias: np.ndarray                    # Mean(estimate - true)
    rmse: np.ndarray                    # Root mean squared error
    mae: np.ndarray                     # Mean absolute error
    
    # Calibration (coverage at different levels)
    coverage_50: np.ndarray             # % in 50% CI
    coverage_90: np.ndarray             # % in 90% CI
    coverage_95: np.ndarray             # % in 95% CI
    
    # Sharpness (posterior width)
    mean_posterior_std: np.ndarray
    
    # Overall
    n_tests: int
    mean_inference_time: float
    
    # RM-Tools comparison (if available)
    rmtools_bias: Optional[np.ndarray] = None
    rmtools_rmse: Optional[np.ndarray] = None
    rmtools_mean_time: Optional[float] = None


class PosteriorValidator:
    """
    Comprehensive posterior validation system.
    
    Generates test cases, runs inference, computes metrics,
    and creates publication-quality plots.
    """
    
    def __init__(
        self,
        posterior_path: Path,
        output_dir: Path,
        device: str = "auto",  # Auto-detect by default
    ):
        """
        Initialize validator.
        
        Parameters
        ----------
        posterior_path : Path
            Path to posterior .pt file or models directory
        output_dir : Path
            Output directory for results and plots
        device : str
            Device for inference ('auto', 'cuda', or 'cpu')
        """
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Auto-detect device
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        
        # Load posterior
        self._load_posterior()
        
        # Storage for test cases and results
        self.test_cases: List[TestCase] = []
        self.metrics: Optional[ValidationMetrics] = None
    
    def _load_posterior(self):
        """Load posterior model and extract metadata."""
        logger.info(f"Loading posterior from {self.posterior_path}")
        
        # Load on CPU first, then move to target device
        checkpoint = torch.load(self.posterior_path, map_location='cpu', weights_only=False)
        
        # Extract metadata
        self.model_type = checkpoint['model_type']
        self.n_components = checkpoint['n_components']
        self.n_params = checkpoint['n_params']
        self.n_freq = checkpoint['n_freq']
        self.params_per_comp = checkpoint['params_per_comp']
        self.lambda_sq = np.array(checkpoint['lambda_sq'])
        self.prior_bounds = checkpoint['prior_bounds']
        
        # Try to get the posterior object
        self.posterior = checkpoint.get('posterior')
        
        # If posterior is None or doesn't work, try to rebuild it
        if self.posterior is None:
            logger.info("  Posterior object not found, attempting to rebuild...")
            self._rebuild_posterior(checkpoint)
        
        # Verify posterior works
        if self.posterior is None:
            raise RuntimeError("Could not load or rebuild posterior from checkpoint")
        
        # Test that posterior has sample method
        if not hasattr(self.posterior, 'sample'):
            raise RuntimeError(f"Posterior object has no 'sample' method: {type(self.posterior)}")
        
        # Move to target device
        # Note: SBI's DirectPosterior.to() modifies in place and may not return self
        if self.device == 'cuda' and torch.cuda.is_available():
            try:
                # Call .to() but don't reassign - it modifies in place
                self.posterior.to(self.device)
                logger.info(f"  Using device: {self.device}")
            except Exception as e:
                logger.warning(f"Could not move posterior to {self.device}: {e}")
                logger.warning("Falling back to CPU")
                self.device = 'cpu'
        else:
            if self.device == 'cuda' and not torch.cuda.is_available():
                logger.warning("CUDA requested but not available, using CPU")
                self.device = 'cpu'
            logger.info(f"  Using device: {self.device}")
        
        logger.info(f"  Model: {self.model_type}, N={self.n_components}")
        logger.info(f"  Parameters: {self.n_params} ({self.params_per_comp} per component)")
        logger.info(f"  Frequencies: {self.n_freq}")
        
        # Build parameter names
        self._build_param_names()
    
    def _rebuild_posterior(self, checkpoint: Dict):
        """Rebuild posterior from saved state dicts."""
        try:
            from sbi.neural_nets.net_builders import build_nsf, build_maf
            from sbi.inference.posteriors import DirectPosterior
            from torch.distributions import Uniform
            import torch
            
            # Get config
            config_used = checkpoint.get('config_used', {})
            sbi_model = config_used.get('sbi_model', 'nsf')
            embedding_dim = config_used.get('embedding_dim', 64)
            hidden_features = config_used.get('hidden_features', 256)
            num_transforms = config_used.get('num_transforms', 15)
            
            # Create sample tensors for building the flow
            theta_sample = torch.randn(100, self.n_params)
            x_sample = torch.randn(100, 2 * self.n_freq)  # Q and U concatenated
            
            # Build embedding net
            from ..training.networks import SpectralEmbedding
            embedding_net = SpectralEmbedding(
                input_dim=2 * self.n_freq,
                output_dim=embedding_dim,
            )
            
            # Load embedding net state if available
            if 'embedding_net_state' in checkpoint:
                embedding_net.load_state_dict(checkpoint['embedding_net_state'])
            
            # Build density estimator
            build_kwargs = {
                'hidden_features': hidden_features,
                'num_transforms': num_transforms,
                'embedding_net': embedding_net,
            }
            
            if sbi_model.lower() == 'nsf':
                build_kwargs['num_bins'] = config_used.get('num_bins', 16)
                density_estimator = build_nsf(theta_sample, x_sample, **build_kwargs)
            else:
                density_estimator = build_maf(theta_sample, x_sample, **build_kwargs)
            
            # Load density estimator state if available
            if 'posterior_state' in checkpoint and checkpoint['posterior_state'] is not None:
                density_estimator.load_state_dict(checkpoint['posterior_state'])
            
            # Build prior
            low = torch.tensor(self.prior_bounds['low'], dtype=torch.float32)
            high = torch.tensor(self.prior_bounds['high'], dtype=torch.float32)
            
            from sbi.utils import BoxUniform
            prior = BoxUniform(low=low, high=high)
            
            # Build posterior
            self.posterior = DirectPosterior(
                posterior_estimator=density_estimator,
                prior=prior,
            )
            
            logger.info("  Successfully rebuilt posterior from state dict")
            
        except Exception as e:
            logger.error(f"Failed to rebuild posterior: {e}")
            import traceback
            traceback.print_exc()
            self.posterior = None
    
    def _build_param_names(self):
        """Build parameter names based on model type."""
        self.param_names = []
        
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                self.param_names.extend([f'RM_{i+1}', f'amp_{i+1}', f'χ₀_{i+1}'])
        elif self.model_type == "burn_slab":
            for i in range(self.n_components):
                self.param_names.extend([f'φ_c_{i+1}', f'Δφ_{i+1}', f'amp_{i+1}', f'χ₀_{i+1}'])
        else:  # external/internal dispersion
            for i in range(self.n_components):
                self.param_names.extend([f'φ_{i+1}', f'σ_φ_{i+1}', f'amp_{i+1}', f'χ₀_{i+1}'])
    
    def generate_test_cases(
        self,
        n_tests: int = 100,
        noise_levels: List[float] = [0.01, 0.05, 0.1],
        missing_fractions: List[float] = [0.0, 0.1, 0.3],
        prior_overrides: Optional[Dict[str, float]] = None,
        seed: int = 42,
    ):
        """
        Generate test cases with known ground truth.
        
        Parameters
        ----------
        n_tests : int
            Number of test cases per noise/missing combination
        noise_levels : List[float]
            Noise levels to test
        missing_fractions : List[float]
            Fractions of missing channels to test
        prior_overrides : Dict, optional
            Override prior bounds (e.g., {'rm_min': 0, 'rm_max': 500})
        seed : int
            Random seed for reproducibility
        """
        np.random.seed(seed)
        
        # Get prior bounds (possibly overridden)
        low = np.array(self.prior_bounds['low'])
        high = np.array(self.prior_bounds['high'])
        
        if prior_overrides:
            # Override specific bounds based on parameter type
            for i in range(self.n_components):
                if self.model_type == "faraday_thin":
                    base_idx = i * 3
                    if 'rm_min' in prior_overrides:
                        low[base_idx] = prior_overrides['rm_min']
                    if 'rm_max' in prior_overrides:
                        high[base_idx] = prior_overrides['rm_max']
                    if 'amp_min' in prior_overrides:
                        low[base_idx + 1] = prior_overrides['amp_min']
                    if 'amp_max' in prior_overrides:
                        high[base_idx + 1] = prior_overrides['amp_max']
                    if 'chi0_min' in prior_overrides:
                        low[base_idx + 2] = prior_overrides['chi0_min']
                    if 'chi0_max' in prior_overrides:
                        high[base_idx + 2] = prior_overrides['chi0_max']
        
        logger.info(f"Generating {n_tests} test cases per condition")
        logger.info(f"  Noise levels: {noise_levels}")
        logger.info(f"  Missing fractions: {missing_fractions}")
        logger.info(f"  Prior bounds: {low} to {high}")
        
        self.test_cases = []
        
        for noise_level in noise_levels:
            for missing_frac in missing_fractions:
                for _ in range(n_tests):
                    # Draw random parameters from prior
                    theta_true = np.random.uniform(low, high)
                    
                    # Generate clean spectrum
                    Q_true, U_true = self._simulate_spectrum(theta_true)
                    
                    # Add noise
                    noise_Q = np.random.normal(0, noise_level, self.n_freq)
                    noise_U = np.random.normal(0, noise_level, self.n_freq)
                    Q_obs = Q_true + noise_Q
                    U_obs = U_true + noise_U
                    
                    # Generate weights with missing channels
                    weights = self._generate_weights(missing_frac)
                    
                    # Create test case
                    test_case = TestCase(
                        theta_true=theta_true,
                        Q_obs=Q_obs,
                        U_obs=U_obs,
                        Q_true=Q_true,
                        U_true=U_true,
                        weights=weights,
                        noise_level=noise_level,
                        missing_fraction=missing_frac,
                        lambda_sq=self.lambda_sq,
                    )
                    self.test_cases.append(test_case)
        
        logger.info(f"Generated {len(self.test_cases)} total test cases")
    
    def _simulate_spectrum(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate Q/U spectrum from parameters."""
        P_complex = np.zeros(self.n_freq, dtype=complex)
        
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                rm = theta[i * 3]
                amp = theta[i * 3 + 1]
                chi0 = theta[i * 3 + 2]
                
                # Faraday thin: P = p * exp(2i * (chi0 + RM * lambda^2))
                phase = 2 * (chi0 + rm * self.lambda_sq)
                P_complex += amp * np.exp(1j * phase)
        
        elif self.model_type == "burn_slab":
            for i in range(self.n_components):
                phi_c = theta[i * 4]
                delta_phi = theta[i * 4 + 1]
                amp = theta[i * 4 + 2]
                chi0 = theta[i * 4 + 3]
                
                # Burn slab: P = p * sinc(delta_phi * lambda^2) * exp(2i * (chi0 + phi_c * lambda^2))
                sinc_arg = delta_phi * self.lambda_sq
                sinc_term = np.sinc(sinc_arg / np.pi)  # np.sinc(x) = sin(pi*x)/(pi*x)
                phase = 2 * (chi0 + phi_c * self.lambda_sq)
                P_complex += amp * sinc_term * np.exp(1j * phase)
        
        elif self.model_type in ["external_dispersion", "internal_dispersion"]:
            for i in range(self.n_components):
                phi = theta[i * 4]
                sigma_phi = theta[i * 4 + 1]
                amp = theta[i * 4 + 2]
                chi0 = theta[i * 4 + 3]
                
                # External: P = p * exp(-2 * sigma^2 * lambda^4) * exp(2i * (chi0 + phi * lambda^2))
                depol = np.exp(-2 * sigma_phi**2 * self.lambda_sq**2)
                phase = 2 * (chi0 + phi * self.lambda_sq)
                P_complex += amp * depol * np.exp(1j * phase)
        
        return P_complex.real, P_complex.imag
    
    def _generate_weights(self, missing_fraction: float) -> np.ndarray:
        """Generate weights with missing channels."""
        weights = np.ones(self.n_freq)
        
        if missing_fraction > 0:
            n_missing = int(self.n_freq * missing_fraction)
            
            # Mix of scattered and contiguous gaps
            n_scattered = n_missing // 2
            n_gap = n_missing - n_scattered
            
            # Scattered missing
            if n_scattered > 0:
                scattered_idx = np.random.choice(self.n_freq, n_scattered, replace=False)
                weights[scattered_idx] = 0
            
            # Contiguous gap
            if n_gap > 0:
                available = np.where(weights > 0)[0]
                if len(available) > n_gap:
                    gap_start = np.random.choice(len(available) - n_gap)
                    gap_idx = available[gap_start:gap_start + n_gap]
                    weights[gap_idx] = 0
        
        return weights
    
    def run_inference(
        self,
        n_samples: int = 5000,
        show_progress: bool = True,
    ):
        """
        Run VROOM-SBI inference on all test cases.
        
        Parameters
        ----------
        n_samples : int
            Number of posterior samples per test case
        show_progress : bool
            Show progress bar
        """
        logger.info(f"Running inference on {len(self.test_cases)} test cases")
        
        iterator = tqdm(self.test_cases, disable=not show_progress, desc="Inference")
        
        for test_case in iterator:
            start_time = datetime.now()
            
            # Prepare observation (concatenate Q and U, apply weights)
            x_obs = np.concatenate([
                test_case.Q_obs * test_case.weights,
                test_case.U_obs * test_case.weights
            ])
            x_obs_tensor = torch.tensor(x_obs, dtype=torch.float32).unsqueeze(0)
            
            # Move to device
            if self.device != 'cpu':
                x_obs_tensor = x_obs_tensor.to(self.device)
            
            # Sample posterior
            try:
                with torch.no_grad():
                    samples = self.posterior.sample((n_samples,), x=x_obs_tensor)
                    if isinstance(samples, torch.Tensor):
                        samples = samples.cpu().numpy()
                    samples = samples.squeeze()
            except Exception as e:
                logger.warning(f"Inference failed: {e}")
                continue
            
            inference_time = (datetime.now() - start_time).total_seconds()
            
            # Store results
            test_case.posterior_samples = samples
            test_case.posterior_mean = np.mean(samples, axis=0)
            test_case.posterior_std = np.std(samples, axis=0)
            test_case.inference_time = inference_time
            
            # Compute credible intervals
            test_case.credible_intervals = {
                '50': np.percentile(samples, [25, 75], axis=0),
                '90': np.percentile(samples, [5, 95], axis=0),
                '95': np.percentile(samples, [2.5, 97.5], axis=0),
            }
        
        logger.info("Inference complete")
    
    def run_rmtools_comparison(
        self,
        show_progress: bool = True,
    ):
        """
        Run RM-Tools QUfit for comparison.
        
        Requires RM-Tools to be installed with `qufit` command available.
        """
        # Check if qufit is available
        try:
            result = subprocess.run(['qufit', '-h'], capture_output=True, timeout=5)
        except (FileNotFoundError, subprocess.TimeoutExpired):
            logger.warning("RM-Tools qufit not found. Skipping comparison.")
            return
        
        logger.info("Running RM-Tools QUfit comparison")
        
        # Map our model to RM-Tools model number
        model_map = {
            ('faraday_thin', 1): 'm1',
            ('faraday_thin', 2): 'm11',
            ('faraday_thin', 3): 'm111',
            ('burn_slab', 1): 'm5',
            ('burn_slab', 2): 'm6',
        }
        
        rmtools_model = model_map.get((self.model_type, self.n_components))
        if rmtools_model is None:
            logger.warning(f"No RM-Tools model for {self.model_type} N={self.n_components}")
            return
        
        iterator = tqdm(self.test_cases, disable=not show_progress, desc="RM-Tools")
        
        for test_case in iterator:
            start_time = datetime.now()
            
            with tempfile.TemporaryDirectory() as tmpdir:
                # Write data file
                data_file = Path(tmpdir) / "data.txt"
                freq_hz = 3e8 / np.sqrt(self.lambda_sq)  # Convert lambda^2 to frequency
                
                # RM-Tools format: freq, I, Q, U, dI, dQ, dU
                # We don't have I, so use 5-column format: freq, Q, U, dQ, dU
                with open(data_file, 'w') as f:
                    for i in range(self.n_freq):
                        if test_case.weights[i] > 0:
                            f.write(f"{freq_hz[i]:.6e} {test_case.Q_obs[i]:.6e} "
                                   f"{test_case.U_obs[i]:.6e} {test_case.noise_level:.6e} "
                                   f"{test_case.noise_level:.6e}\n")
                
                # Run qufit
                try:
                    result = subprocess.run(
                        ['qufit', str(data_file), '-m', rmtools_model[1:], '-v'],
                        capture_output=True,
                        text=True,
                        timeout=60,
                        cwd=tmpdir,
                    )
                    
                    # Parse results (simplified - would need proper parsing)
                    # For now, just record timing
                    test_case.rmtools_time = (datetime.now() - start_time).total_seconds()
                    
                except subprocess.TimeoutExpired:
                    logger.warning("RM-Tools timed out")
                except Exception as e:
                    logger.warning(f"RM-Tools failed: {e}")
    
    def compute_metrics(self) -> ValidationMetrics:
        """Compute validation metrics from test results."""
        logger.info("Computing validation metrics")
        
        # Filter successful test cases
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        n_valid = len(valid_cases)
        
        if n_valid == 0:
            raise RuntimeError("No valid test cases to compute metrics")
        
        # Collect arrays
        theta_true = np.array([tc.theta_true for tc in valid_cases])
        theta_mean = np.array([tc.posterior_mean for tc in valid_cases])
        theta_std = np.array([tc.posterior_std for tc in valid_cases])
        
        # Bias and errors
        errors = theta_mean - theta_true
        bias = np.mean(errors, axis=0)
        rmse = np.sqrt(np.mean(errors**2, axis=0))
        mae = np.mean(np.abs(errors), axis=0)
        
        # Coverage
        def compute_coverage(level: str) -> np.ndarray:
            coverage = np.zeros(self.n_params)
            for p in range(self.n_params):
                in_interval = 0
                for tc in valid_cases:
                    low, high = tc.credible_intervals[level][0, p], tc.credible_intervals[level][1, p]
                    if low <= tc.theta_true[p] <= high:
                        in_interval += 1
                coverage[p] = in_interval / n_valid
            return coverage
        
        coverage_50 = compute_coverage('50')
        coverage_90 = compute_coverage('90')
        coverage_95 = compute_coverage('95')
        
        # Mean posterior width
        mean_posterior_std = np.mean(theta_std, axis=0)
        
        # Timing
        mean_inference_time = np.mean([tc.inference_time for tc in valid_cases])
        
        self.metrics = ValidationMetrics(
            bias=bias,
            rmse=rmse,
            mae=mae,
            coverage_50=coverage_50,
            coverage_90=coverage_90,
            coverage_95=coverage_95,
            mean_posterior_std=mean_posterior_std,
            n_tests=n_valid,
            mean_inference_time=mean_inference_time,
        )
        
        # RM-Tools metrics if available
        rmtools_cases = [tc for tc in valid_cases if tc.rmtools_estimate is not None]
        if rmtools_cases:
            rmtools_true = np.array([tc.theta_true for tc in rmtools_cases])
            rmtools_est = np.array([tc.rmtools_estimate for tc in rmtools_cases])
            rmtools_errors = rmtools_est - rmtools_true
            
            self.metrics.rmtools_bias = np.mean(rmtools_errors, axis=0)
            self.metrics.rmtools_rmse = np.sqrt(np.mean(rmtools_errors**2, axis=0))
            self.metrics.rmtools_mean_time = np.mean([tc.rmtools_time for tc in rmtools_cases])
        
        logger.info(f"Metrics computed from {n_valid} test cases")
        return self.metrics
    
    def create_plots(self):
        """Create all publication-quality validation plots."""
        logger.info("Creating validation plots")
        
        self._plot_calibration()
        self._plot_residuals()
        self._plot_recovery_scatter()
        self._plot_snr_dependence()
        self._plot_missing_data_effect()
        self._plot_example_corners()
        self._plot_example_spectra()
        self._plot_summary()
        
        logger.info(f"Plots saved to {self.output_dir}")
    
    def _plot_calibration(self):
        """Plot calibration (expected vs observed coverage)."""
        fig, ax = plt.subplots(figsize=(8, 8))
        
        expected_levels = [0.50, 0.90, 0.95]
        observed_coverage = [
            np.mean(self.metrics.coverage_50),
            np.mean(self.metrics.coverage_90),
            np.mean(self.metrics.coverage_95),
        ]
        
        # Plot diagonal (perfect calibration)
        ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Perfect calibration')
        
        # Plot observed
        ax.scatter(expected_levels, observed_coverage, s=150, c='#2E86AB', 
                   edgecolors='black', linewidths=1.5, zorder=5)
        
        # Per-parameter coverage
        colors = plt.cm.Set2(np.linspace(0, 1, self.n_params))
        for p in range(self.n_params):
            obs_p = [
                self.metrics.coverage_50[p],
                self.metrics.coverage_90[p],
                self.metrics.coverage_95[p],
            ]
            ax.scatter(expected_levels, obs_p, s=50, c=[colors[p]], 
                       alpha=0.6, marker='o')
        
        ax.set_xlabel('Expected Coverage')
        ax.set_ylabel('Observed Coverage')
        ax.set_title('Posterior Calibration')
        ax.set_xlim(0.4, 1.0)
        ax.set_ylim(0.4, 1.0)
        ax.legend(loc='lower right')
        ax.set_aspect('equal')
        
        # Add calibration quality text
        avg_obs = np.mean(observed_coverage)
        ax.text(0.05, 0.95, f'Mean coverage deviation: {np.mean(np.abs(np.array(observed_coverage) - np.array(expected_levels))):.2%}',
                transform=ax.transAxes, fontsize=11, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'calibration.png')
        plt.close()
    
    def _plot_residuals(self):
        """Plot residual distributions for each parameter."""
        n_cols = min(3, self.n_params)
        n_rows = (self.n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
        if self.n_params == 1:
            axes = np.array([[axes]])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        for p, ax in enumerate(axes[:self.n_params]):
            residuals = [tc.posterior_mean[p] - tc.theta_true[p] for tc in valid_cases]
            
            ax.hist(residuals, bins=30, density=True, alpha=0.7, color='#2E86AB', 
                    edgecolor='black', linewidth=0.5)
            
            # Overlay Gaussian fit
            mu, sigma = np.mean(residuals), np.std(residuals)
            x = np.linspace(min(residuals), max(residuals), 100)
            ax.plot(x, 1/(sigma * np.sqrt(2*np.pi)) * np.exp(-0.5*((x-mu)/sigma)**2),
                    'r-', lw=2, label=f'μ={mu:.3f}\nσ={sigma:.3f}')
            
            ax.axvline(0, color='black', linestyle='--', lw=1.5)
            ax.set_xlabel(f'Residual ({self.param_names[p]})')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right', fontsize=9)
        
        # Hide unused axes
        for ax in axes[self.n_params:]:
            ax.set_visible(False)
        
        plt.suptitle('Residual Distributions (Estimate - True)', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'residuals.png')
        plt.close()
    
    def _plot_recovery_scatter(self):
        """Plot true vs estimated values with error bars."""
        n_cols = min(3, self.n_params)
        n_rows = (self.n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4.5 * n_rows))
        if self.n_params == 1:
            axes = np.array([[axes]])
        axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
        
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        for p, ax in enumerate(axes[:self.n_params]):
            true_vals = [tc.theta_true[p] for tc in valid_cases]
            est_vals = [tc.posterior_mean[p] for tc in valid_cases]
            err_vals = [tc.posterior_std[p] for tc in valid_cases]
            
            # Color by noise level
            noise_levels = [tc.noise_level for tc in valid_cases]
            scatter = ax.scatter(true_vals, est_vals, c=noise_levels, 
                                 cmap='viridis', s=30, alpha=0.6)
            
            # Perfect recovery line
            lims = [min(true_vals + est_vals), max(true_vals + est_vals)]
            ax.plot(lims, lims, 'k--', lw=2, label='Perfect recovery')
            
            ax.set_xlabel(f'True {self.param_names[p]}')
            ax.set_ylabel(f'Estimated {self.param_names[p]}')
            ax.set_title(f'RMSE = {self.metrics.rmse[p]:.4f}')
            
            # Colorbar
            plt.colorbar(scatter, ax=ax, label='Noise level')
        
        # Hide unused axes
        for ax in axes[self.n_params:]:
            ax.set_visible(False)
        
        plt.suptitle('Parameter Recovery', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'recovery_scatter.png')
        plt.close()
    
    def _plot_snr_dependence(self):
        """Plot accuracy vs noise level."""
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        noise_levels = sorted(set(tc.noise_level for tc in valid_cases))
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE vs noise
        ax = axes[0]
        for p in range(self.n_params):
            rmse_by_noise = []
            for noise in noise_levels:
                cases = [tc for tc in valid_cases if tc.noise_level == noise]
                errors = [(tc.posterior_mean[p] - tc.theta_true[p])**2 for tc in cases]
                rmse_by_noise.append(np.sqrt(np.mean(errors)))
            ax.plot(noise_levels, rmse_by_noise, 'o-', label=self.param_names[p], lw=2, markersize=8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('RMSE')
        ax.set_title('Accuracy vs Noise Level')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log')
        
        # Posterior width vs noise
        ax = axes[1]
        for p in range(self.n_params):
            width_by_noise = []
            for noise in noise_levels:
                cases = [tc for tc in valid_cases if tc.noise_level == noise]
                widths = [tc.posterior_std[p] for tc in cases]
                width_by_noise.append(np.mean(widths))
            ax.plot(noise_levels, width_by_noise, 'o-', label=self.param_names[p], lw=2, markersize=8)
        
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Mean Posterior Std')
        ax.set_title('Posterior Width vs Noise Level')
        ax.legend(loc='upper left', fontsize=9)
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'snr_dependence.png')
        plt.close()
    
    def _plot_missing_data_effect(self):
        """Plot accuracy vs fraction of missing data."""
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        missing_fractions = sorted(set(tc.missing_fraction for tc in valid_cases))
        
        if len(missing_fractions) < 2:
            logger.info("Only one missing fraction tested, skipping missing data plot")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # RMSE vs missing fraction
        ax = axes[0]
        for p in range(self.n_params):
            rmse_by_missing = []
            for frac in missing_fractions:
                cases = [tc for tc in valid_cases if tc.missing_fraction == frac]
                errors = [(tc.posterior_mean[p] - tc.theta_true[p])**2 for tc in cases]
                rmse_by_missing.append(np.sqrt(np.mean(errors)))
            ax.plot(missing_fractions, rmse_by_missing, 'o-', label=self.param_names[p], lw=2, markersize=8)
        
        ax.set_xlabel('Fraction of Missing Channels')
        ax.set_ylabel('RMSE')
        ax.set_title('Accuracy vs Missing Data')
        ax.legend(loc='upper left', fontsize=9)
        
        # Coverage vs missing fraction
        ax = axes[1]
        coverage_by_missing = []
        for frac in missing_fractions:
            cases = [tc for tc in valid_cases if tc.missing_fraction == frac]
            in_90ci = 0
            for tc in cases:
                for p in range(self.n_params):
                    low, high = tc.credible_intervals['90'][0, p], tc.credible_intervals['90'][1, p]
                    if low <= tc.theta_true[p] <= high:
                        in_90ci += 1
            coverage_by_missing.append(in_90ci / (len(cases) * self.n_params))
        
        ax.bar(missing_fractions, coverage_by_missing, width=0.05, color='#2E86AB', 
               edgecolor='black', linewidth=1.5)
        ax.axhline(0.90, color='red', linestyle='--', lw=2, label='Expected 90%')
        ax.set_xlabel('Fraction of Missing Channels')
        ax.set_ylabel('90% CI Coverage')
        ax.set_title('Calibration vs Missing Data')
        ax.legend()
        ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'missing_data_effect.png')
        plt.close()
    
    def _plot_example_corners(self, n_examples: int = 3):
        """Plot corner plots for selected example cases."""
        try:
            import corner
        except ImportError:
            logger.warning("corner package not available, skipping corner plots")
            return
        
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        # Select examples from different noise levels
        noise_levels = sorted(set(tc.noise_level for tc in valid_cases))
        
        for noise in noise_levels[:n_examples]:
            cases = [tc for tc in valid_cases if tc.noise_level == noise]
            if not cases:
                continue
            
            tc = cases[0]  # First case at this noise level
            
            fig = corner.corner(
                tc.posterior_samples,
                labels=self.param_names,
                truths=tc.theta_true,
                quantiles=[0.16, 0.5, 0.84],
                show_titles=True,
                title_fmt='.3f',
                truth_color='red',
            )
            fig.suptitle(f'Posterior (noise={noise:.3f})', y=1.02)
            plt.savefig(self.output_dir / f'corner_noise{noise:.3f}.png')
            plt.close()
    
    def _plot_example_spectra(self, n_examples: int = 4):
        """Plot Q/U spectra with posterior predictive."""
        valid_cases = [tc for tc in self.test_cases if tc.posterior_samples is not None]
        
        # Select diverse examples
        examples = valid_cases[:n_examples]
        
        fig, axes = plt.subplots(n_examples, 2, figsize=(12, 3 * n_examples))
        if n_examples == 1:
            axes = axes.reshape(1, -1)
        
        for i, tc in enumerate(examples):
            # Generate posterior predictive samples
            n_pred = min(100, len(tc.posterior_samples))
            Q_pred = []
            U_pred = []
            for j in range(n_pred):
                Q_j, U_j = self._simulate_spectrum(tc.posterior_samples[j])
                Q_pred.append(Q_j)
                U_pred.append(U_j)
            Q_pred = np.array(Q_pred)
            U_pred = np.array(U_pred)
            
            # Plot Q
            ax = axes[i, 0]
            ax.fill_between(tc.lambda_sq, np.percentile(Q_pred, 5, axis=0),
                           np.percentile(Q_pred, 95, axis=0), alpha=0.3, color='blue')
            ax.plot(tc.lambda_sq, tc.Q_true, 'k-', lw=2, label='True')
            ax.scatter(tc.lambda_sq[tc.weights > 0], tc.Q_obs[tc.weights > 0], 
                       s=20, c='red', alpha=0.6, label='Observed')
            ax.set_xlabel('λ² (m²)')
            ax.set_ylabel('Stokes Q')
            if i == 0:
                ax.legend(loc='upper right')
            
            # Plot U
            ax = axes[i, 1]
            ax.fill_between(tc.lambda_sq, np.percentile(U_pred, 5, axis=0),
                           np.percentile(U_pred, 95, axis=0), alpha=0.3, color='blue')
            ax.plot(tc.lambda_sq, tc.U_true, 'k-', lw=2, label='True')
            ax.scatter(tc.lambda_sq[tc.weights > 0], tc.U_obs[tc.weights > 0],
                       s=20, c='red', alpha=0.6, label='Observed')
            ax.set_xlabel('λ² (m²)')
            ax.set_ylabel('Stokes U')
        
        plt.suptitle('Example Spectra with Posterior Predictive', y=1.02)
        plt.tight_layout()
        plt.savefig(self.output_dir / 'example_spectra.png')
        plt.close()
    
    def _plot_summary(self):
        """Create summary figure with key metrics."""
        fig = plt.figure(figsize=(14, 10))
        gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)
        
        # 1. Bias bar chart
        ax = fig.add_subplot(gs[0, 0])
        x = np.arange(self.n_params)
        ax.bar(x, self.metrics.bias, color='#2E86AB', edgecolor='black')
        ax.axhline(0, color='black', linestyle='-', lw=1)
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names, rotation=45, ha='right')
        ax.set_ylabel('Bias')
        ax.set_title('Parameter Bias')
        
        # 2. RMSE bar chart
        ax = fig.add_subplot(gs[0, 1])
        ax.bar(x, self.metrics.rmse, color='#A23B72', edgecolor='black')
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names, rotation=45, ha='right')
        ax.set_ylabel('RMSE')
        ax.set_title('Root Mean Squared Error')
        
        # 3. Coverage bar chart
        ax = fig.add_subplot(gs[0, 2])
        width = 0.25
        ax.bar(x - width, self.metrics.coverage_50, width, label='50% CI', color='#F18F01')
        ax.bar(x, self.metrics.coverage_90, width, label='90% CI', color='#C73E1D')
        ax.bar(x + width, self.metrics.coverage_95, width, label='95% CI', color='#3B1F2B')
        ax.axhline(0.50, color='#F18F01', linestyle='--', alpha=0.5)
        ax.axhline(0.90, color='#C73E1D', linestyle='--', alpha=0.5)
        ax.axhline(0.95, color='#3B1F2B', linestyle='--', alpha=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names, rotation=45, ha='right')
        ax.set_ylabel('Coverage')
        ax.set_title('Credible Interval Coverage')
        ax.legend(loc='lower right')
        ax.set_ylim(0, 1.1)
        
        # 4. Text summary
        ax = fig.add_subplot(gs[1, :])
        ax.axis('off')
        
        summary_text = f"""
        VALIDATION SUMMARY
        ══════════════════════════════════════════════════════════════════
        
        Model: {self.model_type} with {self.n_components} component(s)
        Number of test cases: {self.metrics.n_tests}
        Mean inference time: {self.metrics.mean_inference_time * 1000:.1f} ms
        
        PARAMETER RECOVERY:
        {'─' * 60}
        {'Parameter':<15} {'Bias':>10} {'RMSE':>10} {'MAE':>10} {'90% Cov':>10}
        {'─' * 60}
        """
        
        for p in range(self.n_params):
            summary_text += f"\n        {self.param_names[p]:<15} {self.metrics.bias[p]:>10.4f} {self.metrics.rmse[p]:>10.4f} {self.metrics.mae[p]:>10.4f} {self.metrics.coverage_90[p]:>10.1%}"
        
        summary_text += f"""
        {'─' * 60}
        
        CALIBRATION:
        • 50% CI coverage: {np.mean(self.metrics.coverage_50):.1%} (expected 50%)
        • 90% CI coverage: {np.mean(self.metrics.coverage_90):.1%} (expected 90%)
        • 95% CI coverage: {np.mean(self.metrics.coverage_95):.1%} (expected 95%)
        """
        
        ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=11,
                verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        plt.savefig(self.output_dir / 'summary.png')
        plt.close()
    
    def save_results(self):
        """Save validation results to JSON."""
        results = {
            'model_type': self.model_type,
            'n_components': self.n_components,
            'n_params': self.n_params,
            'param_names': self.param_names,
            'n_tests': self.metrics.n_tests,
            'metrics': {
                'bias': self.metrics.bias.tolist(),
                'rmse': self.metrics.rmse.tolist(),
                'mae': self.metrics.mae.tolist(),
                'coverage_50': self.metrics.coverage_50.tolist(),
                'coverage_90': self.metrics.coverage_90.tolist(),
                'coverage_95': self.metrics.coverage_95.tolist(),
                'mean_posterior_std': self.metrics.mean_posterior_std.tolist(),
                'mean_inference_time_ms': self.metrics.mean_inference_time * 1000,
            },
            'timestamp': datetime.now().isoformat(),
        }
        
        with open(self.output_dir / 'validation_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        logger.info(f"Results saved to {self.output_dir / 'validation_results.json'}")


def run_validation(
    posterior_path: str,
    output_dir: str,
    n_tests: int = 100,
    noise_levels: List[float] = [0.01, 0.05, 0.1],
    missing_fractions: List[float] = [0.0, 0.1, 0.3],
    n_posterior_samples: int = 5000,
    prior_overrides: Optional[Dict[str, float]] = None,
    compare_rmtools: bool = False,
    device: str = "auto",
    seed: int = 42,
):
    """
    Run complete validation pipeline.
    
    Parameters
    ----------
    posterior_path : str
        Path to posterior .pt file
    output_dir : str
        Output directory for results
    n_tests : int
        Number of test cases per noise/missing combination
    noise_levels : List[float]
        Noise levels to test
    missing_fractions : List[float]
        Missing channel fractions to test
    n_posterior_samples : int
        Posterior samples per test case
    prior_overrides : Dict, optional
        Override prior bounds
    compare_rmtools : bool
        Compare with RM-Tools QUfit
    device : str
        Device for inference ('auto', 'cuda', or 'cpu')
    seed : int
        Random seed
    """
    # Create validator
    validator = PosteriorValidator(
        posterior_path=Path(posterior_path),
        output_dir=Path(output_dir),
        device=device,
    )
    
    # Generate test cases
    validator.generate_test_cases(
        n_tests=n_tests,
        noise_levels=noise_levels,
        missing_fractions=missing_fractions,
        prior_overrides=prior_overrides,
        seed=seed,
    )
    
    # Run inference
    validator.run_inference(n_samples=n_posterior_samples)
    
    # Compare with RM-Tools if requested
    if compare_rmtools:
        validator.run_rmtools_comparison()
    
    # Compute metrics
    validator.compute_metrics()
    
    # Create plots
    validator.create_plots()
    
    # Save results
    validator.save_results()
    
    return validator
