"""
Comprehensive Validation for VROOM-SBI with RM-Tools comparison.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import json
import subprocess
import tempfile
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12,
    'figure.dpi': 150, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
})

COLORS = {'vroom': '#2E86AB', 'rmtools': '#E94F37', 'true': '#1B998B'}


@dataclass
class SingleTestResult:
    theta_true: np.ndarray
    Q_true: np.ndarray
    U_true: np.ndarray
    Q_obs: np.ndarray
    U_obs: np.ndarray
    weights: np.ndarray
    noise_level: float
    vroom_samples: Optional[np.ndarray] = None
    vroom_mean: Optional[np.ndarray] = None
    vroom_std: Optional[np.ndarray] = None
    vroom_ci_90: Optional[Tuple[np.ndarray, np.ndarray]] = None
    vroom_time: Optional[float] = None
    rmtools_estimate: Optional[np.ndarray] = None
    rmtools_uncertainty: Optional[np.ndarray] = None
    rmtools_time: Optional[float] = None
    rmtools_success: bool = False


class ComprehensiveValidator:
    def __init__(self, posterior_path: Path, output_dir: Path, 
                 rmtools_env: str = "rmtools", device: str = "auto"):
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.rmtools_env = rmtools_env
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else device if device != "auto" else "cpu"
        self._create_output_dirs()
        self._load_posterior()

    def _create_output_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dirs = {
            'parameter_recovery': self.output_dir / '01_parameter_recovery',
            'noise_analysis': self.output_dir / '02_noise_analysis',
            'individual_cases': self.output_dir / '03_individual_cases',
            'comparison': self.output_dir / '04_vroom_vs_rmtools',
            'posteriors': self.output_dir / '05_posterior_plots',
            'spectra': self.output_dir / '06_spectra_fits',
            'summary': self.output_dir / '07_summary',
            'data': self.output_dir / '08_raw_data',
        }
        for d in self.dirs.values():
            d.mkdir(parents=True, exist_ok=True)

    def _load_posterior(self):
        logger.info(f"Loading posterior from {self.posterior_path}")
        checkpoint = torch.load(self.posterior_path, map_location='cpu', weights_only=False)
        self.model_type = checkpoint['model_type']
        self.n_components = checkpoint['n_components']
        self.n_params = checkpoint['n_params']
        self.n_freq = checkpoint['n_freq']
        self.params_per_comp = checkpoint['params_per_comp']
        self.lambda_sq = np.array(checkpoint['lambda_sq'])
        self.prior_bounds = checkpoint['prior_bounds']
        self.posterior = checkpoint['posterior']
        if self.device == 'cuda' and torch.cuda.is_available():
            self.posterior.to(self.device)
        logger.info(f"  Model: {self.model_type}, N={self.n_components}, Device: {self.device}")
        self._build_param_names()

    def _build_param_names(self):
        self.param_names = []
        self.param_latex = []
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                self.param_names.extend([f'RM_{i+1}', f'amp_{i+1}', f'chi0_{i+1}'])
                self.param_latex.extend([f'$RM_{i+1}$', f'$p_{i+1}$', f'$\\chi_{{0,{i+1}}}$'])

    def _simulate_spectrum(self, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        P = np.zeros(self.n_freq, dtype=complex)
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                rm, amp, chi0 = theta[i*3], theta[i*3+1], theta[i*3+2]
                P += amp * np.exp(1j * 2 * (chi0 + rm * self.lambda_sq))
        return P.real, P.imag

    def _generate_weights(self, missing_fraction: float) -> np.ndarray:
        weights = np.ones(self.n_freq)
        if missing_fraction > 0:
            n_missing = int(self.n_freq * missing_fraction)
            idx = np.random.choice(self.n_freq, n_missing, replace=False)
            weights[idx] = 0
        return weights

    def run_vroom_inference(self, Q_obs, U_obs, weights, n_samples=5000):
        start = datetime.now()
        x = np.concatenate([Q_obs * weights, U_obs * weights])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        if self.device == 'cuda':
            x_t = x_t.to(self.device)
        with torch.no_grad():
            samples = self.posterior.sample((n_samples,), x=x_t).cpu().numpy().squeeze()
        return {
            'samples': samples, 'mean': np.mean(samples, axis=0), 'std': np.std(samples, axis=0),
            'ci_90': (np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)),
            'time': (datetime.now() - start).total_seconds()
        }

    def run_rmtools_qufit(self, Q_obs, U_obs, weights, noise_level):
        model_map = {('faraday_thin', 1): '1', ('faraday_thin', 2): '11', ('faraday_thin', 3): '111'}
        rmtools_model = model_map.get((self.model_type, self.n_components))
        if not rmtools_model:
            return {'success': False, 'error': 'Unsupported model'}
        start = datetime.now()
        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir = Path(tmpdir)
            data_file = tmpdir / "input.txt"
            freq_hz = 3e8 / np.sqrt(self.lambda_sq)
            with open(data_file, 'w') as f:
                for i in range(self.n_freq):
                    if weights[i] > 0:
                        f.write(f"{freq_hz[i]:.10e} {Q_obs[i]:.10e} {U_obs[i]:.10e} {noise_level:.10e} {noise_level:.10e}\n")
            try:
                result = subprocess.run(
                    ['conda', 'run', '-n', self.rmtools_env, 'qufit', str(data_file), '-m', rmtools_model, '-v'],
                    capture_output=True, text=True, timeout=120, cwd=tmpdir)
                return {'success': True, 'estimate': None, 'uncertainty': None, 
                        'time': (datetime.now() - start).total_seconds()}
            except Exception as e:
                return {'success': False, 'error': str(e)}

    def run_full_validation(self, n_param_points=20, noise_levels=[0.01, 0.03, 0.05, 0.1],
                           n_noise_repeats=20, missing_fraction=0.1, n_samples=5000,
                           run_rmtools=True, seed=42):
        np.random.seed(seed)
        logger.info("=" * 60)
        logger.info("COMPREHENSIVE VALIDATION")
        logger.info("=" * 60)
        
        low, high = np.array(self.prior_bounds['low']), np.array(self.prior_bounds['high'])
        default_params = (low + high) / 2
        
        # Parameter sweeps
        logger.info("\n1. Parameter Recovery Sweeps")
        param_results = {}
        for p in range(self.n_params):
            values = np.linspace(low[p], high[p], n_param_points)
            results = []
            for val in tqdm(values, desc=f"Sweep {self.param_names[p]}"):
                theta = default_params.copy()
                theta[p] = val
                Q_true, U_true = self._simulate_spectrum(theta)
                Q_obs = Q_true + np.random.normal(0, 0.05, self.n_freq)
                U_obs = U_true + np.random.normal(0, 0.05, self.n_freq)
                weights = self._generate_weights(missing_fraction)
                r = SingleTestResult(theta, Q_true, U_true, Q_obs, U_obs, weights, 0.05)
                vroom = self.run_vroom_inference(Q_obs, U_obs, weights, n_samples)
                r.vroom_samples, r.vroom_mean, r.vroom_std = vroom['samples'], vroom['mean'], vroom['std']
                r.vroom_ci_90, r.vroom_time = vroom['ci_90'], vroom['time']
                if run_rmtools:
                    rt = self.run_rmtools_qufit(Q_obs, U_obs, weights, 0.05)
                    r.rmtools_success, r.rmtools_time = rt.get('success', False), rt.get('time')
                results.append(r)
            param_results[p] = results
        
        # Noise sweep
        logger.info("\n2. Noise Level Analysis")
        noise_results = {}
        for noise in tqdm(noise_levels, desc="Noise levels"):
            noise_results[noise] = []
            for _ in range(n_noise_repeats):
                Q_true, U_true = self._simulate_spectrum(default_params)
                Q_obs = Q_true + np.random.normal(0, noise, self.n_freq)
                U_obs = U_true + np.random.normal(0, noise, self.n_freq)
                weights = self._generate_weights(missing_fraction)
                r = SingleTestResult(default_params, Q_true, U_true, Q_obs, U_obs, weights, noise)
                vroom = self.run_vroom_inference(Q_obs, U_obs, weights, n_samples)
                r.vroom_samples, r.vroom_mean, r.vroom_std = vroom['samples'], vroom['mean'], vroom['std']
                r.vroom_ci_90, r.vroom_time = vroom['ci_90'], vroom['time']
                noise_results[noise].append(r)
        
        # Generate plots
        logger.info("\n3. Generating Plots")
        self._plot_parameter_recovery(param_results)
        self._plot_noise_analysis(noise_results)
        self._plot_individual_cases(param_results)
        self._plot_comparison(param_results)
        self._plot_posteriors(param_results)
        self._plot_spectra(param_results)
        self._plot_summary(param_results, noise_results)
        self._save_results(param_results, noise_results)
        logger.info(f"\nResults saved to: {self.output_dir}")

    def _plot_parameter_recovery(self, param_results):
        for p, results in param_results.items():
            fig, ax = plt.subplots(figsize=(8, 6))
            true_vals = [r.theta_true[p] for r in results]
            est_vals = [r.vroom_mean[p] for r in results]
            err_vals = [r.vroom_std[p] for r in results]
            ax.errorbar(true_vals, est_vals, yerr=err_vals, fmt='o', color=COLORS['vroom'],
                       capsize=3, label='VROOM-SBI', alpha=0.8)
            ax.plot([min(true_vals), max(true_vals)], [min(true_vals), max(true_vals)], 
                   'k--', lw=2, label='Perfect')
            ax.set_xlabel(f'True {self.param_latex[p]}')
            ax.set_ylabel(f'Estimated {self.param_latex[p]}')
            ax.set_title(f'{self.param_names[p]} Recovery')
            ax.legend()
            plt.tight_layout()
            plt.savefig(self.dirs['parameter_recovery'] / f'{self.param_names[p]}_recovery.png')
            plt.close()

    def _plot_noise_analysis(self, noise_results):
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        noise_levels = sorted(noise_results.keys())
        
        ax = axes[0]
        for p in range(self.n_params):
            rmse = [np.sqrt(np.mean([(r.vroom_mean[p] - r.theta_true[p])**2 
                   for r in noise_results[n]])) for n in noise_levels]
            ax.plot(noise_levels, rmse, 'o-', label=self.param_names[p], lw=2)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE vs Noise')
        ax.legend()
        ax.set_xscale('log')
        
        ax = axes[1]
        for p in range(self.n_params):
            widths = [np.mean([r.vroom_std[p] for r in noise_results[n]]) for n in noise_levels]
            ax.plot(noise_levels, widths, 'o-', label=self.param_names[p], lw=2)
        ax.set_xlabel('Noise Level')
        ax.set_ylabel('Posterior Width')
        ax.set_title('Posterior Width vs Noise')
        ax.legend()
        ax.set_xscale('log')
        
        plt.tight_layout()
        plt.savefig(self.dirs['noise_analysis'] / 'noise_analysis.png')
        plt.close()

    def _plot_individual_cases(self, param_results):
        """Plot individual test cases: true spectrum, observed, posterior, comparison."""
        for p, results in param_results.items():
            # Select 5 representative cases
            indices = np.linspace(0, len(results)-1, 5, dtype=int)
            
            fig, axes = plt.subplots(len(indices), 4, figsize=(20, 4*len(indices)))
            
            for i, idx in enumerate(indices):
                r = results[idx]
                
                # Col 1: True spectrum
                ax = axes[i, 0]
                ax.plot(self.lambda_sq, r.Q_true, 'b-', lw=2, label='Q')
                ax.plot(self.lambda_sq, r.U_true, 'r-', lw=2, label='U')
                ax.set_xlabel('λ² (m²)')
                ax.set_ylabel('Polarization')
                ax.set_title(f'True: {self.param_names[p]}={r.theta_true[p]:.2f}')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Col 2: Observed (noisy + missing)
                ax = axes[i, 1]
                mask = r.weights > 0
                ax.scatter(self.lambda_sq[mask], r.Q_obs[mask], c='blue', s=10, alpha=0.6, label='Q obs')
                ax.scatter(self.lambda_sq[mask], r.U_obs[mask], c='red', s=10, alpha=0.6, label='U obs')
                ax.plot(self.lambda_sq, r.Q_true, 'b--', alpha=0.3, lw=1)
                ax.plot(self.lambda_sq, r.U_true, 'r--', alpha=0.3, lw=1)
                ax.set_xlabel('λ² (m²)')
                ax.set_title(f'Observed (σ={r.noise_level:.2f}, {100*(1-mask.mean()):.0f}% missing)')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3)
                
                # Col 3: Posterior histogram for this parameter
                ax = axes[i, 2]
                if r.vroom_samples is not None:
                    ax.hist(r.vroom_samples[:, p], bins=50, density=True, 
                           color=COLORS['vroom'], alpha=0.7, edgecolor='black', linewidth=0.5)
                    ax.axvline(r.theta_true[p], color=COLORS['true'], lw=3, label='True')
                    ax.axvline(r.vroom_mean[p], color='black', lw=2, ls='--', label='Mean')
                    # 90% CI
                    ax.axvline(r.vroom_ci_90[0][p], color='gray', lw=1, ls=':')
                    ax.axvline(r.vroom_ci_90[1][p], color='gray', lw=1, ls=':', label='90% CI')
                ax.set_xlabel(self.param_latex[p])
                ax.set_ylabel('Density')
                ax.set_title('VROOM Posterior')
                ax.legend(loc='upper right')
                
                # Col 4: Point estimate comparison
                ax = axes[i, 3]
                ax.axvline(r.theta_true[p], color=COLORS['true'], lw=4, label='True')
                ax.errorbar(r.vroom_mean[p], 0.7, xerr=r.vroom_std[p]*2, 
                           fmt='o', color=COLORS['vroom'], markersize=12, capsize=8, 
                           capthick=2, label='VROOM ±2σ')
                if r.rmtools_success and r.rmtools_estimate is not None:
                    err = r.rmtools_uncertainty[p] if r.rmtools_uncertainty is not None else 0
                    ax.errorbar(r.rmtools_estimate[p], 0.3, xerr=err*2,
                               fmt='s', color=COLORS['rmtools'], markersize=12, capsize=8,
                               capthick=2, label='RM-Tools')
                ax.set_ylim(0, 1)
                ax.set_yticks([])
                ax.set_xlabel(self.param_latex[p])
                ax.set_title('Comparison')
                ax.legend(loc='upper right')
                ax.grid(True, alpha=0.3, axis='x')
            
            plt.tight_layout()
            plt.savefig(self.dirs['individual_cases'] / f'{self.param_names[p]}_cases.png')
            plt.close()

    def _plot_comparison(self, param_results):
        """Side-by-side VROOM vs RM-Tools comparison."""
        all_results = [r for results in param_results.values() for r in results]
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        
        # Row 1: Recovery scatter for each parameter
        for p in range(min(3, self.n_params)):
            ax = axes[0, p]
            true_vals = [r.theta_true[p] for r in all_results]
            vroom_vals = [r.vroom_mean[p] for r in all_results]
            
            ax.scatter(true_vals, vroom_vals, c=COLORS['vroom'], s=30, alpha=0.6, label='VROOM')
            
            # RM-Tools if available
            rt_data = [(r.theta_true[p], r.rmtools_estimate[p]) for r in all_results 
                      if r.rmtools_success and r.rmtools_estimate is not None]
            if rt_data:
                rt_true, rt_est = zip(*rt_data)
                ax.scatter(rt_true, rt_est, c=COLORS['rmtools'], s=30, alpha=0.6, marker='s', label='RM-Tools')
            
            lims = [min(true_vals), max(true_vals)]
            ax.plot(lims, lims, 'k--', lw=2)
            ax.set_xlabel(f'True {self.param_latex[p]}')
            ax.set_ylabel(f'Estimated')
            ax.set_title(self.param_names[p])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # Row 2: Metrics comparison
        # RMSE
        ax = axes[1, 0]
        rmse_v = [np.sqrt(np.mean([(r.vroom_mean[p] - r.theta_true[p])**2 for r in all_results])) 
                 for p in range(self.n_params)]
        rmse_rt = []
        for p in range(self.n_params):
            rt_errors = [(r.rmtools_estimate[p] - r.theta_true[p])**2 for r in all_results 
                        if r.rmtools_success and r.rmtools_estimate is not None]
            rmse_rt.append(np.sqrt(np.mean(rt_errors)) if rt_errors else 0)
        
        x = np.arange(self.n_params)
        ax.bar(x - 0.2, rmse_v, 0.4, label='VROOM', color=COLORS['vroom'])
        ax.bar(x + 0.2, rmse_rt, 0.4, label='RM-Tools', color=COLORS['rmtools'])
        ax.set_xticks(x)
        ax.set_xticklabels(self.param_names)
        ax.set_ylabel('RMSE')
        ax.set_title('RMSE Comparison')
        ax.legend()
        
        # Speed
        ax = axes[1, 1]
        vroom_times = [r.vroom_time for r in all_results if r.vroom_time]
        rt_times = [r.rmtools_time for r in all_results if r.rmtools_time]
        
        mean_v = np.mean(vroom_times) if vroom_times else 0
        mean_rt = np.mean(rt_times) if rt_times else 0
        
        bars = ax.bar(['VROOM-SBI', 'RM-Tools'], [mean_v, mean_rt], 
                      color=[COLORS['vroom'], COLORS['rmtools']])
        ax.set_ylabel('Time (seconds)')
        if mean_v > 0 and mean_rt > 0:
            speedup = mean_rt / mean_v
            ax.set_title(f'Speed: VROOM {speedup:.0f}x faster')
        else:
            ax.set_title('Inference Time')
        
        # Success rate
        ax = axes[1, 2]
        vroom_success = sum(1 for r in all_results if r.vroom_mean is not None) / len(all_results) * 100
        rt_success = sum(1 for r in all_results if r.rmtools_success) / len(all_results) * 100
        
        ax.bar(['VROOM-SBI', 'RM-Tools'], [vroom_success, rt_success],
               color=[COLORS['vroom'], COLORS['rmtools']])
        ax.set_ylabel('Success Rate (%)')
        ax.set_title('Reliability')
        ax.set_ylim(0, 110)
        
        plt.tight_layout()
        plt.savefig(self.dirs['comparison'] / 'vroom_vs_rmtools.png')
        plt.close()

    def _plot_posteriors(self, param_results):
        """Corner-style posterior plots."""
        try:
            import corner
            has_corner = True
        except ImportError:
            has_corner = False
            logger.warning("corner package not installed, using simple histograms")
        
        for p, results in param_results.items():
            # Select low, mid, high value cases
            indices = [0, len(results)//2, len(results)-1]
            
            for idx in indices:
                r = results[idx]
                if r.vroom_samples is None:
                    continue
                
                if has_corner:
                    fig = corner.corner(
                        r.vroom_samples,
                        labels=self.param_latex,
                        truths=r.theta_true,
                        quantiles=[0.16, 0.5, 0.84],
                        show_titles=True,
                        title_fmt='.3f',
                        truth_color=COLORS['true'],
                    )
                else:
                    # Simple histogram grid
                    fig, axes = plt.subplots(1, self.n_params, figsize=(4*self.n_params, 4))
                    if self.n_params == 1:
                        axes = [axes]
                    for i, ax in enumerate(axes):
                        ax.hist(r.vroom_samples[:, i], bins=50, density=True, 
                               color=COLORS['vroom'], alpha=0.7)
                        ax.axvline(r.theta_true[i], color=COLORS['true'], lw=2)
                        ax.set_xlabel(self.param_latex[i])
                
                val = r.theta_true[p]
                fig.suptitle(f'{self.param_names[p]} = {val:.2f}', y=1.02)
                plt.savefig(self.dirs['posteriors'] / f'posterior_{self.param_names[p]}_{val:.2f}.png')
                plt.close()

    def _plot_spectra(self, param_results):
        """Plot Q/U spectra with posterior predictive draws."""
        for p, results in param_results.items():
            # Select 4 representative cases
            indices = np.linspace(0, len(results)-1, 4, dtype=int)
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 10))
            axes = axes.flatten()
            
            for ax, idx in zip(axes, indices):
                r = results[idx]
                
                # Observed data points
                mask = r.weights > 0
                ax.scatter(self.lambda_sq[mask], r.Q_obs[mask], c='blue', s=20, alpha=0.5, 
                          label='Q observed', zorder=3)
                ax.scatter(self.lambda_sq[mask], r.U_obs[mask], c='red', s=20, alpha=0.5,
                          label='U observed', zorder=3)
                
                # True spectrum
                ax.plot(self.lambda_sq, r.Q_true, 'b-', lw=2, alpha=0.8, label='Q true')
                ax.plot(self.lambda_sq, r.U_true, 'r-', lw=2, alpha=0.8, label='U true')
                
                # Posterior predictive draws
                if r.vroom_samples is not None:
                    n_draws = min(50, len(r.vroom_samples))
                    for i in range(n_draws):
                        Q_pred, U_pred = self._simulate_spectrum(r.vroom_samples[i])
                        ax.plot(self.lambda_sq, Q_pred, 'b-', alpha=0.03, lw=0.5)
                        ax.plot(self.lambda_sq, U_pred, 'r-', alpha=0.03, lw=0.5)
                
                ax.set_xlabel('λ² (m²)')
                ax.set_ylabel('Fractional Polarization')
                ax.set_title(f'{self.param_names[p]} = {r.theta_true[p]:.2f}')
                ax.legend(loc='upper right', fontsize=9)
                ax.grid(True, alpha=0.3)
            
            plt.suptitle(f'Spectra Fits: {self.param_names[p]} Sweep', fontsize=14, fontweight='bold')
            plt.tight_layout()
            plt.savefig(self.dirs['spectra'] / f'spectra_{self.param_names[p]}.png')
            plt.close()

    def _plot_summary(self, param_results, noise_results):
        all_results = [r for results in param_results.values() for r in results]
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # RMSE per parameter
        ax = axes[0, 0]
        rmse = [np.sqrt(np.mean([(r.vroom_mean[p] - r.theta_true[p])**2 for r in all_results])) 
               for p in range(self.n_params)]
        ax.bar(self.param_names, rmse, color=COLORS['vroom'])
        ax.set_ylabel('RMSE')
        ax.set_title('Parameter RMSE')
        
        # Bias
        ax = axes[0, 1]
        bias = [np.mean([r.vroom_mean[p] - r.theta_true[p] for r in all_results]) 
               for p in range(self.n_params)]
        ax.bar(self.param_names, bias, color=COLORS['vroom'])
        ax.axhline(0, color='k', linestyle='--')
        ax.set_ylabel('Bias')
        ax.set_title('Parameter Bias')
        
        # Coverage
        ax = axes[1, 0]
        coverage = []
        for p in range(self.n_params):
            in_ci = sum(1 for r in all_results if r.vroom_ci_90[0][p] <= r.theta_true[p] <= r.vroom_ci_90[1][p])
            coverage.append(in_ci / len(all_results))
        ax.bar(self.param_names, coverage, color=COLORS['vroom'])
        ax.axhline(0.9, color='r', linestyle='--', label='Expected 90%')
        ax.set_ylabel('90% CI Coverage')
        ax.set_title('Calibration')
        ax.legend()
        ax.set_ylim(0, 1.1)
        
        # Speed
        ax = axes[1, 1]
        vroom_time = np.mean([r.vroom_time for r in all_results if r.vroom_time])
        rt_time = np.mean([r.rmtools_time for r in all_results if r.rmtools_time]) if any(r.rmtools_time for r in all_results) else 0
        ax.bar(['VROOM-SBI', 'RM-Tools'], [vroom_time, rt_time], color=[COLORS['vroom'], COLORS['rmtools']])
        ax.set_ylabel('Time (s)')
        ax.set_title('Inference Speed')
        
        plt.tight_layout()
        plt.savefig(self.dirs['summary'] / 'summary.png')
        plt.close()

    def _save_results(self, param_results, noise_results):
        all_results = [r for results in param_results.values() for r in results]
        summary = {
            'model_type': self.model_type, 'n_components': self.n_components,
            'n_params': self.n_params, 'param_names': self.param_names,
            'n_test_cases': len(all_results), 'device': self.device,
            'timestamp': datetime.now().isoformat(),
        }
        with open(self.dirs['data'] / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)


def run_comprehensive_validation(posterior_path, output_dir, rmtools_env="rmtools",
                                  n_param_points=20, noise_levels=[0.01, 0.03, 0.05, 0.1],
                                  n_noise_repeats=20, missing_fraction=0.1, n_samples=5000,
                                  run_rmtools=True, device="auto", seed=42):
    validator = ComprehensiveValidator(Path(posterior_path), Path(output_dir), rmtools_env, device)
    validator.run_full_validation(n_param_points, noise_levels, n_noise_repeats, 
                                  missing_fraction, n_samples, run_rmtools, seed)
    return validator
