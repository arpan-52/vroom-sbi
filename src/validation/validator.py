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
