"""
Comprehensive Validation for VROOM-SBI with RM-Tools comparison.

Features:
- Parameter sweeps across prior range
- Noise and missing data grid analysis
- Individual case deep dives with side-by-side comparison
- RM-Tools integration via micromamba
- Timing comparison
- Publication-quality plots
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from datetime import datetime
import json
import subprocess
import tempfile
import logging
import re
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore', category=UserWarning)
logger = logging.getLogger(__name__)

plt.rcParams.update({
    'font.size': 11, 'font.family': 'serif', 'axes.labelsize': 12,
    'axes.titlesize': 13, 'figure.dpi': 150, 'savefig.dpi': 300,
    'savefig.bbox': 'tight',
})

COLORS = {'vroom': '#2E86AB', 'rmtools': '#E94F37', 'true': '#1B998B', 'observed': '#FF9F1C'}


@dataclass
class RMToolsResult:
    success: bool = False
    values: Optional[np.ndarray] = None
    err_plus: Optional[np.ndarray] = None
    err_minus: Optional[np.ndarray] = None
    param_names: Optional[List[str]] = None
    chi_sq_red: Optional[float] = None
    time: Optional[float] = None
    error_msg: Optional[str] = None


@dataclass
class TestCase:
    theta_true: np.ndarray
    Q_true: np.ndarray
    U_true: np.ndarray
    Q_obs: np.ndarray
    U_obs: np.ndarray
    weights: np.ndarray
    noise_level: float
    missing_fraction: float
    vroom_samples: Optional[np.ndarray] = None
    vroom_mean: Optional[np.ndarray] = None
    vroom_std: Optional[np.ndarray] = None
    vroom_ci_90: Optional[Tuple[np.ndarray, np.ndarray]] = None
    vroom_time: Optional[float] = None
    rmtools: Optional[RMToolsResult] = None


class ComprehensiveValidator:
    def __init__(self, posterior_path: Path, output_dir: Path, rmtools_model: str = "1", device: str = "auto"):
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.rmtools_model = rmtools_model
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else (device if device != "auto" else "cpu")
        self._create_output_dirs()
        self._load_posterior()

    def _create_output_dirs(self):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dirs = {
            'parameter_sweeps': self.output_dir / 'parameter_sweeps',
            'noise_missing_analysis': self.output_dir / 'noise_missing_analysis',
            'individual_cases': self.output_dir / 'individual_cases',
            'posteriors': self.output_dir / 'posteriors',
            'timing': self.output_dir / 'timing',
            'data': self.output_dir / 'data',
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
                self.param_latex.extend([f'$RM_{{{i+1}}}$', f'$p_{{{i+1}}}$', f'$\\chi_{{0,{i+1}}}$'])

    def _simulate_spectrum(self, theta):
        P = np.zeros(self.n_freq, dtype=complex)
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                rm, amp, chi0 = theta[i*3], theta[i*3+1], theta[i*3+2]
                P += amp * np.exp(1j * 2 * (chi0 + rm * self.lambda_sq))
        return P.real, P.imag

    def _generate_weights(self, missing_fraction):
        weights = np.ones(self.n_freq)
        if missing_fraction > 0:
            n_missing = int(self.n_freq * missing_fraction)
            n_scattered = n_missing // 2
            n_gap = n_missing - n_scattered
            if n_scattered > 0:
                idx = np.random.choice(self.n_freq, n_scattered, replace=False)
                weights[idx] = 0
            if n_gap > 0:
                available = np.where(weights > 0)[0]
                if len(available) > n_gap:
                    start = np.random.randint(0, len(available) - n_gap)
                    weights[available[start:start + n_gap]] = 0
        return weights

    def run_vroom_inference(self, Q_obs, U_obs, weights, n_samples=5000):
        start = datetime.now()
        x = np.concatenate([Q_obs * weights, U_obs * weights])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        if self.device == 'cuda':
            x_t = x_t.to(self.device)
        with torch.no_grad():
            samples = self.posterior.sample((n_samples,), x=x_t).cpu().numpy().squeeze()
        elapsed = (datetime.now() - start).total_seconds()
        return {
            'samples': samples, 'mean': np.mean(samples, axis=0), 'std': np.std(samples, axis=0),
            'ci_90': (np.percentile(samples, 5, axis=0), np.percentile(samples, 95, axis=0)),
            'time': elapsed
        }

    def run_rmtools(self, Q_obs, U_obs, weights, noise_level, case_id=None):
        """Run RM-Tools QUfit. Saves input file to output directory."""
        import shutil
        
        start = datetime.now()
        
        # Create rmtools working directory in output
        rmtools_dir = self.output_dir / "rmtools_work"
        rmtools_dir.mkdir(parents=True, exist_ok=True)
        
        # Use case_id for unique filename, or generate one
        if case_id is None:
            case_id = f"case_{datetime.now().strftime('%H%M%S%f')}"
        
        input_file = rmtools_dir / f"{case_id}.dat"
        freq_hz = 3e8 / np.sqrt(self.lambda_sq)
        
        # Write input file
        with open(input_file, 'w') as f:
            for i in range(self.n_freq):
                if weights[i] > 0:
                    f.write(f"{freq_hz[i]:.10e} {Q_obs[i]:.10e} {U_obs[i]:.10e} {noise_level:.10e} {noise_level:.10e}\n")
        
        cmd = ['micromamba', 'run', '-n', 'rmtool', 'qufit', str(input_file), '-m', self.rmtools_model, '-v']
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300, cwd=rmtools_dir)
            elapsed = (datetime.now() - start).total_seconds()
            
            # RM-Tools creates: {case_id}_m{model}_dynesty.dat
            expected_dat = rmtools_dir / f"{case_id}_m{self.rmtools_model}_dynesty.dat"
            
            if expected_dat.exists():
                return self._parse_rmtools_dat(expected_dat, elapsed)
            
            # Try parsing from stdout
            parsed = self._parse_rmtools_stdout(result.stdout, elapsed)
            if not parsed.success:
                parsed.error_msg = f"No output file. Exit: {result.returncode}"
            return parsed
            
        except subprocess.TimeoutExpired:
            return RMToolsResult(success=False, error_msg="Timeout (300s)", time=300)
        except FileNotFoundError:
            return RMToolsResult(success=False, error_msg="micromamba or qufit not found")
        except Exception as e:
            return RMToolsResult(success=False, error_msg=str(e))

    def _parse_rmtools_dat(self, dat_file, elapsed):
        try:
            content = dat_file.read_text()
            result = RMToolsResult(success=True, time=elapsed)
            def extract_list(name):
                match = re.search(rf"{name}=\[(.*?)\]", content)
                if match:
                    items = match.group(1).split(',')
                    try:
                        return [float(x.strip().strip("'\"")) for x in items if x.strip()]
                    except:
                        return [x.strip().strip("'\"") for x in items if x.strip()]
                return None
            result.param_names = extract_list('parNames')
            result.values = np.array(extract_list('values')) if extract_list('values') else None
            result.err_plus = np.array(extract_list('errPlus')) if extract_list('errPlus') else None
            result.err_minus = np.array(extract_list('errMinus')) if extract_list('errMinus') else None
            match = re.search(r"chiSqRed=([\d.eE+-]+)", content)
            if match:
                result.chi_sq_red = float(match.group(1))
            return result
        except Exception as e:
            return RMToolsResult(success=False, error_msg=str(e), time=elapsed)

    def _parse_rmtools_stdout(self, stdout, elapsed):
        result = RMToolsResult(success=False, time=elapsed)
        try:
            if "RESULTS:" in stdout:
                result.success = True
                values, err_plus, err_minus, param_names = [], [], [], []
                pattern = r"(\w+)\s*=\s*([\d.eE+-]+)\s*\(\+([\d.eE+-]+),\s*-([\d.eE+-]+)\)"
                for match in re.finditer(pattern, stdout):
                    param_names.append(match.group(1))
                    values.append(float(match.group(2)))
                    err_plus.append(float(match.group(3)))
                    err_minus.append(float(match.group(4)))
                if values:
                    result.param_names = param_names
                    result.values = np.array(values)
                    result.err_plus = np.array(err_plus)
                    result.err_minus = np.array(err_minus)
        except Exception as e:
            result.error_msg = str(e)
        return result

    def _convert_rmtools_to_vroom(self, rmtools):
        if not rmtools.success or rmtools.values is None:
            return None
        try:
            if self.model_type == "faraday_thin" and self.n_components == 1:
                frac_pol, psi0_deg, rm = rmtools.values[0], rmtools.values[1], rmtools.values[2]
                chi0 = np.deg2rad(psi0_deg)
                return np.array([rm, frac_pol, chi0])
        except:
            pass
        return None

    def _convert_rmtools_errors(self, rmtools):
        if not rmtools.success or rmtools.err_plus is None:
            return None
        try:
            if self.model_type == "faraday_thin" and self.n_components == 1:
                err_frac = (rmtools.err_plus[0] + rmtools.err_minus[0]) / 2
                err_psi = (rmtools.err_plus[1] + rmtools.err_minus[1]) / 2
                err_rm = (rmtools.err_plus[2] + rmtools.err_minus[2]) / 2
                return np.array([err_rm, err_frac, np.deg2rad(err_psi)])
        except:
            pass
        return None

    def run_parameter_sweeps(self, n_points=20, noise_level=0.05, missing_fraction=0.1, n_samples=5000, run_rmtools=True):
        logger.info("Running parameter sweeps...")
        low, high = np.array(self.prior_bounds['low']), np.array(self.prior_bounds['high'])
        default_params = (low + high) / 2
        self.param_sweep_results = {}
        for p in range(self.n_params):
            logger.info(f"  Sweeping {self.param_names[p]}")
            values = np.linspace(low[p], high[p], n_points)
            results = []
            for val in tqdm(values, desc=self.param_names[p]):
                theta = default_params.copy()
                theta[p] = val
                Q_true, U_true = self._simulate_spectrum(theta)
                Q_obs = Q_true + np.random.normal(0, noise_level, self.n_freq)
                U_obs = U_true + np.random.normal(0, noise_level, self.n_freq)
                weights = self._generate_weights(missing_fraction)
                tc = TestCase(theta, Q_true, U_true, Q_obs, U_obs, weights, noise_level, missing_fraction)
                vroom = self.run_vroom_inference(Q_obs, U_obs, weights, n_samples)
                tc.vroom_samples, tc.vroom_mean, tc.vroom_std = vroom['samples'], vroom['mean'], vroom['std']
                tc.vroom_ci_90, tc.vroom_time = vroom['ci_90'], vroom['time']
                if run_rmtools:
                    tc.rmtools = self.run_rmtools(Q_obs, U_obs, weights, noise_level, case_id=f"sweep_{self.param_names[p]}_{len(results):03d}")
                results.append(tc)
            self.param_sweep_results[p] = results

    def run_noise_missing_grid(self, noise_min=0.001, noise_max=0.5, noise_steps=10, missing_min=0.0, missing_max=0.5, missing_steps=10, n_repeats=5, n_samples=5000, run_rmtools=True):
        logger.info("Running noise/missing grid...")
        noise_levels = np.linspace(noise_min, noise_max, noise_steps)
        missing_fractions = np.linspace(missing_min, missing_max, missing_steps)
        low, high = np.array(self.prior_bounds['low']), np.array(self.prior_bounds['high'])
        default_params = (low + high) / 2
        self.grid_results = {}
        self.noise_levels = noise_levels
        self.missing_fractions = missing_fractions
        total = len(noise_levels) * len(missing_fractions) * n_repeats
        pbar = tqdm(total=total, desc="Grid")
        for noise in noise_levels:
            for missing in missing_fractions:
                key = (noise, missing)
                self.grid_results[key] = []
                for _ in range(n_repeats):
                    Q_true, U_true = self._simulate_spectrum(default_params)
                    Q_obs = Q_true + np.random.normal(0, noise, self.n_freq)
                    U_obs = U_true + np.random.normal(0, noise, self.n_freq)
                    weights = self._generate_weights(missing)
                    tc = TestCase(default_params, Q_true, U_true, Q_obs, U_obs, weights, noise, missing)
                    vroom = self.run_vroom_inference(Q_obs, U_obs, weights, n_samples)
                    tc.vroom_samples, tc.vroom_mean, tc.vroom_std = vroom['samples'], vroom['mean'], vroom['std']
                    tc.vroom_ci_90, tc.vroom_time = vroom['ci_90'], vroom['time']
                    if run_rmtools:
                        grid_case_id = f"grid_n{noise:.3f}_m{missing:.2f}_{len(self.grid_results[key]):02d}"
                        tc.rmtools = self.run_rmtools(Q_obs, U_obs, weights, noise, case_id=grid_case_id)
                    self.grid_results[key].append(tc)
                    pbar.update(1)
        pbar.close()

    def run_individual_cases(self, n_cases=10, n_samples=5000, run_rmtools=True):
        logger.info(f"Running {n_cases} individual cases...")
        low, high = np.array(self.prior_bounds['low']), np.array(self.prior_bounds['high'])
        self.individual_cases = []
        noise_samples = np.linspace(0.01, 0.2, n_cases)
        missing_samples = np.linspace(0.0, 0.3, n_cases)
        np.random.shuffle(noise_samples)
        np.random.shuffle(missing_samples)
        for i in tqdm(range(n_cases), desc="Cases"):
            theta = np.random.uniform(low, high)
            noise, missing = noise_samples[i], missing_samples[i]
            Q_true, U_true = self._simulate_spectrum(theta)
            Q_obs = Q_true + np.random.normal(0, noise, self.n_freq)
            U_obs = U_true + np.random.normal(0, noise, self.n_freq)
            weights = self._generate_weights(missing)
            tc = TestCase(theta, Q_true, U_true, Q_obs, U_obs, weights, noise, missing)
            vroom = self.run_vroom_inference(Q_obs, U_obs, weights, n_samples)
            tc.vroom_samples, tc.vroom_mean, tc.vroom_std = vroom['samples'], vroom['mean'], vroom['std']
            tc.vroom_ci_90, tc.vroom_time = vroom['ci_90'], vroom['time']
            if run_rmtools:
                tc.rmtools = self.run_rmtools(Q_obs, U_obs, weights, noise, case_id=f"individual_{i+1:03d}")
            self.individual_cases.append(tc)

    def plot_parameter_sweeps(self):
        logger.info("Plotting parameter sweeps...")
        for p, results in self.param_sweep_results.items():
            fig, axes = plt.subplots(1, 2, figsize=(14, 5))
            true_vals = [r.theta_true[p] for r in results]
            vroom_means = [r.vroom_mean[p] for r in results]
            vroom_stds = [r.vroom_std[p] for r in results]
            ax = axes[0]
            ax.errorbar(true_vals, vroom_means, yerr=vroom_stds, fmt='o', color=COLORS['vroom'], capsize=3, label='VROOM-SBI', alpha=0.8)
            rt_data = [(r.theta_true[p], self._convert_rmtools_to_vroom(r.rmtools)[p], self._convert_rmtools_errors(r.rmtools)[p] if self._convert_rmtools_errors(r.rmtools) is not None else 0) for r in results if r.rmtools and r.rmtools.success and self._convert_rmtools_to_vroom(r.rmtools) is not None]
            if rt_data:
                rt_true, rt_est, rt_err = zip(*rt_data)
                ax.errorbar(rt_true, rt_est, yerr=rt_err, fmt='s', color=COLORS['rmtools'], capsize=3, label='RM-Tools', alpha=0.8)
            lims = [min(true_vals), max(true_vals)]
            ax.plot(lims, lims, 'k--', lw=2, label='Perfect')
            ax.set_xlabel(f'True {self.param_latex[p]}')
            ax.set_ylabel(f'Estimated {self.param_latex[p]}')
            ax.set_title(f'{self.param_names[p]} Recovery')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax = axes[1]
            residuals_vroom = np.array(vroom_means) - np.array(true_vals)
            ax.scatter(true_vals, residuals_vroom, c=COLORS['vroom'], s=50, alpha=0.7, label='VROOM-SBI')
            if rt_data:
                residuals_rt = np.array(rt_est) - np.array(rt_true)
                ax.scatter(rt_true, residuals_rt, c=COLORS['rmtools'], s=50, alpha=0.7, marker='s', label='RM-Tools')
            ax.axhline(0, color='black', linestyle='--', lw=1.5)
            ax.set_xlabel(f'True {self.param_latex[p]}')
            ax.set_ylabel('Residual')
            ax.set_title(f'{self.param_names[p]} Residuals')
            ax.legend()
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(self.dirs['parameter_sweeps'] / f'{self.param_names[p]}_recovery.png')
            plt.close()

    def plot_noise_missing_heatmaps(self):
        logger.info("Plotting heatmaps...")
        n_noise, n_missing = len(self.noise_levels), len(self.missing_fractions)
        rmse_vroom = np.zeros((n_noise, n_missing, self.n_params))
        rmse_rmtools = np.zeros((n_noise, n_missing, self.n_params))
        rmse_rmtools_valid = np.zeros((n_noise, n_missing), dtype=bool)
        for i, noise in enumerate(self.noise_levels):
            for j, missing in enumerate(self.missing_fractions):
                results = self.grid_results[(noise, missing)]
                for p in range(self.n_params):
                    errors = [(r.vroom_mean[p] - r.theta_true[p])**2 for r in results]
                    rmse_vroom[i, j, p] = np.sqrt(np.mean(errors))
                rt_errors = [[] for _ in range(self.n_params)]
                for r in results:
                    if r.rmtools and r.rmtools.success:
                        rt_vals = self._convert_rmtools_to_vroom(r.rmtools)
                        if rt_vals is not None:
                            for p in range(self.n_params):
                                rt_errors[p].append((rt_vals[p] - r.theta_true[p])**2)
                for p in range(self.n_params):
                    if rt_errors[p]:
                        rmse_rmtools[i, j, p] = np.sqrt(np.mean(rt_errors[p]))
                        rmse_rmtools_valid[i, j] = True
        for p in range(self.n_params):
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            ax = axes[0]
            im = ax.imshow(rmse_vroom[:, :, p], aspect='auto', origin='lower', extent=[self.missing_fractions[0], self.missing_fractions[-1], self.noise_levels[0], self.noise_levels[-1]])
            ax.set_xlabel('Missing Fraction')
            ax.set_ylabel('Noise Level')
            ax.set_title(f'VROOM RMSE: {self.param_names[p]}')
            plt.colorbar(im, ax=ax, label='RMSE')
            ax = axes[1]
            rmse_rt_masked = np.ma.masked_where(~rmse_rmtools_valid, rmse_rmtools[:, :, p])
            im = ax.imshow(rmse_rt_masked, aspect='auto', origin='lower', extent=[self.missing_fractions[0], self.missing_fractions[-1], self.noise_levels[0], self.noise_levels[-1]])
            ax.set_xlabel('Missing Fraction')
            ax.set_ylabel('Noise Level')
            ax.set_title(f'RM-Tools RMSE: {self.param_names[p]}')
            plt.colorbar(im, ax=ax, label='RMSE')
            ax = axes[2]
            ratio = np.ma.masked_where(~rmse_rmtools_valid, rmse_vroom[:, :, p] / (rmse_rmtools[:, :, p] + 1e-10))
            im = ax.imshow(ratio, aspect='auto', origin='lower', cmap='RdBu_r', vmin=0, vmax=2, extent=[self.missing_fractions[0], self.missing_fractions[-1], self.noise_levels[0], self.noise_levels[-1]])
            ax.set_xlabel('Missing Fraction')
            ax.set_ylabel('Noise Level')
            ax.set_title(f'Ratio (VROOM/RM-Tools, <1=VROOM better)')
            plt.colorbar(im, ax=ax, label='Ratio')
            plt.tight_layout()
            plt.savefig(self.dirs['noise_missing_analysis'] / f'{self.param_names[p]}_heatmap.png')
            plt.close()

    def plot_individual_cases(self):
        logger.info("Plotting individual cases...")
        for i, tc in enumerate(self.individual_cases):
            fig = plt.figure(figsize=(20, 14))
            gs = gridspec.GridSpec(3, 4, figure=fig, hspace=0.35, wspace=0.3)
            # Row 1: Spectra
            ax = fig.add_subplot(gs[0, 0])
            ax.plot(self.lambda_sq, tc.Q_true, 'b-', lw=2, label='Q')
            ax.plot(self.lambda_sq, tc.U_true, 'r-', lw=2, label='U')
            ax.set_xlabel('λ² (m²)')
            ax.set_ylabel('Polarization')
            ax.set_title('True Spectrum')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax = fig.add_subplot(gs[0, 1])
            mask = tc.weights > 0
            ax.scatter(self.lambda_sq[mask], tc.Q_obs[mask], c='blue', s=10, alpha=0.6, label='Q')
            ax.scatter(self.lambda_sq[mask], tc.U_obs[mask], c='red', s=10, alpha=0.6, label='U')
            ax.plot(self.lambda_sq, tc.Q_true, 'b--', alpha=0.3)
            ax.plot(self.lambda_sq, tc.U_true, 'r--', alpha=0.3)
            ax.set_xlabel('λ² (m²)')
            ax.set_title(f'Observed (σ={tc.noise_level:.3f}, {tc.missing_fraction*100:.0f}% missing)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax = fig.add_subplot(gs[0, 2])
            ax.scatter(self.lambda_sq[mask], tc.Q_obs[mask], c='blue', s=10, alpha=0.3)
            ax.scatter(self.lambda_sq[mask], tc.U_obs[mask], c='red', s=10, alpha=0.3)
            for j in range(min(30, len(tc.vroom_samples))):
                Q_p, U_p = self._simulate_spectrum(tc.vroom_samples[j])
                ax.plot(self.lambda_sq, Q_p, 'b-', alpha=0.05)
                ax.plot(self.lambda_sq, U_p, 'r-', alpha=0.05)
            Q_m, U_m = self._simulate_spectrum(tc.vroom_mean)
            ax.plot(self.lambda_sq, Q_m, 'b-', lw=2, label='Q fit')
            ax.plot(self.lambda_sq, U_m, 'r-', lw=2, label='U fit')
            ax.set_xlabel('λ² (m²)')
            ax.set_title(f'VROOM Fit ({tc.vroom_time*1000:.1f} ms)')
            ax.legend()
            ax.grid(True, alpha=0.3)
            ax = fig.add_subplot(gs[0, 3])
            ax.scatter(self.lambda_sq[mask], tc.Q_obs[mask], c='blue', s=10, alpha=0.3)
            ax.scatter(self.lambda_sq[mask], tc.U_obs[mask], c='red', s=10, alpha=0.3)
            if tc.rmtools and tc.rmtools.success:
                rt_p = self._convert_rmtools_to_vroom(tc.rmtools)
                if rt_p is not None:
                    Q_rt, U_rt = self._simulate_spectrum(rt_p)
                    ax.plot(self.lambda_sq, Q_rt, 'b-', lw=2)
                    ax.plot(self.lambda_sq, U_rt, 'r-', lw=2)
                    ax.set_title(f'RM-Tools ({tc.rmtools.time:.1f} s)')
                else:
                    ax.set_title('RM-Tools: Parse error')
            else:
                ax.set_title('RM-Tools: Failed')
            ax.set_xlabel('λ² (m²)')
            ax.grid(True, alpha=0.3)
            # Row 2: Parameter comparison + posteriors
            ax = fig.add_subplot(gs[1, :2])
            y_pos = np.arange(self.n_params)
            ax.barh(y_pos - 0.25, tc.theta_true, height=0.2, color=COLORS['true'], label='True', alpha=0.8)
            ax.barh(y_pos, tc.vroom_mean, height=0.2, color=COLORS['vroom'], label='VROOM', alpha=0.8, xerr=tc.vroom_std, capsize=3)
            if tc.rmtools and tc.rmtools.success:
                rt_p = self._convert_rmtools_to_vroom(tc.rmtools)
                rt_e = self._convert_rmtools_errors(tc.rmtools)
                if rt_p is not None:
                    ax.barh(y_pos + 0.25, rt_p, height=0.2, color=COLORS['rmtools'], label='RM-Tools', alpha=0.8, xerr=rt_e if rt_e is not None else None, capsize=3)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(self.param_names)
            ax.set_xlabel('Value')
            ax.set_title('Parameter Comparison')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3, axis='x')
            for p in range(min(self.n_params, 2)):
                ax = fig.add_subplot(gs[1, 2 + p])
                ax.hist(tc.vroom_samples[:, p], bins=50, density=True, color=COLORS['vroom'], alpha=0.7, label='VROOM')
                ax.axvline(tc.theta_true[p], color=COLORS['true'], lw=3, label='True')
                ax.axvline(tc.vroom_mean[p], color=COLORS['vroom'], lw=2, ls='--')
                if tc.rmtools and tc.rmtools.success:
                    rt_p = self._convert_rmtools_to_vroom(tc.rmtools)
                    rt_e = self._convert_rmtools_errors(tc.rmtools)
                    if rt_p is not None:
                        ax.axvline(rt_p[p], color=COLORS['rmtools'], lw=2, ls='--', label='RM-Tools')
                        if rt_e is not None:
                            ax.axvspan(rt_p[p] - rt_e[p], rt_p[p] + rt_e[p], color=COLORS['rmtools'], alpha=0.2)
                ax.set_xlabel(self.param_latex[p])
                ax.set_ylabel('Density')
                ax.set_title(f'{self.param_names[p]} Posterior')
                ax.legend(fontsize=8)
            # Row 3: Summary
            ax = fig.add_subplot(gs[2, :])
            ax.axis('off')
            summary = f"Case {i+1}: Ground Truth vs Estimates\n\n{'Parameter':<15} {'True':>12} {'VROOM':>20} {'RM-Tools':>20}\n" + "─"*70
            for p in range(self.n_params):
                vroom_str = f"{tc.vroom_mean[p]:.4f} ± {tc.vroom_std[p]:.4f}"
                rt_str = "N/A"
                if tc.rmtools and tc.rmtools.success:
                    rt_p = self._convert_rmtools_to_vroom(tc.rmtools)
                    rt_e = self._convert_rmtools_errors(tc.rmtools)
                    if rt_p is not None:
                        err = rt_e[p] if rt_e is not None else 0
                        rt_str = f"{rt_p[p]:.4f} ± {err:.4f}"
                summary += f"\n{self.param_names[p]:<15} {tc.theta_true[p]:>12.4f} {vroom_str:>20} {rt_str:>20}"
            summary += f"\n" + "─"*70 + f"\nTiming: VROOM = {tc.vroom_time*1000:.1f} ms"
            if tc.rmtools and tc.rmtools.time:
                summary += f" | RM-Tools = {tc.rmtools.time:.1f} s | Speedup: {tc.rmtools.time/tc.vroom_time:.0f}x"
            ax.text(0.05, 0.95, summary, transform=ax.transAxes, fontsize=11, verticalalignment='top', fontfamily='monospace', bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))
            plt.suptitle(f'Case {i+1}: σ={tc.noise_level:.3f}, missing={tc.missing_fraction*100:.0f}%', fontsize=14, fontweight='bold')
            plt.savefig(self.dirs['individual_cases'] / f'case_{i+1:03d}.png')
            plt.close()

    def plot_posteriors(self):
        logger.info("Plotting posteriors...")
        try:
            import corner
            has_corner = True
        except ImportError:
            has_corner = False
        for i, tc in enumerate(self.individual_cases):
            if has_corner and tc.vroom_samples is not None:
                fig = corner.corner(tc.vroom_samples, labels=self.param_latex, truths=tc.theta_true, quantiles=[0.16, 0.5, 0.84], show_titles=True, truth_color=COLORS['true'])
                fig.suptitle(f'Case {i+1} VROOM Posterior', y=1.02)
                plt.savefig(self.dirs['posteriors'] / f'case_{i+1:03d}_vroom.png')
                plt.close()

    def plot_timing(self):
        logger.info("Plotting timing...")
        vroom_times, rmtools_times = [], []
        for results in self.param_sweep_results.values():
            for r in results:
                if r.vroom_time:
                    vroom_times.append(r.vroom_time)
                if r.rmtools and r.rmtools.time:
                    rmtools_times.append(r.rmtools.time)
        for results in self.grid_results.values():
            for r in results:
                if r.vroom_time:
                    vroom_times.append(r.vroom_time)
                if r.rmtools and r.rmtools.time:
                    rmtools_times.append(r.rmtools.time)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        ax = axes[0]
        ax.hist(np.array(vroom_times) * 1000, bins=30, color=COLORS['vroom'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('Count')
        ax.set_title('VROOM-SBI Timing')
        ax = axes[1]
        ax.hist(rmtools_times, bins=30, color=COLORS['rmtools'], alpha=0.7, edgecolor='black')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Count')
        ax.set_title('RM-Tools Timing')
        ax = axes[2]
        mean_v = np.mean(vroom_times) * 1000
        mean_rt = np.mean(rmtools_times) * 1000 if rmtools_times else 0
        bars = ax.bar(['VROOM-SBI', 'RM-Tools'], [mean_v, mean_rt], color=[COLORS['vroom'], COLORS['rmtools']])
        ax.set_ylabel('Mean Time (ms)')
        if mean_v > 0 and mean_rt > 0:
            ax.set_title(f'VROOM is {mean_rt/mean_v:.0f}x faster')
        for bar, val in zip(bars, [mean_v, mean_rt]):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, f'{val:.1f}', ha='center', fontsize=11)
        plt.tight_layout()
        plt.savefig(self.dirs['timing'] / 'timing_comparison.png')
        plt.close()

    def save_results(self):
        logger.info("Saving results...")
        vroom_times, rmtools_times = [], []
        for results in self.param_sweep_results.values():
            for r in results:
                if r.vroom_time:
                    vroom_times.append(r.vroom_time)
                if r.rmtools and r.rmtools.time:
                    rmtools_times.append(r.rmtools.time)
        summary = {
            'model_type': self.model_type, 'n_components': self.n_components,
            'rmtools_model': self.rmtools_model, 'device': self.device,
            'timestamp': datetime.now().isoformat(),
            'timing': {
                'vroom_mean_ms': np.mean(vroom_times) * 1000 if vroom_times else None,
                'rmtools_mean_s': np.mean(rmtools_times) if rmtools_times else None,
                'speedup': (np.mean(rmtools_times) / np.mean(vroom_times)) if (vroom_times and rmtools_times) else None,
            }
        }
        with open(self.dirs['data'] / 'summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

    def run_full_validation(self, n_param_points=20, noise_min=0.001, noise_max=0.5, noise_steps=10, missing_min=0.0, missing_max=0.5, missing_steps=10, n_grid_repeats=5, n_individual_cases=10, n_samples=5000, run_rmtools=True, seed=42):
        np.random.seed(seed)
        logger.info("=" * 60)
        logger.info("VROOM-SBI COMPREHENSIVE VALIDATION")
        logger.info("=" * 60)
        logger.info(f"Model: {self.model_type}, N={self.n_components}")
        logger.info(f"RM-Tools model: {self.rmtools_model}")
        logger.info(f"Device: {self.device}")
        logger.info("=" * 60)
        logger.info("\n[1/5] Parameter sweeps...")
        self.run_parameter_sweeps(n_param_points, 0.05, 0.1, n_samples, run_rmtools)
        logger.info("\n[2/5] Noise/missing grid...")
        self.run_noise_missing_grid(noise_min, noise_max, noise_steps, missing_min, missing_max, missing_steps, n_grid_repeats, n_samples, run_rmtools)
        logger.info("\n[3/5] Individual cases...")
        self.run_individual_cases(n_individual_cases, n_samples, run_rmtools)
        logger.info("\n[4/5] Generating plots...")
        self.plot_parameter_sweeps()
        self.plot_noise_missing_heatmaps()
        self.plot_individual_cases()
        self.plot_posteriors()
        self.plot_timing()
        logger.info("\n[5/5] Saving results...")
        self.save_results()
        logger.info(f"\nResults saved to: {self.output_dir}")


def run_comprehensive_validation(posterior_path, output_dir, rmtools_model="1", n_param_points=20, noise_min=0.001, noise_max=0.5, noise_steps=10, missing_min=0.0, missing_max=0.5, missing_steps=10, n_grid_repeats=5, n_individual_cases=10, n_samples=5000, run_rmtools=True, device="auto", seed=42):
    validator = ComprehensiveValidator(Path(posterior_path), Path(output_dir), rmtools_model, device)
    validator.run_full_validation(n_param_points, noise_min, noise_max, noise_steps, missing_min, missing_max, missing_steps, n_grid_repeats, n_individual_cases, n_samples, run_rmtools, seed)
    return validator


