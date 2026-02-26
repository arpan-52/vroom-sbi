"""
VROOM-SBI Validation with percentage-based noise.
Supports reading model info from checkpoint or user-provided values.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime
import json
import logging
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


class Validator:
    """
    VROOM-SBI Validator with parameter sweeps and detailed case plots.
    
    Priority for model info:
    1. User-provided (--model, --n-components, --prior-low, --prior-high)
    2. Read from posterior checkpoint
    3. Error if both fail
    """
    
    # Parameter definitions per model type
    PARAM_DEFS = {
        'faraday_thin': {
            'names': ['RM', 'amp', 'chi0'],
            'n_params': 3,
        },
        'burn_slab': {
            'names': ['phi_c', 'delta_phi', 'amp', 'chi0'],
            'n_params': 4,
        },
        'external_dispersion': {
            'names': ['phi', 'sigma_phi', 'amp', 'chi0'],
            'n_params': 4,
        },
        'internal_dispersion': {
            'names': ['phi', 'sigma_phi', 'amp', 'chi0'],
            'n_params': 4,
        },
    }
    
    def __init__(self, posterior_path, output_dir, model_type=None, n_components=None,
                 prior_low=None, prior_high=None, device="auto"):
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        
        # Store user-provided values (may be None)
        self._user_model_type = model_type
        self._user_n_components = n_components
        self._user_prior_low = prior_low
        self._user_prior_high = prior_high
        
        # Load posterior and resolve model info
        self._load_posterior()
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.dirs = {
            'sweeps': self.output_dir / "parameter_sweeps",
            'cases': self.output_dir / "individual_cases",
            'rmtools': self.output_dir / "rmtools_dat",
            'data': self.output_dir / "data",
        }
        for d in self.dirs.values():
            d.mkdir(exist_ok=True)
    
    def _load_posterior(self):
        logger.info(f"Loading posterior from {self.posterior_path}")
        checkpoint = torch.load(self.posterior_path, map_location='cpu', weights_only=False)
        
        # Load posterior object and basic info
        self.posterior = checkpoint['posterior']
        self.n_freq = checkpoint['n_freq']
        self.lambda_sq = np.array(checkpoint['lambda_sq'])
        
        # Resolve model_type: user > checkpoint > error
        if self._user_model_type:
            self.model_type = self._user_model_type
            logger.info(f"  Using user-provided model: {self.model_type}")
        elif 'model_type' in checkpoint:
            self.model_type = checkpoint['model_type']
            logger.info(f"  Model from checkpoint: {self.model_type}")
        else:
            raise ValueError(
                "Model type not found in checkpoint and not provided by user.\n"
                "Please provide --model (faraday_thin, burn_slab, external_dispersion, internal_dispersion)"
            )
        
        # Resolve n_components: user > checkpoint > error
        if self._user_n_components:
            self.n_components = self._user_n_components
            logger.info(f"  Using user-provided n_components: {self.n_components}")
        elif 'n_components' in checkpoint:
            self.n_components = checkpoint['n_components']
            logger.info(f"  N components from checkpoint: {self.n_components}")
        else:
            raise ValueError(
                "Number of components not found in checkpoint and not provided by user.\n"
                "Please provide --n-components"
            )
        
        # Get parameter info from model type
        if self.model_type not in self.PARAM_DEFS:
            raise ValueError(f"Unknown model type: {self.model_type}. "
                           f"Choose from: {list(self.PARAM_DEFS.keys())}")
        
        self.params_per_comp = self.PARAM_DEFS[self.model_type]['n_params']
        self.n_params = self.params_per_comp * self.n_components
        
        # Generate parameter names
        self.param_names = []
        base_names = self.PARAM_DEFS[self.model_type]['names']
        for i in range(self.n_components):
            for name in base_names:
                self.param_names.append(f"{name}_{i+1}")
        
        # Resolve prior bounds: user > checkpoint > error
        if self._user_prior_low and self._user_prior_high:
            if len(self._user_prior_low) != self.n_params:
                raise ValueError(f"prior_low has {len(self._user_prior_low)} values, expected {self.n_params}")
            if len(self._user_prior_high) != self.n_params:
                raise ValueError(f"prior_high has {len(self._user_prior_high)} values, expected {self.n_params}")
            self.prior_bounds = {
                'low': self._user_prior_low,
                'high': self._user_prior_high,
            }
            logger.info(f"  Using user-provided prior bounds")
        elif 'prior_bounds' in checkpoint:
            self.prior_bounds = checkpoint['prior_bounds']
            logger.info(f"  Prior bounds from checkpoint")
        else:
            raise ValueError(
                "Prior bounds not found in checkpoint and not provided by user.\n"
                "Please provide --prior-low and --prior-high"
            )
        
        # Move posterior to device
        if self.device == 'cuda':
            self.posterior.to(self.device)
        
        # Print summary
        logger.info(f"  Parameters ({self.n_params}): {self.param_names}")
        logger.info(f"  Prior low: {self.prior_bounds['low']}")
        logger.info(f"  Prior high: {self.prior_bounds['high']}")
        logger.info(f"  Frequencies: {self.n_freq}")
        logger.info(f"  Device: {self.device}")
    
    def _simulate_spectrum(self, theta):
        """Generate Q, U, and amplitude |P| from parameters."""
        P = np.zeros(self.n_freq, dtype=complex)
        ppc = self.params_per_comp
        
        if self.model_type == "faraday_thin":
            for i in range(self.n_components):
                rm = theta[i*ppc + 0]
                amp = theta[i*ppc + 1]
                chi0 = theta[i*ppc + 2]
                P += amp * np.exp(2j * (chi0 + rm * self.lambda_sq))
        
        elif self.model_type == "burn_slab":
            for i in range(self.n_components):
                phi_c = theta[i*ppc + 0]
                delta_phi = theta[i*ppc + 1]
                amp = theta[i*ppc + 2]
                chi0 = theta[i*ppc + 3]
                sinc_arg = delta_phi * self.lambda_sq
                sinc_term = np.sinc(sinc_arg / np.pi)
                P += amp * sinc_term * np.exp(2j * (chi0 + phi_c * self.lambda_sq))
        
        elif self.model_type == "external_dispersion":
            # External Faraday dispersion: Gaussian depolarization
            # P = amp × exp(-2σ²λ⁴) × exp(2j(χ₀ + φλ²))
            for i in range(self.n_components):
                phi = theta[i*ppc + 0]
                sigma_phi = theta[i*ppc + 1]
                amp = theta[i*ppc + 2]
                chi0 = theta[i*ppc + 3]
                lambda_sq_squared = self.lambda_sq ** 2  # λ⁴
                depol = np.exp(-2 * sigma_phi**2 * lambda_sq_squared)
                P += amp * depol * np.exp(2j * (chi0 + phi * self.lambda_sq))
        
        elif self.model_type == "internal_dispersion":
            # Internal Faraday dispersion (Sokoloff model):
            # P = amp × [(1 - exp(-S)) / S] × exp(2j × χ₀)
            # where S = 2σ²λ⁴ - 2jφλ² (complex!)
            for i in range(self.n_components):
                phi = theta[i*ppc + 0]
                sigma_phi = theta[i*ppc + 1]
                amp = theta[i*ppc + 2]
                chi0 = theta[i*ppc + 3]
                
                lambda_sq_squared = self.lambda_sq ** 2  # λ⁴
                S = 2 * sigma_phi**2 * lambda_sq_squared - 2j * phi * self.lambda_sq
                
                # Depolarization: (1 - exp(-S)) / S, with limit S→0 → 1
                abs_S = np.abs(S)
                depol = np.where(
                    abs_S < 1e-10,
                    np.ones_like(S),
                    (1 - np.exp(-S)) / S
                )
                
                P += amp * depol * np.exp(2j * chi0)
        
        return P.real, P.imag, np.abs(P)
    
    def _add_noise(self, Q, U, P_amplitude, noise_percent):
        """Add percentage-based Gaussian noise."""
        min_sigma = 1e-6
        sigma = np.maximum(noise_percent / 100.0 * P_amplitude, min_sigma)
        Q_noisy = Q + np.random.normal(0, 1, self.n_freq) * sigma
        U_noisy = U + np.random.normal(0, 1, self.n_freq) * sigma
        return Q_noisy, U_noisy, sigma
    
    def _apply_flagging(self, n_freq, missing_fraction):
        """Generate weights with flagged channels."""
        weights = np.ones(n_freq)
        if missing_fraction <= 0:
            return weights
        
        n_flag = int(n_freq * missing_fraction)
        if n_flag == 0:
            return weights
        
        n_random = n_flag // 2
        n_gap = n_flag - n_random
        
        if n_random > 0:
            idx = np.random.choice(n_freq, n_random, replace=False)
            weights[idx] = 0
        
        if n_gap > 0:
            available = np.where(weights > 0)[0]
            if len(available) > n_gap:
                start = np.random.randint(0, len(available) - n_gap)
                weights[available[start:start+n_gap]] = 0
        
        return weights
    
    def _run_inference(self, Q_obs, U_obs, weights, n_samples=5000):
        """Run VROOM inference."""
        start = datetime.now()
        
        x = np.concatenate([Q_obs * weights, U_obs * weights])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        
        if self.device == 'cuda':
            x_t = x_t.to(self.device)
        
        with torch.no_grad():
            samples = self.posterior.sample((n_samples,), x=x_t).cpu().numpy().squeeze()
        
        elapsed = (datetime.now() - start).total_seconds()
        
        return {
            'samples': samples,
            'mean': samples.mean(axis=0),
            'std': samples.std(axis=0),
            'time': elapsed,
        }
    
    def _save_dat_file(self, case_id, Q_obs, U_obs, weights, sigma):
        """Save .dat file for RM-Tools."""
        dat_path = self.dirs['rmtools'] / f"{case_id}.dat"
        freq_hz = 3e8 / np.sqrt(self.lambda_sq)
        
        with open(dat_path, 'w') as f:
            for i in range(self.n_freq):
                if weights[i] > 0:
                    f.write(f"{freq_hz[i]:.10e} {Q_obs[i]:.10e} {U_obs[i]:.10e} "
                           f"{sigma[i]:.10e} {sigma[i]:.10e}\n")
        return dat_path
    
    def _save_case_data(self, case_id, theta_true, result, noise_percent, missing_fraction):
        """Save case data as JSON."""
        data = {
            'case_id': case_id,
            'model_type': self.model_type,
            'n_components': self.n_components,
            'param_names': self.param_names,
            'theta_true': theta_true.tolist(),
            'vroom_mean': result['mean'].tolist(),
            'vroom_std': result['std'].tolist(),
            'vroom_time_ms': result['time'] * 1000,
            'noise_percent': noise_percent,
            'missing_fraction': missing_fraction,
        }
        json_path = self.dirs['data'] / f"{case_id}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _plot_case(self, case_id, theta_true, Q_true, U_true, Q_obs, U_obs, 
                   weights, result, noise_percent, missing_fraction, sigma):
        """Plot individual case with 4 panels."""
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        Q_pred, U_pred, _ = self._simulate_spectrum(result['mean'])
        mask = weights > 0
        
        # Panel 1: True Q/U
        ax = axes[0, 0]
        ax.plot(self.lambda_sq, Q_true, 'b-', lw=2, label='Q true')
        ax.plot(self.lambda_sq, U_true, 'r-', lw=2, label='U true')
        ax.set_xlabel('λ² (m²)')
        ax.set_ylabel('Polarization')
        ax.set_title('True Spectrum (no noise)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Observed Q/U
        ax = axes[0, 1]
        ax.scatter(self.lambda_sq[mask], Q_obs[mask], c='blue', s=15, alpha=0.6, label='Q obs')
        ax.scatter(self.lambda_sq[mask], U_obs[mask], c='red', s=15, alpha=0.6, label='U obs')
        ax.plot(self.lambda_sq, Q_true, 'b--', alpha=0.3, lw=1)
        ax.plot(self.lambda_sq, U_true, 'r--', alpha=0.3, lw=1)
        ax.set_xlabel('λ² (m²)')
        ax.set_ylabel('Polarization')
        ax.set_title(f'Observed ({noise_percent:.0f}% noise, {missing_fraction*100:.0f}% flagged)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 3: VROOM Predicted
        ax = axes[1, 0]
        ax.scatter(self.lambda_sq[mask], Q_obs[mask], c='blue', s=10, alpha=0.3)
        ax.scatter(self.lambda_sq[mask], U_obs[mask], c='red', s=10, alpha=0.3)
        ax.plot(self.lambda_sq, Q_pred, 'b-', lw=2, label='Q pred')
        ax.plot(self.lambda_sq, U_pred, 'r-', lw=2, label='U pred')
        ax.set_xlabel('λ² (m²)')
        ax.set_ylabel('Polarization')
        ax.set_title(f'VROOM Prediction ({result["time"]*1000:.1f} ms)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 4: Residuals
        ax = axes[1, 1]
        Q_resid = Q_obs - Q_pred
        U_resid = U_obs - U_pred
        ax.scatter(self.lambda_sq[mask], Q_resid[mask], c='blue', s=15, alpha=0.6, label='Q resid')
        ax.scatter(self.lambda_sq[mask], U_resid[mask], c='red', s=15, alpha=0.6, label='U resid')
        ax.axhline(0, color='black', ls='--', lw=1)
        avg_sigma = np.mean(sigma[mask])
        ax.axhline(avg_sigma, color='gray', ls=':', lw=1, label=f'±σ (avg={avg_sigma:.4f})')
        ax.axhline(-avg_sigma, color='gray', ls=':', lw=1)
        ax.set_xlabel('λ² (m²)')
        ax.set_ylabel('Residual')
        rms_q = np.sqrt(np.mean(Q_resid[mask]**2))
        rms_u = np.sqrt(np.mean(U_resid[mask]**2))
        ax.set_title(f'Residuals (RMS: Q={rms_q:.4f}, U={rms_u:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Parameter comparison text
        param_text = f"Model: {self.model_type}, N={self.n_components}\n"
        param_text += f"{'Parameter':15} {'True':>10} {'VROOM':>12}\n"
        param_text += "-" * 40 + "\n"
        for i, name in enumerate(self.param_names):
            param_text += f"{name:15} {theta_true[i]:>10.4f} {result['mean'][i]:>8.4f}±{result['std'][i]:.4f}\n"
        
        fig.text(0.02, 0.02, param_text, fontsize=9, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{case_id}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.15, 1, 0.96])
        plt.savefig(self.dirs['cases'] / f'{case_id}.png', dpi=150)
        plt.close()
    
    def run_parameter_sweeps(self, n_points=20, noise_percent=10.0, missing_fraction=0.1, n_samples=5000):
        """Run parameter sweeps."""
        logger.info("Running parameter sweeps...")
        
        low = np.array(self.prior_bounds['low'])
        high = np.array(self.prior_bounds['high'])
        mid = (low + high) / 2
        
        sweep_results = {}
        
        for p in range(self.n_params):
            param_name = self.param_names[p]
            logger.info(f"  Sweeping {param_name}...")
            
            values = np.linspace(low[p], high[p], n_points)
            true_vals, est_vals, est_stds = [], [], []
            
            for val in tqdm(values, desc=param_name):
                theta = mid.copy()
                theta[p] = val
                
                Q_true, U_true, P_amp = self._simulate_spectrum(theta)
                Q_obs, U_obs, sigma = self._add_noise(Q_true, U_true, P_amp, noise_percent)
                weights = self._apply_flagging(self.n_freq, missing_fraction)
                
                result = self._run_inference(Q_obs, U_obs, weights, n_samples)
                
                true_vals.append(val)
                est_vals.append(result['mean'][p])
                est_stds.append(result['std'][p])
            
            sweep_results[param_name] = {
                'true': np.array(true_vals),
                'est': np.array(est_vals),
                'std': np.array(est_stds),
            }
        
        self._plot_sweeps(sweep_results, noise_percent, missing_fraction)
        return sweep_results
    
    def _plot_sweeps(self, sweep_results, noise_percent, missing_fraction):
        """Plot parameter sweep results."""
        n_params = len(sweep_results)
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        for idx, (param_name, data) in enumerate(sweep_results.items()):
            ax = axes[idx]
            true = data['true']
            est = data['est']
            std = data['std']
            
            ax.errorbar(true, est, yerr=std, fmt='o', capsize=3, alpha=0.7, markersize=5)
            ax.plot([true.min(), true.max()], [true.min(), true.max()], 'k--', lw=2)
            
            rmse = np.sqrt(np.mean((est - true)**2))
            ax.set_xlabel(f'True {param_name}')
            ax.set_ylabel(f'Estimated {param_name}')
            ax.set_title(f'{param_name} (RMSE={rmse:.4f})')
            ax.grid(True, alpha=0.3)
        
        # Hide unused axes
        for idx in range(len(sweep_results), len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Parameter Sweeps ({noise_percent:.0f}% noise, {missing_fraction*100:.0f}% flagged)\n'
                    f'Model: {self.model_type}, N={self.n_components}', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.dirs['sweeps'] / 'parameter_sweeps.png', dpi=150)
        plt.close()
    
    def run_individual_cases(self, n_cases=10, noise_percent=10.0, missing_fraction=0.1, n_samples=5000):
        """Run individual test cases with detailed plots."""
        logger.info(f"Running {n_cases} individual cases...")
        
        low = np.array(self.prior_bounds['low'])
        high = np.array(self.prior_bounds['high'])
        
        all_results = []
        
        for i in tqdm(range(n_cases), desc="Cases"):
            case_id = f"case_{i+1:03d}"
            
            theta = np.random.uniform(low, high)
            
            Q_true, U_true, P_amp = self._simulate_spectrum(theta)
            Q_obs, U_obs, sigma = self._add_noise(Q_true, U_true, P_amp, noise_percent)
            weights = self._apply_flagging(self.n_freq, missing_fraction)
            
            result = self._run_inference(Q_obs, U_obs, weights, n_samples)
            
            self._save_dat_file(case_id, Q_obs, U_obs, weights, sigma)
            self._save_case_data(case_id, theta, result, noise_percent, missing_fraction)
            self._plot_case(case_id, theta, Q_true, U_true, Q_obs, U_obs, 
                           weights, result, noise_percent, missing_fraction, sigma)
            
            all_results.append({
                'case_id': case_id,
                'theta_true': theta,
                'vroom_mean': result['mean'],
                'vroom_std': result['std'],
                'time': result['time'],
            })
        
        self._plot_summary(all_results, noise_percent, missing_fraction)
        return all_results
    
    def _plot_summary(self, results, noise_percent, missing_fraction):
        """Plot summary of all cases."""
        n_params = self.n_params
        n_cols = min(3, n_params)
        n_rows = (n_params + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 5*n_rows))
        axes = np.atleast_1d(axes).flatten()
        
        for p in range(n_params):
            ax = axes[p]
            true_vals = [r['theta_true'][p] for r in results]
            est_vals = [r['vroom_mean'][p] for r in results]
            est_stds = [r['vroom_std'][p] for r in results]
            
            ax.errorbar(true_vals, est_vals, yerr=est_stds, fmt='o', capsize=3, alpha=0.7)
            lim = [min(true_vals), max(true_vals)]
            ax.plot(lim, lim, 'k--', lw=2)
            
            rmse = np.sqrt(np.mean((np.array(est_vals) - np.array(true_vals))**2))
            ax.set_xlabel(f'True {self.param_names[p]}')
            ax.set_ylabel(f'VROOM {self.param_names[p]}')
            ax.set_title(f'{self.param_names[p]} (RMSE={rmse:.4f})')
            ax.grid(True, alpha=0.3)
        
        for idx in range(n_params, len(axes)):
            axes[idx].set_visible(False)
        
        plt.suptitle(f'Individual Cases ({noise_percent:.0f}% noise, {missing_fraction*100:.0f}% flagged)\n'
                    f'Model: {self.model_type}, N={self.n_components}', 
                    fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(self.dirs['cases'] / 'summary.png', dpi=150)
        plt.close()
        
        # Timing histogram
        times = [r['time'] * 1000 for r in results]
        plt.figure(figsize=(8, 5))
        plt.hist(times, bins=15, edgecolor='black', alpha=0.7)
        plt.axvline(np.mean(times), color='red', ls='--', label=f'Mean: {np.mean(times):.1f} ms')
        plt.xlabel('Inference Time (ms)')
        plt.ylabel('Count')
        plt.title('VROOM Inference Timing')
        plt.legend()
        plt.savefig(self.dirs['cases'] / 'timing.png', dpi=150)
        plt.close()
    
    def run_validation(self, n_sweep_points=20, n_cases=10, noise_percent=10.0, 
                       missing_fraction=0.1, n_samples=5000, seed=42):
        """Run full validation."""
        np.random.seed(seed)
        
        logger.info("=" * 50)
        logger.info("VROOM-SBI Validation")
        logger.info("=" * 50)
        logger.info(f"Model: {self.model_type}, N={self.n_components}")
        logger.info(f"Parameters: {self.param_names}")
        logger.info(f"Noise: {noise_percent}% of signal")
        logger.info(f"Missing: {missing_fraction*100:.0f}% flagged")
        logger.info("=" * 50)
        
        logger.info("\n[1/2] Parameter sweeps...")
        self.run_parameter_sweeps(n_sweep_points, noise_percent, missing_fraction, n_samples)
        
        logger.info("\n[2/2] Individual cases...")
        self.run_individual_cases(n_cases, noise_percent, missing_fraction, n_samples)
        
        logger.info("\n" + "=" * 50)
        logger.info("Validation complete!")
        logger.info("=" * 50)
        logger.info(f"Results: {self.output_dir}")
        logger.info(f"  Sweeps: {self.dirs['sweeps']}")
        logger.info(f"  Cases: {self.dirs['cases']}")
        logger.info(f"  RM-Tools .dat: {self.dirs['rmtools']}")


def run_validation(posterior_path, output_dir, model_type=None, n_components=None,
                   prior_low=None, prior_high=None,
                   n_sweep_points=20, n_cases=10,
                   noise_percent=10.0, missing_fraction=0.1, n_samples=5000, 
                   device="auto", seed=42):
    """Run validation."""
    validator = Validator(posterior_path, output_dir, model_type, n_components,
                         prior_low, prior_high, device)
    validator.run_validation(n_sweep_points, n_cases, noise_percent, missing_fraction, n_samples, seed)
    return validator
