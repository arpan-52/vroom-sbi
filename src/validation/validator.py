"""
VROOM-SBI Validation with percentage-based noise.
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
    
    Uses percentage-based noise: sigma = noise_percent/100 * |P|
    """
    
    def __init__(self, posterior_path, output_dir, device="auto"):
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        
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
        
        self._load_posterior()
    
    def _load_posterior(self):
        logger.info(f"Loading posterior from {self.posterior_path}")
        checkpoint = torch.load(self.posterior_path, map_location='cpu', weights_only=False)
        
        self.model_type = checkpoint['model_type']
        self.n_components = checkpoint['n_components']
        self.n_params = checkpoint['n_params']
        self.n_freq = checkpoint['n_freq']
        self.lambda_sq = np.array(checkpoint['lambda_sq'])
        self.prior_bounds = checkpoint['prior_bounds']
        self.posterior = checkpoint['posterior']
        
        if self.device == 'cuda':
            self.posterior.to(self.device)
        
        # Parameter names
        self.param_names = []
        for i in range(self.n_components):
            self.param_names.extend([f'RM_{i+1}', f'amp_{i+1}', f'chi0_{i+1}'])
        
        logger.info(f"  Model: {self.model_type}, N={self.n_components}")
        logger.info(f"  Device: {self.device}")
    
    def _simulate_spectrum(self, theta):
        """Generate Q, U, and amplitude |P| from parameters."""
        P = np.zeros(self.n_freq, dtype=complex)
        for i in range(self.n_components):
            rm, amp, chi0 = theta[i*3], theta[i*3+1], theta[i*3+2]
            P += amp * np.exp(1j * 2 * (chi0 + rm * self.lambda_sq))
        return P.real, P.imag, np.abs(P)
    
    def _add_noise(self, Q, U, P_amplitude, noise_percent):
        """
        Add percentage-based Gaussian noise.
        
        sigma[i] = noise_percent/100 * |P[i]|
        """
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
        
        # Mix of random + contiguous gaps
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
        """Save .dat file for RM-Tools (freq_hz, Q, U, dQ, dU)."""
        dat_path = self.dirs['rmtools'] / f"{case_id}.dat"
        freq_hz = 3e8 / np.sqrt(self.lambda_sq)
        
        with open(dat_path, 'w') as f:
            for i in range(self.n_freq):
                if weights[i] > 0:
                    # dQ = dU = sigma (noise standard deviation)
                    f.write(f"{freq_hz[i]:.10e} {Q_obs[i]:.10e} {U_obs[i]:.10e} "
                           f"{sigma[i]:.10e} {sigma[i]:.10e}\n")
        return dat_path
    
    def _save_case_data(self, case_id, theta_true, result, noise_percent, missing_fraction):
        """Save case data as JSON."""
        data = {
            'case_id': case_id,
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
        
        # Get predicted spectrum from posterior mean
        Q_pred, U_pred, _ = self._simulate_spectrum(result['mean'])
        
        mask = weights > 0
        
        # Panel 1: True Q/U (clean)
        ax = axes[0, 0]
        ax.plot(self.lambda_sq, Q_true, 'b-', lw=2, label='Q true')
        ax.plot(self.lambda_sq, U_true, 'r-', lw=2, label='U true')
        ax.set_xlabel('λ² (m²)')
        ax.set_ylabel('Polarization')
        ax.set_title('True Spectrum (no noise)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Panel 2: Observed Q/U (with noise + flagging)
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
        # Show ±1σ band (average sigma)
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
        
        # Add parameter comparison text
        param_text = "Parameters:\n"
        param_text += f"{'':12} {'True':>10} {'VROOM':>12}\n"
        param_text += "-" * 36 + "\n"
        for i, name in enumerate(self.param_names):
            param_text += f"{name:12} {theta_true[i]:>10.4f} {result['mean'][i]:>8.4f}±{result['std'][i]:.4f}\n"
        
        fig.text(0.02, 0.02, param_text, fontsize=9, family='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.suptitle(f'{case_id}', fontsize=14, fontweight='bold')
        plt.tight_layout(rect=[0, 0.12, 1, 0.96])
        plt.savefig(self.dirs['cases'] / f'{case_id}.png', dpi=150)
        plt.close()
    
    def run_parameter_sweeps(self, n_points=20, noise_percent=10.0, missing_fraction=0.1, n_samples=5000):
        """Run parameter sweeps for RM, amp, chi0."""
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
                
                # Simulate
                Q_true, U_true, P_amp = self._simulate_spectrum(theta)
                Q_obs, U_obs, sigma = self._add_noise(Q_true, U_true, P_amp, noise_percent)
                weights = self._apply_flagging(self.n_freq, missing_fraction)
                
                # Infer
                result = self._run_inference(Q_obs, U_obs, weights, n_samples)
                
                true_vals.append(val)
                est_vals.append(result['mean'][p])
                est_stds.append(result['std'][p])
            
            sweep_results[param_name] = {
                'true': np.array(true_vals),
                'est': np.array(est_vals),
                'std': np.array(est_stds),
            }
        
        # Plot sweeps
        self._plot_sweeps(sweep_results, noise_percent, missing_fraction)
        
        return sweep_results
    
    def _plot_sweeps(self, sweep_results, noise_percent, missing_fraction):
        """Plot parameter sweep results."""
        n_params = len(sweep_results)
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        if n_params == 1:
            axes = [axes]
        
        for ax, (param_name, data) in zip(axes, sweep_results.items()):
            true = data['true']
            est = data['est']
            std = data['std']
            
            ax.errorbar(true, est, yerr=std, fmt='o', capsize=3, alpha=0.7, markersize=5)
            ax.plot([true.min(), true.max()], [true.min(), true.max()], 'k--', lw=2, label='Perfect')
            
            rmse = np.sqrt(np.mean((est - true)**2))
            ax.set_xlabel(f'True {param_name}')
            ax.set_ylabel(f'Estimated {param_name}')
            ax.set_title(f'{param_name} Recovery (RMSE={rmse:.4f})')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.suptitle(f'Parameter Sweeps ({noise_percent:.0f}% noise, {missing_fraction*100:.0f}% flagged)', 
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
            
            # Random parameters from prior
            theta = np.random.uniform(low, high)
            
            # Simulate
            Q_true, U_true, P_amp = self._simulate_spectrum(theta)
            Q_obs, U_obs, sigma = self._add_noise(Q_true, U_true, P_amp, noise_percent)
            weights = self._apply_flagging(self.n_freq, missing_fraction)
            
            # Infer
            result = self._run_inference(Q_obs, U_obs, weights, n_samples)
            
            # Save .dat file for RM-Tools
            self._save_dat_file(case_id, Q_obs, U_obs, weights, sigma)
            
            # Save case data
            self._save_case_data(case_id, theta, result, noise_percent, missing_fraction)
            
            # Plot
            self._plot_case(case_id, theta, Q_true, U_true, Q_obs, U_obs, 
                           weights, result, noise_percent, missing_fraction, sigma)
            
            all_results.append({
                'case_id': case_id,
                'theta_true': theta,
                'vroom_mean': result['mean'],
                'vroom_std': result['std'],
                'time': result['time'],
            })
        
        # Summary plot
        self._plot_summary(all_results, noise_percent, missing_fraction)
        
        return all_results
    
    def _plot_summary(self, results, noise_percent, missing_fraction):
        """Plot summary of all cases."""
        n_params = self.n_params
        fig, axes = plt.subplots(1, n_params, figsize=(5*n_params, 5))
        if n_params == 1:
            axes = [axes]
        
        for p, ax in enumerate(axes):
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
        
        plt.suptitle(f'Individual Cases ({noise_percent:.0f}% noise, {missing_fraction*100:.0f}% flagged)', 
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
        """
        Run full validation.
        
        Args:
            n_sweep_points: Points per parameter sweep
            n_cases: Number of individual test cases
            noise_percent: Noise as percentage of signal (e.g., 10 = 10%)
            missing_fraction: Fraction of channels to flag
            n_samples: Posterior samples per inference
            seed: Random seed
        """
        np.random.seed(seed)
        
        logger.info("=" * 50)
        logger.info("VROOM-SBI Validation")
        logger.info("=" * 50)
        logger.info(f"Noise: {noise_percent}% of signal amplitude")
        logger.info(f"Missing: {missing_fraction*100:.0f}% flagged channels")
        logger.info(f"Sweep points: {n_sweep_points}")
        logger.info(f"Individual cases: {n_cases}")
        logger.info("=" * 50)
        
        # Parameter sweeps
        logger.info("\n[1/2] Parameter sweeps...")
        self.run_parameter_sweeps(n_sweep_points, noise_percent, missing_fraction, n_samples)
        
        # Individual cases
        logger.info("\n[2/2] Individual cases...")
        self.run_individual_cases(n_cases, noise_percent, missing_fraction, n_samples)
        
        logger.info("\n" + "=" * 50)
        logger.info("Validation complete!")
        logger.info("=" * 50)
        logger.info(f"Results: {self.output_dir}")
        logger.info(f"  Sweeps: {self.dirs['sweeps']}")
        logger.info(f"  Cases: {self.dirs['cases']}")
        logger.info(f"  RM-Tools .dat: {self.dirs['rmtools']}")


def run_validation(posterior_path, output_dir, n_sweep_points=20, n_cases=10,
                   noise_percent=10.0, missing_fraction=0.1, n_samples=5000, 
                   device="auto", seed=42):
    """Run validation."""
    validator = Validator(posterior_path, output_dir, device)
    validator.run_validation(n_sweep_points, n_cases, noise_percent, missing_fraction, n_samples, seed)
    return validator
