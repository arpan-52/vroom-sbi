"""
Simple VROOM-SBI Validation with .dat file export for RM-Tools.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime
import json
import logging
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class TestCase:
    case_id: str
    theta_true: np.ndarray
    vroom_mean: np.ndarray
    vroom_std: np.ndarray
    vroom_time: float
    noise_level: float
    missing_fraction: float


class SimpleValidator:
    def __init__(self, posterior_path, output_dir, device="auto"):
        self.posterior_path = Path(posterior_path)
        self.output_dir = Path(output_dir)
        self.device = "cuda" if device == "auto" and torch.cuda.is_available() else "cpu"
        
        # Create directories
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "rmtools_input").mkdir(exist_ok=True)
        (self.output_dir / "ground_truth").mkdir(exist_ok=True)
        (self.output_dir / "plots").mkdir(exist_ok=True)
        
        self._load_posterior()
        self.results = []
    
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
        
        logger.info(f"  Model: {self.model_type}, N={self.n_components}, Device: {self.device}")
    
    def _simulate_spectrum(self, theta):
        P = np.zeros(self.n_freq, dtype=complex)
        for i in range(self.n_components):
            rm, amp, chi0 = theta[i*3], theta[i*3+1], theta[i*3+2]
            P += amp * np.exp(1j * 2 * (chi0 + rm * self.lambda_sq))
        return P.real, P.imag
    
    def _add_noise_and_gaps(self, Q, U, noise_level, missing_fraction):
        Q_obs = Q + np.random.normal(0, noise_level, self.n_freq)
        U_obs = U + np.random.normal(0, noise_level, self.n_freq)
        
        weights = np.ones(self.n_freq)
        if missing_fraction > 0:
            n_missing = int(self.n_freq * missing_fraction)
            idx = np.random.choice(self.n_freq, n_missing, replace=False)
            weights[idx] = 0
        
        return Q_obs, U_obs, weights
    
    def _run_vroom(self, Q_obs, U_obs, weights, n_samples=5000):
        start = datetime.now()
        
        x = np.concatenate([Q_obs * weights, U_obs * weights])
        x_t = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        if self.device == 'cuda':
            x_t = x_t.to(self.device)
        
        with torch.no_grad():
            samples = self.posterior.sample((n_samples,), x=x_t).cpu().numpy().squeeze()
        
        elapsed = (datetime.now() - start).total_seconds()
        return samples.mean(axis=0), samples.std(axis=0), elapsed
    
    def _save_dat_file(self, case_id, Q_obs, U_obs, weights, noise_level):
        """Save .dat file for RM-Tools."""
        dat_path = self.output_dir / "rmtools_input" / f"{case_id}.dat"
        freq_hz = 3e8 / np.sqrt(self.lambda_sq)
        
        with open(dat_path, 'w') as f:
            for i in range(self.n_freq):
                if weights[i] > 0:
                    f.write(f"{freq_hz[i]:.10e} {Q_obs[i]:.10e} {U_obs[i]:.10e} {noise_level:.10e} {noise_level:.10e}\n")
        
        return dat_path
    
    def _save_ground_truth(self, case_id, theta_true, vroom_mean, vroom_std, vroom_time):
        """Save ground truth JSON."""
        json_path = self.output_dir / "ground_truth" / f"{case_id}.json"
        
        data = {
            'case_id': case_id,
            'param_names': self.param_names,
            'theta_true': theta_true.tolist(),
            'vroom_mean': vroom_mean.tolist(),
            'vroom_std': vroom_std.tolist(),
            'vroom_time_ms': vroom_time * 1000,
        }
        
        with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_validation(self, n_cases=20, n_samples=5000, seed=42):
        """Run validation and save .dat files."""
        np.random.seed(seed)
        
        logger.info("=" * 50)
        logger.info("VROOM-SBI Validation")
        logger.info("=" * 50)
        
        low = np.array(self.prior_bounds['low'])
        high = np.array(self.prior_bounds['high'])
        
        # Generate test cases with varying noise and missing
        noise_levels = np.linspace(0.01, 0.2, n_cases)
        missing_fracs = np.linspace(0.0, 0.3, n_cases)
        np.random.shuffle(noise_levels)
        np.random.shuffle(missing_fracs)
        
        logger.info(f"Running {n_cases} test cases...")
        
        for i in tqdm(range(n_cases), desc="Validating"):
            case_id = f"case_{i+1:03d}"
            
            # Random parameters
            theta = np.random.uniform(low, high)
            noise = noise_levels[i]
            missing = missing_fracs[i]
            
            # Simulate
            Q_true, U_true = self._simulate_spectrum(theta)
            Q_obs, U_obs, weights = self._add_noise_and_gaps(Q_true, U_true, noise, missing)
            
            # Run VROOM
            vroom_mean, vroom_std, vroom_time = self._run_vroom(Q_obs, U_obs, weights, n_samples)
            
            # Save .dat file
            self._save_dat_file(case_id, Q_obs, U_obs, weights, noise)
            
            # Save ground truth
            self._save_ground_truth(case_id, theta, vroom_mean, vroom_std, vroom_time)
            
            self.results.append(TestCase(case_id, theta, vroom_mean, vroom_std, vroom_time, noise, missing))
        
        # Save bash script
        self._save_rmtools_script()
        
        # Save comparison python script
        self._save_comparison_script()
        
        # Plot VROOM results
        self._plot_vroom_results()
        
        logger.info(f"\nDone! Results saved to: {self.output_dir}")
        logger.info(f"  .dat files: {self.output_dir}/rmtools_input/")
        logger.info(f"  Ground truth: {self.output_dir}/ground_truth/")
        logger.info(f"\nTo run RM-Tools comparison:")
        logger.info(f"  1. Edit compare_rmtools.sh to set RMTOOLS_ENV path")
        logger.info(f"  2. bash {self.output_dir}/compare_rmtools.sh")
        logger.info(f"  3. python {self.output_dir}/compare_results.py")
    
    def _plot_vroom_results(self):
        """Plot VROOM-only results."""
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for p, ax in enumerate(axes):
            true_vals = [r.theta_true[p] for r in self.results]
            est_vals = [r.vroom_mean[p] for r in self.results]
            est_stds = [r.vroom_std[p] for r in self.results]
            
            ax.errorbar(true_vals, est_vals, yerr=est_stds, fmt='o', alpha=0.6, capsize=2)
            lims = [min(true_vals), max(true_vals)]
            ax.plot(lims, lims, 'k--', lw=2)
            ax.set_xlabel(f'True {self.param_names[p]}')
            ax.set_ylabel(f'VROOM {self.param_names[p]}')
            ax.set_title(f'{self.param_names[p]} Recovery')
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / "plots" / "vroom_recovery.png", dpi=150)
        plt.close()
        
        # Timing histogram
        times = [r.vroom_time * 1000 for r in self.results]
        plt.figure(figsize=(8, 5))
        plt.hist(times, bins=20, edgecolor='black', alpha=0.7)
        plt.xlabel('Time (ms)')
        plt.ylabel('Count')
        plt.title(f'VROOM-SBI Inference Time (mean: {np.mean(times):.1f} ms)')
        plt.savefig(self.output_dir / "plots" / "vroom_timing.png", dpi=150)
        plt.close()
    
    def _save_rmtools_script(self):
        """Save bash script to run RM-Tools."""
        script = '''#!/bin/bash
# RM-Tools Comparison Script
# Edit RMTOOLS_ENV to point to your RM-Tools environment

RMTOOLS_ENV="/media/volume/vroom-training-1/y/envs/rmtools"  # <-- EDIT THIS
RMTOOLS_MODEL="1"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INPUT_DIR="${SCRIPT_DIR}/rmtools_input"
OUTPUT_DIR="${SCRIPT_DIR}/rmtools_output"

mkdir -p "$OUTPUT_DIR"

echo "Running RM-Tools on $(ls -1 $INPUT_DIR/*.dat | wc -l) files..."
echo "Environment: $RMTOOLS_ENV"
echo ""

for f in "$INPUT_DIR"/*.dat; do
    BASENAME=$(basename "$f" .dat)
    echo "Processing: $BASENAME"
    
    START=$(date +%s.%N)
    micromamba run -p "$RMTOOLS_ENV" qufit "$f" -m "$RMTOOLS_MODEL" -v > "$OUTPUT_DIR/${BASENAME}_log.txt" 2>&1
    END=$(date +%s.%N)
    
    # Move output files
    OUTFILE="${f%.dat}_m${RMTOOLS_MODEL}_dynesty.dat"
    if [ -f "$OUTFILE" ]; then
        mv "$OUTFILE" "$OUTPUT_DIR/"
        echo "  OK ($(echo "$END - $START" | bc)s)"
    else
        echo "  FAILED - check $OUTPUT_DIR/${BASENAME}_log.txt"
    fi
done

echo ""
echo "Done! Run: python compare_results.py"
'''
        script_path = self.output_dir / "compare_rmtools.sh"
        with open(script_path, 'w') as f:
            f.write(script)
        import os
        os.chmod(script_path, 0o755)
    
    def _save_comparison_script(self):
        """Save Python comparison script."""
        script = '''#!/usr/bin/env python3
"""Compare VROOM-SBI vs RM-Tools results."""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import re

SCRIPT_DIR = Path(__file__).parent
TRUTH_DIR = SCRIPT_DIR / "ground_truth"
RMTOOLS_DIR = SCRIPT_DIR / "rmtools_output"
PLOTS_DIR = SCRIPT_DIR / "plots"

def parse_rmtools(dat_file):
    try:
        content = dat_file.read_text()
        values = re.search(r"values=\\[([^\\]]+)\\]", content)
        if values:
            vals = [float(x.strip()) for x in values.group(1).split(',')]
            # RM-Tools: [fracPol, psi0_deg, RM] -> VROOM: [RM, amp, chi0]
            return np.array([vals[2], vals[0], np.deg2rad(vals[1])])
    except:
        pass
    return None

def main():
    truth_files = sorted(TRUTH_DIR.glob("*.json"))
    print(f"Found {len(truth_files)} cases")
    
    vroom_err, rmtools_err, vroom_times, rmtools_count = [], [], [], 0
    true_all, vroom_all, rmtools_all = [], [], []
    
    for tf in truth_files:
        with open(tf) as f:
            t = json.load(f)
        
        theta = np.array(t['theta_true'])
        vroom = np.array(t['vroom_mean'])
        vroom_times.append(t['vroom_time_ms'])
        
        true_all.append(theta)
        vroom_all.append(vroom)
        vroom_err.append(vroom - theta)
        
        # Check for RM-Tools result
        rt_file = RMTOOLS_DIR / f"{t['case_id']}_m1_dynesty.dat"
        if rt_file.exists():
            rt = parse_rmtools(rt_file)
            if rt is not None:
                rmtools_all.append(rt)
                rmtools_err.append(rt - theta)
                rmtools_count += 1
    
    true_all = np.array(true_all)
    vroom_all = np.array(vroom_all)
    vroom_err = np.array(vroom_err)
    
    print(f"RM-Tools successful: {rmtools_count}/{len(truth_files)}")
    print(f"\\nVROOM mean time: {np.mean(vroom_times):.1f} ms")
    
    names = ['RM', 'amp', 'chi0']
    print("\\nRMSE:")
    for i, n in enumerate(names):
        v_rmse = np.sqrt(np.mean(vroom_err[:, i]**2))
        print(f"  {n} VROOM: {v_rmse:.4f}", end="")
        if rmtools_err:
            r_rmse = np.sqrt(np.mean(np.array(rmtools_err)[:, i]**2))
            print(f"  RM-Tools: {r_rmse:.4f}")
        else:
            print()
    
    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, (ax, n) in enumerate(zip(axes, names)):
        ax.scatter(true_all[:, i], vroom_all[:, i], alpha=0.6, label='VROOM', s=30)
        if rmtools_all:
            rmtools_arr = np.array(rmtools_all)
            ax.scatter(true_all[:len(rmtools_arr), i], rmtools_arr[:, i], 
                      alpha=0.6, label='RM-Tools', marker='s', s=30)
        lim = [true_all[:, i].min(), true_all[:, i].max()]
        ax.plot(lim, lim, 'k--')
        ax.set_xlabel(f'True {n}')
        ax.set_ylabel(f'Estimated {n}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "comparison.png", dpi=150)
    print(f"\\nSaved: {PLOTS_DIR / 'comparison.png'}")

if __name__ == "__main__":
    main()
'''
        script_path = self.output_dir / "compare_results.py"
        with open(script_path, 'w') as f:
            f.write(script)
        import os
        os.chmod(script_path, 0o755)


def run_validation(posterior_path, output_dir, n_cases=20, n_samples=5000, device="auto", seed=42):
    validator = SimpleValidator(posterior_path, output_dir, device)
    validator.run_validation(n_cases, n_samples, seed)
    return validator
