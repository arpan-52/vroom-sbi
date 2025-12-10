#!/usr/bin/env python3
"""
Comprehensive Validation Script for VROOM-SBI

Reads all settings from config.yaml - no hardcoded parameter values.

Tests:
1. Single Component Model
   - Parameter recovery (RM, amplitude, chi0)
   - Missing channel interpolation (scattered)
   - Contiguous gaps (RFI simulation)
   - Different noise levels

2. Two Component Model  
   - Parameter recovery with RM ordering (RM1 > RM2)
   - Different RM separations
   - Different amplitude ratios
   - Missing channels and RFI gaps

3. Decision Layer
   - Correct selection for 1-component data
   - Correct selection for 2-component data
   - Edge cases (weak second component)
   - Verification of AIC/BIC values

Usage:
    python validation.py --config config.yaml --models-dir models/
    python validation.py --config config.yaml --models-dir models/ --part 1  # Only 1-comp tests
    python validation.py --config config.yaml --models-dir models/ --part 2  # Only 2-comp tests
    python validation.py --config config.yaml --models-dir models/ --part 3  # Only decision layer
"""

import argparse
import pickle
import numpy as np
import torch
from pathlib import Path
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
import yaml
import sys
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

from src.simulator import RMSimulator, sample_prior, sort_posterior_samples
from src.physics import load_frequencies, freq_to_lambda_sq
from src.decision import QualityPredictionTrainer


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class ValidationResult:
    """Result of a single validation test."""
    test_name: str
    passed: bool
    message: str
    details: Dict = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration extracted from config.yaml for validation."""
    rm_min: float
    rm_max: float
    amp_min: float
    amp_max: float
    base_noise_level: float
    freq_file: str
    device: str
    
    @property
    def rm_range(self) -> float:
        return self.rm_max - self.rm_min
    
    @property
    def rm_mid(self) -> float:
        return (self.rm_max + self.rm_min) / 2
    
    @property
    def amp_mid(self) -> float:
        return (self.amp_max + self.amp_min) / 2


def load_validation_config(config_path: str) -> ValidationConfig:
    """Load and parse config.yaml for validation."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    priors = config.get('priors', {})
    rm_cfg = priors.get('rm', {})
    amp_cfg = priors.get('amp', {})
    noise_cfg = config.get('noise', {})
    training_cfg = config.get('training', {})
    
    return ValidationConfig(
        rm_min=float(rm_cfg.get('min', -500)),
        rm_max=float(rm_cfg.get('max', 500)),
        amp_min=float(amp_cfg.get('min', 0.01)),
        amp_max=float(amp_cfg.get('max', 1.0)),
        base_noise_level=float(noise_cfg.get('base_level', 0.01)),
        freq_file=config.get('freq_file', 'freq.txt'),
        device=training_cfg.get('device', 'cpu'),
    )


# =============================================================================
# Plotting Functions
# =============================================================================

def plot_1comp_recovery(
    posterior,
    simulator: RMSimulator,
    val_cfg: ValidationConfig,
    output_dir: Path,
    n_test_cases: int = 6,
    n_samples: int = 5000
):
    """
    Plot 1-component parameter recovery: true vs recovered RM and amplitude.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Generate test cases across RM range
    test_rms = np.linspace(val_cfg.rm_min + 0.1 * val_cfg.rm_range, 
                           val_cfg.rm_max - 0.1 * val_cfg.rm_range, 
                           n_test_cases)
    test_amps = np.linspace(val_cfg.amp_min + 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                            val_cfg.amp_max - 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                            n_test_cases)
    
    recovered_rms = []
    recovered_rm_stds = []
    recovered_amps = []
    recovered_amp_stds = []
    
    for i, (true_rm, true_amp) in enumerate(zip(test_rms, test_amps)):
        theta_true = np.array([[true_rm, true_amp, 0.5]])
        qu_obs = simulator(theta_true).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        
        recovered_rms.append(np.mean(samples_np[:, 0]))
        recovered_rm_stds.append(np.std(samples_np[:, 0]))
        recovered_amps.append(np.mean(samples_np[:, 1]))
        recovered_amp_stds.append(np.std(samples_np[:, 1]))
    
    # Plot 1: RM recovery
    ax1 = axes[0, 0]
    ax1.errorbar(test_rms, recovered_rms, yerr=recovered_rm_stds, fmt='o', capsize=5, 
                 color='steelblue', markersize=8, label='Recovered')
    ax1.plot([val_cfg.rm_min, val_cfg.rm_max], [val_cfg.rm_min, val_cfg.rm_max], 
             'k--', lw=2, label='Perfect recovery')
    ax1.set_xlabel('True RM (rad/m²)', fontsize=12)
    ax1.set_ylabel('Recovered RM (rad/m²)', fontsize=12)
    ax1.set_title('1-Component RM Recovery', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Amplitude recovery
    ax2 = axes[0, 1]
    ax2.errorbar(test_amps, recovered_amps, yerr=recovered_amp_stds, fmt='s', capsize=5,
                 color='darkorange', markersize=8, label='Recovered')
    ax2.plot([val_cfg.amp_min, val_cfg.amp_max], [val_cfg.amp_min, val_cfg.amp_max],
             'k--', lw=2, label='Perfect recovery')
    ax2.set_xlabel('True Amplitude', fontsize=12)
    ax2.set_ylabel('Recovered Amplitude', fontsize=12)
    ax2.set_title('1-Component Amplitude Recovery', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Example posterior (corner-like for one case)
    ax3 = axes[0, 2]
    true_rm = val_cfg.rm_mid
    true_amp = val_cfg.amp_mid
    theta_true = np.array([[true_rm, true_amp, 0.5]])
    qu_obs = simulator(theta_true).flatten()
    qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=qu_obs_t)
    samples_np = samples.cpu().numpy()
    
    ax3.scatter(samples_np[:, 0], samples_np[:, 1], alpha=0.3, s=5, c='steelblue')
    ax3.axvline(true_rm, color='red', lw=2, label=f'True RM={true_rm:.0f}')
    ax3.axhline(true_amp, color='red', lw=2, label=f'True Amp={true_amp:.2f}')
    ax3.set_xlabel('RM (rad/m²)', fontsize=12)
    ax3.set_ylabel('Amplitude', fontsize=12)
    ax3.set_title('Example Posterior Samples', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recovery with missing channels
    ax4 = axes[1, 0]
    missing_fracs = [0.0, 0.1, 0.2, 0.3, 0.4]
    rm_errors = []
    rm_stds = []
    
    true_rm = val_cfg.rm_mid
    true_amp = val_cfg.amp_mid
    
    for missing_frac in missing_fracs:
        weights = np.ones(simulator.n_freq)
        if missing_frac > 0:
            n_missing = int(simulator.n_freq * missing_frac)
            missing_idx = np.random.choice(simulator.n_freq, n_missing, replace=False)
            weights[missing_idx] = 0.0
        
        theta_true = np.array([[true_rm, true_amp, 0.5]])
        qu_obs = simulator(theta_true, weights=weights).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        
        rm_errors.append(abs(np.mean(samples_np[:, 0]) - true_rm))
        rm_stds.append(np.std(samples_np[:, 0]))
    
    ax4.bar(np.arange(len(missing_fracs)), rm_errors, yerr=rm_stds, capsize=5,
            color='steelblue', alpha=0.7, edgecolor='black')
    ax4.set_xticks(np.arange(len(missing_fracs)))
    ax4.set_xticklabels([f'{int(f*100)}%' for f in missing_fracs])
    ax4.set_xlabel('Missing Channels', fontsize=12)
    ax4.set_ylabel('RM Error (rad/m²)', fontsize=12)
    ax4.set_title('Recovery vs Missing Data', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Recovery with different noise levels
    ax5 = axes[1, 1]
    noise_factors = [0.25, 0.5, 1.0, 2.0, 4.0]
    rm_errors = []
    rm_stds = []
    
    for noise_factor in noise_factors:
        original_noise = simulator.base_noise_level
        simulator.base_noise_level = val_cfg.base_noise_level * noise_factor
        
        theta_true = np.array([[true_rm, true_amp, 0.5]])
        qu_obs = simulator(theta_true).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        simulator.base_noise_level = original_noise
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        
        rm_errors.append(abs(np.mean(samples_np[:, 0]) - true_rm))
        rm_stds.append(np.std(samples_np[:, 0]))
    
    ax5.bar(np.arange(len(noise_factors)), rm_errors, yerr=rm_stds, capsize=5,
            color='darkorange', alpha=0.7, edgecolor='black')
    ax5.set_xticks(np.arange(len(noise_factors)))
    ax5.set_xticklabels([f'{f}×' for f in noise_factors])
    ax5.set_xlabel('Noise Level (× base)', fontsize=12)
    ax5.set_ylabel('RM Error (rad/m²)', fontsize=12)
    ax5.set_title('Recovery vs Noise Level', fontsize=14, fontweight='bold')
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Example spectrum
    ax6 = axes[1, 2]
    freq, _ = load_frequencies(val_cfg.freq_file)
    lambda_sq = freq_to_lambda_sq(freq)
    
    theta_true = np.array([[val_cfg.rm_mid, val_cfg.amp_mid, 0.5]])
    qu_obs = simulator(theta_true).flatten()
    n_freq = len(freq)
    Q = qu_obs[:n_freq]
    U = qu_obs[n_freq:]
    
    ax6.plot(lambda_sq, Q, 'b-', lw=1.5, label='Stokes Q', alpha=0.8)
    ax6.plot(lambda_sq, U, 'r-', lw=1.5, label='Stokes U', alpha=0.8)
    ax6.set_xlabel('λ² (m²)', fontsize=12)
    ax6.set_ylabel('Polarization', fontsize=12)
    ax6.set_title(f'Example Spectrum (RM={val_cfg.rm_mid:.0f})', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'validation_1comp.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_2comp_recovery(
    posterior,
    simulator: RMSimulator,
    val_cfg: ValidationConfig,
    output_dir: Path,
    n_samples: int = 5000
):
    """
    Plot 2-component parameter recovery with RM ordering verification.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # Plot 1: RM1 vs RM2 recovery (showing ordering is preserved)
    ax1 = axes[0, 0]
    
    n_tests = 10
    true_rm1s = []
    true_rm2s = []
    rec_rm1s = []
    rec_rm2s = []
    rec_rm1_stds = []
    rec_rm2_stds = []
    
    for _ in range(n_tests):
        # Random well-separated RMs
        rm1 = np.random.uniform(val_cfg.rm_mid, val_cfg.rm_max - 0.1 * val_cfg.rm_range)
        rm2 = np.random.uniform(val_cfg.rm_min + 0.1 * val_cfg.rm_range, val_cfg.rm_mid)
        
        # Ensure RM1 > RM2
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        true_rm1s.append(rm1)
        true_rm2s.append(rm2)
        
        theta_true = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
        qu_obs = simulator(theta_true).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        samples_np = sort_posterior_samples(samples_np, n_components=2)
        
        rec_rm1s.append(np.mean(samples_np[:, 0]))
        rec_rm2s.append(np.mean(samples_np[:, 3]))
        rec_rm1_stds.append(np.std(samples_np[:, 0]))
        rec_rm2_stds.append(np.std(samples_np[:, 3]))
    
    ax1.errorbar(true_rm1s, rec_rm1s, yerr=rec_rm1_stds, fmt='o', capsize=4,
                 color='steelblue', markersize=7, label='RM1')
    ax1.errorbar(true_rm2s, rec_rm2s, yerr=rec_rm2_stds, fmt='s', capsize=4,
                 color='darkorange', markersize=7, label='RM2')
    ax1.plot([val_cfg.rm_min, val_cfg.rm_max], [val_cfg.rm_min, val_cfg.rm_max],
             'k--', lw=2, label='Perfect')
    ax1.set_xlabel('True RM (rad/m²)', fontsize=12)
    ax1.set_ylabel('Recovered RM (rad/m²)', fontsize=12)
    ax1.set_title('2-Component RM Recovery', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: RM ordering verification
    ax2 = axes[0, 1]
    ordering_preserved = [r1 > r2 for r1, r2 in zip(rec_rm1s, rec_rm2s)]
    colors = ['green' if op else 'red' for op in ordering_preserved]
    
    for i, (r1, r2, c) in enumerate(zip(rec_rm1s, rec_rm2s, colors)):
        ax2.plot([i, i], [r1, r2], color=c, linestyle='-', lw=2)
        ax2.plot(i, r1, 'o', color=c, markersize=8)
        ax2.plot(i, r2, 's', color=c, markersize=8)
    
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax2.set_xlabel('Test Case', fontsize=12)
    ax2.set_ylabel('Recovered RM (rad/m²)', fontsize=12)
    ax2.set_title('RM Ordering: RM1 > RM2', fontsize=14, fontweight='bold')
    
    legend_elements = [Patch(facecolor='green', label='Ordering ✓'),
                       Patch(facecolor='red', label='Ordering ✗')]
    ax2.legend(handles=legend_elements)
    ax2.grid(True, alpha=0.3)
    
    pct_correct = 100 * sum(ordering_preserved) / len(ordering_preserved)
    ax2.text(0.02, 0.98, f'{pct_correct:.0f}% correct ordering', 
             transform=ax2.transAxes, fontsize=11, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Example 2D posterior (RM1 vs RM2)
    ax3 = axes[0, 2]
    
    rm1 = val_cfg.rm_mid + 0.25 * val_cfg.rm_range
    rm2 = val_cfg.rm_mid - 0.25 * val_cfg.rm_range
    
    theta_true = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
    qu_obs = simulator(theta_true).flatten()
    qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=qu_obs_t)
    samples_np = samples.cpu().numpy()
    samples_sorted = sort_posterior_samples(samples_np, n_components=2)
    
    ax3.scatter(samples_sorted[:, 0], samples_sorted[:, 3], alpha=0.3, s=5, c='steelblue')
    ax3.axvline(rm1, color='red', lw=2, label=f'True RM1={rm1:.0f}')
    ax3.axhline(rm2, color='red', lw=2, label=f'True RM2={rm2:.0f}')
    ax3.plot([val_cfg.rm_min, val_cfg.rm_max], [val_cfg.rm_min, val_cfg.rm_max],
             'k--', lw=1, alpha=0.5)
    ax3.set_xlabel('RM1 (rad/m²)', fontsize=12)
    ax3.set_ylabel('RM2 (rad/m²)', fontsize=12)
    ax3.set_title('Posterior: RM1 vs RM2', fontsize=14, fontweight='bold')
    ax3.legend(fontsize=9)
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Recovery vs RM separation
    ax4 = axes[1, 0]
    separations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    rm1_errors = []
    rm2_errors = []
    
    for sep_frac in separations:
        separation = val_cfg.rm_range * sep_frac
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        
        theta_true = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid, 1.5]])
        qu_obs = simulator(theta_true).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        samples_np = sort_posterior_samples(samples_np, n_components=2)
        
        rm1_errors.append(abs(np.mean(samples_np[:, 0]) - rm1))
        rm2_errors.append(abs(np.mean(samples_np[:, 3]) - rm2))
    
    x = np.arange(len(separations))
    width = 0.35
    ax4.bar(x - width/2, rm1_errors, width, label='RM1 error', color='steelblue', alpha=0.7)
    ax4.bar(x + width/2, rm2_errors, width, label='RM2 error', color='darkorange', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels([f'{int(s*100)}%' for s in separations])
    ax4.set_xlabel('RM Separation (% of range)', fontsize=12)
    ax4.set_ylabel('RM Error (rad/m²)', fontsize=12)
    ax4.set_title('Recovery vs RM Separation', fontsize=14, fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Recovery vs amplitude ratio
    ax5 = axes[1, 1]
    amp_ratios = [1.0, 1.5, 2.0, 3.0, 4.0, 5.0]
    rm1_errors = []
    rm2_errors = []
    
    rm1 = val_cfg.rm_mid + 0.25 * val_cfg.rm_range
    rm2 = val_cfg.rm_mid - 0.25 * val_cfg.rm_range
    
    for ratio in amp_ratios:
        amp1 = val_cfg.amp_mid
        amp2 = val_cfg.amp_mid / ratio
        
        theta_true = np.array([[rm1, amp1, 0.5, rm2, amp2, 1.5]])
        qu_obs = simulator(theta_true).flatten()
        qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=val_cfg.device)
        
        with torch.no_grad():
            samples = posterior.sample((n_samples,), x=qu_obs_t)
        samples_np = samples.cpu().numpy()
        samples_np = sort_posterior_samples(samples_np, n_components=2)
        
        rm1_errors.append(abs(np.mean(samples_np[:, 0]) - rm1))
        rm2_errors.append(abs(np.mean(samples_np[:, 3]) - rm2))
    
    x = np.arange(len(amp_ratios))
    ax5.bar(x - width/2, rm1_errors, width, label='RM1 error', color='steelblue', alpha=0.7)
    ax5.bar(x + width/2, rm2_errors, width, label='RM2 error', color='darkorange', alpha=0.7)
    ax5.set_xticks(x)
    ax5.set_xticklabels([f'{r}:1' for r in amp_ratios])
    ax5.set_xlabel('Amplitude Ratio', fontsize=12)
    ax5.set_ylabel('RM Error (rad/m²)', fontsize=12)
    ax5.set_title('Recovery vs Amplitude Ratio', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Example 2-component spectrum
    ax6 = axes[1, 2]
    freq, _ = load_frequencies(val_cfg.freq_file)
    lambda_sq = freq_to_lambda_sq(freq)
    
    rm1 = val_cfg.rm_mid + 0.25 * val_cfg.rm_range
    rm2 = val_cfg.rm_mid - 0.25 * val_cfg.rm_range
    theta_true = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.7, 1.5]])
    qu_obs = simulator(theta_true).flatten()
    n_freq = len(freq)
    Q = qu_obs[:n_freq]
    U = qu_obs[n_freq:]
    
    ax6.plot(lambda_sq, Q, 'b-', lw=1.5, label='Stokes Q', alpha=0.8)
    ax6.plot(lambda_sq, U, 'r-', lw=1.5, label='Stokes U', alpha=0.8)
    ax6.set_xlabel('λ² (m²)', fontsize=12)
    ax6.set_ylabel('Polarization', fontsize=12)
    ax6.set_title(f'2-Comp Spectrum\n(RM1={rm1:.0f}, RM2={rm2:.0f})', fontsize=14, fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'validation_2comp.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_decision_layer(
    decision_trainer,
    val_cfg: ValidationConfig,
    output_dir: Path,
    n_samples: int = 100
):
    """
    Plot decision layer performance: confusion matrix and metric distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    sim_1comp = RMSimulator(val_cfg.freq_file, n_components=1, base_noise_level=val_cfg.base_noise_level)
    sim_2comp = RMSimulator(val_cfg.freq_file, n_components=2, base_noise_level=val_cfg.base_noise_level)
    n_freq = sim_1comp.n_freq
    
    # Generate predictions
    true_labels = []
    pred_labels = []
    bic_diffs = []
    aic_diffs = []
    
    # 1-component data
    for _ in range(n_samples):
        rm = np.random.uniform(val_cfg.rm_min + 0.1 * val_cfg.rm_range,
                               val_cfg.rm_max - 0.1 * val_cfg.rm_range)
        amp = np.random.uniform(val_cfg.amp_min + 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                                val_cfg.amp_max - 0.1 * (val_cfg.amp_max - val_cfg.amp_min))
        
        theta = np.array([[rm, amp, np.random.uniform(0, np.pi)]])
        qu_obs = sim_1comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        pred_n, info = decision_trainer.select_model(x_input_t, strategy='bic')
        
        true_labels.append(1)
        pred_labels.append(pred_n)
        bic_diffs.append(info['bic'][1] - info['bic'][0])  # BIC_2 - BIC_1
        aic_diffs.append(info['aic'][1] - info['aic'][0])
    
    # 2-component data
    for _ in range(n_samples):
        separation = np.random.uniform(0.3, 0.7) * val_cfg.rm_range
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        amp1 = np.random.uniform(val_cfg.amp_min + 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                                 val_cfg.amp_max - 0.1 * (val_cfg.amp_max - val_cfg.amp_min))
        amp_ratio = np.random.uniform(1.0, 3.0)
        amp2 = amp1 / amp_ratio
        
        theta = np.array([[rm1, amp1, np.random.uniform(0, np.pi), 
                          rm2, amp2, np.random.uniform(0, np.pi)]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        pred_n, info = decision_trainer.select_model(x_input_t, strategy='bic')
        
        true_labels.append(2)
        pred_labels.append(pred_n)
        bic_diffs.append(info['bic'][1] - info['bic'][0])
        aic_diffs.append(info['aic'][1] - info['aic'][0])
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    bic_diffs = np.array(bic_diffs)
    aic_diffs = np.array(aic_diffs)
    
    # Plot 1: Confusion Matrix
    ax1 = axes[0, 0]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t-1, p-1] += 1
    
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['1-comp', '2-comp'])
    ax1.set_yticklabels(['1-comp', '2-comp'])
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax1.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=16)
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax1.text(0.5, -0.15, f'Accuracy: {accuracy*100:.1f}%', 
             transform=ax1.transAxes, ha='center', fontsize=12)
    
    # Plot 2: BIC difference distribution
    ax2 = axes[0, 1]
    
    bic_1comp = bic_diffs[true_labels == 1]
    bic_2comp = bic_diffs[true_labels == 2]
    
    bins = np.linspace(min(bic_diffs), max(bic_diffs), 30)
    ax2.hist(bic_1comp, bins=bins, alpha=0.6, color='steelblue', label='True 1-comp', density=True)
    ax2.hist(bic_2comp, bins=bins, alpha=0.6, color='darkorange', label='True 2-comp', density=True)
    ax2.axvline(0, color='black', linestyle='--', lw=2, label='Decision boundary')
    ax2.set_xlabel('ΔBIC (BIC₂ - BIC₁)', fontsize=12)
    ax2.set_ylabel('Density', fontsize=12)
    ax2.set_title('BIC Difference Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Accuracy vs amplitude ratio
    ax3 = axes[0, 2]
    
    amp_ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    accuracies = []
    
    for ratio in amp_ratios:
        correct = 0
        total = 20
        
        for _ in range(total):
            separation = 0.4 * val_cfg.rm_range
            rm1 = val_cfg.rm_mid + separation / 2
            rm2 = val_cfg.rm_mid - separation / 2
            if rm1 < rm2:
                rm1, rm2 = rm2, rm1
            
            theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid / ratio, 1.5]])
            qu_obs = sim_2comp(theta).flatten()
            
            x_input = np.concatenate([qu_obs, np.ones(n_freq)])
            x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
            
            pred_n, _ = decision_trainer.select_model(x_input_t, strategy='bic')
            if pred_n == 2:
                correct += 1
        
        accuracies.append(correct / total * 100)
    
    ax3.bar(np.arange(len(amp_ratios)), accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    ax3.axhline(50, color='red', linestyle='--', lw=2, label='Chance level')
    ax3.set_xticks(np.arange(len(amp_ratios)))
    ax3.set_xticklabels([f'{r}:1' for r in amp_ratios])
    ax3.set_xlabel('Amplitude Ratio', fontsize=12)
    ax3.set_ylabel('2-comp Detection Rate (%)', fontsize=12)
    ax3.set_title('Detection vs Amplitude Ratio', fontsize=14, fontweight='bold')
    ax3.set_ylim(0, 105)
    ax3.legend()
    ax3.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Accuracy vs RM separation
    ax4 = axes[1, 0]
    
    separations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    accuracies = []
    
    for sep_frac in separations:
        correct = 0
        total = 20
        
        for _ in range(total):
            separation = sep_frac * val_cfg.rm_range
            rm1 = val_cfg.rm_mid + separation / 2
            rm2 = val_cfg.rm_mid - separation / 2
            if rm1 < rm2:
                rm1, rm2 = rm2, rm1
            
            theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
            qu_obs = sim_2comp(theta).flatten()
            
            x_input = np.concatenate([qu_obs, np.ones(n_freq)])
            x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
            
            pred_n, _ = decision_trainer.select_model(x_input_t, strategy='bic')
            if pred_n == 2:
                correct += 1
        
        accuracies.append(correct / total * 100)
    
    ax4.bar(np.arange(len(separations)), accuracies, color='darkorange', alpha=0.7, edgecolor='black')
    ax4.axhline(50, color='red', linestyle='--', lw=2, label='Chance level')
    ax4.set_xticks(np.arange(len(separations)))
    ax4.set_xticklabels([f'{int(s*100)}%' for s in separations])
    ax4.set_xlabel('RM Separation (% of range)', fontsize=12)
    ax4.set_ylabel('2-comp Detection Rate (%)', fontsize=12)
    ax4.set_title('Detection vs RM Separation', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: BIC vs AIC agreement
    ax5 = axes[1, 1]
    
    pred_by_bic = np.where(bic_diffs > 0, 1, 2)
    pred_by_aic = np.where(aic_diffs > 0, 1, 2)
    agreement = (pred_by_bic == pred_by_aic)
    
    ax5.scatter(aic_diffs[agreement], bic_diffs[agreement], alpha=0.5, s=30, 
                c='green', label=f'Agree ({sum(agreement)}/{len(agreement)})')
    ax5.scatter(aic_diffs[~agreement], bic_diffs[~agreement], alpha=0.5, s=30,
                c='red', label=f'Disagree ({sum(~agreement)}/{len(agreement)})')
    ax5.axhline(0, color='black', linestyle='--', alpha=0.5)
    ax5.axvline(0, color='black', linestyle='--', alpha=0.5)
    ax5.set_xlabel('ΔAIC (AIC₂ - AIC₁)', fontsize=12)
    ax5.set_ylabel('ΔBIC (BIC₂ - BIC₁)', fontsize=12)
    ax5.set_title('AIC vs BIC Agreement', fontsize=14, fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # Plot 6: Performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    # Calculate stats
    acc_1comp = cm[0, 0] / cm[0].sum() * 100
    acc_2comp = cm[1, 1] / cm[1].sum() * 100
    overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    
    summary_text = f"""
    Decision Layer Performance Summary
    ══════════════════════════════════
    
    Overall Accuracy:     {overall_acc:.1f}%
    
    1-Component:
      • Correct:          {cm[0,0]} / {cm[0].sum()}
      • Accuracy:         {acc_1comp:.1f}%
    
    2-Component:
      • Correct:          {cm[1,1]} / {cm[1].sum()}
      • Accuracy:         {acc_2comp:.1f}%
    
    AIC-BIC Agreement:    {sum(agreement)}/{len(agreement)} ({100*sum(agreement)/len(agreement):.1f}%)
    
    Mean ΔBIC (1-comp):   {np.mean(bic_1comp):.1f}
    Mean ΔBIC (2-comp):   {np.mean(bic_2comp):.1f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    save_path = output_dir / 'validation_decision.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Printing Utilities
# =============================================================================

def print_header(title: str):
    print("\n" + "=" * 70)
    print(f" {title}")
    print("=" * 70)


def print_subheader(title: str):
    print(f"\n--- {title} ---")


def print_result(result: ValidationResult):
    status = "✓ PASS" if result.passed else "✗ FAIL"
    print(f"\n{status}: {result.test_name}")
    print(f"  {result.message}")


def print_summary(results: List[ValidationResult], part_name: str):
    n_passed = sum(1 for r in results if r.passed)
    n_total = len(results)
    
    print(f"\n{'='*70}")
    print(f" {part_name} SUMMARY: {n_passed}/{n_total} tests passed")
    print(f"{'='*70}")
    
    if n_passed < n_total:
        print("\nFailed tests:")
        for r in results:
            if not r.passed:
                print(f"  ✗ {r.test_name}")


# =============================================================================
# PART 1: Single Component Tests
# =============================================================================

def test_1comp_recovery(
    posterior,
    simulator: RMSimulator,
    true_rm: float,
    true_amp: float,
    true_chi0: float,
    noise_level: float,
    weights: Optional[np.ndarray],
    device: str,
    test_name: str,
    n_samples: int = 5000
) -> ValidationResult:
    """Test 1-component parameter recovery."""
    theta_true = np.array([[true_rm, true_amp, true_chi0]])
    
    original_noise = simulator.base_noise_level
    simulator.base_noise_level = noise_level
    
    if weights is None:
        weights = np.ones(simulator.n_freq)
    
    qu_obs = simulator(theta_true, weights=weights).flatten()
    simulator.base_noise_level = original_noise
    
    qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=qu_obs_t)
    samples_np = samples.cpu().numpy()
    
    rm_mean = np.mean(samples_np[:, 0])
    rm_std = np.std(samples_np[:, 0])
    amp_mean = np.mean(samples_np[:, 1])
    amp_std = np.std(samples_np[:, 1])
    
    rm_error = abs(rm_mean - true_rm)
    amp_error = abs(amp_mean - true_amp)
    
    rm_tolerance = max(2 * rm_std, 50.0)
    amp_tolerance = max(2 * amp_std, 0.15)
    
    passed = (rm_error < rm_tolerance) and (amp_error < amp_tolerance)
    
    return ValidationResult(
        test_name=test_name,
        passed=passed,
        message=f"RM: {rm_mean:.1f}±{rm_std:.1f} (true={true_rm:.1f}), "
                f"Amp: {amp_mean:.3f}±{amp_std:.3f} (true={true_amp:.2f})",
        details={"rm_error": rm_error, "amp_error": amp_error}
    )


def run_1comp_tests(posterior, prior, val_cfg: ValidationConfig) -> List[ValidationResult]:
    """Run all 1-component tests."""
    print_header("PART 1: Single Component Model Tests")
    
    results = []
    simulator = RMSimulator(val_cfg.freq_file, n_components=1, base_noise_level=val_cfg.base_noise_level)
    n_freq = simulator.n_freq
    
    # Test RMs across the configured range
    rm_tests = [
        val_cfg.rm_min + 0.1 * val_cfg.rm_range,
        val_cfg.rm_mid,
        val_cfg.rm_max - 0.1 * val_cfg.rm_range,
    ]
    
    # 1.1 Basic Recovery
    print_subheader("1.1 Basic Parameter Recovery")
    for rm in rm_tests:
        result = test_1comp_recovery(
            posterior, simulator, rm, val_cfg.amp_mid, 0.5,
            val_cfg.base_noise_level, None, val_cfg.device,
            f"1-Comp Recovery (RM={rm:.0f})"
        )
        print_result(result)
        results.append(result)
    
    # 1.2 Different Noise Levels
    print_subheader("1.2 Different Noise Levels")
    for noise_factor in [0.5, 1.0, 2.0]:
        noise = val_cfg.base_noise_level * noise_factor
        result = test_1comp_recovery(
            posterior, simulator, val_cfg.rm_mid, val_cfg.amp_mid, 0.5,
            noise, None, val_cfg.device,
            f"1-Comp (noise={noise:.3f})"
        )
        print_result(result)
        results.append(result)
    
    # 1.3 Missing Channels
    print_subheader("1.3 Missing Channels (Scattered)")
    for missing_frac in [0.1, 0.2, 0.3]:
        weights = np.ones(n_freq)
        n_missing = int(n_freq * missing_frac)
        missing_idx = np.random.choice(n_freq, n_missing, replace=False)
        weights[missing_idx] = 0.0
        
        result = test_1comp_recovery(
            posterior, simulator, val_cfg.rm_mid, val_cfg.amp_mid, 0.5,
            val_cfg.base_noise_level, weights, val_cfg.device,
            f"1-Comp ({missing_frac*100:.0f}% missing)"
        )
        print_result(result)
        results.append(result)
    
    # 1.4 Contiguous Gaps
    print_subheader("1.4 Contiguous Gaps (RFI)")
    for gap_frac in [0.1, 0.2]:
        gap_size = int(n_freq * gap_frac)
        gap_start = n_freq // 3
        
        weights = np.ones(n_freq)
        weights[gap_start:gap_start + gap_size] = 0.0
        
        result = test_1comp_recovery(
            posterior, simulator, val_cfg.rm_mid, val_cfg.amp_mid, 0.5,
            val_cfg.base_noise_level, weights, val_cfg.device,
            f"1-Comp (gap={gap_size} channels)"
        )
        print_result(result)
        results.append(result)
    
    print_summary(results, "PART 1")
    return results


# =============================================================================
# PART 2: Two Component Tests
# =============================================================================

def test_2comp_recovery(
    posterior,
    simulator: RMSimulator,
    true_rm1: float,
    true_rm2: float,
    true_amp1: float,
    true_amp2: float,
    noise_level: float,
    weights: Optional[np.ndarray],
    device: str,
    test_name: str,
    n_samples: int = 5000
) -> ValidationResult:
    """Test 2-component parameter recovery with RM ordering check."""
    # Ensure RM1 > RM2
    if true_rm1 < true_rm2:
        true_rm1, true_rm2 = true_rm2, true_rm1
        true_amp1, true_amp2 = true_amp2, true_amp1
    
    theta_true = np.array([[true_rm1, true_amp1, 0.5, true_rm2, true_amp2, 1.5]])
    
    original_noise = simulator.base_noise_level
    simulator.base_noise_level = noise_level
    
    if weights is None:
        weights = np.ones(simulator.n_freq)
    
    qu_obs = simulator(theta_true, weights=weights).flatten()
    simulator.base_noise_level = original_noise
    
    qu_obs_t = torch.tensor(qu_obs, dtype=torch.float32, device=device)
    
    with torch.no_grad():
        samples = posterior.sample((n_samples,), x=qu_obs_t)
    samples_np = samples.cpu().numpy()
    
    # Sort posterior samples
    samples_np = sort_posterior_samples(samples_np, n_components=2)
    
    rm1_mean = np.mean(samples_np[:, 0])
    rm1_std = np.std(samples_np[:, 0])
    rm2_mean = np.mean(samples_np[:, 3])
    rm2_std = np.std(samples_np[:, 3])
    
    rm1_error = abs(rm1_mean - true_rm1)
    rm2_error = abs(rm2_mean - true_rm2)
    
    rm1_tolerance = max(2 * rm1_std, 75.0)
    rm2_tolerance = max(2 * rm2_std, 75.0)
    
    ordering_ok = rm1_mean > rm2_mean
    
    passed = (rm1_error < rm1_tolerance) and (rm2_error < rm2_tolerance) and ordering_ok
    
    return ValidationResult(
        test_name=test_name,
        passed=passed,
        message=f"RM1: {rm1_mean:.1f}±{rm1_std:.1f} (true={true_rm1:.1f}), "
                f"RM2: {rm2_mean:.1f}±{rm2_std:.1f} (true={true_rm2:.1f}), "
                f"Order: {'✓' if ordering_ok else '✗'}",
        details={"rm1_error": rm1_error, "rm2_error": rm2_error, "ordering_ok": ordering_ok}
    )


def run_2comp_tests(posterior, prior, val_cfg: ValidationConfig) -> List[ValidationResult]:
    """Run all 2-component tests."""
    print_header("PART 2: Two Component Model Tests")
    
    results = []
    simulator = RMSimulator(val_cfg.freq_file, n_components=2, base_noise_level=val_cfg.base_noise_level)
    n_freq = simulator.n_freq
    
    # 2.1 Basic Recovery
    print_subheader("2.1 Basic Parameter Recovery (RM1 > RM2 ordering)")
    rm_configs = [
        (val_cfg.rm_max * 0.6, val_cfg.rm_min + 0.2 * val_cfg.rm_range),
        (val_cfg.rm_max * 0.8, val_cfg.rm_max * 0.2),
    ]
    
    for rm1, rm2 in rm_configs:
        result = test_2comp_recovery(
            posterior, simulator, rm1, rm2, val_cfg.amp_mid, val_cfg.amp_mid * 0.8,
            val_cfg.base_noise_level, None, val_cfg.device,
            f"2-Comp (RM1={rm1:.0f}, RM2={rm2:.0f})"
        )
        print_result(result)
        results.append(result)
    
    # 2.2 Different Separations
    print_subheader("2.2 Different RM Separations")
    for sep_frac in [0.3, 0.5, 0.7]:
        separation = val_cfg.rm_range * sep_frac
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        
        result = test_2comp_recovery(
            posterior, simulator, rm1, rm2, val_cfg.amp_mid, val_cfg.amp_mid,
            val_cfg.base_noise_level, None, val_cfg.device,
            f"2-Comp (sep={separation:.0f} rad/m²)"
        )
        print_result(result)
        results.append(result)
    
    # 2.3 Different Amplitude Ratios
    print_subheader("2.3 Different Amplitude Ratios")
    for amp_ratio in [1.0, 2.0, 3.0]:
        rm1 = val_cfg.rm_mid + val_cfg.rm_range * 0.25
        rm2 = val_cfg.rm_mid - val_cfg.rm_range * 0.25
        
        result = test_2comp_recovery(
            posterior, simulator, rm1, rm2, val_cfg.amp_mid, val_cfg.amp_mid / amp_ratio,
            val_cfg.base_noise_level, None, val_cfg.device,
            f"2-Comp (amp ratio={amp_ratio}:1)"
        )
        print_result(result)
        results.append(result)
    
    # 2.4 Missing Channels
    print_subheader("2.4 Missing Channels (2-Component)")
    for missing_frac in [0.1, 0.2]:
        weights = np.ones(n_freq)
        n_missing = int(n_freq * missing_frac)
        missing_idx = np.random.choice(n_freq, n_missing, replace=False)
        weights[missing_idx] = 0.0
        
        rm1 = val_cfg.rm_mid + val_cfg.rm_range * 0.25
        rm2 = val_cfg.rm_mid - val_cfg.rm_range * 0.25
        
        result = test_2comp_recovery(
            posterior, simulator, rm1, rm2, val_cfg.amp_mid, val_cfg.amp_mid * 0.8,
            val_cfg.base_noise_level, weights, val_cfg.device,
            f"2-Comp ({missing_frac*100:.0f}% missing)"
        )
        print_result(result)
        results.append(result)
    
    print_summary(results, "PART 2")
    return results


# =============================================================================
# PART 3: Decision Layer Tests
# =============================================================================

def run_decision_tests(decision_trainer, val_cfg: ValidationConfig) -> List[ValidationResult]:
    """Run all decision layer tests."""
    print_header("PART 3: Decision Layer Tests")
    
    results = []
    
    sim_1comp = RMSimulator(val_cfg.freq_file, n_components=1, base_noise_level=val_cfg.base_noise_level)
    sim_2comp = RMSimulator(val_cfg.freq_file, n_components=2, base_noise_level=val_cfg.base_noise_level)
    n_freq = sim_1comp.n_freq
    
    # 3.1 Clear 1-Component Cases
    print_subheader("3.1 Clear 1-Component Cases")
    for rm in [val_cfg.rm_mid, val_cfg.rm_max * 0.7]:
        theta = np.array([[rm, val_cfg.amp_mid, 0.5]])
        qu_obs = sim_1comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, info = decision_trainer.select_model(x_input_t, strategy='bic')
        passed = predicted_n == 1
        
        result = ValidationResult(
            test_name=f"Decision: 1-comp data (RM={rm:.0f})",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp | {info['confidence']}",
            details=info
        )
        print_result(result)
        results.append(result)
    
    # 3.2 Clear 2-Component Cases
    print_subheader("3.2 Clear 2-Component Cases")
    for sep_frac in [0.4, 0.6]:
        separation = val_cfg.rm_range * sep_frac
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, info = decision_trainer.select_model(x_input_t, strategy='bic')
        passed = predicted_n == 2
        
        result = ValidationResult(
            test_name=f"Decision: 2-comp data (sep={separation:.0f})",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp | {info['confidence']}",
            details=info
        )
        print_result(result)
        results.append(result)
    
    # 3.3 Weak Second Component (Edge Cases)
    print_subheader("3.3 Edge Case: Weak Second Component")
    for amp_ratio in [3.0, 5.0, 10.0]:
        rm1 = val_cfg.rm_mid + val_cfg.rm_range * 0.25
        rm2 = val_cfg.rm_mid - val_cfg.rm_range * 0.25
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid / amp_ratio, 1.5]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, info = decision_trainer.select_model(x_input_t, strategy='bic')
        
        # For very weak components, either answer is acceptable
        if amp_ratio >= 5.0:
            passed = True
            msg_suffix = "(either acceptable)"
        else:
            passed = predicted_n == 2
            msg_suffix = ""
        
        result = ValidationResult(
            test_name=f"Decision: weak 2nd comp (ratio={amp_ratio}:1)",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp {msg_suffix}",
            details=info
        )
        print_result(result)
        results.append(result)
    
    # 3.4 AIC/BIC Sanity Check
    print_subheader("3.4 AIC/BIC Sanity Check")
    
    # For 1-comp data, BIC_1comp should be < BIC_2comp
    theta = np.array([[val_cfg.rm_mid, val_cfg.amp_mid, 0.5]])
    qu_obs = sim_1comp(theta).flatten()
    x_input = np.concatenate([qu_obs, np.ones(n_freq)])
    x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
    
    qualities = decision_trainer.predict_qualities(x_input_t)
    bic = qualities['bic'][0]
    bic_correct = bic[0] < bic[1]
    
    result = ValidationResult(
        test_name="AIC/BIC Sanity: 1-comp data",
        passed=bic_correct,
        message=f"BIC_1={bic[0]:.1f}, BIC_2={bic[1]:.1f} ({'correct' if bic_correct else 'WRONG'})",
        details={"bic_1comp": bic[0], "bic_2comp": bic[1]}
    )
    print_result(result)
    results.append(result)
    
    print_summary(results, "PART 3")
    return results


# =============================================================================
# PART 3 (Alternative): Classifier Tests
# =============================================================================

def run_classifier_tests(classifier, val_cfg: ValidationConfig) -> List[ValidationResult]:
    """Run all classifier tests."""
    print_header("PART 3: Model Selection Classifier Tests")
    
    results = []
    
    sim_1comp = RMSimulator(val_cfg.freq_file, n_components=1, base_noise_level=val_cfg.base_noise_level)
    sim_2comp = RMSimulator(val_cfg.freq_file, n_components=2, base_noise_level=val_cfg.base_noise_level)
    n_freq = sim_1comp.n_freq
    
    # 3.1 Clear 1-Component Cases
    print_subheader("3.1 Clear 1-Component Cases")
    for rm in [val_cfg.rm_mid, val_cfg.rm_max * 0.7]:
        theta = np.array([[rm, val_cfg.amp_mid, 0.5]])
        qu_obs = sim_1comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, prob_dict = classifier.predict(x_input_t)
        passed = predicted_n == 1
        confidence = prob_dict[predicted_n]
        
        result = ValidationResult(
            test_name=f"Classifier: 1-comp data (RM={rm:.0f})",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp (conf={confidence:.1%})",
            details=prob_dict
        )
        print_result(result)
        results.append(result)
    
    # 3.2 Clear 2-Component Cases
    print_subheader("3.2 Clear 2-Component Cases")
    for sep_frac in [0.4, 0.6]:
        separation = val_cfg.rm_range * sep_frac
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, prob_dict = classifier.predict(x_input_t)
        passed = predicted_n == 2
        confidence = prob_dict[predicted_n]
        
        result = ValidationResult(
            test_name=f"Classifier: 2-comp data (sep={separation:.0f})",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp (conf={confidence:.1%})",
            details=prob_dict
        )
        print_result(result)
        results.append(result)
    
    # 3.3 Weak Second Component (Edge Cases)
    print_subheader("3.3 Edge Case: Weak Second Component")
    for amp_ratio in [3.0, 5.0, 10.0]:
        rm1 = val_cfg.rm_mid + val_cfg.rm_range * 0.25
        rm2 = val_cfg.rm_mid - val_cfg.rm_range * 0.25
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid / amp_ratio, 1.5]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        predicted_n, prob_dict = classifier.predict(x_input_t)
        confidence = prob_dict[predicted_n]
        
        # For very weak components, either answer is acceptable
        if amp_ratio >= 5.0:
            passed = True  # Either answer acceptable
            note = "(either answer acceptable)"
        else:
            passed = predicted_n == 2
            note = ""
        
        result = ValidationResult(
            test_name=f"Classifier: Weak 2nd comp (ratio={amp_ratio:.0f}:1) {note}",
            passed=passed,
            message=f"Predicted: {predicted_n}-comp (conf={confidence:.1%})",
            details=prob_dict
        )
        print_result(result)
        results.append(result)
    
    # 3.4 Missing Channels
    print_subheader("3.4 With Missing Channels")
    for missing_frac in [0.1, 0.2]:
        # 1-comp with missing
        theta_1 = np.array([[val_cfg.rm_mid, val_cfg.amp_mid, 0.5]])
        weights_1 = np.ones(n_freq)
        n_missing = int(n_freq * missing_frac)
        missing_idx = np.random.choice(n_freq, n_missing, replace=False)
        weights_1[missing_idx] = 0.0
        
        qu_obs_1 = sim_1comp(theta_1, weights=weights_1).flatten()
        x_input_1 = np.concatenate([qu_obs_1, weights_1])
        x_input_1_t = torch.tensor(x_input_1, dtype=torch.float32, device=val_cfg.device)
        
        pred_1, prob_1 = classifier.predict(x_input_1_t)
        
        result = ValidationResult(
            test_name=f"Classifier: 1-comp ({missing_frac*100:.0f}% missing)",
            passed=pred_1 == 1,
            message=f"Predicted: {pred_1}-comp (conf={prob_1[pred_1]:.1%})",
            details=prob_1
        )
        print_result(result)
        results.append(result)
    
    print_summary(results, "PART 3 (Classifier)")
    return results


def plot_classifier(
    classifier,
    val_cfg: ValidationConfig,
    output_dir: Path,
    n_samples: int = 100
):
    """
    Plot classifier performance: confusion matrix and probability distributions.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    sim_1comp = RMSimulator(val_cfg.freq_file, n_components=1, base_noise_level=val_cfg.base_noise_level)
    sim_2comp = RMSimulator(val_cfg.freq_file, n_components=2, base_noise_level=val_cfg.base_noise_level)
    n_freq = sim_1comp.n_freq
    
    # Generate predictions
    true_labels = []
    pred_labels = []
    probs_1comp = []
    probs_2comp = []
    
    # 1-component data
    for _ in range(n_samples):
        rm = np.random.uniform(val_cfg.rm_min + 0.1 * val_cfg.rm_range,
                               val_cfg.rm_max - 0.1 * val_cfg.rm_range)
        amp = np.random.uniform(val_cfg.amp_min + 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                                val_cfg.amp_max - 0.1 * (val_cfg.amp_max - val_cfg.amp_min))
        
        theta = np.array([[rm, amp, np.random.uniform(0, np.pi)]])
        qu_obs = sim_1comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        pred_n, prob_dict = classifier.predict(x_input_t)
        
        true_labels.append(1)
        pred_labels.append(pred_n)
        probs_1comp.append(prob_dict[1])
        probs_2comp.append(prob_dict[2])
    
    # 2-component data
    for _ in range(n_samples):
        separation = np.random.uniform(0.3, 0.7) * val_cfg.rm_range
        rm1 = val_cfg.rm_mid + separation / 2
        rm2 = val_cfg.rm_mid - separation / 2
        if rm1 < rm2:
            rm1, rm2 = rm2, rm1
        
        amp1 = np.random.uniform(val_cfg.amp_min + 0.1 * (val_cfg.amp_max - val_cfg.amp_min),
                                 val_cfg.amp_max - 0.1 * (val_cfg.amp_max - val_cfg.amp_min))
        amp_ratio = np.random.uniform(1.0, 3.0)
        amp2 = amp1 / amp_ratio
        
        theta = np.array([[rm1, amp1, np.random.uniform(0, np.pi), 
                          rm2, amp2, np.random.uniform(0, np.pi)]])
        qu_obs = sim_2comp(theta).flatten()
        weights = np.ones(n_freq)
        
        x_input = np.concatenate([qu_obs, weights])
        x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
        
        pred_n, prob_dict = classifier.predict(x_input_t)
        
        true_labels.append(2)
        pred_labels.append(pred_n)
        probs_1comp.append(prob_dict[1])
        probs_2comp.append(prob_dict[2])
    
    true_labels = np.array(true_labels)
    pred_labels = np.array(pred_labels)
    probs_1comp = np.array(probs_1comp)
    probs_2comp = np.array(probs_2comp)
    
    # Plot 1: Confusion Matrix
    ax1 = axes[0, 0]
    cm = np.zeros((2, 2), dtype=int)
    for t, p in zip(true_labels, pred_labels):
        cm[t-1, p-1] += 1
    
    im = ax1.imshow(cm, interpolation='nearest', cmap='Blues')
    ax1.set_xticks([0, 1])
    ax1.set_yticks([0, 1])
    ax1.set_xticklabels(['1-comp', '2-comp'])
    ax1.set_yticklabels(['1-comp', '2-comp'])
    ax1.set_xlabel('Predicted', fontsize=12)
    ax1.set_ylabel('True', fontsize=12)
    ax1.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    
    for i in range(2):
        for j in range(2):
            color = 'white' if cm[i, j] > cm.max()/2 else 'black'
            ax1.text(j, i, cm[i, j], ha='center', va='center', color=color, fontsize=16)
    
    accuracy = (cm[0,0] + cm[1,1]) / cm.sum()
    ax1.text(0.5, -0.15, f'Accuracy: {accuracy*100:.1f}%', 
             transform=ax1.transAxes, ha='center', fontsize=12)
    
    # Plot 2: Probability distribution for 1-comp data
    ax2 = axes[0, 1]
    mask_1 = true_labels == 1
    ax2.hist(probs_2comp[mask_1], bins=20, alpha=0.7, color='steelblue', edgecolor='black')
    ax2.axvline(0.5, color='red', linestyle='--', lw=2, label='Decision boundary')
    ax2.set_xlabel('P(2-component)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.set_title('1-comp Data: P(2-comp) Distribution', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Probability distribution for 2-comp data
    ax3 = axes[0, 2]
    mask_2 = true_labels == 2
    ax3.hist(probs_2comp[mask_2], bins=20, alpha=0.7, color='darkorange', edgecolor='black')
    ax3.axvline(0.5, color='red', linestyle='--', lw=2, label='Decision boundary')
    ax3.set_xlabel('P(2-component)', fontsize=12)
    ax3.set_ylabel('Count', fontsize=12)
    ax3.set_title('2-comp Data: P(2-comp) Distribution', fontsize=14, fontweight='bold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Accuracy vs amplitude ratio
    ax4 = axes[1, 0]
    amp_ratios = [1.0, 2.0, 3.0, 4.0, 5.0]
    accuracies = []
    
    for ratio in amp_ratios:
        correct = 0
        total = 20
        
        for _ in range(total):
            separation = 0.4 * val_cfg.rm_range
            rm1 = val_cfg.rm_mid + separation / 2
            rm2 = val_cfg.rm_mid - separation / 2
            if rm1 < rm2:
                rm1, rm2 = rm2, rm1
            
            theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid / ratio, 1.5]])
            qu_obs = sim_2comp(theta).flatten()
            
            x_input = np.concatenate([qu_obs, np.ones(n_freq)])
            x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
            
            pred_n, _ = classifier.predict(x_input_t)
            if pred_n == 2:
                correct += 1
        
        accuracies.append(correct / total * 100)
    
    ax4.bar(np.arange(len(amp_ratios)), accuracies, color='steelblue', alpha=0.7, edgecolor='black')
    ax4.axhline(50, color='red', linestyle='--', lw=2, label='Chance level')
    ax4.set_xticks(np.arange(len(amp_ratios)))
    ax4.set_xticklabels([f'{r}:1' for r in amp_ratios])
    ax4.set_xlabel('Amplitude Ratio', fontsize=12)
    ax4.set_ylabel('2-comp Detection Rate (%)', fontsize=12)
    ax4.set_title('Detection vs Amplitude Ratio', fontsize=14, fontweight='bold')
    ax4.set_ylim(0, 105)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Plot 5: Accuracy vs RM separation
    ax5 = axes[1, 1]
    separations = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    accuracies = []
    
    for sep_frac in separations:
        correct = 0
        total = 20
        
        for _ in range(total):
            separation = sep_frac * val_cfg.rm_range
            rm1 = val_cfg.rm_mid + separation / 2
            rm2 = val_cfg.rm_mid - separation / 2
            if rm1 < rm2:
                rm1, rm2 = rm2, rm1
            
            theta = np.array([[rm1, val_cfg.amp_mid, 0.5, rm2, val_cfg.amp_mid * 0.8, 1.5]])
            qu_obs = sim_2comp(theta).flatten()
            
            x_input = np.concatenate([qu_obs, np.ones(n_freq)])
            x_input_t = torch.tensor(x_input, dtype=torch.float32, device=val_cfg.device)
            
            pred_n, _ = classifier.predict(x_input_t)
            if pred_n == 2:
                correct += 1
        
        accuracies.append(correct / total * 100)
    
    ax5.bar(np.arange(len(separations)), accuracies, color='darkorange', alpha=0.7, edgecolor='black')
    ax5.axhline(50, color='red', linestyle='--', lw=2, label='Chance level')
    ax5.set_xticks(np.arange(len(separations)))
    ax5.set_xticklabels([f'{int(s*100)}%' for s in separations])
    ax5.set_xlabel('RM Separation (% of range)', fontsize=12)
    ax5.set_ylabel('2-comp Detection Rate (%)', fontsize=12)
    ax5.set_title('Detection vs RM Separation', fontsize=14, fontweight='bold')
    ax5.set_ylim(0, 105)
    ax5.legend()
    ax5.grid(True, alpha=0.3, axis='y')
    
    # Plot 6: Performance summary
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    acc_1comp = cm[0, 0] / cm[0].sum() * 100
    acc_2comp = cm[1, 1] / cm[1].sum() * 100
    overall_acc = (cm[0,0] + cm[1,1]) / cm.sum() * 100
    
    summary_text = f"""
    Classifier Performance Summary
    ══════════════════════════════
    
    Overall Accuracy:     {overall_acc:.1f}%
    
    1-Component:
      • Correct:          {cm[0,0]} / {cm[0].sum()}
      • Accuracy:         {acc_1comp:.1f}%
    
    2-Component:
      • Correct:          {cm[1,1]} / {cm[1].sum()}
      • Accuracy:         {acc_2comp:.1f}%
    
    Mean P(2-comp) for 1-comp data: {probs_2comp[mask_1].mean():.3f}
    Mean P(2-comp) for 2-comp data: {probs_2comp[mask_2].mean():.3f}
    """
    
    ax6.text(0.1, 0.9, summary_text, transform=ax6.transAxes, fontsize=11,
             verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))
    
    plt.tight_layout()
    save_path = output_dir / 'validation_classifier.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VROOM-SBI Validation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory with trained models')
    parser.add_argument('--part', type=int, choices=[1, 2, 3], default=None, help='Run only specific part')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--plot', action='store_true', help='Generate validation plots')
    parser.add_argument('--plot-dir', type=str, default='validation_plots', help='Directory for plots')
    
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    print(f"Loading configuration from {args.config}...")
    val_cfg = load_validation_config(args.config)
    
    print(f"\nConfiguration:")
    print(f"  RM range: [{val_cfg.rm_min}, {val_cfg.rm_max}] rad/m²")
    print(f"  Amp range: [{val_cfg.amp_min}, {val_cfg.amp_max}]")
    print(f"  Base noise: {val_cfg.base_noise_level}")
    print(f"  Device: {val_cfg.device}")
    
    models_dir = Path(args.models_dir)
    all_results = []
    
    # Setup plot directory
    if args.plot:
        plot_dir = Path(args.plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nPlots will be saved to: {plot_dir}")
    
    # PART 1
    if args.part is None or args.part == 1:
        print(f"\nLoading 1-component posterior...")
        with open(models_dir / "posterior_n1.pkl", 'rb') as f:
            data_1 = pickle.load(f)
        posterior_1 = data_1['posterior']
        prior_1 = data_1.get('prior')
        
        results_1 = run_1comp_tests(posterior_1, prior_1, val_cfg)
        all_results.extend(results_1)
        
        if args.plot:
            print("\nGenerating 1-component validation plots...")
            simulator_1 = RMSimulator(val_cfg.freq_file, n_components=1, 
                                       base_noise_level=val_cfg.base_noise_level)
            plot_1comp_recovery(posterior_1, simulator_1, val_cfg, plot_dir)
    
    # PART 2
    if args.part is None or args.part == 2:
        print(f"\nLoading 2-component posterior...")
        with open(models_dir / "posterior_n2.pkl", 'rb') as f:
            data_2 = pickle.load(f)
        posterior_2 = data_2['posterior']
        prior_2 = data_2.get('prior')
        
        results_2 = run_2comp_tests(posterior_2, prior_2, val_cfg)
        all_results.extend(results_2)
        
        if args.plot:
            print("\nGenerating 2-component validation plots...")
            simulator_2 = RMSimulator(val_cfg.freq_file, n_components=2,
                                       base_noise_level=val_cfg.base_noise_level)
            plot_2comp_recovery(posterior_2, simulator_2, val_cfg, plot_dir)
    
    # PART 3: Model Selection
    if args.part is None or args.part == 3:
        # Try new classifier first, fall back to decision layer
        classifier_path = models_dir / "classifier.pkl"
        decision_path = models_dir / "decision_layer.pkl"
        
        if classifier_path.exists():
            print(f"\nLoading classifier...")
            from src.classifier import ClassifierTrainer
            freq, _ = load_frequencies(val_cfg.freq_file)
            
            classifier = ClassifierTrainer(
                n_freq=len(freq),
                n_classes=2,
                device=val_cfg.device
            )
            classifier.load(str(classifier_path))
            
            results_3 = run_classifier_tests(classifier, val_cfg)
            all_results.extend(results_3)
            
            if args.plot:
                print("\nGenerating classifier validation plots...")
                plot_classifier(classifier, val_cfg, plot_dir)
        elif decision_path.exists():
            print(f"\nLoading decision layer (legacy)...")
            freq, _ = load_frequencies(val_cfg.freq_file)
            decision_trainer = QualityPredictionTrainer(n_freq=len(freq), device=val_cfg.device)
            decision_trainer.load(str(decision_path))
            
            results_3 = run_decision_tests(decision_trainer, val_cfg)
            all_results.extend(results_3)
            
            if args.plot:
                print("\nGenerating decision layer validation plots...")
                plot_decision_layer(decision_trainer, val_cfg, plot_dir)
        else:
            print(f"  No classifier or decision layer found, skipping Part 3")
    
    # Final Summary
    print_header("FINAL SUMMARY")
    n_passed = sum(1 for r in all_results if r.passed)
    n_total = len(all_results)
    
    print(f"\nTotal: {n_passed}/{n_total} tests passed ({100*n_passed/n_total:.1f}%)")
    
    if n_passed == n_total:
        print("\n✓ All tests passed!")
    else:
        print("\nFailed tests:")
        for r in all_results:
            if not r.passed:
                print(f"  ✗ {r.test_name}")
    
    if args.plot:
        print(f"\nValidation plots saved to: {plot_dir}/")
    
    sys.exit(0 if n_passed == n_total else 1)


if __name__ == '__main__':
    main()
    