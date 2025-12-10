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
from typing import Dict, List, Optional, Any
import yaml
import sys

from src.simulator import RMSimulator, sample_prior, sort_posterior_samples
from src.physics import load_frequencies
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
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description="VROOM-SBI Validation")
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config.yaml')
    parser.add_argument('--models-dir', type=str, default='models', help='Directory with trained models')
    parser.add_argument('--part', type=int, choices=[1, 2, 3], default=None, help='Run only specific part')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
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
    
    # PART 1
    if args.part is None or args.part == 1:
        print(f"\nLoading 1-component posterior...")
        with open(models_dir / "posterior_n1.pkl", 'rb') as f:
            data_1 = pickle.load(f)
        posterior_1 = data_1['posterior']
        prior_1 = data_1.get('prior')
        
        results_1 = run_1comp_tests(posterior_1, prior_1, val_cfg)
        all_results.extend(results_1)
    
    # PART 2
    if args.part is None or args.part == 2:
        print(f"\nLoading 2-component posterior...")
        with open(models_dir / "posterior_n2.pkl", 'rb') as f:
            data_2 = pickle.load(f)
        posterior_2 = data_2['posterior']
        prior_2 = data_2.get('prior')
        
        results_2 = run_2comp_tests(posterior_2, prior_2, val_cfg)
        all_results.extend(results_2)
    
    # PART 3
    if args.part is None or args.part == 3:
        print(f"\nLoading decision layer...")
        decision_path = models_dir / "decision_layer.pkl"
        if decision_path.exists():
            freq, _ = load_frequencies(val_cfg.freq_file)
            decision_trainer = QualityPredictionTrainer(n_freq=len(freq), device=val_cfg.device)
            decision_trainer.load(str(decision_path))
            
            results_3 = run_decision_tests(decision_trainer, val_cfg)
            all_results.extend(results_3)
        else:
            print(f"  Decision layer not found at {decision_path}, skipping Part 3")
    
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
    
    sys.exit(0 if n_passed == n_total else 1)


if __name__ == '__main__':
    main()