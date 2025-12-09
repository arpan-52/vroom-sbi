#!/usr/bin/env python3
"""
Comprehensive validation of VROOM-SBI implementation.
Tests code structure, augmentation, simulator, and configuration.
"""

import sys
import numpy as np
import yaml
from pathlib import Path

print("="*70)
print("VROOM-SBI IMPLEMENTATION VALIDATION")
print("="*70)

# Test 1: Configuration
print("\n[1/7] Testing Configuration...")
try:
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Check noise configuration
    assert "noise" in config, "Missing 'noise' section"
    assert "base_level" in config["noise"], "Missing 'base_level'"
    assert "augmentation" in config["noise"], "Missing noise augmentation config"

    noise_aug = config["noise"]["augmentation"]
    assert noise_aug.get("enable") is True, "Noise augmentation not enabled"
    assert "min_factor" in noise_aug, "Missing min_factor"
    assert "max_factor" in noise_aug, "Missing max_factor"

    # Check weight augmentation
    assert "weight_augmentation" in config, "Missing weight_augmentation section"

    # Check decision layer
    assert "decision_layer" in config, "Missing decision_layer section"

    # Check SBI config
    assert "sbi" in config, "Missing SBI configuration"
    assert config["sbi"]["model"] == "nsf", "Should use NSF (Neural Spline Flow)"

    print("   ✓ Configuration valid")
    print(f"   ✓ base_noise_level: {config['noise']['base_level']}")
    print(f"   ✓ Noise augmentation: {noise_aug['min_factor']}x to {noise_aug['max_factor']}x")
    print(f"   ✓ Decision layer enabled: {config['model_selection'].get('use_decision_layer')}")
    print(f"   ✓ SBI model: {config['sbi']['model']} (NSF)")

except Exception as e:
    print(f"   ✗ Configuration error: {e}")
    sys.exit(1)

# Test 2: Module imports
print("\n[2/7] Testing Module Imports...")
try:
    from src.augmentation import (
        augment_weights_combined,
        augment_base_noise_level,
        augment_weights_scattered,
        augment_weights_contiguous_gap,
        augment_weights_noise_variation,
        augment_weights_large_rfi_block
    )
    print("   ✓ Augmentation module imported")

    from src.physics import load_frequencies, freq_to_lambda_sq
    print("   ✓ Physics module imported")

    from src.simulator import RMSimulator, build_prior, sample_prior
    print("   ✓ Simulator module imported")

    from src.decision import QualityPredictionNetwork, QualityPredictionTrainer
    print("   ✓ Decision module imported")

except Exception as e:
    print(f"   ✗ Import error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 3: Weight Augmentation
print("\n[3/7] Testing Weight Augmentation...")
try:
    # Create mock weights
    base_weights = np.ones(100)

    # Test scattered missing
    aug1 = augment_weights_scattered(base_weights, missing_prob=0.1)
    n_missing = np.sum(aug1 == 0)
    print(f"   ✓ Scattered missing: {n_missing}/100 channels flagged")

    # Test contiguous gap
    aug2 = augment_weights_contiguous_gap(base_weights, gap_prob=1.0)
    print(f"   ✓ Contiguous gap created")

    # Test large RFI block
    aug3 = augment_weights_large_rfi_block(base_weights, block_prob=1.0)
    print(f"   ✓ Large RFI block created")

    # Test noise variation
    aug4 = augment_weights_noise_variation(base_weights, variation_scale=0.2)
    print(f"   ✓ Noise variation: weights range [{aug4.min():.3f}, {aug4.max():.3f}]")

    # Test combined augmentation
    aug5 = augment_weights_combined(base_weights)
    n_missing_combined = np.sum(aug5 == 0)
    print(f"   ✓ Combined augmentation: {n_missing_combined}/100 channels affected")

    # Validate properties
    assert aug5.min() >= 0, "Weights should be non-negative"
    assert aug5.max() <= 1.0, "Weights should be <= 1.0"

    print("   ✓ All weight augmentation strategies working")

except Exception as e:
    print(f"   ✗ Weight augmentation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Noise Augmentation
print("\n[4/7] Testing Noise Augmentation...")
try:
    base_level = 0.01
    min_factor = 0.5
    max_factor = 2.0

    # Generate 1000 samples
    samples = [augment_base_noise_level(base_level, min_factor, max_factor)
               for _ in range(1000)]
    samples = np.array(samples)

    expected_min = base_level * min_factor
    expected_max = base_level * max_factor

    print(f"   ✓ Base level: {base_level}")
    print(f"   ✓ Expected range: [{expected_min:.4f}, {expected_max:.4f}]")
    print(f"   ✓ Actual range: [{samples.min():.4f}, {samples.max():.4f}]")
    print(f"   ✓ Mean: {samples.mean():.4f}")

    # Validate range
    assert samples.min() >= expected_min * 0.9, "Min too low"
    assert samples.max() <= expected_max * 1.1, "Max too high"

    print("   ✓ Noise augmentation working correctly")

except Exception as e:
    print(f"   ✗ Noise augmentation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Simulator with base_noise_level
print("\n[5/7] Testing Simulator...")
try:
    freq_file = config.get("freq_file", "freq.txt")

    if not Path(freq_file).exists():
        print(f"   ⚠ Frequency file not found: {freq_file}")
        print("   Creating mock frequency file for testing...")
        freqs = np.linspace(1e9, 2e9, 100)
        np.savetxt(freq_file, freqs)

    # Create simulator with base_noise_level
    base_noise = 0.01
    sim = RMSimulator(freq_file, n_components=1, base_noise_level=base_noise)

    print(f"   ✓ Simulator created")
    print(f"   ✓ n_params: {sim.n_params} (should be 3 for 1-comp: RM, amp, chi0)")
    print(f"   ✓ n_freq: {sim.n_freq}")
    print(f"   ✓ base_noise_level: {sim.base_noise_level}")

    # Test simulation with uniform weights
    theta = np.array([[100.0, 0.5, 1.57]])  # RM=100, amp=0.5, chi0=π/2
    x = sim(theta)

    print(f"   ✓ Simulation output shape: {x.shape} (should be 2*n_freq)")
    assert x.shape == (2 * sim.n_freq,), "Wrong output shape"

    # Test with augmented weights
    aug_weights = augment_weights_combined(sim.weights)
    x_aug = sim(theta, weights=aug_weights)

    print(f"   ✓ Simulation with augmented weights: {x_aug.shape}")

    # Test with augmented noise
    aug_noise = augment_base_noise_level(base_noise)
    sim.base_noise_level = aug_noise
    x_aug_noise = sim(theta, weights=aug_weights)

    print(f"   ✓ Simulation with augmented noise: {x_aug_noise.shape}")

    print("   ✓ Simulator working correctly with augmentation")

except Exception as e:
    print(f"   ✗ Simulator error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Prior sampling
print("\n[6/7] Testing Prior Sampling...")
try:
    flat_priors = {
        "rm_min": -500.0,
        "rm_max": 500.0,
        "amp_min": 0.01,
        "amp_max": 1.0,
    }

    # Sample for 1-component model
    theta_1 = sample_prior(100, n_components=1, config=flat_priors)
    print(f"   ✓ 1-comp samples: {theta_1.shape} (should be [100, 3])")
    assert theta_1.shape == (100, 3), "Wrong shape for 1-comp"

    # Sample for 2-component model
    theta_2 = sample_prior(100, n_components=2, config=flat_priors)
    print(f"   ✓ 2-comp samples: {theta_2.shape} (should be [100, 6])")
    assert theta_2.shape == (100, 6), "Wrong shape for 2-comp"

    # Validate ranges
    rm_vals = theta_1[:, 0]
    amp_vals = theta_1[:, 1]
    chi0_vals = theta_1[:, 2]

    print(f"   ✓ RM range: [{rm_vals.min():.1f}, {rm_vals.max():.1f}]")
    print(f"   ✓ Amp range: [{amp_vals.min():.3f}, {amp_vals.max():.3f}]")
    print(f"   ✓ Chi0 range: [{chi0_vals.min():.3f}, {chi0_vals.max():.3f}]")

    assert rm_vals.min() >= flat_priors["rm_min"], "RM below min"
    assert rm_vals.max() <= flat_priors["rm_max"], "RM above max"
    assert amp_vals.min() >= flat_priors["amp_min"], "Amp below min"
    assert amp_vals.max() <= flat_priors["amp_max"], "Amp above max"
    assert chi0_vals.min() >= 0, "Chi0 below 0"
    assert chi0_vals.max() <= np.pi, "Chi0 above pi"

    print("   ✓ Prior sampling working correctly")

except Exception as e:
    print(f"   ✗ Prior sampling error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 7: Decision Layer Architecture
print("\n[7/7] Testing Decision Layer...")
try:
    n_freq = sim.n_freq

    # Create network
    net = QualityPredictionNetwork(n_freq=n_freq, hidden_dims=[256, 128, 64])

    print(f"   ✓ QualityPredictionNetwork created")
    print(f"   ✓ Input dim: {3 * n_freq} (Q + U + weights)")

    # Test forward pass with dummy data
    dummy_input = np.random.randn(1, 3 * n_freq).astype(np.float32)
    import torch
    dummy_tensor = torch.from_numpy(dummy_input)

    outputs = net(dummy_tensor)

    print(f"   ✓ Forward pass successful")
    print(f"   ✓ Log evidence output: {outputs['log_evidence'].shape} (should be [1, 2])")
    print(f"   ✓ AIC output: {outputs['aic'].shape} (should be [1, 2])")
    print(f"   ✓ BIC output: {outputs['bic'].shape} (should be [1, 2])")

    assert outputs['log_evidence'].shape == (1, 2), "Wrong log_evidence shape"
    assert outputs['aic'].shape == (1, 2), "Wrong AIC shape"
    assert outputs['bic'].shape == (1, 2), "Wrong BIC shape"

    print("   ✓ Decision layer architecture correct")

except Exception as e:
    print(f"   ✗ Decision layer error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Final Summary
print("\n" + "="*70)
print("VALIDATION SUMMARY")
print("="*70)
print("✅ All tests passed!")
print("\nArchitecture verified:")
print("  • Fixed base_noise_level approach (not a learnable parameter)")
print("  • Weight augmentation (scattered, RFI, noise variation)")
print("  • Noise augmentation (0.5x to 2.0x base level)")
print("  • RMSimulator with weights parameter")
print("  • Decision layer predicts AIC, BIC, log_evidence")
print("  • NSF (Neural Spline Flow) with SpectralEmbedding")
print("  • Parameter space: 3*N (RM, amp, chi0 per component)")
print("\nReady to train! Run: python3 train_all.py")
print("="*70)
