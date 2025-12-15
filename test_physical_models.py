#!/usr/bin/env python3
"""
Test script to verify all physical models work correctly.

Tests:
1. All physical models can be instantiated
2. Simulation produces expected output shapes
3. Prior sampling works for all model types
4. RM sorting works correctly
"""

import numpy as np
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.simulator import RMSimulator, sample_prior, build_prior


def test_model(model_type: str, n_components: int = 2):
    """Test a specific physical model."""
    print(f"\n{'='*60}")
    print(f"Testing: {model_type} with {n_components} components")
    print(f"{'='*60}")

    freq_file = "freq.txt"
    base_noise_level = 0.01

    # Create simulator
    try:
        simulator = RMSimulator(
            freq_file=freq_file,
            n_components=n_components,
            base_noise_level=base_noise_level,
            model_type=model_type
        )
        print(f"✓ Simulator created")
        print(f"  n_params: {simulator.n_params}")
        print(f"  params_per_comp: {simulator.params_per_comp}")
        print(f"  n_freq: {simulator.n_freq}")
    except Exception as e:
        print(f"✗ Failed to create simulator: {e}")
        return False

    # Sample from prior
    config = {
        "rm_min": 0.0,
        "rm_max": 500.0,
        "amp_min": 0.01,
        "amp_max": 1.0
    }

    try:
        theta = sample_prior(10, n_components, config, model_type=model_type)
        print(f"✓ Prior sampling successful")
        print(f"  theta shape: {theta.shape}")
        print(f"  Expected: (10, {simulator.n_params})")

        if theta.shape[1] != simulator.n_params:
            print(f"✗ Shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Failed to sample prior: {e}")
        return False

    # Simulate
    try:
        simulations = simulator(theta)
        print(f"✓ Simulation successful")
        print(f"  Output shape: {simulations.shape}")
        print(f"  Expected: (10, {2 * simulator.n_freq})")

        if simulations.shape != (10, 2 * simulator.n_freq):
            print(f"✗ Simulation output shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Failed to simulate: {e}")
        import traceback
        traceback.print_exc()
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(simulations)) or np.any(np.isinf(simulations)):
        print(f"✗ Simulation contains NaN or Inf values!")
        return False
    else:
        print(f"✓ No NaN/Inf values detected")

    # Verify RM sorting (for multi-component)
    if n_components >= 2:
        try:
            first_params = theta[:, [i * simulator.params_per_comp for i in range(n_components)]]
            is_sorted = np.all(np.diff(first_params, axis=1) <= 0)  # Descending
            if is_sorted:
                print(f"✓ RM/phi properly sorted (descending)")
            else:
                print(f"✗ RM/phi not properly sorted!")
                print(f"  Sample: {first_params[0]}")
                return False
        except Exception as e:
            print(f"✗ Failed to verify sorting: {e}")
            return False

    print(f"✓ All tests passed for {model_type}!")
    return True


def main():
    """Run all tests."""
    print("="*60)
    print("TESTING PHYSICAL MODELS")
    print("="*60)

    models_to_test = [
        "faraday_thin",
        "burn_slab",
        "external_dispersion",
        "internal_dispersion",
    ]

    component_counts = [1, 2, 3, 5]

    results = {}

    for model_type in models_to_test:
        results[model_type] = {}
        for n_comp in component_counts:
            success = test_model(model_type, n_comp)
            results[model_type][n_comp] = success

    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)

    all_passed = True
    for model_type in models_to_test:
        print(f"\n{model_type}:")
        for n_comp in component_counts:
            status = "✓ PASS" if results[model_type][n_comp] else "✗ FAIL"
            print(f"  {n_comp} components: {status}")
            if not results[model_type][n_comp]:
                all_passed = False

    print("\n" + "="*60)
    if all_passed:
        print("ALL TESTS PASSED! ✓")
        print("="*60)
        return 0
    else:
        print("SOME TESTS FAILED! ✗")
        print("="*60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
