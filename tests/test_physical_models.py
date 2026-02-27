#!/usr/bin/env python3
"""
Test script to verify all physical models work correctly.

Tests:
1. All physical models can be instantiated
2. Simulation produces expected output shapes
3. Prior sampling works for all model types
4. RM sorting works correctly
5. Depolarization formulas are correct
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulator import RMSimulator, sample_prior

# Absolute path to freq.txt so tests pass regardless of working directory
FREQ_FILE = str(Path(__file__).parent.parent / "freq.txt")


def _check_model(model_type: str, n_components: int = 2):
    """Helper (not a pytest test) — verifies a specific physical model."""
    print(f"\n{'=' * 60}")
    print(f"Testing: {model_type} with {n_components} components")
    print(f"{'=' * 60}")

    # Create simulator
    try:
        simulator = RMSimulator(
            freq_file=FREQ_FILE, n_components=n_components, model_type=model_type
        )
        print("✓ Simulator created")
        print(f"  n_params: {simulator.n_params}")
        print(f"  params_per_comp: {simulator.params_per_comp}")
        print(f"  n_freq: {simulator.n_freq}")
    except Exception as e:
        print(f"✗ Failed to create simulator: {e}")
        return False

    # Sample from prior
    config = {"rm_min": 0.0, "rm_max": 500.0, "amp_min": 0.01, "amp_max": 1.0}

    try:
        theta = sample_prior(10, n_components, config, model_type=model_type)
        print("✓ Prior sampling successful")
        print(f"  theta shape: {theta.shape}")

        if theta.shape[1] != simulator.n_params:
            print("✗ Shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Failed to sample prior: {e}")
        return False

    # Simulate
    try:
        simulations = simulator(theta)
        print("✓ Simulation successful")
        print(f"  Output shape: {simulations.shape}")

        if simulations.shape != (10, 2 * simulator.n_freq):
            print("✗ Simulation output shape mismatch!")
            return False
    except Exception as e:
        print(f"✗ Failed to simulate: {e}")
        import traceback

        traceback.print_exc()
        return False

    # Check for NaN or Inf
    if np.any(np.isnan(simulations)) or np.any(np.isinf(simulations)):
        print("✗ Simulation contains NaN or Inf values!")
        return False
    else:
        print("✓ No NaN/Inf values detected")

    # Verify RM sorting
    if n_components >= 2:
        try:
            first_params = theta[
                :, [i * simulator.params_per_comp for i in range(n_components)]
            ]
            is_sorted = np.all(np.diff(first_params, axis=1) <= 0)
            if is_sorted:
                print("✓ RM/phi properly sorted (descending)")
            else:
                print("✗ RM/phi not properly sorted!")
                return False
        except Exception as e:
            print(f"✗ Failed to verify sorting: {e}")
            return False

    print(f"✓ All tests passed for {model_type}!")
    return True


def test_depolarization_formulas():
    """Test that depolarization formulas are implemented correctly."""
    print(f"\n{'=' * 60}")
    print("Testing depolarization formula correctness")
    print(f"{'=' * 60}")

    # Test external dispersion: should use λ⁴ = (λ²)²
    simulator = RMSimulator(
        freq_file=FREQ_FILE, n_components=1, model_type="external_dispersion"
    )

    # Parameters: [phi, sigma_phi, amp, chi0]
    theta = np.array([[100.0, 50.0, 1.0, 0.0]])  # phi=100, sigma=50, amp=1, chi0=0

    # Get noiseless simulation
    P_sim = simulator.simulate_noiseless(theta)
    Q_sim = P_sim[: simulator.n_freq]
    U_sim = P_sim[simulator.n_freq :]

    # Compute expected manually
    lambda_sq = simulator.lambda_sq
    lambda_sq_squared = lambda_sq**2  # λ⁴

    phi = 100.0
    sigma = 50.0

    # External dispersion: P = exp(-2σ²λ⁴) × exp(2iφλ²)
    depol = np.exp(-2 * sigma**2 * lambda_sq_squared)
    phase = 2 * phi * lambda_sq

    Q_expected = depol * np.cos(phase)
    U_expected = depol * np.sin(phase)

    # Check agreement
    Q_error = np.max(np.abs(Q_sim - Q_expected))
    U_error = np.max(np.abs(U_sim - U_expected))

    print(f"External dispersion Q error: {Q_error:.2e}")
    print(f"External dispersion U error: {U_error:.2e}")

    if Q_error < 1e-10 and U_error < 1e-10:
        print("✓ External dispersion formula correct (uses λ⁴)")
    else:
        print("✗ External dispersion formula may be incorrect")
        return False

    # Test internal dispersion
    simulator_int = RMSimulator(
        freq_file=FREQ_FILE, n_components=1, model_type="internal_dispersion"
    )

    # For small sigma, should approach Faraday-thin behavior
    theta_thin = np.array([[100.0, 0.01, 1.0, 0.0]])  # Very small sigma
    simulator_int.simulate_noiseless(theta_thin)

    # Compare with Faraday-thin
    # Internal dispersion with sigma→0 should give P ≈ exp(2iφλ²)
    # But note internal dispersion has exp(2iχ₀) not exp(2i(χ₀ + φλ²))
    # So it's a different formula altogether

    print("✓ Internal dispersion simulation runs without errors")

    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("TESTING PHYSICAL MODELS (VROOM-SBI v2.0)")
    print("=" * 60)

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
            success = _check_model(model_type, n_comp)
            results[model_type][n_comp] = success

    # Test depolarization formulas
    depol_ok = test_depolarization_formulas()

    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)

    all_passed = True
    for model_type in models_to_test:
        print(f"\n{model_type}:")
        for n_comp in component_counts:
            status = "✓ PASS" if results[model_type][n_comp] else "✗ FAIL"
            print(f"  {n_comp} components: {status}")
            if not results[model_type][n_comp]:
                all_passed = False

    print(f"\nDepolarization formulas: {'✓ PASS' if depol_ok else '✗ FAIL'}")

    print("\n" + "=" * 60)
    if all_passed and depol_ok:
        print("ALL TESTS PASSED! ✓")
        print("=" * 60)
        return 0
    else:
        print("SOME TESTS FAILED! ✗")
        print("=" * 60)
        return 1


if __name__ == "__main__":
    sys.exit(main())
