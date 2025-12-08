"""
Tests for RM component recovery.

Tests that the simulator and inference pipeline can correctly recover
known RM components from synthetic observations.
"""

import pytest
import numpy as np
import torch
import yaml
import os
import tempfile
import shutil

from src.simulator import RMSimulator, build_prior, sample_prior
from src.train import train_model
from src.inference import RMInference


@pytest.fixture(scope="module")
def freq_file():
    """Create a temporary frequency file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
        # Use the standard frequency channels
        freqs = [908e6, 952e6, 996e6, 1044e6, 1093e6, 1145e6, 
                1318e6, 1382e6, 1448e6, 1482e6, 1594e6, 1656e6]
        for freq in freqs:
            f.write(f"{freq}\n")
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)


@pytest.fixture(scope="module")
def model_dir():
    """Create a temporary directory for models."""
    temp_dir = tempfile.mkdtemp()
    yield temp_dir
    # Cleanup
    shutil.rmtree(temp_dir)


@pytest.fixture(scope="module")
def config():
    """Test configuration."""
    return {
        'priors': {
            'rm': {'min': -500.0, 'max': 500.0},
            'amp': {'min': 0.0, 'max': 1.0},
        },
        'noise': {
            'base_level': 0.01
        },
        'training': {
            'batch_size': 50,
            'validation_fraction': 0.1
        }
    }


def test_single_component_recovery(freq_file, model_dir, config):
    """
    Test recovery of a single RM component.
    
    Create synthetic data with known parameters, train a model,
    and verify the parameters can be recovered.
    """
    print("\n" + "="*60)
    print("Testing single component recovery")
    print("="*60)
    
    # True parameters
    true_rm = 100.0
    true_q = 0.5
    true_u = 0.3
    base_noise_level = 0.01
    
    # Create simulator
    simulator = RMSimulator(freq_file, n_components=1, base_noise_level=base_noise_level)
    
    # Generate observation
    theta_true = torch.tensor([true_rm, true_q, true_u], dtype=torch.float32)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    qu_obs = simulator(theta_true)
    qu_obs_np = qu_obs.numpy()
    
    print(f"\nTrue parameters:")
    print(f"  RM = {true_rm} rad/m²")
    print(f"  q = {true_q}")
    print(f"  u = {true_u}")
    print(f"  base_noise_level = {base_noise_level}")
    
    # Train model with fewer simulations for speed
    print("\nTraining model (this may take a few minutes)...")
    device = 'cpu'  # Use CPU for testing to avoid GPU requirements
    posterior = train_model(
        freq_file=freq_file,
        n_components=1,
        n_simulations=1000,  # Reduced for testing speed
        config=config,
        device=device
    )
    
    # Save model
    model_path = os.path.join(model_dir, 'model_n1.pkl')
    torch.save(posterior, model_path)
    
    # Run inference
    print("\nRunning inference...")
    inference = RMInference(model_dir=model_dir, device=device)
    inference.load_models(max_components=1)
    
    # Get posterior samples
    qu_obs_torch = torch.tensor(qu_obs_np, dtype=torch.float32)
    samples = posterior.sample((1000,), x=qu_obs_torch)
    samples_np = samples.cpu().numpy()
    
    # Extract parameters
    rm_recovered = np.mean(samples_np[:, 0])
    q_recovered = np.mean(samples_np[:, 1])
    u_recovered = np.mean(samples_np[:, 2])
    
    print(f"\nRecovered parameters:")
    print(f"  RM = {rm_recovered:.1f} rad/m² (true: {true_rm})")
    print(f"  q = {q_recovered:.3f} (true: {true_q})")
    print(f"  u = {u_recovered:.3f} (true: {true_u})")
    
    # Check recovery (allow for some tolerance due to noise and finite training)
    print("\nVerifying recovery...")
    assert abs(rm_recovered - true_rm) < 50, f"RM not recovered well: {rm_recovered} vs {true_rm}"
    assert abs(q_recovered - true_q) < 0.2, f"q not recovered well: {q_recovered} vs {true_q}"
    assert abs(u_recovered - true_u) < 0.2, f"u not recovered well: {u_recovered} vs {true_u}"
    
    print("✓ Single component recovery test passed!")


def test_two_component_recovery(freq_file, model_dir, config):
    """
    Test recovery of two RM components.
    
    Create synthetic data with two known components and verify
    they can be recovered.
    """
    print("\n" + "="*60)
    print("Testing two component recovery")
    print("="*60)
    
    # True parameters: [RM1, amp1, chi01, RM2, amp2, chi02]
    true_rm1 = -200.0
    true_amp1 = 0.4
    true_chi01 = 0.5
    true_rm2 = 150.0
    true_amp2 = 0.3
    true_chi02 = 1.2
    base_noise_level = 0.01
    
    # Create simulator
    simulator = RMSimulator(freq_file, n_components=2, base_noise_level=base_noise_level)
    
    # Generate observation
    theta_true = torch.tensor([
        true_rm1, true_amp1, true_chi01,
        true_rm2, true_amp2, true_chi02
    ], dtype=torch.float32)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    qu_obs = simulator(theta_true)
    qu_obs_np = qu_obs.numpy()
    
    print(f"\nTrue parameters:")
    print(f"  Component 1: RM = {true_rm1} rad/m², amp = {true_amp1}, chi0 = {true_chi01}")
    print(f"  Component 2: RM = {true_rm2} rad/m², amp = {true_amp2}, chi0 = {true_chi02}")
    print(f"  Base noise level = {base_noise_level}")
    
    # Train model
    print("\nTraining model (this may take a few minutes)...")
    device = 'cpu'
    posterior = train_model(
        freq_file=freq_file,
        n_components=2,
        n_simulations=1000,  # Reduced for testing speed
        config=config,
        device=device
    )
    
    # Save model
    model_path = os.path.join(model_dir, 'model_n2.pkl')
    torch.save(posterior, model_path)
    
    # Run inference
    print("\nRunning inference...")
    qu_obs_torch = torch.tensor(qu_obs_np, dtype=torch.float32)
    samples = posterior.sample((1000,), x=qu_obs_torch)
    samples_np = samples.cpu().numpy()
    
    # Extract parameters
    rm1_recovered = np.mean(samples_np[:, 0])
    rm2_recovered = np.mean(samples_np[:, 1])
    
    print(f"\nRecovered RM values:")
    print(f"  RM1 = {rm1_recovered:.1f} rad/m²")
    print(f"  RM2 = {rm2_recovered:.1f} rad/m²")
    
    # Check that we recovered two distinct RM values
    # Note: order may be swapped, so check both possibilities
    rm_true_set = {true_rm1, true_rm2}
    rm_recovered_set = {rm1_recovered, rm2_recovered}
    
    # Find closest matches
    print("\nVerifying recovery...")
    min_dist = float('inf')
    for true_rm in rm_true_set:
        for recovered_rm in rm_recovered_set:
            dist = abs(true_rm - recovered_rm)
            min_dist = min(min_dist, dist)
    
    # Allow for more tolerance with two components due to increased difficulty
    assert min_dist < 100, f"RM components not recovered well: min distance = {min_dist}"
    
    print("✓ Two component recovery test passed!")


def test_simulator_forward_model(freq_file):
    """Test that the forward model produces expected output shapes."""
    print("\n" + "="*60)
    print("Testing simulator forward model")
    print("="*60)
    
    for n_components in [1, 2, 3]:
        simulator = RMSimulator(freq_file, n_components=n_components)
        
        # Create random parameters (no noise parameter)
        n_params = 3 * n_components
        theta = torch.randn(n_params)
        
        # Run forward model
        qu = simulator(theta)
        
        # Check output shape
        expected_shape = simulator.n_freq * 2
        assert len(qu) == expected_shape, \
            f"Expected output shape {expected_shape}, got {len(qu)}"
        
        print(f"✓ Forward model for {n_components} component(s): shape = {qu.shape}")
    
    print("✓ Simulator forward model test passed!")


def test_prior_sampling():
    """Test prior construction and sampling."""
    print("\n" + "="*60)
    print("Testing prior sampling")
    print("="*60)
    
    flat_priors = {
        "rm_min": -500.0,
        "rm_max": 500.0,
        "amp_min": 0.01,
        "amp_max": 1.0,
    }
    
    for n_components in [1, 2, 3]:
        # Sample from prior
        samples = sample_prior(n_samples=100, n_components=n_components, config=flat_priors)
        
        # Check shape (no noise parameter)
        expected_shape = (100, 3 * n_components)
        assert samples.shape == expected_shape, \
            f"Expected shape {expected_shape}, got {samples.shape}"
        
        # Check ranges (rough check)
        rm_samples = samples[:, ::3]  # Every third column starting from 0
        assert np.all(rm_samples >= -500) and np.all(rm_samples <= 500), \
            "RM samples outside expected range"
        
        print(f"✓ Prior for {n_components} component(s): shape = {samples.shape}")
    
    print("✓ Prior sampling test passed!")


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v', '-s'])
