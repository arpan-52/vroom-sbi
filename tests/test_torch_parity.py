"""
Parity tests for the GPU on-the-fly path against the legacy NumPy path.

- Physics: Torch ``compute_polarization`` must match NumPy
  ``RMSimulator.simulate_noiseless`` to float32 tolerance (exact-math parity).
- RM sorting: Torch vectorized sort must match the NumPy per-sample loop exactly.
- Augmentation: distributional/structural parity only (different RNGs).
- GPUSimulator: data contract (shapes, finiteness, [Q,U,w] layout).
"""

from pathlib import Path

import numpy as np
import pytest
import torch

from src.simulator import RMSimulator, sample_prior
from src.simulator.physics import freq_to_lambda_sq, load_frequencies
from src.simulator.prior import sort_components_by_rm as np_sort
from src.simulator.torch_augmentation import augment_weights_continuous_batch
from src.simulator.torch_physics import compute_polarization, params_per_comp
from src.simulator.torch_prior import sort_components_by_rm as torch_sort

FREQ_FILE = str(Path(__file__).parent.parent / "freq.txt")
ALL_MODELS = ["faraday_thin", "burn_slab", "external_dispersion", "internal_dispersion"]
COMPONENT_COUNTS = [1, 2, 3]
PRIOR_CONFIG = {
    "rm_min": -500.0, "rm_max": 500.0,
    "amp_min": 0.01, "amp_max": 1.0,
    "sigma_phi_min": 0.0, "sigma_phi_max": 50.0,
    "delta_phi_min": 0.0, "delta_phi_max": 50.0,
}


@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("n_components", COMPONENT_COUNTS)
def test_physics_parity(model_type, n_components):
    sim = RMSimulator(FREQ_FILE, n_components, model_type)
    theta = sample_prior(64, n_components, PRIOR_CONFIG, model_type=model_type)

    qu_np = sim.simulate_noiseless(theta)  # (64, 2*n_freq)
    n_freq = sim.n_freq
    Q_np, U_np = qu_np[:, :n_freq], qu_np[:, n_freq:]

    lsq = torch.tensor(freq_to_lambda_sq(sim.freq), dtype=torch.float32)
    theta_t = torch.tensor(theta, dtype=torch.float32)
    P = compute_polarization(theta_t, lsq, n_components, model_type)
    Q_t, U_t = P.real.numpy(), P.imag.numpy()

    np.testing.assert_allclose(Q_t, Q_np, rtol=1e-4, atol=1e-5)
    np.testing.assert_allclose(U_t, U_np, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("n_components", [2, 3])
def test_rm_sort_parity(model_type, n_components):
    ppc = params_per_comp(model_type)
    rng = np.random.default_rng(0)
    theta = rng.uniform(-100, 100, size=(50, ppc * n_components)).astype(np.float32)

    sorted_np = np_sort(theta.copy(), n_components, ppc)
    sorted_t = torch_sort(
        torch.tensor(theta), n_components, ppc
    ).numpy()

    np.testing.assert_allclose(sorted_t, sorted_np, rtol=0, atol=1e-6)


def test_augmentation_structure():
    _, base_w = load_frequencies(FREQ_FILE)
    base = torch.tensor(base_w, dtype=torch.float32)
    w = augment_weights_continuous_batch(base, 256)

    assert w.shape == (256, len(base_w))
    assert torch.all(w >= 0) and torch.all(w <= 1.0)
    # Channels flagged in the base must stay zero
    base_zero = base == 0
    assert torch.all(w[:, base_zero] == 0)
    # Some augmentation actually happened (not all rows identical to base)
    assert not torch.allclose(w, base[None, :].expand_as(w))


def test_gpu_simulator_contract():
    from src.config import Configuration

    cfg = Configuration.from_yaml(
        str(Path(__file__).parent.parent / "config.yaml")
    )
    cfg.training.device = "cpu"
    from src.simulator.gpu_simulator import GPUSimulator

    sim = GPUSimulator(cfg, "faraday_thin", 2, device="cpu")
    theta, x = sim.generate_batch(32)

    assert theta.shape == (32, 2 * 3)
    assert x.shape == (32, 3 * sim.n_freq)
    assert torch.all(torch.isfinite(theta))
    assert torch.all(torch.isfinite(x))
