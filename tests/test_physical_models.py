"""
Tests for physical simulators, prior sampling, and RM sorting.

Covers:
- All four RM model types x component counts: instantiation, output shape, NaN/Inf
- RM sorting (label-switching fix) for multi-component priors
- Sobol quasi-random prior sampling: bounds compliance and coverage
- Depolarization formula correctness (external and internal dispersion)
- Spectral shape simulator: 3-param model, F(nu0)=1 normalization
- Spectral shape prior: Sobol sampling, correct parameter names
"""

from pathlib import Path

import numpy as np
import pytest

from src.config.configuration import SpectralShapeConfig
from src.simulator import RMSimulator, sample_prior
from src.simulator.prior import sample_spectral_shape_prior
from src.simulator.spectral_simulator import SpectralShapeSimulator

FREQ_FILE = str(Path(__file__).parent.parent / "freq.txt")

ALL_MODELS = ["faraday_thin", "burn_slab", "external_dispersion", "internal_dispersion"]
COMPONENT_COUNTS = [1, 2, 3, 5]

PRIOR_CONFIG = {"rm_min": -500.0, "rm_max": 500.0, "amp_min": 0.01, "amp_max": 1.0}


# ---------------------------------------------------------------------------
# 1. Simulator instantiation and output shape
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("n_components", COMPONENT_COUNTS)
def test_simulator_output_shape(model_type, n_components):
    sim = RMSimulator(freq_file=FREQ_FILE, n_components=n_components, model_type=model_type)
    theta = sample_prior(8, n_components, PRIOR_CONFIG, model_type=model_type)
    assert theta.shape == (8, sim.n_params)
    out = sim(theta)
    assert out.shape == (8, 2 * sim.n_freq)


# ---------------------------------------------------------------------------
# 2. No NaN or Inf in simulated output
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("n_components", COMPONENT_COUNTS)
def test_simulator_no_nan_inf(model_type, n_components):
    sim = RMSimulator(freq_file=FREQ_FILE, n_components=n_components, model_type=model_type)
    theta = sample_prior(20, n_components, PRIOR_CONFIG, model_type=model_type)
    out = sim(theta)
    assert np.all(np.isfinite(out)), f"NaN/Inf in {model_type} N={n_components}"


# ---------------------------------------------------------------------------
# 3. RM sorting: multi-component priors must have RM1 >= RM2 >= ...
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ALL_MODELS)
@pytest.mark.parametrize("n_components", [2, 3, 5])
def test_prior_rm_sorted_descending(model_type, n_components):
    theta = sample_prior(200, n_components, PRIOR_CONFIG, model_type=model_type)
    params_per_comp = theta.shape[1] // n_components
    rm_cols = theta[:, [i * params_per_comp for i in range(n_components)]]
    # Each row: RM values should be non-increasing left to right
    assert np.all(np.diff(rm_cols, axis=1) <= 0), (
        f"RM not sorted descending for {model_type} N={n_components}"
    )


# ---------------------------------------------------------------------------
# 4. Sobol prior sampling: samples stay within declared bounds
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("model_type", ALL_MODELS)
def test_sobol_prior_within_bounds(model_type):
    from src.simulator.prior import _build_bounds_from_dict
    n = 512
    theta = sample_prior(n, 1, PRIOR_CONFIG, model_type=model_type)
    low, high = _build_bounds_from_dict(PRIOR_CONFIG, model_type, 1)
    assert np.all(theta >= low - 1e-9), "Sobol samples below lower bound"
    assert np.all(theta <= high + 1e-9), "Sobol samples above upper bound"


def test_sobol_prior_covers_space_better_than_uniform():
    """Sobol discrepancy should be lower than uniform random at n=256."""
    from scipy.stats import qmc
    n, d = 256, 3
    theta_sobol = sample_prior(n, 1, PRIOR_CONFIG, model_type="faraday_thin")
    # Normalise to [0,1]^d for discrepancy calculation
    low = np.array([PRIOR_CONFIG["rm_min"], PRIOR_CONFIG["amp_min"], 0.0])
    high = np.array([PRIOR_CONFIG["rm_max"], PRIOR_CONFIG["amp_max"], np.pi])
    u_sobol = (theta_sobol - low) / (high - low)
    rng = np.random.default_rng(0)
    u_uniform = rng.uniform(size=(n, d))
    disc_sobol = qmc.discrepancy(u_sobol)
    disc_uniform = qmc.discrepancy(u_uniform)
    assert disc_sobol < disc_uniform, (
        f"Sobol discrepancy {disc_sobol:.4f} not better than uniform {disc_uniform:.4f}"
    )


# ---------------------------------------------------------------------------
# 5. Depolarization formula correctness
# ---------------------------------------------------------------------------

def test_external_dispersion_formula():
    """P = amp * exp(-2 sigma^2 lambda^4) * exp(2i (chi0 + phi lambda^2))"""
    sim = RMSimulator(freq_file=FREQ_FILE, n_components=1, model_type="external_dispersion")
    phi, sigma, amp, chi0 = 100.0, 50.0, 1.0, 0.0
    theta = np.array([[phi, sigma, amp, chi0]])
    P_sim = sim.simulate_noiseless(theta)
    Q_sim, U_sim = P_sim[:sim.n_freq], P_sim[sim.n_freq:]

    lsq = sim.lambda_sq
    depol = np.exp(-2.0 * sigma**2 * lsq**2)
    phase = 2.0 * (chi0 + phi * lsq)
    assert np.max(np.abs(Q_sim - depol * np.cos(phase))) < 1e-10
    assert np.max(np.abs(U_sim - depol * np.sin(phase))) < 1e-10


def test_internal_dispersion_limit_no_nan():
    """Sokoloff model with near-zero sigma should produce finite output."""
    sim = RMSimulator(freq_file=FREQ_FILE, n_components=1, model_type="internal_dispersion")
    theta = np.array([[100.0, 1e-12, 1.0, 0.0]])  # sigma_phi -> 0
    out = sim.simulate_noiseless(theta)
    assert np.all(np.isfinite(out)), "Internal dispersion blows up near sigma=0"


def test_faraday_thin_noiseless_amplitude():
    """Single component faraday_thin: |P| should equal amp at all frequencies."""
    sim = RMSimulator(freq_file=FREQ_FILE, n_components=1, model_type="faraday_thin")
    amp = 0.7
    theta = np.array([[200.0, amp, 0.5]])
    out = sim.simulate_noiseless(theta)
    Q, U = out[:sim.n_freq], out[sim.n_freq:]
    pol_intensity = np.sqrt(Q**2 + U**2)
    np.testing.assert_allclose(pol_intensity, amp, atol=1e-10)


# ---------------------------------------------------------------------------
# 6. Spectral shape simulator: 3-param model, F(nu0) = 1
# ---------------------------------------------------------------------------

def test_spectral_shape_param_count():
    sim = SpectralShapeSimulator(FREQ_FILE)
    assert sim.n_params == 3
    assert sim.get_param_names() == ["alpha", "beta", "gamma"]


def test_spectral_shape_f_nu0_equals_one():
    """Noiseless output at the reference channel must equal 1 for any theta."""
    sim = SpectralShapeSimulator(FREQ_FILE)
    rng = np.random.default_rng(42)
    theta = rng.uniform([-3, -1, -0.5], [1, 1, 0.5], size=(50, 3))
    out = sim.simulate_noiseless(theta)  # (50, n_freq) — F values
    mid = sim.mid_idx
    np.testing.assert_allclose(out[:, mid], 1.0, atol=1e-10,
                               err_msg="F(nu0) != 1 for spectral shape model")


def test_spectral_shape_output_shape():
    sim = SpectralShapeSimulator(FREQ_FILE)
    rng = np.random.default_rng(0)
    theta = rng.uniform([-1, -0.5, -0.1], [0.5, 0.5, 0.1], size=(12, 3))
    out = sim.simulate_noiseless(theta)
    assert out.shape == (12, sim.n_freq)
    assert np.all(np.isfinite(out))


def test_spectral_shape_prior_sobol_bounds():
    config = SpectralShapeConfig()
    samples = sample_spectral_shape_prior(256, config)
    assert samples.shape == (256, 3)
    assert np.all(samples[:, 0] >= config.alpha_min)
    assert np.all(samples[:, 0] <= config.alpha_max)
    assert np.all(samples[:, 1] >= config.beta_min)
    assert np.all(samples[:, 1] <= config.beta_max)
    assert np.all(samples[:, 2] >= config.gamma_min)
    assert np.all(samples[:, 2] <= config.gamma_max)


# ---------------------------------------------------------------------------
# RM sorting API
# ---------------------------------------------------------------------------


def test_sort_components_by_rm_is_callable():
    from src.simulator.prior import sort_components_by_rm

    assert callable(sort_components_by_rm)


def test_sort_posterior_samples_wrapper_removed():
    import src.simulator.prior as prior_mod

    assert not hasattr(prior_mod, "sort_posterior_samples")
