"""Tests for the circular chi0 reparameterization (simulator/chi0_reparam.py)."""

import math

import pytest
import torch

from src.simulator.chi0_reparam import (
    contract_theta,
    expand_bounds,
    expand_theta,
    flow_n_params,
)
from src.simulator.torch_physics import params_per_comp

MODEL_TYPES = [
    "faraday_thin",
    "burn_slab",
    "external_dispersion",
    "internal_dispersion",
]


def _random_theta(model_type: str, n_components: int, batch: int = 64):
    """Physical theta with chi0 in [0, pi) in every component's last slot."""
    ppc = params_per_comp(model_type)
    t = torch.randn(batch, n_components, ppc)
    t[:, :, -1] = torch.rand(batch, n_components) * math.pi
    return t.reshape(batch, n_components * ppc)


@pytest.mark.parametrize("model_type", MODEL_TYPES)
@pytest.mark.parametrize("n_components", [1, 3])
def test_expand_contract_roundtrip(model_type, n_components):
    theta = _random_theta(model_type, n_components)
    flow = expand_theta(theta, n_components, model_type)
    assert flow.shape == (theta.shape[0], flow_n_params(model_type, n_components))
    back = contract_theta(flow, n_components, model_type)
    torch.testing.assert_close(back, theta, atol=1e-6, rtol=1e-6)


def test_expand_is_continuous_across_wrap():
    """chi0 -> 0+ and chi0 -> pi- must map to nearby flow points."""
    ppc = params_per_comp("faraday_thin")
    eps = 1e-4
    lo = torch.zeros(1, ppc)
    hi = torch.zeros(1, ppc)
    lo[0, -1] = eps
    hi[0, -1] = math.pi - eps
    f_lo = expand_theta(lo, 1, "faraday_thin")
    f_hi = expand_theta(hi, 1, "faraday_thin")
    assert torch.norm(f_lo - f_hi) < 10 * eps


def test_contract_projects_off_circle_points():
    """Flow samples off the unit circle contract to the same angle as on it."""
    theta = _random_theta("faraday_thin", 1)
    flow = expand_theta(theta, 1, "faraday_thin")
    scaled = flow.clone()
    scaled[:, -2:] = scaled[:, -2:] * 3.7  # radial scaling, angle unchanged
    torch.testing.assert_close(
        contract_theta(scaled, 1, "faraday_thin"),
        contract_theta(flow, 1, "faraday_thin"),
        atol=1e-6,
        rtol=1e-6,
    )


def test_contract_range_is_zero_to_pi():
    flow = torch.randn(1000, flow_n_params("faraday_thin", 1))
    chi0 = contract_theta(flow, 1, "faraday_thin")[:, -1]
    assert (chi0 >= 0).all() and (chi0 < math.pi).all()


@pytest.mark.parametrize("model_type", MODEL_TYPES)
def test_expand_bounds_layout(model_type):
    ppc = params_per_comp(model_type)
    n_components = 2
    low = torch.arange(0, n_components * ppc, dtype=torch.float32)
    high = low + 10.0
    lo_f, hi_f = expand_bounds(low, high, n_components, model_type)
    assert lo_f.shape == (flow_n_params(model_type, n_components),)
    fppc = ppc + 1
    for c in range(n_components):
        # Non-chi0 slots pass through untouched.
        torch.testing.assert_close(
            lo_f[c * fppc : c * fppc + ppc - 1],
            low[c * ppc : c * ppc + ppc - 1],
        )
        # sin/cos slots are [-1, 1].
        assert (lo_f[c * fppc + ppc - 1 : c * fppc + ppc + 1] == -1.0).all()
        assert (hi_f[c * fppc + ppc - 1 : c * fppc + ppc + 1] == 1.0).all()


def test_multi_component_batched_shapes():
    """contract handles (n_samples, dim) sample tensors from posterior.sample."""
    theta = _random_theta("burn_slab", 4, batch=17)
    flow = expand_theta(theta, 4, "burn_slab")
    back = contract_theta(flow, 4, "burn_slab")
    assert back.shape == theta.shape
    torch.testing.assert_close(back, theta, atol=1e-6, rtol=1e-6)
