"""
GPU-resident Torch forward models for RM synthesis.

Direct Torch ports of the four NumPy polarization models in
``base_simulator.py``. Each function takes a parameter tensor ``theta`` of
shape ``(B, params_per_comp * n_components)`` and a ``lambda_sq`` tensor of
shape ``(F,)``, both on the same device, and returns the complex polarization
``P`` of shape ``(B, F)``.

Parity with the NumPy implementation is enforced by ``tests/test_torch_parity.py``.
The math mirrors ``RMSimulator._compute_polarization_*`` exactly.
"""

import torch

# Models with 3 params/component vs 4 params/component
_PARAMS_PER_COMP = {
    "faraday_thin": 3,
    "burn_slab": 4,
    "external_dispersion": 4,
    "internal_dispersion": 4,
}


def params_per_comp(model_type: str) -> int:
    """Number of parameters per component for a model type."""
    if model_type not in _PARAMS_PER_COMP:
        raise ValueError(f"Unknown model type: {model_type}")
    return _PARAMS_PER_COMP[model_type]


def _reshape(theta: torch.Tensor, n_components: int, ppc: int) -> torch.Tensor:
    """(B, ppc*C) -> (B, C, ppc)."""
    return theta.reshape(theta.shape[0], n_components, ppc)


def polarization_faraday_thin(
    theta: torch.Tensor, lambda_sq: torch.Tensor, n_components: int
) -> torch.Tensor:
    """P = Sum_j amp_j exp[2i(chi0_j + rm_j * lambda^2)]. Params: [RM, amp, chi0]."""
    t = _reshape(theta, n_components, 3)
    rm = t[:, :, 0]
    amp = t[:, :, 1]
    chi0 = t[:, :, 2]

    lsq = lambda_sq[None, None, :]  # (1, 1, F)
    phase = 2.0 * (chi0[:, :, None] + rm[:, :, None] * lsq)  # (B, C, F)
    return (amp[:, :, None] * torch.exp(1j * phase)).sum(dim=1)  # (B, F)


def polarization_burn_slab(
    theta: torch.Tensor, lambda_sq: torch.Tensor, n_components: int
) -> torch.Tensor:
    """P = amp * sinc(delta_phi * lambda^2) * exp[2i(chi0 + phi_c * lambda^2)].

    sinc(x) = sin(x)/x. Params: [phi_c, delta_phi, amp, chi0].
    """
    t = _reshape(theta, n_components, 4)
    phi_c = t[:, :, 0]
    delta_phi = t[:, :, 1]
    amp = t[:, :, 2]
    chi0 = t[:, :, 3]

    lsq = lambda_sq[None, None, :]
    arg = delta_phi[:, :, None] * lsq
    sinc_term = torch.where(
        arg.abs() < 1e-10,
        torch.ones_like(arg),
        torch.sin(arg) / arg,
    )
    phase = 2.0 * (chi0[:, :, None] + phi_c[:, :, None] * lsq)
    return (amp[:, :, None] * sinc_term * torch.exp(1j * phase)).sum(dim=1)


def polarization_external_dispersion(
    theta: torch.Tensor, lambda_sq: torch.Tensor, n_components: int
) -> torch.Tensor:
    """P = amp * exp(-2 sigma_phi^2 lambda^4) * exp[2i(chi0 + phi lambda^2)].

    Params: [phi, sigma_phi, amp, chi0].
    """
    t = _reshape(theta, n_components, 4)
    phi = t[:, :, 0]
    sigma_phi = t[:, :, 1]
    amp = t[:, :, 2]
    chi0 = t[:, :, 3]

    lsq = lambda_sq[None, None, :]
    lsq4 = lsq**2
    depol = torch.exp(-2.0 * sigma_phi[:, :, None] ** 2 * lsq4)
    phase = 2.0 * (chi0[:, :, None] + phi[:, :, None] * lsq)
    return (amp[:, :, None] * depol * torch.exp(1j * phase)).sum(dim=1)


def polarization_internal_dispersion(
    theta: torch.Tensor, lambda_sq: torch.Tensor, n_components: int
) -> torch.Tensor:
    """Sokoloff internal dispersion.

    P = amp * [(1 - exp(-S)) / S] * exp(2i chi0),  S = 2 sigma_phi^2 lambda^4 - 2i phi lambda^2.
    Params: [phi, sigma_phi, amp, chi0].
    """
    t = _reshape(theta, n_components, 4)
    phi = t[:, :, 0]
    sigma_phi = t[:, :, 1]
    amp = t[:, :, 2]
    chi0 = t[:, :, 3]

    lsq = lambda_sq[None, None, :]
    lsq4 = lsq**2
    S = 2.0 * sigma_phi[:, :, None] ** 2 * lsq4 - 2j * phi[:, :, None] * lsq
    depol = torch.where(
        S.abs() < 1e-10,
        torch.ones_like(S),
        (1 - torch.exp(-S)) / S,
    )
    return (amp[:, :, None] * depol * torch.exp(2j * chi0[:, :, None])).sum(dim=1)


_DISPATCH = {
    "faraday_thin": polarization_faraday_thin,
    "burn_slab": polarization_burn_slab,
    "external_dispersion": polarization_external_dispersion,
    "internal_dispersion": polarization_internal_dispersion,
}


def compute_polarization(
    theta: torch.Tensor,
    lambda_sq: torch.Tensor,
    n_components: int,
    model_type: str,
) -> torch.Tensor:
    """Dispatch to the requested model. Returns complex P of shape (B, F)."""
    mt = "internal_dispersion" if model_type == "sokoloff" else model_type
    if mt not in _DISPATCH:
        raise ValueError(f"Unknown model type: {model_type}")
    P = _DISPATCH[mt](theta, lambda_sq, n_components)
    return torch.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)
