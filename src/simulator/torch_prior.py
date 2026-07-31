"""
GPU-resident prior sampling for VROOM-SBI.

Samples uniformly inside the prior box on-device and applies the RM-ordering
constraint (RM1/phi1 > RM2/phi2 > ...) to break label-switching symmetry.

Default sampling is plain per-batch uniform (``torch.rand``); a Sobol fallback
(``torch.quasirandom.SobolEngine``) is available via ``method="sobol"`` and
preserves the low-discrepancy coverage of the legacy NumPy path. SBC is the
arbiter of calibration regardless of sampler.

The ordering convention and bound layout mirror ``simulator/prior.py`` exactly.
"""

import torch

from .torch_physics import params_per_comp


def sort_components_by_rm(
    theta: torch.Tensor, n_components: int, ppc: int
) -> torch.Tensor:
    """Sort components so first param (RM/phi) is descending, keeping tuples intact.

    Vectorized replacement for the per-sample Python loop in ``prior.py``.

    Parameters
    ----------
    theta : torch.Tensor, shape (B, ppc * n_components)
    n_components : int
    ppc : int
        Parameters per component (3 or 4).
    """
    if n_components < 2:
        return theta
    B = theta.shape[0]
    t = theta.reshape(B, n_components, ppc)  # (B, C, ppc)
    keys = t[:, :, 0]  # (B, C)
    order = torch.argsort(keys, dim=1, descending=True)  # (B, C)
    idx = order[:, :, None].expand(-1, -1, ppc)  # (B, C, ppc)
    t_sorted = torch.gather(t, 1, idx)
    return t_sorted.reshape(B, n_components * ppc)


def sample_prior(
    n_samples: int,
    low: torch.Tensor,
    high: torch.Tensor,
    n_components: int,
    model_type: str,
    method: str = "uniform",
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Sample theta on the device of ``low``/``high`` with RM ordering.

    Parameters
    ----------
    n_samples : int
    low, high : torch.Tensor, shape (n_params,)
        Prior box bounds, already on the target device.
    n_components : int
    model_type : str
    method : {"uniform", "sobol"}
        "uniform" (default): plain per-batch ``torch.rand``.
        "sobol": low-discrepancy Sobol fallback.
    generator : torch.Generator, optional
        For reproducible sampling (used by parity tests and val-set generation).
    """
    device = low.device
    n_params = low.shape[0]
    ppc = params_per_comp(model_type)

    if method == "uniform":
        u = torch.rand(n_samples, n_params, device=device, generator=generator)
    elif method == "sobol":
        # SobolEngine runs on CPU; move the draw to the target device.
        seed = None
        if generator is not None:
            seed = int(torch.randint(0, 2**31 - 1, (1,), generator=generator).item())
        engine = torch.quasirandom.SobolEngine(
            dimension=n_params, scramble=True, seed=seed
        )
        u = engine.draw(n_samples).to(device)
    else:
        raise ValueError(f"Unknown sampling method: {method}")

    theta = low + u * (high - low)

    if n_components >= 2:
        theta = sort_components_by_rm(theta, n_components, ppc)

    return theta
