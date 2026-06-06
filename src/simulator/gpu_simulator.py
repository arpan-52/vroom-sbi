"""
GPU-resident forward simulator: prior -> physics -> augmentation -> noise.

``GPUSimulator.generate_batch(B)`` returns ``(theta, x)`` already on-device,
with ``x = [Q, U, weights]`` of width ``3 * n_freq`` — the same data contract
the chunked path writes to disk (``trainer._generate_simulation_chunks``,
continuous-weights branch). Nothing is serialized; this is the on-the-fly
source for ``training/online_trainer.py``.

The noise model matches the chunked continuous path exactly:
per-channel additive Gaussian with ``sigma_k = sigma_base / sqrt(w_k)``,
zero on flagged channels.
"""

import numpy as np
import torch

from ..config import Configuration
from .physics import freq_to_lambda_sq, load_frequencies
from .prior import _build_bounds_from_dict
from .torch_augmentation import augment_weights_continuous_batch
from .torch_physics import compute_polarization, params_per_comp
from .torch_prior import sample_prior


class GPUSimulator:
    """On-device generator of ``(theta, x=[Q,U,w])`` batches.

    Parameters
    ----------
    config : Configuration
        Full configuration (prior bounds, noise, weight augmentation).
    model_type : str
    n_components : int
    device : str
    sampling_method : {"uniform", "sobol"}
        Prior sampler. "uniform" is the default; "sobol" is the fallback.
    """

    def __init__(
        self,
        config: Configuration,
        model_type: str,
        n_components: int,
        device: str = "cuda",
        sampling_method: str = "uniform",
    ):
        self.config = config
        self.model_type = (
            "internal_dispersion" if model_type == "sokoloff" else model_type
        )
        self.n_components = n_components
        self.device = device
        self.sampling_method = sampling_method
        self.ppc = params_per_comp(self.model_type)
        self.n_params = self.ppc * n_components

        freq, base_w = load_frequencies(config.freq_file)
        self.n_freq = len(freq)
        lambda_sq = freq_to_lambda_sq(freq)
        self.lambda_sq = torch.tensor(lambda_sq, dtype=torch.float32, device=device)
        self.base_weights = torch.tensor(base_w, dtype=torch.float32, device=device)

        # Prior bounds — single source of truth via PriorConfig / dict helper
        if hasattr(config.priors, "get_bounds_for_model"):
            low, high = config.priors.get_bounds_for_model(
                self.model_type, n_components
            )
        else:
            flat = config.priors.to_flat_dict()
            low, high = _build_bounds_from_dict(flat, self.model_type, n_components)
        self.low = torch.tensor(np.asarray(low), dtype=torch.float32, device=device)
        self.high = torch.tensor(np.asarray(high), dtype=torch.float32, device=device)

    def _sigma_base(self, B: int, generator: torch.Generator | None) -> torch.Tensor:
        """Per-sample base noise sigma, matching the chunked continuous path."""
        noise = self.config.noise
        if noise.augmentation_enable:
            lo = noise.sigma_min * noise.augmentation_min_factor
            hi = noise.sigma_max * noise.augmentation_max_factor
            u = torch.rand(B, device=self.device, generator=generator)
            return lo + (hi - lo) * u
        mid = (noise.sigma_min + noise.sigma_max) / 2.0
        return torch.full((B,), mid, device=self.device)

    def generate_batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate ``(theta, x)`` on device. ``x`` width = ``3 * n_freq``."""
        B = batch_size
        wa = self.config.weight_augmentation

        theta = sample_prior(
            B,
            self.low,
            self.high,
            self.n_components,
            self.model_type,
            method=self.sampling_method,
            generator=generator,
        )

        weights = augment_weights_continuous_batch(
            self.base_weights,
            B,
            noise_ratio_min=getattr(wa, "noise_ratio_min", 2.0),
            noise_ratio_max=getattr(wa, "noise_ratio_max", 300.0),
            scattered_prob=wa.scattered_prob,
            gap_prob=wa.gap_prob,
            large_block_prob=wa.large_block_prob,
            generator=generator,
        )  # (B, F)

        good = weights > 0
        sigma_base = self._sigma_base(B, generator)  # (B,)
        sigma_per_chan = torch.where(
            good,
            sigma_base[:, None] / torch.sqrt(weights + 1e-12),
            torch.zeros_like(weights),
        )

        # Noiseless polarization, masked to good channels
        P = compute_polarization(
            theta, self.lambda_sq, self.n_components, self.model_type
        )  # (B, F) complex
        Q_nl = torch.where(good, P.real, torch.zeros_like(weights))
        U_nl = torch.where(good, P.imag, torch.zeros_like(weights))

        noise_Q = (
            torch.randn(B, self.n_freq, device=self.device, generator=generator)
            * sigma_per_chan
        )
        noise_U = (
            torch.randn(B, self.n_freq, device=self.device, generator=generator)
            * sigma_per_chan
        )
        Q_obs = torch.where(good, Q_nl + noise_Q, torch.zeros_like(weights))
        U_obs = torch.where(good, U_nl + noise_U, torch.zeros_like(weights))

        x = torch.cat([Q_obs, U_obs, weights], dim=1).to(torch.float32)
        return theta.to(torch.float32), x
