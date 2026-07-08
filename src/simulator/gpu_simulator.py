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
from .torch_augmentation import (
    apply_rfi_flagging_batch,
    augment_weights_continuous_batch,
)
from .torch_physics import compute_polarization, params_per_comp, spectral_shape_flux
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

        # Third channel. Default: relative weights (network must infer the
        # absolute noise scale -> conservative, under-confident posteriors).
        # condition_on_noise: expose the *absolute* per-channel noise as masked
        # log-precision so the posterior conditions on the (known-at-inference)
        # noise instead of marginalizing the 200x sigma_base range. Shifted
        # strictly positive so 0 uniquely marks a flagged channel, /4 -> O(1).
        if getattr(self.config.noise, "condition_on_noise", False):
            log_prec = -torch.log10(sigma_per_chan + 1e-30)  # high where noise low
            chan3 = torch.where(good, (log_prec + 1.0) / 4.0, torch.zeros_like(weights))
        else:
            chan3 = weights

        x = torch.cat([Q_obs, U_obs, chan3], dim=1).to(torch.float32)
        return theta.to(torch.float32), x


class GPUSpectralSimulator:
    """On-device generator of (theta, x) batches for the spectral-shape model.

    theta = [alpha, beta, gamma]; x = F(nu) of width ``n_freq`` (real flux),
    with flagged channels zeroed. Matches the chunked spectral data contract
    (``SpectralShapeTrainer._generate_chunks``): F(nu0)=1 normalization,
    binary RFI flagging mask, additive Gaussian noise.

    Noise note: the chunked path draws one sigma per *chunk*; here we draw one
    per *sample*, which removes within-batch theta/noise correlation (the pol
    path already does per-sample). SBC remains the calibration arbiter.
    """

    def __init__(
        self,
        config: Configuration,
        device: str = "cuda",
        sampling_method: str = "uniform",
    ):
        self.config = config
        self.device = device
        self.sampling_method = sampling_method
        self.n_params = 3  # alpha, beta, gamma

        freq, base_w = load_frequencies(config.freq_file)
        self.n_freq = len(freq)
        nu0 = freq[len(freq) // 2]
        log_nu_ratio = np.log(freq / nu0)
        self.log_nu_ratio = torch.tensor(
            log_nu_ratio, dtype=torch.float32, device=device
        )
        self.base_weights = torch.tensor(base_w, dtype=torch.float32, device=device)

        ss = config.spectral_shape
        self.low = torch.tensor(
            [ss.alpha_min, ss.beta_min, ss.gamma_min],
            dtype=torch.float32,
            device=device,
        )
        self.high = torch.tensor(
            [ss.alpha_max, ss.beta_max, ss.gamma_max],
            dtype=torch.float32,
            device=device,
        )

    def _sample_theta(self, B, generator):
        if self.sampling_method == "sobol":
            seed = None
            if generator is not None:
                seed = int(
                    torch.randint(0, 2**31 - 1, (1,), generator=generator).item()
                )
            engine = torch.quasirandom.SobolEngine(
                dimension=self.n_params, scramble=True, seed=seed
            )
            u = engine.draw(B).to(self.device)
        else:
            u = torch.rand(B, self.n_params, device=self.device, generator=generator)
        return self.low + u * (self.high - self.low)

    def _noise_sigma(self, B, generator):
        ss = self.config.spectral_shape
        noise = self.config.noise
        if noise.augmentation_enable:
            lo = ss.sigma_min * noise.augmentation_min_factor
            hi = ss.sigma_max * noise.augmentation_max_factor
            u = torch.rand(B, device=self.device, generator=generator)
            return lo + (hi - lo) * u
        mid = (ss.sigma_min + ss.sigma_max) / 2.0
        return torch.full((B,), mid, device=self.device)

    def generate_batch(
        self, batch_size: int, generator: torch.Generator | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Generate (theta, x) on device. x width = n_freq."""
        B = batch_size
        wa = self.config.weight_augmentation

        theta = self._sample_theta(B, generator)

        weights = apply_rfi_flagging_batch(
            self.base_weights,
            B,
            scattered_prob=wa.scattered_prob,
            gap_prob=wa.gap_prob,
            large_block_prob=wa.large_block_prob,
            generator=generator,
        )
        good = weights > 0

        F_nl = spectral_shape_flux(theta, self.log_nu_ratio)  # (B, F)
        sigma = self._noise_sigma(B, generator)[:, None]  # (B, 1)
        noise = torch.randn(
            B, self.n_freq, device=self.device, generator=generator
        ) * sigma
        x = torch.where(good, F_nl + noise, torch.zeros_like(F_nl))

        return theta.to(torch.float32), x.to(torch.float32)
