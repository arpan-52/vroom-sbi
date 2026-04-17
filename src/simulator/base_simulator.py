"""
Forward simulator for RM synthesis.

Generates synthetic Q, U spectra from physical parameters.

Supports multiple physical models:
- faraday_thin: Simple external Faraday rotation (no depolarization)
- burn_slab: Burn slab depolarization (sinc function)
- external_dispersion: External Faraday dispersion (turbulent foreground)
- internal_dispersion: Internal Faraday dispersion (Sokoloff model)

FIXED: The external and internal dispersion models now use correct formulas.
"""

import warnings

import numpy as np

from ..core.base_classes import BaseSimulator
from .physics import freq_to_lambda_sq, load_frequencies


class RMSimulator(BaseSimulator):
    """
    Simulator for N-component RM models with various physical models.

    Each component has parameters depending on model type:
    - faraday_thin: [RM, amp, chi0] → 3 params/component
    - burn_slab: [phi_c, delta_phi, amp, chi0] → 4 params/component
    - external_dispersion: [phi, sigma_phi, amp, chi0] → 4 params/component
    - internal_dispersion: [phi, sigma_phi, amp, chi0] → 4 params/component

    The amplitude 'amp' is the intrinsic polarized intensity.
    chi0 is the intrinsic polarization angle.

    Noise is percentage-based: sigma = noise_percent/100 * |P|

    Parameters
    ----------
    freq_file : str
        Path to frequency file
    n_components : int
        Number of RM components
    model_type : str
        Physical model type
    """

    VALID_MODEL_TYPES = [
        "faraday_thin",
        "burn_slab",
        "external_dispersion",
        "internal_dispersion",
        "sokoloff",  # Alias for internal_dispersion
    ]

    def __init__(
        self,
        freq_file: str,
        n_components: int,
        model_type: str = "faraday_thin",
    ):
        self.freq, self._weights = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.freq)
        self._n_freq = len(self.freq)
        self.n_components = n_components
        self.model_type = model_type.lower()

        # Validate model type
        if self.model_type not in self.VALID_MODEL_TYPES:
            raise ValueError(
                f"Unknown model type: {model_type}. "
                f"Valid types: {self.VALID_MODEL_TYPES}"
            )

        # Handle sokoloff alias
        if self.model_type == "sokoloff":
            self.model_type = "internal_dispersion"

        # Determine parameters per component
        if self.model_type == "faraday_thin":
            self._params_per_comp = 3  # [RM, amp, chi0]
        else:
            self._params_per_comp = 4  # [phi, sigma/delta, amp, chi0]

        self._n_params = self._params_per_comp * n_components

    @property
    def n_params(self) -> int:
        return self._n_params

    @property
    def n_freq(self) -> int:
        return self._n_freq

    @property
    def params_per_comp(self) -> int:
        return self._params_per_comp

    @property
    def weights(self) -> np.ndarray:
        return self._weights

    def _compute_polarization_faraday_thin(self, theta: np.ndarray) -> np.ndarray:
        """
        Faraday-thin model: P = Σⱼ pⱼ exp[2i(χ₀,ⱼ + φⱼλ²)]

        Vectorized over (batch, component, freq).
        Parameters: [RM, amp, chi0] per component (3N total)
        """
        batch_size = theta.shape[0]
        theta_r = theta.reshape(batch_size, self.n_components, 3)
        rm = theta_r[:, :, 0]    # (B, C)
        amp = theta_r[:, :, 1]
        chi0 = theta_r[:, :, 2]

        lsq = self.lambda_sq[None, None, :]  # (1, 1, F)
        phase = 2.0 * (chi0[:, :, None] + rm[:, :, None] * lsq)  # (B, C, F)
        return (amp[:, :, None] * np.exp(1j * phase)).sum(axis=1)  # (B, F)

    def _compute_polarization_burn_slab(self, theta: np.ndarray) -> np.ndarray:
        """
        Burn slab model: P = p₀ × sinc(Δφ × λ²) × exp[2i(χ₀ + φ_c × λ²)]
        where sinc(x) = sin(x)/x.

        Vectorized over (batch, component, freq).
        Parameters: [phi_c, delta_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        theta_r = theta.reshape(batch_size, self.n_components, 4)
        phi_c = theta_r[:, :, 0]
        delta_phi = theta_r[:, :, 1]
        amp = theta_r[:, :, 2]
        chi0 = theta_r[:, :, 3]

        lsq = self.lambda_sq[None, None, :]  # (1, 1, F)
        arg = delta_phi[:, :, None] * lsq    # (B, C, F)
        with np.errstate(divide="ignore", invalid="ignore"):
            sinc_term = np.where(np.abs(arg) < 1e-10, 1.0, np.sin(arg) / arg)

        phase = 2.0 * (chi0[:, :, None] + phi_c[:, :, None] * lsq)
        return (amp[:, :, None] * sinc_term * np.exp(1j * phase)).sum(axis=1)

    def _compute_polarization_external_dispersion(
        self, theta: np.ndarray
    ) -> np.ndarray:
        """
        External Faraday dispersion: P = p₀ × exp(-2σ_φ² × λ⁴) × exp[2i(χ₀ + φλ²)].

        Vectorized over (batch, component, freq).
        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        theta_r = theta.reshape(batch_size, self.n_components, 4)
        phi = theta_r[:, :, 0]
        sigma_phi = theta_r[:, :, 1]
        amp = theta_r[:, :, 2]
        chi0 = theta_r[:, :, 3]

        lsq = self.lambda_sq[None, None, :]  # (1, 1, F)
        lsq4 = lsq**2  # λ⁴
        depol = np.exp(-2.0 * sigma_phi[:, :, None] ** 2 * lsq4)  # (B, C, F)
        phase = 2.0 * (chi0[:, :, None] + phi[:, :, None] * lsq)
        return (amp[:, :, None] * depol * np.exp(1j * phase)).sum(axis=1)

    def _compute_polarization_internal_dispersion(
        self, theta: np.ndarray
    ) -> np.ndarray:
        """
        Internal Faraday dispersion (Sokoloff):
            P = p₀ × [(1 - exp(-S)) / S] × exp(2i × χ₀)
        with S = 2σ_φ²λ⁴ - 2iφλ² (complex).

        Vectorized over (batch, component, freq).
        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        theta_r = theta.reshape(batch_size, self.n_components, 4)
        phi = theta_r[:, :, 0]
        sigma_phi = theta_r[:, :, 1]
        amp = theta_r[:, :, 2]
        chi0 = theta_r[:, :, 3]

        lsq = self.lambda_sq[None, None, :]  # (1, 1, F)
        lsq4 = lsq**2
        S = 2.0 * sigma_phi[:, :, None] ** 2 * lsq4 - 2j * phi[:, :, None] * lsq
        abs_S = np.abs(S)
        with np.errstate(divide="ignore", invalid="ignore"):
            depol = np.where(abs_S < 1e-10, 1.0 + 0j, (1 - np.exp(-S)) / S)

        return (amp[:, :, None] * depol * np.exp(2j * chi0[:, :, None])).sum(axis=1)

    def simulate(
        self,
        theta: np.ndarray,
        weights: np.ndarray | None = None,
        noise_percent: float = 10.0,
        noise_sigma: float | None = None,
    ) -> np.ndarray:
        """
        Simulate Q, U spectra from parameters with noise.

        Parameters
        ----------
        theta : np.ndarray
            Parameters of shape (batch, n_params) or (n_params,)
        weights : np.ndarray, optional
            Channel weights of shape (n_freq,). If None, uses loaded weights.
        noise_percent : float
            Noise as percentage of signal amplitude (used when noise_sigma is None).
        noise_sigma : float, optional
            If given, use additive Gaussian noise with this fixed std dev instead
            of percentage-based noise.

        Returns
        -------
        np.ndarray
            Simulated [Q, U] of shape (batch, 2*n_freq) or (2*n_freq,)
        """
        theta = np.atleast_2d(theta)
        batch_size = theta.shape[0]

        if weights is None:
            weights = self._weights

        # Compute complex polarization based on model type
        if self.model_type == "faraday_thin":
            P = self._compute_polarization_faraday_thin(theta)
        elif self.model_type == "burn_slab":
            P = self._compute_polarization_burn_slab(theta)
        elif self.model_type == "external_dispersion":
            P = self._compute_polarization_external_dispersion(theta)
        elif self.model_type == "internal_dispersion":
            P = self._compute_polarization_internal_dispersion(theta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Check for numerical issues
        if np.any(~np.isfinite(P)):
            warnings.warn(
                f"Non-finite values in polarization computation for {self.model_type}. "
                f"Check parameters for extreme values."
            )
            P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        Q = np.zeros((batch_size, self._n_freq))
        U = np.zeros((batch_size, self._n_freq))

        if noise_sigma is not None:
            # Additive noise: fixed sigma for all channels
            for b in range(batch_size):
                for j in range(self._n_freq):
                    if weights[j] > 0:
                        Q[b, j] = P[b, j].real + np.random.normal(0, noise_sigma)
                        U[b, j] = P[b, j].imag + np.random.normal(0, noise_sigma)
        else:
            # Percentage-based noise: sigma = noise_percent/100 * |P|
            P_amplitude = np.abs(P)  # (batch, n_freq)
            min_sigma = 1e-6
            for b in range(batch_size):
                for j in range(self._n_freq):
                    if weights[j] > 0:
                        sigma = max(noise_percent / 100.0 * P_amplitude[b, j], min_sigma)
                        Q[b, j] = P[b, j].real + np.random.normal(0, sigma)
                        U[b, j] = P[b, j].imag + np.random.normal(0, sigma)

        # Concatenate [Q, U]
        x = np.hstack([Q, U])

        return x.squeeze()

    def simulate_batch(
        self,
        theta: np.ndarray,
        weights_batch: np.ndarray,
        noise_percent: float = 10.0,
        noise_sigma: float | None = None,
    ) -> np.ndarray:
        """
        Simulate batch with per-sample weights and noise.

        Parameters
        ----------
        theta : np.ndarray
            Parameters of shape (batch, n_params)
        weights_batch : np.ndarray
            Per-sample weights of shape (batch, n_freq)
        noise_percent : float
            Noise as percentage of signal amplitude (used when noise_sigma is None).
        noise_sigma : float, optional
            If given, use additive Gaussian noise with this fixed std dev instead
            of percentage-based noise.

        Returns
        -------
        np.ndarray
            Simulated [Q, U] of shape (batch, 2*n_freq)
        """
        theta = np.atleast_2d(theta)
        batch_size = theta.shape[0]

        # Compute complex polarization based on model type
        if self.model_type == "faraday_thin":
            P = self._compute_polarization_faraday_thin(theta)
        elif self.model_type == "burn_slab":
            P = self._compute_polarization_burn_slab(theta)
        elif self.model_type == "external_dispersion":
            P = self._compute_polarization_external_dispersion(theta)
        elif self.model_type == "internal_dispersion":
            P = self._compute_polarization_internal_dispersion(theta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Check for numerical issues
        if np.any(~np.isfinite(P)):
            P = np.nan_to_num(P, nan=0.0, posinf=0.0, neginf=0.0)

        # Apply weights mask (zero where weight is zero)
        mask = weights_batch > 0

        if noise_sigma is not None:
            # Additive noise: same fixed sigma for all channels
            noise_Q = np.random.normal(0, noise_sigma, (batch_size, self._n_freq))
            noise_U = np.random.normal(0, noise_sigma, (batch_size, self._n_freq))
        else:
            # Percentage-based noise: sigma = noise_percent/100 * |P|
            P_amplitude = np.abs(P)  # (batch, n_freq)
            min_sigma = 1e-6
            sigma = np.maximum(noise_percent / 100.0 * P_amplitude, min_sigma)
            noise_Q = np.random.normal(0, 1, (batch_size, self._n_freq)) * sigma
            noise_U = np.random.normal(0, 1, (batch_size, self._n_freq)) * sigma

        Q = np.where(mask, P.real + noise_Q, 0.0)
        U = np.where(mask, P.imag + noise_U, 0.0)

        # Concatenate [Q, U]
        return np.hstack([Q, U])

    def simulate_noiseless(self, theta: np.ndarray) -> np.ndarray:
        """
        Simulate noiseless Q, U spectra.

        Useful for model comparison and visualization.

        Parameters
        ----------
        theta : np.ndarray
            Parameters of shape (batch, n_params) or (n_params,)

        Returns
        -------
        np.ndarray
            Noiseless [Q, U] of shape (batch, 2*n_freq) or (2*n_freq,)
        """
        theta = np.atleast_2d(theta)

        # Compute complex polarization
        if self.model_type == "faraday_thin":
            P = self._compute_polarization_faraday_thin(theta)
        elif self.model_type == "burn_slab":
            P = self._compute_polarization_burn_slab(theta)
        elif self.model_type == "external_dispersion":
            P = self._compute_polarization_external_dispersion(theta)
        elif self.model_type == "internal_dispersion":
            P = self._compute_polarization_internal_dispersion(theta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        Q = P.real
        U = P.imag
        x = np.hstack([Q, U])

        return x.squeeze()

    def get_param_names(self) -> list:
        """Get parameter names for this model."""
        names = []
        for i in range(self.n_components):
            if self.model_type == "faraday_thin":
                names.extend([f"RM_{i + 1}", f"amp_{i + 1}", f"chi0_{i + 1}"])
            elif self.model_type == "burn_slab":
                names.extend(
                    [
                        f"phi_c_{i + 1}",
                        f"delta_phi_{i + 1}",
                        f"amp_{i + 1}",
                        f"chi0_{i + 1}",
                    ]
                )
            else:  # external_dispersion, internal_dispersion
                names.extend(
                    [
                        f"phi_{i + 1}",
                        f"sigma_phi_{i + 1}",
                        f"amp_{i + 1}",
                        f"chi0_{i + 1}",
                    ]
                )
        return names
