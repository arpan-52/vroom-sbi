"""
Spectral shape simulator for VROOM-SBI.

Models the total intensity spectral energy distribution (SED) as a polynomial
in log-log space:

    log F(ν) = log_F0 + alpha * x + beta * x^2 + gamma * x^3

where x = log(ν / ν₀) and ν₀ is the middle frequency of the grid.

Parameters (4 total):
    log_F0 : log flux at the reference frequency ν₀
    alpha  : spectral index
    beta   : spectral curvature (second order)
    gamma  : third-order curvature

Output: F(ν) values at each frequency channel — shape (batch, n_freq).
Noise: additive Gaussian with sigma drawn per simulation.
"""

import numpy as np

from ..core.base_classes import BaseSimulator
from .physics import load_frequencies


class SpectralShapeSimulator(BaseSimulator):
    """
    Simulator for total intensity spectral shape using a log-log polynomial.

    log F(ν) = log_F0 + alpha * x + beta * x^2 + gamma * x^3
    where x = log(ν / ν₀), ν₀ = middle frequency of the grid.

    Parameters
    ----------
    freq_file : str
        Path to frequency file (same format as used by RMSimulator)
    """

    def __init__(self, freq_file: str):
        self.freq, self._weights = load_frequencies(freq_file)
        self._n_freq = len(self.freq)
        # Reference frequency: middle channel of the grid
        self.mid_idx = self._n_freq // 2
        self.nu0 = self.freq[self.mid_idx]
        # Precompute log(ν/ν₀) for all channels — shape (n_freq,)
        self._log_nu_ratio = np.log(self.freq / self.nu0)
        # log_F0 is NOT a network parameter — normalise by F(ν₀) so it's
        # always 1 at the reference channel.  Only [alpha, beta, gamma] are inferred.
        self._n_params = 3
        self._params_per_comp = 3

    # ------------------------------------------------------------------
    # BaseSimulator interface
    # ------------------------------------------------------------------

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

    def get_param_names(self) -> list[str]:
        return ["alpha", "beta", "gamma"]

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _compute_spectrum(self, theta: np.ndarray) -> np.ndarray:
        """
        Compute noiseless spectra.

        Parameters
        ----------
        theta : np.ndarray, shape (batch, 4)

        Returns
        -------
        np.ndarray, shape (batch, n_freq)
        """
        theta = np.atleast_2d(theta)
        x = self._log_nu_ratio  # (n_freq,)

        alpha = theta[:, 0:1]  # (batch, 1)
        beta  = theta[:, 1:2]
        gamma = theta[:, 2:3]

        # F(ν₀) = exp(0) = 1 by construction
        log_F = alpha * x + beta * x**2 + gamma * x**3  # (batch, n_freq)
        return np.exp(log_F)

    # ------------------------------------------------------------------
    # Public simulation methods
    # ------------------------------------------------------------------

    def simulate(
        self,
        theta: np.ndarray,
        weights: np.ndarray | None = None,
        noise_sigma: float = 0.01,
    ) -> np.ndarray:
        """
        Simulate spectra for a single or small batch of parameters.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,) or (batch, n_params)
        weights : np.ndarray, optional
            Channel weights (n_freq,). If None, uses loaded weights.
        noise_sigma : float
            Additive Gaussian noise standard deviation.

        Returns
        -------
        np.ndarray
            Simulated F(ν), shape (batch, n_freq) or (n_freq,) for single input.
        """
        theta = np.atleast_2d(theta)
        if weights is None:
            weights = self._weights

        F = self._compute_spectrum(theta)  # (batch, n_freq)
        mask = weights > 0  # (n_freq,)
        F = F * mask
        noise = np.random.normal(0, noise_sigma, F.shape)
        result = F + noise * mask
        return result.squeeze()

    def simulate_batch(
        self,
        theta: np.ndarray,
        weights_batch: np.ndarray,
        noise_sigma: float = 0.01,
    ) -> np.ndarray:
        """
        Simulate batch with per-sample weights and additive noise.

        Parameters
        ----------
        theta : np.ndarray, shape (batch, 4)
        weights_batch : np.ndarray, shape (batch, n_freq)
            Per-sample channel weights.
        noise_sigma : float
            Additive Gaussian noise standard deviation (same for all channels).

        Returns
        -------
        np.ndarray, shape (batch, n_freq)
        """
        theta = np.atleast_2d(theta)
        F = self._compute_spectrum(theta)  # (batch, n_freq)
        mask = weights_batch > 0  # (batch, n_freq)
        F = F * mask
        noise = np.random.normal(0, noise_sigma, F.shape)
        return F + noise * mask

    def simulate_noiseless(self, theta: np.ndarray) -> np.ndarray:
        """
        Simulate noiseless spectra.

        Parameters
        ----------
        theta : np.ndarray, shape (n_params,) or (batch, n_params)

        Returns
        -------
        np.ndarray, shape (batch, n_freq) or (n_freq,) for single input.
        """
        theta = np.atleast_2d(theta)
        return self._compute_spectrum(theta).squeeze()
