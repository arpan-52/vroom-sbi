"""
Forward simulator: θ → Q, U
"""
import numpy as np
import torch

from .physics import load_frequencies, freq_to_lambda_sq


class RMSimulator:
    """
    Simulator for N Faraday-thin components.

    Parameters per component: RM, amplitude, chi0
    Plus: noise level

    Total params = 3*N + 1
    """

    def __init__(self, freq_file: str, n_components: int):
        self.freq, self.weights = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.freq)
        self.n_freq = len(self.freq)
        self.n_components = n_components
        self.n_params = 3 * n_components + 1

    def __call__(self, theta: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Simulate Q, U from parameters with optional weighted channels.

        Parameters
        ----------
        theta : array (n_params,) or (batch, n_params)
        weights : array (n_freq,), optional
            Channel weights (1.0 = best, 0.0 = missing). If None, uses self.weights.

        Returns
        -------
        x : array (2*n_freq,) or (batch, 2*n_freq)
            Simulated Q and U with weighted noise
        """
        theta = np.atleast_2d(theta)
        batch_size = theta.shape[0]

        if weights is None:
            weights = self.weights

        Q = np.zeros((batch_size, self.n_freq))
        U = np.zeros((batch_size, self.n_freq))

        for b in range(batch_size):
            P = np.zeros(self.n_freq, dtype=complex)

            for i in range(self.n_components):
                rm = theta[b, 3 * i]
                amp = theta[b, 3 * i + 1]
                chi0 = theta[b, 3 * i + 2]

                phase = 2 * (chi0 + rm * self.lambda_sq)
                P += amp * np.exp(1j * phase)

            base_noise = theta[b, -1]
            
            # Apply weighted noise: σ = base_noise / weight for weight > 0
            # For weight = 0 (missing channels), set to 0
            for j in range(self.n_freq):
                if weights[j] > 0:
                    # σ = base_noise / weight, but weight = 1/σ_relative
                    # So actual σ = base_noise / weight
                    sigma = base_noise / weights[j]
                    Q[b, j] = P[j].real + np.random.normal(0, sigma)
                    U[b, j] = P[j].imag + np.random.normal(0, sigma)
                else:
                    # Missing channel - set to zero
                    Q[b, j] = 0.0
                    U[b, j] = 0.0

        x = np.hstack([Q, U])
        return x.squeeze()


def build_prior(n_components: int, config: dict, device: str = "cpu"):
    """
    Build SBI BoxUniform prior for N components.
    """
    from sbi.utils import BoxUniform

    low = []
    high = []

    for _ in range(n_components):
        low.extend([
            config["rm_min"],
            config["amp_min"],
            0.0,
        ])
        high.extend([
            config["rm_max"],
            config["amp_max"],
            np.pi,
        ])

    low.append(config["noise_min"])
    high.append(config["noise_max"])

    low_t = torch.tensor(low, dtype=torch.float32, device=device)
    high_t = torch.tensor(high, dtype=torch.float32, device=device)

    return BoxUniform(low=low_t, high=high_t)


def sample_prior(n_samples: int, n_components: int, config: dict) -> np.ndarray:
    """
    Sample from prior - UNIFORM for all to match BoxUniform prior.
    """
    theta = np.zeros((n_samples, 3 * n_components + 1))

    for i in range(n_components):
        # RM: uniform
        theta[:, 3 * i] = np.random.uniform(
            config["rm_min"], config["rm_max"], n_samples
        )
        # Amplitude: uniform (matches BoxUniform)
        theta[:, 3 * i + 1] = np.random.uniform(
            config["amp_min"], config["amp_max"], n_samples
        )
        # Chi0: uniform [0, π]
        theta[:, 3 * i + 2] = np.random.uniform(0, np.pi, n_samples)

    # Noise: uniform (matches BoxUniform)
    theta[:, -1] = np.random.uniform(
        config["noise_min"], config["noise_max"], n_samples
    )

    return theta