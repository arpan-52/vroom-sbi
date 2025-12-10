"""
Forward simulator: θ → Q, U

Fixed version with RM sorting for 2-component model to break label switching symmetry.
Convention: RM1 > RM2 always (for n_components >= 2)
"""
import numpy as np
import torch

from .physics import load_frequencies, freq_to_lambda_sq


class RMSimulator:
    """
    Simulator for N Faraday-thin components.

    Parameters per component: RM, amplitude, chi0
    Layout: [RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...]
    
    Noise is encoded in weights, not as a parameter.
    Total params = 3*N
    """

    def __init__(self, freq_file: str, n_components: int, base_noise_level: float = 0.01):
        self.freq, self.weights = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.freq)
        self.n_freq = len(self.freq)
        self.n_components = n_components
        self.n_params = 3 * n_components
        self.base_noise_level = base_noise_level

    def __call__(self, theta: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Simulate Q, U from parameters with optional weighted channels.

        Parameters
        ----------
        theta : array (n_params,) or (batch, n_params)
            Layout: [RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...]
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
            
            # Apply weighted noise based on channel quality
            for j in range(self.n_freq):
                if weights[j] > 0:
                    sigma = self.base_noise_level / weights[j]
                    Q[b, j] = P[j].real + np.random.normal(0, sigma)
                    U[b, j] = P[j].imag + np.random.normal(0, sigma)
                else:
                    # Missing channel - set to zero for network interpolation
                    Q[b, j] = 0.0
                    U[b, j] = 0.0

        x = np.hstack([Q, U])
        return x.squeeze()


def build_prior(n_components: int, config: dict, device: str = "cpu"):
    """
    Build SBI BoxUniform prior for N components.
    
    Note: For n_components >= 2, we sample uniformly and then sort,
    so the prior bounds are the same for all components.
    The sorting happens in sample_prior, not here.
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

    low_t = torch.tensor(low, dtype=torch.float32, device=device)
    high_t = torch.tensor(high, dtype=torch.float32, device=device)

    return BoxUniform(low=low_t, high=high_t)


def sample_prior(n_samples: int, n_components: int, config: dict) -> np.ndarray:
    """
    Sample from prior with RM ordering constraint for n_components >= 2.
    
    Convention: RM1 > RM2 > RM3 > ... (descending order)
    
    This breaks the label switching symmetry by ensuring a unique ordering.
    
    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_components : int
        Number of RM components
    config : dict
        Prior configuration with rm_min, rm_max, amp_min, amp_max
        
    Returns
    -------
    theta : np.ndarray, shape (n_samples, 3 * n_components)
        Parameter samples with layout [RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...]
        Guaranteed: RM_1 > RM_2 > ... for n_components >= 2
    """
    theta = np.zeros((n_samples, 3 * n_components))

    # Sample all parameters
    for i in range(n_components):
        # RM: uniform
        theta[:, 3 * i] = np.random.uniform(
            config["rm_min"], config["rm_max"], n_samples
        )
        # Amplitude: uniform
        theta[:, 3 * i + 1] = np.random.uniform(
            config["amp_min"], config["amp_max"], n_samples
        )
        # Chi0: uniform [0, π]
        theta[:, 3 * i + 2] = np.random.uniform(0, np.pi, n_samples)

    # Sort components by RM (descending) to break label switching
    if n_components >= 2:
        theta = sort_components_by_rm(theta, n_components)

    return theta


def sort_components_by_rm(theta: np.ndarray, n_components: int) -> np.ndarray:
    """
    Sort components so that RM1 > RM2 > RM3 > ...
    
    This ensures a unique ordering and breaks the label switching symmetry.
    Each component's (RM, amp, chi0) triplet stays together.
    
    Parameters
    ----------
    theta : np.ndarray, shape (n_samples, 3 * n_components)
        Parameter array with layout [RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...]
    n_components : int
        Number of components
        
    Returns
    -------
    theta_sorted : np.ndarray
        Same shape, but with components sorted by RM (descending)
    """
    theta = np.atleast_2d(theta)
    n_samples = theta.shape[0]
    theta_sorted = np.zeros_like(theta)
    
    for s in range(n_samples):
        # Extract RMs for this sample
        rms = np.array([theta[s, 3 * i] for i in range(n_components)])
        
        # Get indices that would sort RMs in descending order
        sort_idx = np.argsort(rms)[::-1]  # descending
        
        # Reorder all components according to this sorting
        for new_pos, old_pos in enumerate(sort_idx):
            # Copy the entire component (RM, amp, chi0) to new position
            theta_sorted[s, 3 * new_pos] = theta[s, 3 * old_pos]          # RM
            theta_sorted[s, 3 * new_pos + 1] = theta[s, 3 * old_pos + 1]  # amp
            theta_sorted[s, 3 * new_pos + 2] = theta[s, 3 * old_pos + 2]  # chi0
    
    return theta_sorted


def sort_posterior_samples(samples: np.ndarray, n_components: int) -> np.ndarray:
    """
    Sort posterior samples to ensure RM1 > RM2 > ...
    
    Use this after sampling from the posterior to ensure consistent ordering.
    This is needed because even though we train with sorted samples,
    the posterior might still occasionally produce unsorted outputs.
    
    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, 3 * n_components)
        Posterior samples
    n_components : int
        Number of components
        
    Returns
    -------
    samples_sorted : np.ndarray
        Sorted samples
    """
    return sort_components_by_rm(samples, n_components)