"""
Forward simulator: θ → Q, U

Supports multiple physical models:
- Faraday thin: Simple external rotation (current default)
- Faraday thick: Internal rotation with depolarization
- Burn slab: External Faraday screen with wavelength-dependent depolarization
- Sokoloff: Turbulent magnetic field model

Fixed version with RM sorting for multi-component models to break label switching symmetry.
Convention: RM1 > RM2 > RM3 > ... always (for n_components >= 2)
"""
import numpy as np
import torch

from .physics import load_frequencies, freq_to_lambda_sq


class RMSimulator:
    """
    Simulator for N components with multiple physical models.

    Parameters per component: RM, amplitude, chi0
    Layout: [RM_1, amp_1, chi0_1, RM_2, amp_2, chi0_2, ...]

    Supports:
    - faraday_thin: Simple external rotation (default)
    - faraday_thick: Internal rotation with depolarization
    - burn_slab: External screen with wavelength-dependent depolarization
    - sokoloff: Turbulent galactic magnetic field model

    Noise is encoded in weights, not as a parameter.
    Total params = 3*N
    """

    def __init__(
        self,
        freq_file: str,
        n_components: int,
        base_noise_level: float = 0.01,
        model_type: str = "faraday_thin",
        model_params: dict = None
    ):
        self.freq, self.weights = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.freq)
        self.n_freq = len(self.freq)
        self.n_components = n_components
        self.model_type = model_type
        self.model_params = model_params if model_params is not None else {}

        # Determine number of parameters based on model type
        if model_type == "faraday_thin":
            self.n_params = 3 * n_components  # [RM, amp, chi0] per component
            self.params_per_comp = 3
        else:
            self.n_params = 4 * n_components  # [phi, sigma/delta, amp, chi0] per component
            self.params_per_comp = 4

        self.base_noise_level = base_noise_level

    def _compute_polarization_faraday_thin(self, theta: np.ndarray) -> np.ndarray:
        """
        Faraday-thin model: P = Σⱼ pⱼ exp[2i(χ₀,ⱼ + φⱼλ²)]

        Parameters: [RM, amp, chi0] per component (3N total)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self.n_freq), dtype=complex)

        for b in range(batch_size):
            for i in range(self.n_components):
                rm = theta[b, 3 * i]
                amp = theta[b, 3 * i + 1]
                chi0 = theta[b, 3 * i + 2]

                phase = 2 * (chi0 + rm * self.lambda_sq)
                P[b] += amp * np.exp(1j * phase)

        return P

    def _compute_polarization_burn_slab(self, theta: np.ndarray) -> np.ndarray:
        """
        Burn slab model: P = p₀ [sin(Δφλ²)/(Δφλ²)] exp[2i(χ₀ + φ_c λ²)]

        Parameters: [phi_c, delta_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self.n_freq), dtype=complex)

        for b in range(batch_size):
            for i in range(self.n_components):
                phi_c = theta[b, 4 * i]          # Central RM
                delta_phi = theta[b, 4 * i + 1]  # Slab depth/thickness
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]

                # Sinc depolarization term
                arg = delta_phi * self.lambda_sq
                # Handle sinc(0) = 1 case
                sinc_term = np.where(np.abs(arg) < 1e-10, 1.0, np.sin(arg) / arg)

                # Rotation term
                phase = 2 * (chi0 + phi_c * self.lambda_sq)

                P[b] += amp * sinc_term * np.exp(1j * phase)

        return P

    def _compute_polarization_external_dispersion(self, theta: np.ndarray) -> np.ndarray:
        """
        External Faraday dispersion: P = p₀ exp(-2σ_φ² λ⁴) exp[2i(χ₀ + φλ²)]

        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self.n_freq), dtype=complex)

        for b in range(batch_size):
            for i in range(self.n_components):
                phi = theta[b, 4 * i]            # Mean RM
                sigma_phi = theta[b, 4 * i + 1]  # RM dispersion
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]

                # Gaussian depolarization (turbulent foreground screen)
                depol = np.exp(-2 * sigma_phi**2 * self.lambda_sq**2)

                # Rotation term
                phase = 2 * (chi0 + phi * self.lambda_sq)

                P[b] += amp * depol * np.exp(1j * phase)

        return P

    def _compute_polarization_internal_dispersion(self, theta: np.ndarray) -> np.ndarray:
        """
        Internal Faraday dispersion (Sokoloff): P = p₀ [(1-exp(-S))/S] exp(2iχ₀)
        where S = 2σ_φ² λ⁴ - 2iφλ²

        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self.n_freq), dtype=complex)

        for b in range(batch_size):
            for i in range(self.n_components):
                phi = theta[b, 4 * i]            # Mean RM
                sigma_phi = theta[b, 4 * i + 1]  # RM dispersion
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]

                # Complex depolarization function S
                S = 2 * sigma_phi**2 * self.lambda_sq**2 - 2j * phi * self.lambda_sq

                # Handle S → 0 case: lim (1-exp(-S))/S = 1
                depol = np.where(
                    np.abs(S) < 1e-10,
                    1.0,
                    (1 - np.exp(-S)) / S
                )

                P[b] += amp * depol * np.exp(2j * chi0)

        return P

    def __call__(self, theta: np.ndarray, weights: np.ndarray = None) -> np.ndarray:
        """
        Simulate Q, U from parameters with optional weighted channels.

        Parameters
        ----------
        theta : array (n_params,) or (batch, n_params)
            Parameter layout depends on model_type:
            - faraday_thin: [RM, amp, chi0] per component (3N params)
            - burn_slab: [phi_c, delta_phi, amp, chi0] per component (4N params)
            - external_dispersion: [phi, sigma_phi, amp, chi0] per component (4N params)
            - internal_dispersion: [phi, sigma_phi, amp, chi0] per component (4N params)
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

        # Compute complex polarization based on model type
        if self.model_type == "faraday_thin":
            P = self._compute_polarization_faraday_thin(theta)
        elif self.model_type == "burn_slab":
            P = self._compute_polarization_burn_slab(theta)
        elif self.model_type == "external_dispersion":
            P = self._compute_polarization_external_dispersion(theta)
        elif self.model_type == "internal_dispersion" or self.model_type == "sokoloff":
            P = self._compute_polarization_internal_dispersion(theta)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

        # Convert to Stokes Q, U with weighted noise
        Q = np.zeros((batch_size, self.n_freq))
        U = np.zeros((batch_size, self.n_freq))

        for b in range(batch_size):
            for j in range(self.n_freq):
                if weights[j] > 0:
                    sigma = self.base_noise_level / weights[j]
                    Q[b, j] = P[b, j].real + np.random.normal(0, sigma)
                    U[b, j] = P[b, j].imag + np.random.normal(0, sigma)
                else:
                    # Missing channel - set to zero for network interpolation
                    Q[b, j] = 0.0
                    U[b, j] = 0.0

        x = np.hstack([Q, U])
        return x.squeeze()


def build_prior(n_components: int, config: dict, device: str = "cpu", model_type: str = "faraday_thin", model_params: dict = None):
    """
    Build SBI BoxUniform prior for N components.

    Parameter layout depends on model_type:
    - faraday_thin: [RM, amp, chi0] → 3N params
    - burn_slab: [phi_c, delta_phi, amp, chi0] → 4N params
    - external_dispersion: [phi, sigma_phi, amp, chi0] → 4N params
    - internal_dispersion: [phi, sigma_phi, amp, chi0] → 4N params

    Note: For n_components >= 2, we sample uniformly and then sort,
    so the prior bounds are the same for all components.
    The sorting happens in sample_prior, not here.
    """
    from sbi.utils import BoxUniform

    if model_params is None:
        model_params = {}

    low = []
    high = []

    if model_type == "faraday_thin":
        # 3 params per component: [RM, amp, chi0]
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
    elif model_type == "burn_slab":
        # 4 params per component: [phi_c, delta_phi, amp, chi0]
        max_delta_phi = model_params.get("max_delta_phi", 200.0)  # Read from config!
        for _ in range(n_components):
            low.extend([
                config["rm_min"],           # phi_c (central RM)
                0.0,                         # delta_phi (slab thickness, non-negative)
                config["amp_min"],
                0.0,                         # chi0
            ])
            high.extend([
                config["rm_max"],
                max_delta_phi,               # Actually use config parameter!
                config["amp_max"],
                np.pi,
            ])
    else:  # external_dispersion or internal_dispersion
        # 4 params per component: [phi, sigma_phi, amp, chi0]
        max_sigma_phi = model_params.get("max_sigma_phi", 200.0)  # Read from config!
        for _ in range(n_components):
            low.extend([
                config["rm_min"],           # phi (mean RM)
                0.0,                         # sigma_phi (RM dispersion, non-negative)
                config["amp_min"],
                0.0,                         # chi0
            ])
            high.extend([
                config["rm_max"],
                max_sigma_phi,               # Actually use config parameter!
                config["amp_max"],
                np.pi,
            ])

    low_t = torch.tensor(low, dtype=torch.float32, device=device)
    high_t = torch.tensor(high, dtype=torch.float32, device=device)

    return BoxUniform(low=low_t, high=high_t)


def sample_prior(n_samples: int, n_components: int, config: dict, model_type: str = "faraday_thin", model_params: dict = None) -> np.ndarray:
    """
    Sample from prior with RM ordering constraint for n_components >= 2.

    Convention: RM1/phi1 > RM2/phi2 > ... (descending order)

    This breaks the label switching symmetry by ensuring a unique ordering.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_components : int
        Number of components
    config : dict
        Prior configuration with rm_min, rm_max, amp_min, amp_max
    model_type : str
        Physical model type
    model_params : dict, optional
        Model-specific parameters (e.g., max_sigma_phi, max_delta_phi)

    Returns
    -------
    theta : np.ndarray
        Parameter samples with layout depending on model_type:
        - faraday_thin: shape (n_samples, 3*n_components) - [RM, amp, chi0, ...]
        - others: shape (n_samples, 4*n_components) - [phi, sigma/delta, amp, chi0, ...]
        Guaranteed: first param (RM/phi) sorted descending for n_components >= 2
    """
    if model_params is None:
        model_params = {}
    if model_type == "faraday_thin":
        params_per_comp = 3
        theta = np.zeros((n_samples, params_per_comp * n_components))

        # Sample all parameters
        for i in range(n_components):
            # RM: uniform
            theta[:, params_per_comp * i] = np.random.uniform(
                config["rm_min"], config["rm_max"], n_samples
            )
            # Amplitude: uniform
            theta[:, params_per_comp * i + 1] = np.random.uniform(
                config["amp_min"], config["amp_max"], n_samples
            )
            # Chi0: uniform [0, π]
            theta[:, params_per_comp * i + 2] = np.random.uniform(0, np.pi, n_samples)

        # Sort components by RM (descending) to break label switching
        if n_components >= 2:
            theta = sort_components_by_rm(theta, n_components, params_per_comp)

    else:
        # burn_slab, external_dispersion, internal_dispersion
        params_per_comp = 4
        theta = np.zeros((n_samples, params_per_comp * n_components))

        # Get the appropriate max value based on model type
        if model_type == "burn_slab":
            max_second_param = model_params.get("max_delta_phi", 200.0)  # Slab thickness
        else:  # external_dispersion or internal_dispersion
            max_second_param = model_params.get("max_sigma_phi", 200.0)  # RM dispersion

        for i in range(n_components):
            # phi/phi_c: uniform
            theta[:, params_per_comp * i] = np.random.uniform(
                config["rm_min"], config["rm_max"], n_samples
            )
            # sigma_phi or delta_phi: uniform [0, max]
            theta[:, params_per_comp * i + 1] = np.random.uniform(
                0.0, max_second_param, n_samples  # Use config parameter!
            )
            # Amplitude: uniform
            theta[:, params_per_comp * i + 2] = np.random.uniform(
                config["amp_min"], config["amp_max"], n_samples
            )
            # Chi0: uniform [0, π]
            theta[:, params_per_comp * i + 3] = np.random.uniform(0, np.pi, n_samples)

        # Sort components by phi (descending) to break label switching
        if n_components >= 2:
            theta = sort_components_by_rm(theta, n_components, params_per_comp)

    return theta


def sort_components_by_rm(theta: np.ndarray, n_components: int, params_per_comp: int = 3) -> np.ndarray:
    """
    Sort components so that RM1/phi1 > RM2/phi2 > ...

    This ensures a unique ordering and breaks the label switching symmetry.
    Each component's parameter tuple stays together.

    Parameters
    ----------
    theta : np.ndarray, shape (n_samples, params_per_comp * n_components)
        Parameter array with first param being RM/phi
    n_components : int
        Number of components
    params_per_comp : int
        Number of parameters per component (3 or 4)

    Returns
    -------
    theta_sorted : np.ndarray
        Same shape, but with components sorted by RM/phi (descending)
    """
    theta = np.atleast_2d(theta)
    n_samples = theta.shape[0]
    theta_sorted = np.zeros_like(theta)

    for s in range(n_samples):
        # Extract first parameter (RM or phi) for this sample
        first_params = np.array([theta[s, params_per_comp * i] for i in range(n_components)])

        # Get indices that would sort in descending order
        sort_idx = np.argsort(first_params)[::-1]  # descending

        # Reorder all components according to this sorting
        for new_pos, old_pos in enumerate(sort_idx):
            # Copy the entire component tuple to new position
            for p in range(params_per_comp):
                theta_sorted[s, params_per_comp * new_pos + p] = theta[s, params_per_comp * old_pos + p]

    return theta_sorted


def sort_posterior_samples(samples: np.ndarray, n_components: int, params_per_comp: int = 3) -> np.ndarray:
    """
    Sort posterior samples to ensure RM1/phi1 > RM2/phi2 > ...

    Use this after sampling from the posterior to ensure consistent ordering.
    This is needed because even though we train with sorted samples,
    the posterior might still occasionally produce unsorted outputs.

    Parameters
    ----------
    samples : np.ndarray, shape (n_samples, params_per_comp * n_components)
        Posterior samples
    n_components : int
        Number of components
    params_per_comp : int
        Number of parameters per component (3 or 4)

    Returns
    -------
    samples_sorted : np.ndarray
        Sorted samples
    """
    return sort_components_by_rm(samples, n_components, params_per_comp)