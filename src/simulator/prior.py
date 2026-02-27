"""
Prior sampling and construction for VROOM-SBI.

Handles:
- Building SBI-compatible priors from centralized PriorConfig
- Sampling from priors with RM ordering constraint
- Sorting posterior samples to break label switching
"""

import numpy as np
import torch


def build_prior(
    n_components: int,
    prior_config,  # PriorConfig or dict
    device: str = "cpu",
    model_type: str = "faraday_thin",
):
    """
    Build SBI BoxUniform prior for N components.

    Uses centralized PriorConfig for ALL parameter bounds.

    Parameter layout depends on model_type:
    - faraday_thin: [RM, amp, chi0] → 3N params
    - burn_slab: [phi_c, delta_phi, amp, chi0] → 4N params
    - external_dispersion: [phi, sigma_phi, amp, chi0] → 4N params
    - internal_dispersion: [phi, sigma_phi, amp, chi0] → 4N params

    For n_components >= 2, we sample uniformly and then sort by RM/phi
    to break label switching symmetry.

    Parameters
    ----------
    n_components : int
        Number of RM components
    prior_config : PriorConfig or dict
        Prior configuration with all parameter bounds
    device : str
        Torch device
    model_type : str
        Physical model type

    Returns
    -------
    BoxUniform
        SBI prior distribution
    """
    from sbi.utils import BoxUniform

    # Handle both PriorConfig object and dict
    if hasattr(prior_config, "get_bounds_for_model"):
        # It's a PriorConfig object - use its method
        low, high = prior_config.get_bounds_for_model(model_type, n_components)
    else:
        # It's a dict - build bounds manually
        low, high = _build_bounds_from_dict(prior_config, model_type, n_components)

    low_t = torch.tensor(low, dtype=torch.float32, device=device)
    high_t = torch.tensor(high, dtype=torch.float32, device=device)

    return BoxUniform(low=low_t, high=high_t)


def _build_bounds_from_dict(config: dict, model_type: str, n_components: int):
    """Build bounds arrays from a flat config dict."""
    # Extract bounds with defaults
    rm_min = config.get("rm_min", -500.0)
    rm_max = config.get("rm_max", 500.0)
    amp_min = config.get("amp_min", 0.01)
    amp_max = config.get("amp_max", 1.0)
    chi0_min = config.get("chi0_min", 0.0)
    chi0_max = config.get("chi0_max", np.pi)
    sigma_phi_min = config.get("sigma_phi_min", 0.0)
    sigma_phi_max = config.get("sigma_phi_max", config.get("max_sigma_phi", 100.0))
    delta_phi_min = config.get("delta_phi_min", 0.0)
    delta_phi_max = config.get("delta_phi_max", config.get("max_delta_phi", 100.0))

    low = []
    high = []

    if model_type == "faraday_thin":
        for _ in range(n_components):
            low.extend([rm_min, amp_min, chi0_min])
            high.extend([rm_max, amp_max, chi0_max])
    elif model_type == "burn_slab":
        for _ in range(n_components):
            low.extend([rm_min, delta_phi_min, amp_min, chi0_min])
            high.extend([rm_max, delta_phi_max, amp_max, chi0_max])
    else:  # external_dispersion, internal_dispersion
        for _ in range(n_components):
            low.extend([rm_min, sigma_phi_min, amp_min, chi0_min])
            high.extend([rm_max, sigma_phi_max, amp_max, chi0_max])

    return np.array(low), np.array(high)


def sample_prior(
    n_samples: int,
    n_components: int,
    prior_config,  # PriorConfig or dict
    model_type: str = "faraday_thin",
) -> np.ndarray:
    """
    Sample from prior with RM ordering constraint.

    Convention: RM1/phi1 > RM2/phi2 > ... (descending order)

    This breaks label switching symmetry by ensuring a unique ordering
    of components by their Faraday depth parameter.

    Parameters
    ----------
    n_samples : int
        Number of samples to generate
    n_components : int
        Number of components
    prior_config : PriorConfig or dict
        Prior configuration with all parameter bounds
    model_type : str
        Physical model type

    Returns
    -------
    theta : np.ndarray
        Parameter samples with shape (n_samples, n_params)
        Guaranteed: first param (RM/phi) sorted descending for n_components >= 2
    """
    # Handle both PriorConfig object and dict
    if hasattr(prior_config, "get_bounds_for_model"):
        low, high = prior_config.get_bounds_for_model(model_type, n_components)
    else:
        low, high = _build_bounds_from_dict(prior_config, model_type, n_components)

    params_per_comp = get_params_per_component(model_type)
    n_params = params_per_comp * n_components

    # Sample uniformly
    theta = np.random.uniform(low, high, size=(n_samples, n_params))

    # Sort components by RM/phi (descending) to break label switching
    if n_components >= 2:
        theta = sort_components_by_rm(theta, n_components, params_per_comp)

    return theta


def sort_components_by_rm(
    theta: np.ndarray, n_components: int, params_per_comp: int = 3
) -> np.ndarray:
    """
    Sort components so that RM1/phi1 > RM2/phi2 > ... (descending).

    This ensures a unique ordering and breaks label switching symmetry.
    Each component's parameter tuple stays together.

    Parameters
    ----------
    theta : np.ndarray
        Parameter array of shape (n_samples, params_per_comp * n_components)
    n_components : int
        Number of components
    params_per_comp : int
        Number of parameters per component (3 or 4)

    Returns
    -------
    theta_sorted : np.ndarray
        Same shape, with components sorted by RM/phi (descending)
    """
    theta = np.atleast_2d(theta)
    n_samples = theta.shape[0]
    theta_sorted = np.zeros_like(theta)

    for s in range(n_samples):
        # Extract first parameter (RM or phi) for each component
        first_params = np.array(
            [theta[s, params_per_comp * i] for i in range(n_components)]
        )

        # Get indices that would sort in descending order
        sort_idx = np.argsort(first_params)[::-1]

        # Reorder all components according to this sorting
        for new_pos, old_pos in enumerate(sort_idx):
            for p in range(params_per_comp):
                theta_sorted[s, params_per_comp * new_pos + p] = theta[
                    s, params_per_comp * old_pos + p
                ]

    return theta_sorted


def sort_posterior_samples(
    samples: np.ndarray, n_components: int, params_per_comp: int = 3
) -> np.ndarray:
    """
    Sort posterior samples to ensure RM1/phi1 > RM2/phi2 > ...

    Use this after sampling from the posterior to ensure consistent ordering.
    This is needed because even though we train with sorted samples,
    the posterior might occasionally produce unsorted outputs.

    Parameters
    ----------
    samples : np.ndarray
        Posterior samples of shape (n_samples, n_params)
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


def get_params_per_component(model_type: str) -> int:
    """Get number of parameters per component for a model type."""
    if model_type == "faraday_thin":
        return 3
    else:
        return 4


def get_param_names(model_type: str, n_components: int) -> list:
    """Get parameter names for a model configuration."""
    names = []
    for i in range(n_components):
        if model_type == "faraday_thin":
            names.extend([f"RM_{i + 1}", f"amp_{i + 1}", f"chi0_{i + 1}"])
        elif model_type == "burn_slab":
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
                [f"phi_{i + 1}", f"sigma_phi_{i + 1}", f"amp_{i + 1}", f"chi0_{i + 1}"]
            )
    return names
