"""
Prior sampling and construction for VROOM-SBI.

Handles:
- Building SBI-compatible priors
- Sampling from priors with RM ordering constraint
- Sorting posterior samples to break label switching
"""

import numpy as np
import torch
from typing import Dict, Optional


def build_prior(
    n_components: int, 
    config: Dict[str, float], 
    device: str = "cpu",
    model_type: str = "faraday_thin",
    model_params: Optional[Dict] = None
):
    """
    Build SBI BoxUniform prior for N components.
    
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
    config : dict
        Prior configuration with rm_min, rm_max, amp_min, amp_max
    device : str
        Torch device
    model_type : str
        Physical model type
    model_params : dict, optional
        Model-specific parameters (max_delta_phi, max_sigma_phi)
        
    Returns
    -------
    BoxUniform
        SBI prior distribution
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
                0.0,  # chi0 min
            ])
            high.extend([
                config["rm_max"],
                config["amp_max"],
                np.pi,  # chi0 max
            ])
    elif model_type == "burn_slab":
        # 4 params per component: [phi_c, delta_phi, amp, chi0]
        max_delta_phi = model_params.get("max_delta_phi", 200.0)
        for _ in range(n_components):
            low.extend([
                config["rm_min"],           # phi_c
                0.0,                         # delta_phi (non-negative)
                config["amp_min"],
                0.0,                         # chi0
            ])
            high.extend([
                config["rm_max"],
                max_delta_phi,
                config["amp_max"],
                np.pi,
            ])
    else:  # external_dispersion or internal_dispersion
        # 4 params per component: [phi, sigma_phi, amp, chi0]
        max_sigma_phi = model_params.get("max_sigma_phi", 200.0)
        for _ in range(n_components):
            low.extend([
                config["rm_min"],           # phi
                0.0,                         # sigma_phi (non-negative)
                config["amp_min"],
                0.0,                         # chi0
            ])
            high.extend([
                config["rm_max"],
                max_sigma_phi,
                config["amp_max"],
                np.pi,
            ])
    
    low_t = torch.tensor(low, dtype=torch.float32, device=device)
    high_t = torch.tensor(high, dtype=torch.float32, device=device)
    
    return BoxUniform(low=low_t, high=high_t)


def sample_prior(
    n_samples: int, 
    n_components: int, 
    config: Dict[str, float],
    model_type: str = "faraday_thin",
    model_params: Optional[Dict] = None
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
    config : dict
        Prior configuration with rm_min, rm_max, amp_min, amp_max
    model_type : str
        Physical model type
    model_params : dict, optional
        Model-specific parameters
        
    Returns
    -------
    theta : np.ndarray
        Parameter samples with shape (n_samples, n_params)
        Guaranteed: first param (RM/phi) sorted descending for n_components >= 2
    """
    if model_params is None:
        model_params = {}
    
    if model_type == "faraday_thin":
        params_per_comp = 3
        theta = np.zeros((n_samples, params_per_comp * n_components))
        
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
        
        # Sort components by RM (descending)
        if n_components >= 2:
            theta = sort_components_by_rm(theta, n_components, params_per_comp)
    
    else:
        # burn_slab, external_dispersion, internal_dispersion
        params_per_comp = 4
        theta = np.zeros((n_samples, params_per_comp * n_components))
        
        # Get the appropriate max value based on model type
        if model_type == "burn_slab":
            max_second_param = model_params.get("max_delta_phi", 200.0)
        else:  # external_dispersion or internal_dispersion
            max_second_param = model_params.get("max_sigma_phi", 200.0)
        
        for i in range(n_components):
            # phi/phi_c: uniform
            theta[:, params_per_comp * i] = np.random.uniform(
                config["rm_min"], config["rm_max"], n_samples
            )
            # sigma_phi or delta_phi: uniform [0, max]
            theta[:, params_per_comp * i + 1] = np.random.uniform(
                0.0, max_second_param, n_samples
            )
            # Amplitude: uniform
            theta[:, params_per_comp * i + 2] = np.random.uniform(
                config["amp_min"], config["amp_max"], n_samples
            )
            # Chi0: uniform [0, π]
            theta[:, params_per_comp * i + 3] = np.random.uniform(0, np.pi, n_samples)
        
        # Sort components by phi (descending)
        if n_components >= 2:
            theta = sort_components_by_rm(theta, n_components, params_per_comp)
    
    return theta


def sort_components_by_rm(
    theta: np.ndarray, 
    n_components: int, 
    params_per_comp: int = 3
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
        # Extract first parameter (RM or phi) for this sample
        first_params = np.array([
            theta[s, params_per_comp * i] 
            for i in range(n_components)
        ])
        
        # Get indices that would sort in descending order
        sort_idx = np.argsort(first_params)[::-1]
        
        # Reorder all components according to this sorting
        for new_pos, old_pos in enumerate(sort_idx):
            for p in range(params_per_comp):
                theta_sorted[s, params_per_comp * new_pos + p] = \
                    theta[s, params_per_comp * old_pos + p]
    
    return theta_sorted


def sort_posterior_samples(
    samples: np.ndarray, 
    n_components: int, 
    params_per_comp: int = 3
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
            names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
        elif model_type == "burn_slab":
            names.extend([f"phi_c_{i+1}", f"delta_phi_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
        else:  # external_dispersion, internal_dispersion
            names.extend([f"phi_{i+1}", f"sigma_phi_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
    return names
