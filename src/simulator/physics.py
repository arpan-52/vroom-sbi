"""
Physics functions for RM Synthesis.

Contains wavelength conversion, RMSF computation, and related utilities.
"""

import numpy as np
from astropy import constants as const


def load_frequencies(freq_file: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load frequency channels from a file.

    Parameters
    ----------
    freq_file : str
        Path to file containing frequencies in Hz (one per line)
        Can have optional weights column:
        - Single column: frequencies only (weights default to 1.0)
        - Two columns: frequencies and weights

    Returns
    -------
    frequencies : np.ndarray
        Array of frequencies in Hz
    weights : np.ndarray
        Array of weights (1.0 = best quality, 0.0 = missing/flagged)
    """
    try:
        data = np.loadtxt(freq_file)
        if data.ndim == 1:
            # Single column - frequencies only
            frequencies = data
            weights = np.ones_like(frequencies)
        else:
            # Two columns - frequencies and weights
            frequencies = data[:, 0]
            weights = data[:, 1]
    except Exception:
        # Fallback: try single column
        frequencies = np.loadtxt(freq_file)
        weights = np.ones_like(frequencies)

    return frequencies, weights


def freq_to_lambda_sq(frequencies: np.ndarray) -> np.ndarray:
    """
    Convert frequencies to squared wavelengths.

    Parameters
    ----------
    frequencies : np.ndarray
        Array of frequencies in Hz

    Returns
    -------
    lambda_sq : np.ndarray
        Array of squared wavelengths in m^2
    """
    c = const.c.value  # Speed of light in m/s
    wavelengths = c / frequencies
    lambda_sq = wavelengths**2
    return lambda_sq


def compute_rmsf(
    lambda_sq: np.ndarray, phi: np.ndarray, weights: np.ndarray = None
) -> np.ndarray:
    """
    Compute the Rotation Measure Spread Function (RMSF).

    The RMSF is the Fourier transform of the wavelength sampling function.

    Parameters
    ----------
    lambda_sq : np.ndarray
        Array of squared wavelengths in m^2
    phi : np.ndarray
        Array of Faraday depth values in rad/m^2
    weights : np.ndarray, optional
        Channel weights (1.0 = good, 0.0 = flagged)

    Returns
    -------
    rmsf : np.ndarray (complex)
        The RMSF as a function of phi
    """
    if weights is None:
        weights = np.ones_like(lambda_sq)

    # Compute weighted mean lambda squared
    good_mask = weights > 0
    if not np.any(good_mask):
        raise ValueError("No valid channels (all weights are zero)")

    lambda_sq_mean = np.average(lambda_sq[good_mask], weights=weights[good_mask])

    # Initialize RMSF
    n_phi = len(phi)
    rmsf = np.zeros(n_phi, dtype=complex)

    # Compute RMSF with weights
    K = np.sum(weights[good_mask])  # Normalization

    for i, phi_val in enumerate(phi):
        rmsf[i] = (
            np.sum(weights * np.exp(-2j * phi_val * (lambda_sq - lambda_sq_mean))) / K
        )

    return rmsf


def get_rmsf_properties(
    lambda_sq: np.ndarray, weights: np.ndarray = None
) -> dict[str, float]:
    """
    Compute properties of the RMSF.

    Parameters
    ----------
    lambda_sq : np.ndarray
        Array of squared wavelengths in m^2
    weights : np.ndarray, optional
        Channel weights

    Returns
    -------
    dict
        Dictionary containing:
        - fwhm: Full width at half maximum in rad/m^2
        - max_scale: Maximum detectable scale in rad/m^2
        - lambda_sq_mean: Mean squared wavelength
        - delta_lambda_sq: Range of squared wavelengths
    """
    if weights is None:
        weights = np.ones_like(lambda_sq)

    good_mask = weights > 0
    lambda_sq_good = lambda_sq[good_mask]

    lambda_sq_mean = np.average(lambda_sq_good, weights=weights[good_mask])
    delta_lambda_sq = np.max(lambda_sq_good) - np.min(lambda_sq_good)

    # FWHM of RMSF (approximate)
    fwhm = 2 * np.sqrt(3) / delta_lambda_sq

    # Maximum detectable scale (limited by minimum sampling in lambda^2)
    lambda_sq_sorted = np.sort(lambda_sq_good)
    min_spacing = np.min(np.abs(np.diff(lambda_sq_sorted)))
    max_scale = np.pi / min_spacing if min_spacing > 0 else np.inf

    return {
        "fwhm": fwhm,
        "max_scale": max_scale,
        "lambda_sq_mean": lambda_sq_mean,
        "delta_lambda_sq": delta_lambda_sq,
    }


def compute_faraday_spectrum(
    qu_obs: np.ndarray,
    lambda_sq: np.ndarray,
    phi: np.ndarray,
    weights: np.ndarray = None,
) -> np.ndarray:
    """
    Compute Faraday spectrum via RM synthesis.

    Parameters
    ----------
    qu_obs : np.ndarray
        Observed [Q, U] spectrum of shape (2 * n_freq,)
    lambda_sq : np.ndarray
        Squared wavelengths in m^2
    phi : np.ndarray
        Faraday depth values in rad/m^2
    weights : np.ndarray, optional
        Channel weights

    Returns
    -------
    faraday_spectrum : np.ndarray (complex)
        F(phi) at each Faraday depth
    """
    if weights is None:
        weights = np.ones_like(lambda_sq)

    n_freq = len(lambda_sq)
    Q = qu_obs[:n_freq]
    U = qu_obs[n_freq:]
    P = Q + 1j * U

    # Compute weighted mean lambda squared
    good_mask = weights > 0
    lambda_sq_mean = np.average(lambda_sq[good_mask], weights=weights[good_mask])

    # Compute Faraday spectrum
    K = np.sum(weights)
    n_phi = len(phi)
    F = np.zeros(n_phi, dtype=complex)

    for i, phi_val in enumerate(phi):
        F[i] = (
            np.sum(weights * P * np.exp(-2j * phi_val * (lambda_sq - lambda_sq_mean)))
            / K
        )

    return F
