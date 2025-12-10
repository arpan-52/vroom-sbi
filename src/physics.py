"""
Physics functions for RM Synthesis
"""

import numpy as np
from astropy import constants as const


def load_frequencies(freq_file):
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
    # Try loading with two columns first
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


def freq_to_lambda_sq(frequencies):
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
    lambda_sq = wavelengths ** 2
    return lambda_sq


def compute_rmsf(lambda_sq, phi):
    """
    Compute the Rotation Measure Spread Function (RMSF).
    
    The RMSF is the Fourier transform of the wavelength sampling function.
    
    Parameters
    ----------
    lambda_sq : np.ndarray
        Array of squared wavelengths in m^2
    phi : np.ndarray
        Array of Faraday depth values in rad/m^2
        
    Returns
    -------
    rmsf : np.ndarray (complex)
        The RMSF as a function of phi
    """
    # Compute mean lambda squared
    lambda_sq_mean = np.mean(lambda_sq)
    
    # Initialize RMSF
    n_phi = len(phi)
    n_lambda = len(lambda_sq)
    rmsf = np.zeros(n_phi, dtype=complex)
    
    # Compute RMSF
    for i, phi_val in enumerate(phi):
        rmsf[i] = np.sum(np.exp(-2j * phi_val * (lambda_sq - lambda_sq_mean))) / n_lambda
    
    return rmsf


def get_rmsf_properties(lambda_sq):
    """
    Compute properties of the RMSF.
    
    Parameters
    ----------
    lambda_sq : np.ndarray
        Array of squared wavelengths in m^2
        
    Returns
    -------
    dict
        Dictionary containing:
        - fwhm: Full width at half maximum in rad/m^2
        - max_scale: Maximum detectable scale in rad/m^2
        - lambda_sq_mean: Mean squared wavelength
        - delta_lambda_sq: Range of squared wavelengths
    """
    lambda_sq_mean = np.mean(lambda_sq)
    delta_lambda_sq = np.max(lambda_sq) - np.min(lambda_sq)
    
    # FWHM of RMSF (approximate)
    fwhm = 2 * np.sqrt(3) / delta_lambda_sq
    
    # Maximum detectable scale
    max_scale = np.pi / np.min(np.abs(np.diff(np.sort(lambda_sq))))
    
    return {
        'fwhm': fwhm,
        'max_scale': max_scale,
        'lambda_sq_mean': lambda_sq_mean,
        'delta_lambda_sq': delta_lambda_sq
    }
