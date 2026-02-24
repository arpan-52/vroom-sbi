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

import numpy as np
from typing import Optional, Dict
import warnings

from .physics import load_frequencies, freq_to_lambda_sq
from ..core.base_classes import BaseSimulator


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
    
    Parameters
    ----------
    freq_file : str
        Path to frequency file
    n_components : int
        Number of RM components
    base_noise_level : float
        Base noise standard deviation
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
        base_noise_level: float = 0.01,
        model_type: str = "faraday_thin",
    ):
        self.freq, self._weights = load_frequencies(freq_file)
        self.lambda_sq = freq_to_lambda_sq(self.freq)
        self._n_freq = len(self.freq)
        self.n_components = n_components
        self.model_type = model_type.lower()
        self.base_noise_level = base_noise_level
        
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
        
        This is pure external Faraday rotation with no depolarization.
        
        Parameters: [RM, amp, chi0] per component (3N total)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self._n_freq), dtype=complex)
        
        for b in range(batch_size):
            for i in range(self.n_components):
                rm = theta[b, 3 * i]
                amp = theta[b, 3 * i + 1]
                chi0 = theta[b, 3 * i + 2]
                
                # Phase rotation: 2 * (chi0 + RM * lambda^2)
                phase = 2 * (chi0 + rm * self.lambda_sq)
                P[b] += amp * np.exp(1j * phase)
        
        return P
    
    def _compute_polarization_burn_slab(self, theta: np.ndarray) -> np.ndarray:
        """
        Burn slab model (uniform Faraday-rotating medium):
        
        P = p₀ × sinc(Δφ × λ²) × exp[2i(χ₀ + φ_c × λ²)]
        
        where sinc(x) = sin(x) / x
        
        The sinc depolarization arises from a uniform slab of
        thickness Δφ in Faraday depth.
        
        Parameters: [phi_c, delta_phi, amp, chi0] per component (4N total)
        - phi_c: Central RM (rad/m²)
        - delta_phi: Slab thickness / 2 (rad/m²), so full width = 2*delta_phi
        - amp: Intrinsic polarized intensity
        - chi0: Intrinsic polarization angle (rad)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self._n_freq), dtype=complex)
        
        for b in range(batch_size):
            for i in range(self.n_components):
                phi_c = theta[b, 4 * i]          # Central RM
                delta_phi = theta[b, 4 * i + 1]  # Slab half-width
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]
                
                # Sinc depolarization: sinc(delta_phi * lambda^2)
                arg = delta_phi * self.lambda_sq
                # Use np.sinc which computes sin(pi*x)/(pi*x), so we need arg/pi
                # Actually sinc(x) = sin(x)/x, np.sinc(x) = sin(pi*x)/(pi*x)
                # So we want sin(arg)/arg
                with np.errstate(divide='ignore', invalid='ignore'):
                    sinc_term = np.where(
                        np.abs(arg) < 1e-10, 
                        1.0, 
                        np.sin(arg) / arg
                    )
                
                # Rotation phase: 2 * (chi0 + phi_c * lambda^2)
                phase = 2 * (chi0 + phi_c * self.lambda_sq)
                
                P[b] += amp * sinc_term * np.exp(1j * phase)
        
        return P
    
    def _compute_polarization_external_dispersion(self, theta: np.ndarray) -> np.ndarray:
        """
        External Faraday dispersion (turbulent foreground screen):
        
        P = p₀ × exp(-2σ_φ² × λ⁴) × exp[2i(χ₀ + φ × λ²)]
        
        This models depolarization from a turbulent foreground with
        Gaussian RM distribution of width σ_φ.
        
        The exp(-2σ²λ⁴) term is the Fourier transform of a Gaussian
        RM distribution convolved with the source.
        
        FIXED: The formula uses λ⁴ = (λ²)², not λ²
        
        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        - phi: Mean RM (rad/m²)
        - sigma_phi: RM dispersion (rad/m²)
        - amp: Intrinsic polarized intensity  
        - chi0: Intrinsic polarization angle (rad)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self._n_freq), dtype=complex)
        
        for b in range(batch_size):
            for i in range(self.n_components):
                phi = theta[b, 4 * i]            # Mean RM
                sigma_phi = theta[b, 4 * i + 1]  # RM dispersion
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]
                
                # FIXED: Gaussian depolarization uses λ⁴ = (λ²)²
                # Formula: exp(-2 * sigma^2 * lambda^4)
                lambda_sq_squared = self.lambda_sq ** 2  # This is λ⁴
                depol = np.exp(-2 * sigma_phi**2 * lambda_sq_squared)
                
                # Rotation phase: 2 * (chi0 + phi * lambda^2)
                phase = 2 * (chi0 + phi * self.lambda_sq)
                
                P[b] += amp * depol * np.exp(1j * phase)
        
        return P
    
    def _compute_polarization_internal_dispersion(self, theta: np.ndarray) -> np.ndarray:
        """
        Internal Faraday dispersion (Sokoloff model):
        
        P = p₀ × [(1 - exp(-S)) / S] × exp(2i × χ₀)
        
        where S = 2σ_φ²λ⁴ - 2iφλ² (complex!)
        
        This models a source with internal turbulence where Faraday
        rotation and emission are mixed along the line of sight.
        
        Note: The Sokoloff formula doesn't have an exp(2i*phi*lambda^2)
        term because the RM variation is internal, not external.
        
        FIXED: 
        1. Use λ⁴ = (λ²)² in the real part of S
        2. Properly handle complex S and the limit S → 0
        
        Parameters: [phi, sigma_phi, amp, chi0] per component (4N total)
        - phi: Mean RM (rad/m²)
        - sigma_phi: RM dispersion (rad/m²)
        - amp: Intrinsic polarized intensity
        - chi0: Intrinsic polarization angle (rad)
        """
        batch_size = theta.shape[0]
        P = np.zeros((batch_size, self._n_freq), dtype=complex)
        
        for b in range(batch_size):
            for i in range(self.n_components):
                phi = theta[b, 4 * i]            # Mean RM
                sigma_phi = theta[b, 4 * i + 1]  # RM dispersion
                amp = theta[b, 4 * i + 2]
                chi0 = theta[b, 4 * i + 3]
                
                # Complex depolarization parameter S
                # S = 2σ²λ⁴ - 2iφλ²
                # FIXED: λ⁴ = (λ²)², not (λ²)
                lambda_sq_squared = self.lambda_sq ** 2  # This is λ⁴
                S = 2 * sigma_phi**2 * lambda_sq_squared - 2j * phi * self.lambda_sq
                
                # Depolarization function: (1 - exp(-S)) / S
                # Handle S → 0 limit where (1-exp(-S))/S → 1
                abs_S = np.abs(S)
                depol = np.where(
                    abs_S < 1e-10,
                    np.ones_like(S),  # Limit as S → 0
                    (1 - np.exp(-S)) / S
                )
                
                # Note: Internal dispersion includes intrinsic angle only
                # The RM rotation is already encoded in the complex S
                P[b] += amp * depol * np.exp(2j * chi0)
        
        return P
    
    def simulate(
        self, 
        theta: np.ndarray, 
        weights: Optional[np.ndarray] = None,
        noise_percent: float = 10.0
    ) -> np.ndarray:
        """
        Simulate Q, U spectra from parameters with percentage-based noise.
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters of shape (batch, n_params) or (n_params,)
        weights : np.ndarray, optional
            Channel weights of shape (n_freq,). If None, uses loaded weights.
        noise_percent : float
            Noise as percentage of signal amplitude (default: 10 means 10%)
            
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
        
        # Percentage-based noise: sigma = noise_percent/100 * |P|
        P_amplitude = np.abs(P)  # (batch, n_freq)
        min_sigma = 1e-6
        
        Q = np.zeros((batch_size, self._n_freq))
        U = np.zeros((batch_size, self._n_freq))
        
        for b in range(batch_size):
            for j in range(self._n_freq):
                if weights[j] > 0:
                    sigma = max(noise_percent / 100.0 * P_amplitude[b, j], min_sigma)
                    Q[b, j] = P[b, j].real + np.random.normal(0, sigma)
                    U[b, j] = P[b, j].imag + np.random.normal(0, sigma)
                else:
                    Q[b, j] = 0.0
                    U[b, j] = 0.0
        
        # Concatenate [Q, U]
        x = np.hstack([Q, U])
        
        return x.squeeze()
    
    def simulate_batch(
        self,
        theta: np.ndarray,
        weights_batch: np.ndarray,
        noise_percent: float = 10.0,
    ) -> np.ndarray:
        """
        Simulate batch with per-sample weights and percentage-based noise.
        
        Noise is computed as a percentage of the signal amplitude at each channel:
            sigma = noise_percent/100 * |P|
        
        Parameters
        ----------
        theta : np.ndarray
            Parameters of shape (batch, n_params)
        weights_batch : np.ndarray
            Per-sample weights of shape (batch, n_freq)
        noise_percent : float
            Noise as percentage of signal amplitude (default: 10 means 10%)
            
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
        
        # Percentage-based noise: sigma = noise_percent/100 * |P|
        # |P| is the polarized intensity at each channel
        P_amplitude = np.abs(P)  # (batch, n_freq)
        
        # Avoid zero sigma where signal is zero - use small floor
        min_sigma = 1e-6
        sigma = np.maximum(noise_percent / 100.0 * P_amplitude, min_sigma)
        
        # Generate noise
        noise_Q = np.random.normal(0, 1, (batch_size, self._n_freq)) * sigma
        noise_U = np.random.normal(0, 1, (batch_size, self._n_freq)) * sigma
        
        # Apply weights mask (zero where weight is zero)
        mask = weights_batch > 0
        
        Q = np.where(mask, P.real + noise_Q, 0.0)
        U = np.where(mask, P.imag + noise_U, 0.0)
        
        # Concatenate [Q, U]
        return np.hstack([Q, U])
    
    def simulate_noiseless(
        self, 
        theta: np.ndarray
    ) -> np.ndarray:
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
                names.extend([f"RM_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
            elif self.model_type == "burn_slab":
                names.extend([f"phi_c_{i+1}", f"delta_phi_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
            else:  # external_dispersion, internal_dispersion
                names.extend([f"phi_{i+1}", f"sigma_phi_{i+1}", f"amp_{i+1}", f"chi0_{i+1}"])
        return names
