"""
Abstract base classes for VROOM-SBI components.

Defines interfaces for simulators, posteriors, and inference engines.
"""

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import torch


class BaseSimulator(ABC):
    """
    Abstract base class for RM simulators.

    Simulators generate synthetic Q, U spectra from physical parameters.
    """

    @property
    @abstractmethod
    def n_params(self) -> int:
        """Total number of parameters."""
        pass

    @property
    @abstractmethod
    def n_freq(self) -> int:
        """Number of frequency channels."""
        pass

    @property
    @abstractmethod
    def params_per_comp(self) -> int:
        """Parameters per component."""
        pass

    @abstractmethod
    def simulate(
        self, theta: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """
        Simulate Q, U spectra from parameters.

        Parameters
        ----------
        theta : np.ndarray
            Parameter array of shape (batch, n_params) or (n_params,)
        weights : np.ndarray, optional
            Channel weights of shape (n_freq,)

        Returns
        -------
        np.ndarray
            Simulated [Q, U] of shape (batch, 2*n_freq) or (2*n_freq,)
        """
        pass

    def __call__(
        self, theta: np.ndarray, weights: np.ndarray | None = None
    ) -> np.ndarray:
        """Alias for simulate()."""
        return self.simulate(theta, weights)


class PosteriorInterface(ABC):
    """
    Abstract interface for posterior distributions.

    Wraps SBI posteriors with a consistent interface.
    """

    @abstractmethod
    def sample(self, n_samples: int, x_obs: torch.Tensor) -> torch.Tensor:
        """
        Sample from posterior given observation.

        Parameters
        ----------
        n_samples : int
            Number of samples to draw
        x_obs : torch.Tensor
            Observation to condition on

        Returns
        -------
        torch.Tensor
            Posterior samples of shape (n_samples, n_params)
        """
        pass

    @abstractmethod
    def log_prob(self, theta: torch.Tensor, x_obs: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability of parameters given observation.

        Parameters
        ----------
        theta : torch.Tensor
            Parameters to evaluate
        x_obs : torch.Tensor
            Observation to condition on

        Returns
        -------
        torch.Tensor
            Log probabilities
        """
        pass

    @abstractmethod
    def to(self, device: str) -> "PosteriorInterface":
        """Move posterior to device."""
        pass


class InferenceEngineInterface(ABC):
    """
    Abstract interface for inference engines.

    Inference engines coordinate model loading and posterior inference.
    """

    @abstractmethod
    def load_models(self, max_components: int = 5):
        """Load trained posterior models."""
        pass

    @abstractmethod
    def run_inference(
        self,
        qu_obs: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 10000,
    ) -> tuple[dict[int, Any], int]:
        """
        Run inference and return results.

        Parameters
        ----------
        qu_obs : np.ndarray
            Observed Q, U spectrum
        weights : np.ndarray, optional
            Channel weights
        n_samples : int
            Number of posterior samples

        Returns
        -------
        Tuple[Dict, int]
            (results_dict, best_n_components)
        """
        pass

    @abstractmethod
    def get_model_for_n(self, n_components: int) -> PosteriorInterface | None:
        """Get posterior model for given component count."""
        pass
