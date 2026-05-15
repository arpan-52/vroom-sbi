"""
Abstract base classes for VROOM-SBI components.

Defines interfaces for simulators and inference engines.
"""

from abc import ABC, abstractmethod

import numpy as np


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
    def infer(
        self,
        qu_obs: np.ndarray,
        weights: np.ndarray | None = None,
        n_samples: int = 10000,
    ):
        """Run inference and return results."""
        pass

    @abstractmethod
    def run_inference_cube_chunked(self, *args, **kwargs):
        """Run inference over a spatial cube in chunks."""
        pass
