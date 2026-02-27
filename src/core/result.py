"""
Result dataclasses for VROOM-SBI.

Contains structured results from training and inference.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import numpy as np


@dataclass
class ComponentResult:
    """
    Results for a single RM component.

    Contains posterior statistics and samples.
    """

    rm_mean: float
    rm_std: float
    q_mean: float  # Note: using q instead of 'amp' for Faraday-thin compatibility
    q_std: float
    u_mean: float
    u_std: float
    samples: np.ndarray = field(repr=False)

    # Additional stats for extended models
    sigma_phi_mean: float | None = None
    sigma_phi_std: float | None = None
    delta_phi_mean: float | None = None
    delta_phi_std: float | None = None
    chi0_mean: float | None = None
    chi0_std: float | None = None


@dataclass
class InferenceResult:
    """
    Complete inference results for a single model.

    Contains all component results and model evidence.
    """

    n_components: int
    model_type: str
    log_evidence: float
    components: list[ComponentResult]
    all_samples: np.ndarray = field(repr=False)

    # Optional classifier info
    classifier_probability: float | None = None

    # Metadata
    inference_time_seconds: float | None = None
    n_posterior_samples: int | None = None

    @property
    def noise_mean(self) -> float:
        """Mean noise estimate (if available)."""
        # For backwards compatibility
        return 0.0

    @property
    def noise_std(self) -> float:
        """Noise std estimate (if available)."""
        return 0.0


@dataclass
class TrainingMetrics:
    """
    Metrics from a single training epoch.
    """

    epoch: int
    train_loss: float
    val_loss: float | None = None
    learning_rate: float = 0.0
    timestamp: datetime = field(default_factory=datetime.now)

    # Additional metrics
    grad_norm: float | None = None
    memory_used_gb: float | None = None


@dataclass
class TrainingResult:
    """
    Complete training result for a single model.

    Contains training history and final model info.
    """

    model_type: str
    n_components: int
    n_simulations: int
    n_params: int

    # Training history
    metrics_history: list[TrainingMetrics] = field(default_factory=list)

    # Final metrics
    final_train_loss: float = 0.0
    final_val_loss: float | None = None
    best_epoch: int = 0

    # Paths
    model_path: str | None = None
    checkpoint_path: str | None = None

    # Timing
    training_time_seconds: float = 0.0

    # Configuration used
    config_snapshot: dict[str, Any] = field(default_factory=dict)

    def get_loss_history(self) -> dict[str, list[float]]:
        """Extract loss history as dict of lists."""
        return {
            "train_loss": [m.train_loss for m in self.metrics_history],
            "val_loss": [
                m.val_loss for m in self.metrics_history if m.val_loss is not None
            ],
            "epochs": [m.epoch for m in self.metrics_history],
        }

    def add_metric(self, metric: TrainingMetrics):
        """Add a new training metric."""
        self.metrics_history.append(metric)


@dataclass
class ClassifierResult:
    """
    Result from model selection classifier.
    """

    predicted_n_components: int
    predicted_model_type: str | None
    probabilities: dict[str, float]  # Key is "model_type_N" or just "N"
    confidence: float

    @property
    def is_high_confidence(self) -> bool:
        return self.confidence > 0.9

    @property
    def is_medium_confidence(self) -> bool:
        return 0.7 < self.confidence <= 0.9

    @property
    def is_low_confidence(self) -> bool:
        return self.confidence <= 0.7


@dataclass
class ValidationResult:
    """
    Results from posterior validation/testing.
    """

    model_type: str
    n_components: int
    n_tests: int

    # Per-parameter metrics
    param_names: list[str]
    mae: np.ndarray  # Mean absolute error
    rmse: np.ndarray  # Root mean square error
    mean_posterior_std: np.ndarray  # Average posterior width

    # Overall metrics
    overall_mae: float = 0.0
    overall_rmse: float = 0.0
    coverage_68: float = 0.0  # Fraction within 1-sigma
    coverage_95: float = 0.0  # Fraction within 2-sigma
