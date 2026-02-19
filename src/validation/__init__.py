"""
Validation module for VROOM-SBI.

Provides comprehensive validation with publication-quality plots.
"""

from .validator import (
    PosteriorValidator,
    ValidationMetrics,
    TestCase,
    run_validation,
)

__all__ = [
    'PosteriorValidator',
    'ValidationMetrics',
    'TestCase',
    'run_validation',
]
