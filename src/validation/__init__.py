"""
Validation module for VROOM-SBI.

Provides comprehensive validation with publication-quality plots.
"""

from .validator import (
    ComprehensiveValidator,
    SingleTestResult,
    run_comprehensive_validation,
)

__all__ = [
    'ComprehensiveValidator',
    'SingleTestResult',
    'run_comprehensive_validation',
]
