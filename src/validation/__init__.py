"""
Validation module for VROOM-SBI.

Provides comprehensive validation with publication-quality plots.
"""

from .validator import (
    ComprehensiveValidator,
    TestCase,
    RMToolsResult,
    run_comprehensive_validation,
)

__all__ = [
    'ComprehensiveValidator',
    'TestCase',
    'RMToolsResult',
    'run_comprehensive_validation',
]
