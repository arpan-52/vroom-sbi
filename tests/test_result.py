"""
Tests for src/core/result.py.
"""

import numpy as np
import pytest

from src.core.result import ComponentResult, InferenceResult


class TestComponentResult:
    def _fields(self):
        return {f.name for f in ComponentResult.__dataclass_fields__.values()}

    def test_has_amp_mean(self):
        assert "amp_mean" in self._fields()

    def test_has_amp_std(self):
        assert "amp_std" in self._fields()

    def test_no_q_mean(self):
        assert "q_mean" not in self._fields()

    def test_no_u_mean(self):
        assert "u_mean" not in self._fields()

    def test_can_instantiate_with_new_fields(self):
        comp = ComponentResult(
            rm_mean=25.0,
            rm_std=1.0,
            amp_mean=0.3,
            amp_std=0.05,
            samples=np.zeros((10, 3)),
        )
        assert comp.amp_mean == pytest.approx(0.3)


class TestInferenceResult:
    def test_no_noise_mean_stub(self):
        assert not hasattr(InferenceResult, "noise_mean")

    def test_no_noise_std_stub(self):
        assert not hasattr(InferenceResult, "noise_std")
