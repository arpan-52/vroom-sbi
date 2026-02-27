"""
Tests for InferenceEngine.run_inference_cube()

The InferenceEngine is used with a mocked ``infer()`` method so that no
trained posterior files are required.  Tests verify that the cube loop
correctly aggregates pixel results into 2D maps, respects masks, and
handles error maps (p16/p84).
"""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.inference.engine import InferenceEngine
from tests.conftest import (
    N_DEC,
    N_FREQ,
    N_RA,
    make_fake_inference_result,
)

spectral_cube = pytest.importorskip(
    "spectral_cube",
    reason="spectral_cube not installed; run: pip install 'vroom-sbi[io]'",
)


# ---------------------------------------------------------------------------
# Helper: build a minimal InferenceEngine with mocked posterior
# ---------------------------------------------------------------------------


def _make_engine():
    """Return an InferenceEngine with one fake posterior loaded."""
    engine = InferenceEngine.__new__(InferenceEngine)
    engine.config = None
    engine.model_dir = None
    engine.device = "cpu"
    engine.posteriors = {"faraday_thin_n1": MagicMock()}
    engine.posterior_metadata = {"faraday_thin_n1": {}}
    engine.classifier = None
    from src.config.configuration import MemoryConfig

    engine.memory_config = MemoryConfig()
    engine._models_on_device = {"faraday_thin_n1": "cpu"}
    return engine


# ---------------------------------------------------------------------------
# 9. Full IQUV path: result map has non-NaN values at all valid pixels
# ---------------------------------------------------------------------------


def test_cube_inference_iquv(q_cube, u_cube):
    engine = _make_engine()
    fake_result = make_fake_inference_result()

    with patch.object(
        engine, "infer", return_value=(fake_result, {"faraday_thin_n1": fake_result})
    ):
        results = engine.run_inference_cube(q_cube, u_cube, n_samples=10)

    rm_map = results["rm_mean_comp1"]
    assert rm_map.shape == (N_DEC, N_RA)
    assert np.all(np.isfinite(rm_map)), "All valid pixels should have finite RM"
    assert rm_map[0, 0] == pytest.approx(fake_result.components[0].rm_mean, rel=1e-6)


# ---------------------------------------------------------------------------
# 10. Q+U only path (no IQUV): same behaviour with normalised inputs
# ---------------------------------------------------------------------------


def test_cube_inference_qu_only(q_cube, u_cube):
    engine = _make_engine()
    fake_result = make_fake_inference_result()

    with patch.object(engine, "infer", return_value=(fake_result, {})):
        results = engine.run_inference_cube(q_cube, u_cube, n_samples=10)

    assert results["rm_mean_comp1"].shape == (N_DEC, N_RA)
    assert np.all(np.isfinite(results["rm_mean_comp1"]))


# ---------------------------------------------------------------------------
# 11. NaN masking: pixels with NaN in Q or U stay NaN in all outputs
# ---------------------------------------------------------------------------


def test_cube_inference_nan_masking(q_cube, u_cube):
    q_nan = q_cube.copy()
    q_nan[:, 1, 2] = np.nan  # flag pixel (1, 2) in all channels

    engine = _make_engine()
    fake_result = make_fake_inference_result()

    with patch.object(engine, "infer", return_value=(fake_result, {})):
        results = engine.run_inference_cube(q_nan, u_cube, n_samples=10)

    rm_map = results["rm_mean_comp1"]
    assert np.isnan(rm_map[1, 2]), "NaN pixel should remain NaN in output"
    # All other pixels should be finite
    mask = np.ones((N_DEC, N_RA), dtype=bool)
    mask[1, 2] = False
    assert np.all(np.isfinite(rm_map[mask]))


# ---------------------------------------------------------------------------
# 12. User mask: only unmasked pixels (mask==True) are processed
# ---------------------------------------------------------------------------


def test_cube_inference_user_mask(q_cube, u_cube):
    # Mask: only the right half of pixels
    mask = np.zeros((N_DEC, N_RA), dtype=bool)
    mask[:, N_RA // 2 :] = True

    engine = _make_engine()
    fake_result = make_fake_inference_result()

    with patch.object(engine, "infer", return_value=(fake_result, {})):
        results = engine.run_inference_cube(q_cube, u_cube, mask=mask, n_samples=10)

    rm_map = results["rm_mean_comp1"]
    # Left half should be NaN (not processed)
    assert np.all(np.isnan(rm_map[:, : N_RA // 2])), "masked-out pixels should be NaN"
    # Right half should be finite
    assert np.all(np.isfinite(rm_map[:, N_RA // 2 :])), (
        "unmasked pixels should be finite"
    )


# ---------------------------------------------------------------------------
# 13. SNR threshold: low-SNR pixels are excluded
# ---------------------------------------------------------------------------


def test_cube_inference_snr_threshold(q_cube, u_cube):
    # Add a zero-signal pixel (SNR = 0) at (0, 0)
    q_zero = q_cube.copy()
    u_zero = u_cube.copy()
    q_zero[:, 0, 0] = 0.0
    u_zero[:, 0, 0] = 0.0

    engine = _make_engine()
    fake_result = make_fake_inference_result()

    # Use a high SNR threshold; the zero pixel should be excluded
    with patch.object(engine, "infer", return_value=(fake_result, {})):
        results = engine.run_inference_cube(
            q_zero, u_zero, snr_threshold=1.0, n_samples=10
        )

    rm_map = results["rm_mean_comp1"]
    assert np.isnan(rm_map[0, 0]), "zero-SNR pixel should be excluded (NaN)"


# ---------------------------------------------------------------------------
# 14. Noise weights: spatially varying weights are passed to infer()
# ---------------------------------------------------------------------------


def test_cube_inference_noise_weights(q_cube, u_cube):
    """Per-pixel weights (3D array) are sliced correctly per pixel."""
    weights_3d = np.ones((N_FREQ, N_DEC, N_RA), dtype=np.float32)
    weights_3d[:, 0, 0] = 0.0  # flag all channels for pixel (0,0)

    engine = _make_engine()
    fake_result = make_fake_inference_result()

    captured_weights = []

    def mock_infer(qu_obs, weights=None, n_samples=1000, **kw):
        captured_weights.append(weights)
        return fake_result, {}

    with patch.object(engine, "infer", side_effect=mock_infer):
        engine.run_inference_cube(q_cube, u_cube, weights=weights_3d, n_samples=10)

    # At least one call should have received the per-pixel zero-weight slice
    zero_weight_calls = [
        w for w in captured_weights if w is not None and np.all(w == 0)
    ]
    assert len(zero_weight_calls) > 0, "pixel (0,0) should receive all-zero weights"


# ---------------------------------------------------------------------------
# 15. Error maps: rm_std, rm_p16, rm_p84 are all present and ordered
# ---------------------------------------------------------------------------


def test_cube_inference_error_maps(q_cube, u_cube):
    engine = _make_engine()
    fake_result = make_fake_inference_result()

    with patch.object(engine, "infer", return_value=(fake_result, {})):
        results = engine.run_inference_cube(q_cube, u_cube, n_samples=10)

    rm_mean = results["rm_mean_comp1"]
    rm_std = results["rm_std_comp1"]
    rm_p16 = results["rm_p16_comp1"]
    rm_p84 = results["rm_p84_comp1"]

    # All maps must exist and be finite for non-NaN pixels
    for name, arr in [("rm_std", rm_std), ("rm_p16", rm_p16), ("rm_p84", rm_p84)]:
        assert arr.shape == (N_DEC, N_RA), f"{name} wrong shape"
        assert np.all(np.isfinite(arr)), f"{name} has NaN/Inf"

    # Ordering: p16 <= mean <= p84  (element-wise)
    assert np.all(rm_p16 <= rm_mean + 1e-6), "p16 should be <= mean"
    assert np.all(rm_mean <= rm_p84 + 1e-6), "mean should be <= p84"
    assert np.all(rm_std >= 0), "std must be non-negative"
