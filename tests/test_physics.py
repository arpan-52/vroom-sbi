"""
Tests for src/simulator/physics.py.

Covers:
- load_frequencies: error handling for malformed files
- compute_rmsf_vec: vectorized implementation matches loop reference
- compute_faraday_spectrum_vec: vectorized implementation matches loop reference
"""

from pathlib import Path

import numpy as np
import pytest

from src.simulator.physics import (
    compute_faraday_spectrum,
    compute_faraday_spectrum_vec,
    compute_rmsf,
    compute_rmsf_vec,
    freq_to_lambda_sq,
    load_frequencies,
)

RNG = np.random.default_rng(42)
FREQS = np.linspace(1.0e9, 3.0e9, 64)
LAMBDA_SQ = freq_to_lambda_sq(FREQS)
PHI = np.linspace(-500.0, 500.0, 256)


# ---------------------------------------------------------------------------
# load_frequencies
# ---------------------------------------------------------------------------


class TestLoadFrequencies:
    def test_malformed_file_raises_with_filename_in_message(self, tmp_path):
        bad = tmp_path / "bad_freq.txt"
        bad.write_text("not_a_number\nalso_bad\n")
        with pytest.raises(Exception, match=str(bad)):
            load_frequencies(str(bad))

    def test_single_column_file(self, tmp_path):
        f = tmp_path / "freq.txt"
        f.write_text("1e9\n2e9\n3e9\n")
        freqs, weights = load_frequencies(str(f))
        assert len(freqs) == 3
        np.testing.assert_array_equal(weights, np.ones(3))

    def test_two_column_file(self, tmp_path):
        f = tmp_path / "freq.txt"
        f.write_text("1e9 1.0\n2e9 0.5\n3e9 0.0\n")
        freqs, weights = load_frequencies(str(f))
        assert weights[1] == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# compute_rmsf_vec
# ---------------------------------------------------------------------------


class TestComputeRmsfVec:
    def test_uniform_weights_matches_loop(self):
        w = np.ones_like(LAMBDA_SQ)
        np.testing.assert_allclose(
            compute_rmsf_vec(LAMBDA_SQ, PHI, w),
            compute_rmsf(LAMBDA_SQ, PHI, w),
            rtol=1e-10, atol=1e-14,
        )

    def test_nonuniform_weights_matches_loop(self):
        w = RNG.uniform(0.0, 1.0, size=len(LAMBDA_SQ))
        w[:4] = 0.0
        np.testing.assert_allclose(
            compute_rmsf_vec(LAMBDA_SQ, PHI, w),
            compute_rmsf(LAMBDA_SQ, PHI, w),
            rtol=1e-10, atol=1e-14,
        )

    def test_none_weights_matches_loop(self):
        np.testing.assert_allclose(
            compute_rmsf_vec(LAMBDA_SQ, PHI),
            compute_rmsf(LAMBDA_SQ, PHI),
            rtol=1e-10, atol=1e-14,
        )

    def test_output_shape_and_dtype(self):
        out = compute_rmsf_vec(LAMBDA_SQ, PHI)
        assert out.shape == (len(PHI),)
        assert np.iscomplexobj(out)

    def test_all_zero_weights_raises(self):
        with pytest.raises(ValueError, match="all weights are zero"):
            compute_rmsf_vec(LAMBDA_SQ, PHI, np.zeros_like(LAMBDA_SQ))

    def test_peak_at_zero_phi(self):
        phi_dense = np.linspace(-10.0, 10.0, 1001)
        out = compute_rmsf_vec(LAMBDA_SQ, phi_dense)
        assert abs(phi_dense[np.argmax(np.abs(out))]) < 0.1


# ---------------------------------------------------------------------------
# compute_faraday_spectrum_vec
# ---------------------------------------------------------------------------


def _make_qu(rm: float = 100.0, amp: float = 0.5) -> np.ndarray:
    lsq_mean = np.mean(LAMBDA_SQ)
    phase = 2.0 * rm * (LAMBDA_SQ - lsq_mean)
    return np.concatenate([amp * np.cos(phase), amp * np.sin(phase)])


class TestComputeFaradaySpectrumVec:
    def test_uniform_weights_matches_loop(self):
        qu = _make_qu()
        w = np.ones_like(LAMBDA_SQ)
        np.testing.assert_allclose(
            compute_faraday_spectrum_vec(qu, LAMBDA_SQ, PHI, w),
            compute_faraday_spectrum(qu, LAMBDA_SQ, PHI, w),
            rtol=1e-10, atol=1e-14,
        )

    def test_nonuniform_weights_matches_loop(self):
        qu = _make_qu(rm=200.0)
        w = RNG.uniform(0.5, 1.0, size=len(LAMBDA_SQ))
        np.testing.assert_allclose(
            compute_faraday_spectrum_vec(qu, LAMBDA_SQ, PHI, w),
            compute_faraday_spectrum(qu, LAMBDA_SQ, PHI, w),
            rtol=1e-10, atol=1e-14,
        )

    def test_none_weights_matches_loop(self):
        qu = _make_qu()
        np.testing.assert_allclose(
            compute_faraday_spectrum_vec(qu, LAMBDA_SQ, PHI),
            compute_faraday_spectrum(qu, LAMBDA_SQ, PHI),
            rtol=1e-10, atol=1e-14,
        )

    def test_output_shape_and_dtype(self):
        out = compute_faraday_spectrum_vec(_make_qu(), LAMBDA_SQ, PHI)
        assert out.shape == (len(PHI),)
        assert np.iscomplexobj(out)

    def test_peak_near_injected_rm(self):
        rm_true = 150.0
        phi_dense = np.linspace(-500.0, 500.0, 2001)
        out = compute_faraday_spectrum_vec(_make_qu(rm=rm_true), LAMBDA_SQ, phi_dense)
        assert abs(phi_dense[np.argmax(np.abs(out))] - rm_true) < 5.0
