"""
Tests for src/io/cube_io.py

All FITS cubes are generated synthetically by the fixtures in conftest.py.
No external data files are required.

Requires: pip install 'vroom-sbi[io]'
"""

import numpy as np
import pytest

spectral_cube = pytest.importorskip(
    "spectral_cube",
    reason="spectral_cube not installed; run: pip install 'vroom-sbi[io]'",
)

from src.io.cube_io import (  # noqa: E402
    compute_weights,
    normalize_qu_by_i,
    read_iquv_cube,
    read_qu_cubes,
    write_results_maps,
)
from tests.conftest import N_DEC, N_FREQ, N_RA  # noqa: E402

# ---------------------------------------------------------------------------
# 1. read_iquv_cube — standard StokesSpectralCube path
# ---------------------------------------------------------------------------


def test_read_iquv_cube_stokes_format(fits_iquv_path, frequencies):
    q_data, u_data, freqs, wcs_2d, i_data = read_iquv_cube(fits_iquv_path)

    assert q_data.shape == (N_FREQ, N_DEC, N_RA), f"Q shape wrong: {q_data.shape}"
    assert u_data.shape == (N_FREQ, N_DEC, N_RA), f"U shape wrong: {u_data.shape}"
    assert i_data.shape == (N_FREQ, N_DEC, N_RA), f"I shape wrong: {i_data.shape}"
    assert freqs.shape == (N_FREQ,), f"freq shape wrong: {freqs.shape}"
    assert freqs[0] == pytest.approx(frequencies[0], rel=1e-3), "first freq mismatch"
    assert wcs_2d.naxis == 2, "wcs_2d should be 2D (spatial only)"


# ---------------------------------------------------------------------------
# 2. read_iquv_cube — fallback for non-standard header
# ---------------------------------------------------------------------------


def test_read_iquv_cube_fallback(fits_iquv_nonstandard_path):
    """Non-standard header triggers astropy fallback; shapes still correct."""
    q_data, u_data, freqs, wcs_2d, i_data = read_iquv_cube(fits_iquv_nonstandard_path)

    assert q_data.shape == (N_FREQ, N_DEC, N_RA)
    assert u_data.shape == (N_FREQ, N_DEC, N_RA)
    assert i_data.shape == (N_FREQ, N_DEC, N_RA)
    assert freqs.shape == (N_FREQ,)
    assert wcs_2d.naxis == 2


# ---------------------------------------------------------------------------
# 3. read_qu_cubes — separate Q and U cubes
# ---------------------------------------------------------------------------


def test_read_qu_cubes_separate(fits_q_path, fits_u_path, frequencies):
    with pytest.warns(UserWarning, match="spectrally normalised"):
        q_data, u_data, freqs, wcs_2d = read_qu_cubes(fits_q_path, fits_u_path)

    assert q_data.shape == (N_FREQ, N_DEC, N_RA)
    assert u_data.shape == (N_FREQ, N_DEC, N_RA)
    assert freqs.shape == (N_FREQ,)
    assert freqs[0] == pytest.approx(frequencies[0], rel=1e-3)
    assert wcs_2d.naxis == 2


# ---------------------------------------------------------------------------
# 4. normalize_qu_by_i
# ---------------------------------------------------------------------------


def test_normalize_qu_by_i_correct_ratio():
    q = np.ones((10, 4, 5)) * 0.4
    u = np.ones((10, 4, 5)) * 0.3
    i = np.ones((10, 4, 5)) * 2.0

    q_n, u_n = normalize_qu_by_i(q, u, i)

    assert q_n == pytest.approx(0.2, rel=1e-6)
    assert u_n == pytest.approx(0.15, rel=1e-6)


def test_normalize_qu_by_i_zero_i_becomes_nan():
    q = np.ones((4, 2, 2))
    u = np.ones((4, 2, 2))
    i = np.ones((4, 2, 2))
    i[:, 0, 0] = 0.0  # one pixel has I=0 in all channels

    q_n, u_n = normalize_qu_by_i(q, u, i)

    assert np.all(np.isnan(q_n[:, 0, 0])), "I=0 pixel should yield NaN"
    assert np.all(np.isfinite(q_n[:, 0, 1])), "normal pixel should be finite"


# ---------------------------------------------------------------------------
# 5. compute_weights — auto estimation from Q cube
# ---------------------------------------------------------------------------


def test_compute_weights_auto(q_cube, u_cube):
    weights = compute_weights(q_cube, u_cube)

    assert weights.shape == (N_FREQ,), "global weights should be (n_freq,)"
    assert weights.max() == pytest.approx(1.0), "weights should be normalised to 1"
    assert np.all(weights >= 0), "weights must be non-negative"


def test_compute_weights_auto_nan_channel(q_cube, u_cube):
    q_nan = q_cube.copy()
    q_nan[0, :, :] = np.nan  # flag first channel

    weights = compute_weights(q_nan, u_cube)

    assert weights[0] == 0.0, "NaN channel should have weight=0"


# ---------------------------------------------------------------------------
# 6. compute_weights — from provided noise cube
# ---------------------------------------------------------------------------


def test_compute_weights_from_noise_cube(fits_noise_path, q_cube, u_cube):
    from astropy.io import fits as afits

    noise_cube = afits.getdata(fits_noise_path).astype(np.float64)

    weights = compute_weights(q_cube, u_cube, noise_cube=noise_cube)

    # noise cube has shape (n_freq, n_dec, n_ra) → weights same shape
    assert weights.shape == (N_FREQ, N_DEC, N_RA)
    assert weights.max() == pytest.approx(1.0)
    assert np.all(weights >= 0)


def test_compute_weights_zero_noise_flagged(q_cube, u_cube):
    noise = np.full((N_FREQ,), 0.1)
    noise[5] = 0.0  # one flagged channel

    weights = compute_weights(q_cube, u_cube, noise_cube=noise)

    assert weights[5] == 0.0, "zero-noise channel must have weight=0"
    assert weights.max() == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 7. write_results_maps — FITS output with correct WCS
# ---------------------------------------------------------------------------


def test_write_results_maps_fits(tmp_path, fits_q_path):
    from astropy.io import fits as afits

    wcs_2d = read_qu_cubes.__module__  # import gymnastics not needed; just build a wcs
    # Build a 2D WCS directly
    from astropy.wcs import WCS as AWcs

    wcs_2d = AWcs(naxis=2)
    wcs_2d.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_2d.wcs.crpix = [N_RA // 2 + 1, N_DEC // 2 + 1]
    wcs_2d.wcs.crval = [150.0, -30.0]
    wcs_2d.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0]

    results = {
        "rm_mean_comp1": np.full((N_DEC, N_RA), 50.0, dtype=np.float32),
        "rm_std_comp1": np.full((N_DEC, N_RA), 5.0, dtype=np.float32),
    }
    write_results_maps(results, wcs_2d, str(tmp_path / "out"))

    out_dir = tmp_path / "out"
    for name in results:
        fpath = out_dir / f"{name}.fits"
        assert fpath.exists(), f"{fpath} not written"
        data = afits.getdata(str(fpath))
        assert data.shape == (N_DEC, N_RA), f"Wrong shape in {fpath}"
        assert np.all(np.isfinite(data)), f"NaN in {fpath}"


# ---------------------------------------------------------------------------
# 8. write_results_maps — NPZ archive
# ---------------------------------------------------------------------------


def test_write_results_maps_npz(tmp_path):
    from astropy.wcs import WCS as AWcs

    wcs_2d = AWcs(naxis=2)
    wcs_2d.wcs.ctype = ["RA---SIN", "DEC--SIN"]
    wcs_2d.wcs.crpix = [1, 1]
    wcs_2d.wcs.crval = [0.0, 0.0]
    wcs_2d.wcs.cdelt = [1.0, 1.0]

    results = {
        "rm_mean_comp1": np.ones((N_DEC, N_RA)),
        "log_evidence": np.zeros((N_DEC, N_RA)),
    }
    write_results_maps(results, wcs_2d, str(tmp_path / "npz_out"))

    npz_path = tmp_path / "npz_out" / "results.npz"
    assert npz_path.exists()
    loaded = np.load(str(npz_path))
    assert set(loaded.keys()) == set(results.keys())
    np.testing.assert_array_equal(loaded["rm_mean_comp1"], results["rm_mean_comp1"])
