"""
Shared pytest fixtures for vroom-sbi tests.

Synthetic FITS cubes are generated entirely in-memory using numpy;
no external data files are required.
"""

import numpy as np
import pytest
from astropy.io import fits
from astropy.wcs import WCS


# ---------------------------------------------------------------------------
# Cube dimensions
# ---------------------------------------------------------------------------

N_FREQ = 32
N_DEC = 4
N_RA = 5
FREQ_START = 1.0e9   # Hz
FREQ_STEP = 30.0e6   # Hz (30 MHz per channel)


@pytest.fixture(scope="session")
def n_freq():
    return N_FREQ


@pytest.fixture(scope="session")
def n_dec():
    return N_DEC


@pytest.fixture(scope="session")
def n_ra():
    return N_RA


@pytest.fixture(scope="session")
def frequencies():
    return FREQ_START + np.arange(N_FREQ) * FREQ_STEP


@pytest.fixture(scope="session")
def lambda_sq(frequencies):
    c = 299792458.0
    return (c / frequencies) ** 2


# ---------------------------------------------------------------------------
# Synthetic Stokes data (known parameters for assertion)
# ---------------------------------------------------------------------------

# Ground-truth parameters for the synthetic cubes
GT_RM = 50.0      # rad/m²
GT_AMP = 0.5      # fractional polarisation
GT_CHI0 = 0.3     # rad


@pytest.fixture(scope="session")
def q_arr(lambda_sq):
    """1D Q spectrum (n_freq,) for a single Faraday-thin component."""
    return GT_AMP * np.cos(2.0 * (GT_CHI0 + GT_RM * lambda_sq))


@pytest.fixture(scope="session")
def u_arr(lambda_sq):
    """1D U spectrum (n_freq,) for a single Faraday-thin component."""
    return GT_AMP * np.sin(2.0 * (GT_CHI0 + GT_RM * lambda_sq))


@pytest.fixture(scope="session")
def i_arr(frequencies):
    """1D I spectrum with a power-law SED: I ∝ (f/f0)^{-0.7}."""
    f0 = frequencies[N_FREQ // 2]
    return (frequencies / f0) ** (-0.7)


@pytest.fixture(scope="session")
def q_cube(q_arr):
    """3D Q cube (n_freq, n_dec, n_ra); same spectrum at every pixel."""
    return np.broadcast_to(q_arr[:, None, None], (N_FREQ, N_DEC, N_RA)).copy()


@pytest.fixture(scope="session")
def u_cube(u_arr):
    """3D U cube (n_freq, n_dec, n_ra); same spectrum at every pixel."""
    return np.broadcast_to(u_arr[:, None, None], (N_FREQ, N_DEC, N_RA)).copy()


@pytest.fixture(scope="session")
def i_cube(i_arr):
    """3D I cube (n_freq, n_dec, n_ra); same SED at every pixel."""
    return np.broadcast_to(i_arr[:, None, None], (N_FREQ, N_DEC, N_RA)).copy()


# ---------------------------------------------------------------------------
# WCS helpers
# ---------------------------------------------------------------------------

def _make_wcs_3d() -> WCS:
    """Minimal 3D WCS (RA, Dec, Freq) for tests."""
    wcs = WCS(naxis=3)
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ']
    wcs.wcs.crpix = [N_RA // 2 + 1, N_DEC // 2 + 1, N_FREQ // 2 + 1]
    wcs.wcs.crval = [150.0, -30.0, FREQ_START + (N_FREQ // 2) * FREQ_STEP]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0, FREQ_STEP]
    wcs.wcs.cunit = ['deg', 'deg', 'Hz']
    return wcs


def _make_wcs_4d() -> WCS:
    """4D WCS (RA, Dec, Freq, Stokes) for IQUV cubes."""
    wcs = WCS(naxis=4)
    wcs.wcs.ctype = ['RA---SIN', 'DEC--SIN', 'FREQ', 'STOKES']
    wcs.wcs.crpix = [N_RA // 2 + 1, N_DEC // 2 + 1, N_FREQ // 2 + 1, 1.0]
    wcs.wcs.crval = [150.0, -30.0, FREQ_START + (N_FREQ // 2) * FREQ_STEP, 1.0]
    wcs.wcs.cdelt = [-1.0 / 3600.0, 1.0 / 3600.0, FREQ_STEP, 1.0]
    wcs.wcs.cunit = ['deg', 'deg', 'Hz', '']
    return wcs


# ---------------------------------------------------------------------------
# FITS file fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def fits_q_path(tmp_path, q_cube):
    """3D Q FITS cube on disk."""
    wcs = _make_wcs_3d()
    hdr = wcs.to_header()
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdr['NAXIS3'] = N_FREQ
    hdu = fits.PrimaryHDU(data=q_cube.astype(np.float32), header=hdr)
    path = tmp_path / "Q_cube.fits"
    hdu.writeto(str(path))
    return str(path)


@pytest.fixture
def fits_u_path(tmp_path, u_cube):
    """3D U FITS cube on disk."""
    wcs = _make_wcs_3d()
    hdr = wcs.to_header()
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdr['NAXIS3'] = N_FREQ
    hdu = fits.PrimaryHDU(data=u_cube.astype(np.float32), header=hdr)
    path = tmp_path / "U_cube.fits"
    hdu.writeto(str(path))
    return str(path)


@pytest.fixture
def fits_iquv_path(tmp_path, q_cube, u_cube, i_cube):
    """
    4D IQUV FITS cube (standard StokesSpectralCube format).

    Numpy array shape: (4, n_freq, n_dec, n_ra)
    FITS axis order:   NAXIS1=n_ra, NAXIS2=n_dec, NAXIS3=n_freq, NAXIS4=4
    FITS Stokes convention: 1=I, 2=Q, 3=U, 4=V
    """
    wcs = _make_wcs_4d()
    v_cube = np.zeros_like(q_cube)
    # numpy shape (4, n_freq, n_dec, n_ra) → FITS NAXIS4=4 (Stokes)
    iquv = np.stack([i_cube, q_cube, u_cube, v_cube], axis=0).astype(np.float32)

    hdr = wcs.to_header()
    hdr['NAXIS'] = 4
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdr['NAXIS3'] = N_FREQ
    hdr['NAXIS4'] = 4
    hdu = fits.PrimaryHDU(data=iquv, header=hdr)
    path = tmp_path / "IQUV_cube.fits"
    hdu.writeto(str(path))
    return str(path)


@pytest.fixture
def fits_iquv_nonstandard_path(tmp_path, q_cube, u_cube, i_cube):
    """
    4D IQUV cube with a non-standard header that will fail StokesSpectralCube
    but succeed with the astropy fallback reader.

    We mark the Stokes CTYPE as 'STOKES' (which the fallback reads) but
    omit the SPECSYS keyword that StokesSpectralCube requires, so the
    primary path raises and triggers the fallback.
    """
    v_cube = np.zeros_like(q_cube)
    iquv = np.stack([i_cube, q_cube, u_cube, v_cube], axis=0).astype(np.float32)

    hdr = fits.Header()
    hdr['NAXIS'] = 4
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdr['NAXIS3'] = N_FREQ
    hdr['NAXIS4'] = 4
    # Minimal WCS without SPECSYS / full Stokes metadata to trip up StokesSpectralCube
    hdr['CTYPE1'] = 'RA---SIN'
    hdr['CTYPE2'] = 'DEC--SIN'
    hdr['CTYPE3'] = 'FREQ'
    hdr['CTYPE4'] = 'STOKES'
    hdr['CRPIX1'] = N_RA // 2 + 1
    hdr['CRPIX2'] = N_DEC // 2 + 1
    hdr['CRPIX3'] = N_FREQ // 2 + 1
    hdr['CRPIX4'] = 1.0
    hdr['CRVAL1'] = 150.0
    hdr['CRVAL2'] = -30.0
    hdr['CRVAL3'] = FREQ_START + (N_FREQ // 2) * FREQ_STEP
    hdr['CRVAL4'] = 1.0      # I=1 at reference pixel
    hdr['CDELT1'] = -1.0 / 3600.0
    hdr['CDELT2'] = 1.0 / 3600.0
    hdr['CDELT3'] = FREQ_STEP
    hdr['CDELT4'] = 1.0
    hdu = fits.PrimaryHDU(data=iquv, header=hdr)
    path = tmp_path / "IQUV_nonstandard.fits"
    hdu.writeto(str(path))
    return str(path)


@pytest.fixture
def fits_noise_path(tmp_path):
    """3D noise cube (n_freq, n_dec, n_ra) with uniform noise = 0.1."""
    wcs = _make_wcs_3d()
    hdr = wcs.to_header()
    hdr['NAXIS'] = 3
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdr['NAXIS3'] = N_FREQ
    noise = np.full((N_FREQ, N_DEC, N_RA), 0.1, dtype=np.float32)
    hdu = fits.PrimaryHDU(data=noise, header=hdr)
    path = tmp_path / "noise_cube.fits"
    hdu.writeto(str(path))
    return str(path)


@pytest.fixture
def fits_mask_path(tmp_path):
    """2D spatial mask (n_dec, n_ra): right half of pixels = 1, left half = 0."""
    mask = np.zeros((N_DEC, N_RA), dtype=np.int16)
    mask[:, N_RA // 2:] = 1   # right half valid
    wcs2d = _make_wcs_3d().celestial
    hdr = wcs2d.to_header()
    hdr['NAXIS'] = 2
    hdr['NAXIS1'] = N_RA
    hdr['NAXIS2'] = N_DEC
    hdu = fits.PrimaryHDU(data=mask, header=hdr)
    path = tmp_path / "mask.fits"
    hdu.writeto(str(path))
    return str(path)


# ---------------------------------------------------------------------------
# Fake InferenceResult for engine mocking
# ---------------------------------------------------------------------------

def make_fake_inference_result(rm_true=GT_RM, n_samples=50):
    """Return an InferenceResult with known RM for testing cube aggregation."""
    from src.core.result import InferenceResult, ComponentResult

    rng = np.random.default_rng(42)
    rm_samp = rng.normal(rm_true, 3.0, n_samples)
    amp_samp = np.abs(rng.normal(GT_AMP, 0.05, n_samples))
    chi0_samp = rng.normal(GT_CHI0, 0.05, n_samples)

    comp = ComponentResult(
        rm_mean=float(np.mean(rm_samp)),
        rm_std=float(np.std(rm_samp)),
        q_mean=float(np.mean(amp_samp * np.cos(2 * chi0_samp))),
        q_std=0.05,
        u_mean=float(np.mean(amp_samp * np.sin(2 * chi0_samp))),
        u_std=0.05,
        samples=np.column_stack([rm_samp, amp_samp, chi0_samp]),
        chi0_mean=float(np.mean(chi0_samp)),
        chi0_std=float(np.std(chi0_samp)),
    )
    return InferenceResult(
        n_components=1,
        model_type='faraday_thin',
        log_evidence=-10.0,
        components=[comp],
        all_samples=comp.samples,
        n_posterior_samples=n_samples,
        inference_time_seconds=0.01,
    )
