"""
SpectralCube-based FITS I/O for VROOM-SBI cube inference.

Handles reading of:
  - Separate Stokes Q and U cubes (3D each: freq, dec, ra)
  - Full 4D IQUV Stokes cubes

Handles writing of:
  - 2D FITS maps per inferred parameter with spatial WCS
  - NPZ archive of all result arrays
"""

import logging
import warnings
from pathlib import Path

import astropy.units as u
import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

logger = logging.getLogger(__name__)


def _get_spectral_cube():
    """Lazy import of spectral_cube with a helpful error message."""
    try:
        import spectral_cube

        return spectral_cube
    except ImportError:
        raise ImportError(
            "spectral_cube is required for FITS cube I/O.\n"
            "Install it with:  pip install 'vroom-sbi[io]'\n"
            "or:               pip install spectral_cube"
        )


# ---------------------------------------------------------------------------
# Reading
# ---------------------------------------------------------------------------


def read_iquv_cube(
    fits_path: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WCS, np.ndarray]:
    """
    Read a 4D IQUV FITS cube and return Q, U, I data arrays.

    Tries ``StokesSpectralCube`` first; falls back to ``astropy.io.fits``
    for non-standard headers.

    Parameters
    ----------
    fits_path : str or Path
        Path to a 4D Stokes FITS cube.

    Returns
    -------
    q_data : np.ndarray, shape (n_freq, n_dec, n_ra)
    u_data : np.ndarray, shape (n_freq, n_dec, n_ra)
    frequencies_hz : np.ndarray, shape (n_freq,)
        Frequencies in Hz extracted from the FITS WCS.
    wcs_2d : astropy.wcs.WCS
        2D celestial WCS (RA, Dec) for writing output maps.
    i_data : np.ndarray, shape (n_freq, n_dec, n_ra)
        Stokes I, used for Q/I, U/I normalisation.
    """
    sc_mod = _get_spectral_cube()
    fits_path = str(fits_path)

    # Primary: StokesSpectralCube
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sc = sc_mod.StokesSpectralCube.read(fits_path)

        q_cube = sc["Q"]
        u_cube = sc["U"]
        i_cube = sc["I"]

        q_data = q_cube.unmasked_data[:, :, :].value
        u_data = u_cube.unmasked_data[:, :, :].value
        i_data = i_cube.unmasked_data[:, :, :].value
        frequencies_hz = q_cube.spectral_axis.to(u.Hz).value
        wcs_2d = q_cube.wcs.celestial

        logger.info(
            f"Read IQUV cube via StokesSpectralCube: "
            f"shape {q_data.shape}, {len(frequencies_hz)} channels"
        )
        return q_data, u_data, frequencies_hz, wcs_2d, i_data

    except Exception as primary_exc:
        logger.warning(
            f"StokesSpectralCube.read failed ({primary_exc}); "
            "falling back to astropy.io.fits"
        )
        return _read_iquv_fallback(fits_path, sc_mod)


def _read_iquv_fallback(
    fits_path: str, sc_mod
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WCS, np.ndarray]:
    """
    Fallback reader for IQUV cubes with non-standard headers.

    Locates the Stokes axis via CTYPEn = 'STOKES', slices Q, U, I along
    it, and builds a 3D SpectralCube for frequency/WCS extraction.
    """
    with fits.open(fits_path) as hdul:
        data = hdul[0].data.astype(np.float64)
        header = hdul[0].header

    naxis = header["NAXIS"]

    # Find Stokes axis (1-based FITS axis numbering)
    stokes_fits_ax = None
    for ax in range(1, naxis + 1):
        if "STOKES" in header.get(f"CTYPE{ax}", "").upper():
            stokes_fits_ax = ax
            break

    if stokes_fits_ax is None:
        raise ValueError(
            f"No Stokes axis found in '{fits_path}'. "
            "Expected a header keyword CTYPEn = 'STOKES'."
        )

    # FITS axis order is reversed relative to numpy
    numpy_stokes_ax = naxis - stokes_fits_ax

    # Compute Stokes value at each pixel along that axis
    crpix = float(header.get(f"CRPIX{stokes_fits_ax}", 1))
    crval = float(header.get(f"CRVAL{stokes_fits_ax}", 1))
    cdelt = float(header.get(f"CDELT{stokes_fits_ax}", 1))
    n_stokes = data.shape[numpy_stokes_ax]
    stokes_vals = crval + (np.arange(1, n_stokes + 1) - crpix) * cdelt

    # FITS Stokes convention: I=1, Q=2, U=3, V=4
    def _find_stokes(target: float) -> int:
        dists = np.abs(stokes_vals - target)
        idx = int(np.argmin(dists))
        if dists[idx] > 0.5:
            raise ValueError(
                f"Stokes {target} not found in cube (values: {stokes_vals}). "
                f"Is the Stokes axis correctly described in the FITS header?"
            )
        return idx

    i_idx = _find_stokes(1)
    q_idx = _find_stokes(2)
    u_idx = _find_stokes(3)

    q_3d = np.take(data, q_idx, axis=numpy_stokes_ax)
    u_3d = np.take(data, u_idx, axis=numpy_stokes_ax)
    i_3d = np.take(data, i_idx, axis=numpy_stokes_ax)

    # Build 3D WCS by dropping the Stokes axis (0-based for dropaxis)
    wcs_full = WCS(header)
    wcs_3d = wcs_full.dropaxis(stokes_fits_ax - 1)
    wcs_2d = wcs_3d.celestial

    # Build a SpectralCube from the 3D Q slice to extract frequencies cleanly
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_cube = sc_mod.SpectralCube(
            data=q_3d * u.dimensionless_unscaled,
            wcs=wcs_3d,
        )
    frequencies_hz = q_cube.spectral_axis.to(u.Hz).value

    logger.info(
        f"Read IQUV cube via fallback: shape {q_3d.shape}, "
        f"{len(frequencies_hz)} channels"
    )
    return q_3d, u_3d, frequencies_hz, wcs_2d, i_3d


def read_qu_cubes(
    q_path: str, u_path: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, WCS]:
    """
    Read separate Q and U 3D spectral cubes.

    Issues a UserWarning that Q and U are assumed to be spectrally
    normalised (fractional polarisation) since Stokes I is unavailable.

    Parameters
    ----------
    q_path, u_path : str or Path
        Paths to the Q and U FITS cubes respectively.

    Returns
    -------
    q_data : np.ndarray, shape (n_freq, n_dec, n_ra)
    u_data : np.ndarray, shape (n_freq, n_dec, n_ra)
    frequencies_hz : np.ndarray, shape (n_freq,)
    wcs_2d : astropy.wcs.WCS
    """
    sc_mod = _get_spectral_cube()

    warnings.warn(
        "No Stokes I provided — assuming Q and U are already spectrally "
        "normalised (fractional polarisation p = Q/I, U/I). "
        "If they are not, provide a full IQUV cube with --cube instead.",
        UserWarning,
        stacklevel=2,
    )

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        q_cube = sc_mod.SpectralCube.read(str(q_path))
        u_cube = sc_mod.SpectralCube.read(str(u_path))

    q_data = q_cube.unmasked_data[:, :, :].value
    u_data = u_cube.unmasked_data[:, :, :].value

    if q_data.shape != u_data.shape:
        raise ValueError(
            f"Q and U cubes have incompatible shapes: {q_data.shape} vs {u_data.shape}"
        )

    frequencies_hz = q_cube.spectral_axis.to(u.Hz).value
    wcs_2d = q_cube.wcs.celestial

    logger.info(f"Read Q/U cubes: shape {q_data.shape}, {len(frequencies_hz)} channels")
    return q_data, u_data, frequencies_hz, wcs_2d


# ---------------------------------------------------------------------------
# Normalisation
# ---------------------------------------------------------------------------


def normalize_qu_by_i(
    q_data: np.ndarray,
    u_data: np.ndarray,
    i_data: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Normalise Q and U by Stokes I to obtain fractional polarisation.

    Removes the spectral dependence of total intensity (e.g. spectral
    index).  Voxels where I <= 0 or I is NaN are set to NaN; they are
    excluded automatically by the downstream NaN masking layer.

    Parameters
    ----------
    q_data, u_data, i_data : np.ndarray, shape (n_freq, n_dec, n_ra)

    Returns
    -------
    q_norm, u_norm : np.ndarray, shape (n_freq, n_dec, n_ra)
        Fractional polarisation: Q/I, U/I.
    """
    valid_i = (i_data > 0) & np.isfinite(i_data)

    with np.errstate(divide="ignore", invalid="ignore"):
        q_norm = np.where(valid_i, q_data / i_data, np.nan)
        u_norm = np.where(valid_i, u_data / i_data, np.nan)

    n_bad = int(np.sum(~valid_i))
    if n_bad > 0:
        logger.info(f"normalize_qu_by_i: {n_bad} voxels with I <= 0 or NaN set to NaN")
    return q_norm, u_norm


# ---------------------------------------------------------------------------
# Weights
# ---------------------------------------------------------------------------


def compute_weights(
    q_data: np.ndarray,
    u_data: np.ndarray,
    noise_cube: np.ndarray | None = None,
) -> np.ndarray:
    """
    Compute per-channel inverse-variance weights.

    Parameters
    ----------
    q_data, u_data : np.ndarray, shape (n_freq, n_dec, n_ra)
    noise_cube : np.ndarray, optional
        Per-channel noise values.
        - Shape ``(n_freq,)`` → global weights (same for every pixel).
        - Shape ``(n_freq, n_dec, n_ra)`` → spatially varying weights.
        If *None*, noise is estimated per channel from the spatial standard
        deviation of the Q cube.

    Returns
    -------
    weights : np.ndarray
        Inverse-variance weights, normalised to [0, 1].
        Shape is ``(n_freq,)`` for global or ``(n_freq, n_dec, n_ra)``
        for spatially varying.  Flagged channels (noise == 0 or NaN)
        receive weight 0.
    """
    if noise_cube is not None:
        noise = np.asarray(noise_cube, dtype=np.float64)
        logger.info(f"Using provided noise cube, shape {noise.shape}")
    else:
        # Estimate per-channel noise from the spatial std of Q
        noise = np.nanstd(q_data, axis=(1, 2))  # shape (n_freq,)
        logger.info("Estimating per-channel noise from spatial std of Q cube")

    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where((noise > 0) & np.isfinite(noise), 1.0 / noise**2, 0.0)

    w_max = float(np.nanmax(weights))
    if w_max > 0:
        weights = weights / w_max

    n_flagged = int(np.sum(weights == 0))
    n_total = weights.size
    logger.info(f"Weights: {n_flagged}/{n_total} flagged (weight=0)")
    return weights


# ---------------------------------------------------------------------------
# Writing
# ---------------------------------------------------------------------------


def write_results_maps(
    results_dict: dict[str, np.ndarray],
    wcs_2d: WCS,
    output_dir: str,
    overwrite: bool = True,
) -> None:
    """
    Write inference result maps to FITS files and a NPZ archive.

    One FITS file is created per entry in ``results_dict``; all share the
    same 2D spatial WCS from the input cube.  A ``results.npz`` containing
    all arrays is also written for downstream numpy analysis.

    Parameters
    ----------
    results_dict : dict
        ``{name: 2D array}`` mapping.  ``name`` becomes the stem of the
        output filename (e.g. ``"rm_mean_comp1"`` → ``rm_mean_comp1.fits``).
    wcs_2d : astropy.wcs.WCS
        2D celestial WCS carried over from the input cube.
    output_dir : str or Path
        Output directory (created if it does not exist).
    overwrite : bool
        Overwrite existing files.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    header = wcs_2d.to_header()

    for name, data in results_dict.items():
        out_path = output_dir / f"{name}.fits"
        hdu = fits.PrimaryHDU(data=data.astype(np.float32), header=header)
        hdu.writeto(str(out_path), overwrite=overwrite)
        logger.debug(f"Wrote {out_path}")

    npz_path = output_dir / "results.npz"
    np.savez(str(npz_path), **results_dict)

    logger.info(
        f"Wrote {len(results_dict)} parameter maps and results.npz to {output_dir}"
    )
