#!/usr/bin/env python
"""
run_rmtools.py — Parallel RMtools QUfitting (model m1) over a spectral cube.

Pre-processing mirrors vroom-sbi cube-infer exactly:
  - Pass 1a: per-channel noise from spatial std of Q → good_chans mask
  - Pass 1b: collapsed P map over good channels → SNR threshold
  - Frequency reordering: ascending order (same convention as vroom-sbi)
  - I normalisation: Q/I, U/I per chunk when --cube-i is supplied

For each active pixel:
  - writes a unique ASCII spectrum (7-col with I, 5-col without)
  - calls qufit -m 1 in a subprocess
  - parses the bilby result JSON
  - deletes all intermediate products

Output: 2D FITS maps in the same format as cube-infer.

Usage
-----
    python run_rmtools.py \
        --cube-q cube_Q.fits --cube-u cube_U.fits [--cube-i cube_I.fits] \
        --output-dir rmtools_results \
        --ncores 16 --snr-threshold 3.0 [--nlive 300]
"""

import argparse
import json
import logging
import shutil
import subprocess
import sys
import warnings as _warnings
from pathlib import Path
import multiprocessing as mp

import numpy as np
import psutil
from tqdm import tqdm

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Standalone cube helpers (no vroom-sbi import needed)
# ---------------------------------------------------------------------------

def _open_cube(path: str):
    try:
        import spectral_cube
    except ImportError:
        logger.error("spectral_cube is required: pip install spectral_cube")
        sys.exit(1)
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return spectral_cube.SpectralCube.read(str(path))


def _load_chunk(cube, y0, y1, x0, x1) -> np.ndarray:
    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore")
        return cube[:, y0:y1, x0:x1].unmasked_data[:, :, :].value.astype(np.float32)


def _normalize_qu_by_i(q, u, i):
    """Q/I, U/I — voxels where I <= 0 or NaN become NaN."""
    valid = (i > 0) & np.isfinite(i)
    with np.errstate(divide="ignore", invalid="ignore"):
        q_n = np.where(valid, q / i, np.nan)
        u_n = np.where(valid, u / i, np.nan)
    return q_n, u_n


# ---------------------------------------------------------------------------
# Worker (top-level for multiprocessing pickling)
# ---------------------------------------------------------------------------

def _parse_qufit_result(out_dir: Path) -> dict | None:
    """Parse bilby result JSON from qufit output directory."""
    jsons = list(out_dir.glob("*_result.json"))
    if not jsons:
        return None
    try:
        with open(jsons[0]) as f:
            data = json.load(f)
    except Exception:
        return None

    posterior = data.get("posterior", {})
    if isinstance(posterior, dict) and "content" in posterior:
        posterior = posterior["content"]

    def _get(d, *keys):
        for k in keys:
            if k in d:
                return np.asarray(d[k], dtype=float)
        return None

    rm_s  = _get(posterior, "RM0",  "rm",   "RM",  "rm0")
    amp_s = _get(posterior, "amp0", "p0",   "amp", "A0")
    chi_s = _get(posterior, "pa0",  "psi0", "chi0","PA0", "psi")

    if rm_s is None or len(rm_s) == 0:
        return None

    result = {
        "rm_mean":      float(np.mean(rm_s)),
        "rm_std":       float(np.std(rm_s)),
        "rm_p16":       float(np.percentile(rm_s, 16)),
        "rm_p84":       float(np.percentile(rm_s, 84)),
        "log_evidence": float(data.get("log_evidence", np.nan)),
    }
    if amp_s is not None and len(amp_s):
        result["amp_mean"] = float(np.mean(amp_s))
        result["amp_std"]  = float(np.std(amp_s))
        result["amp_p16"]  = float(np.percentile(amp_s, 16))
        result["amp_p84"]  = float(np.percentile(amp_s, 84))
    if chi_s is not None and len(chi_s):
        result["chi0_mean"] = float(np.mean(chi_s))
        result["chi0_std"]  = float(np.std(chi_s))
        result["chi0_p16"]  = float(np.percentile(chi_s, 16))
        result["chi0_p84"]  = float(np.percentile(chi_s, 84))
    return result


def _run_pixel(args):
    """
    Worker: write ASCII → run qufit → parse → clean up.
    Returns (dec_idx, ra_idx, params_dict_or_None).
    """
    (dec_idx, ra_idx,
     freq_hz, Q_pix, U_pix, I_pix,
     noise_q, noise_i, good_chans,
     tmp_dir, nlive) = args

    tmp_dir    = Path(tmp_dir)
    pix_name   = f"pix_{dec_idx}_{ra_idx}"
    ascii_path = tmp_dir / f"{pix_name}.dat"
    out_dir    = tmp_dir / pix_name   # qufit dumps here (cwd=tmp_dir)

    try:
        mask = good_chans & np.isfinite(Q_pix) & np.isfinite(U_pix)
        if I_pix is not None:
            mask &= np.isfinite(I_pix)
        if mask.sum() < 10:
            return dec_idx, ra_idx, None

        if I_pix is not None:
            rows = np.column_stack([
                freq_hz[mask],
                I_pix[mask],  Q_pix[mask], U_pix[mask],
                noise_i[mask], noise_q[mask], noise_q[mask],
            ])
        else:
            rows = np.column_stack([
                freq_hz[mask],
                Q_pix[mask], U_pix[mask],
                noise_q[mask], noise_q[mask],
            ])
        np.savetxt(ascii_path, rows)

        cmd = ["qufit", str(ascii_path), "-m", "1",
               "--nlive", str(nlive), "--ncores", "1"]
        if I_pix is None:
            cmd.append("-i")

        subprocess.run(cmd, capture_output=True, cwd=str(tmp_dir), check=False)

        return dec_idx, ra_idx, _parse_qufit_result(out_dir)

    except Exception:
        return dec_idx, ra_idx, None

    finally:
        if ascii_path.exists():
            ascii_path.unlink()
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Parallel RMtools QUfitting (m1) over a spectral cube."
    )
    parser.add_argument("--cube-q",        required=True, metavar="PATH")
    parser.add_argument("--cube-u",        required=True, metavar="PATH")
    parser.add_argument("--cube-i",        default=None,  metavar="PATH",
                        help="Stokes I cube (optional; enables Q/I, U/I normalisation)")
    parser.add_argument("--snr-threshold", type=float, default=3.0)
    parser.add_argument("--output-dir",    default="rmtools_results")
    parser.add_argument("--ncores",        type=int, default=1)
    parser.add_argument("--nlive",         type=int, default=300)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir = output_dir / "tmp"
    tmp_dir.mkdir(exist_ok=True)

    import astropy.units as u
    from astropy.io import fits as _fits

    # -- Open cubes lazily ---------------------------------------------------
    q_cube = _open_cube(args.cube_q)
    u_cube = _open_cube(args.cube_u)
    i_cube = _open_cube(args.cube_i) if args.cube_i else None

    if i_cube is None:
        _warnings.warn(
            "No Stokes I provided — Q and U assumed already spectrally normalised.",
            UserWarning, stacklevel=2,
        )

    frequencies_hz = q_cube.spectral_axis.to(u.Hz).value
    wcs_2d         = q_cube.wcs.celestial
    shape          = tuple(int(x) for x in q_cube.shape)   # (n_freq, n_dec, n_ra)
    n_freq, n_dec, n_ra = shape

    # -- Frequency reordering: ensure ascending (matches vroom-sbi convention)
    freq_sort_idx = None
    if frequencies_hz[0] > frequencies_hz[-1]:
        freq_sort_idx = np.arange(n_freq)[::-1]
        frequencies_hz = frequencies_hz[freq_sort_idx]
        logger.info("Cube frequencies are descending — reordering to ascending.")

    # -- Chunk size from available RAM ---------------------------------------
    n_cubes         = 3 if i_cube is not None else 2
    bytes_per_pixel = n_freq * 4 * n_cubes
    available       = psutil.virtual_memory().available
    chunk_side      = max(64, int(np.sqrt(available * 0.25 / bytes_per_pixel)))
    chunk_side      = min(chunk_side, n_dec, n_ra)
    logger.info(
        f"Available RAM: {available / 2**30:.1f} GB  →  "
        f"chunk {chunk_side}×{chunk_side} "
        f"({chunk_side**2 * bytes_per_pixel / 2**20:.0f} MB per chunk)"
    )

    # -- Pass 1a: per-channel noise → good_chans ----------------------------
    logger.info("Pass 1a: estimating per-channel noise...")
    noise_sum  = np.zeros(n_freq, dtype=np.float64)
    noise_sum2 = np.zeros(n_freq, dtype=np.float64)
    n_pix_total = 0

    for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1a (rows)", unit="row"):
        y1 = min(y0 + chunk_side, n_dec)
        for x0 in range(0, n_ra, chunk_side):
            x1 = min(x0 + chunk_side, n_ra)
            q_ch = _load_chunk(q_cube, y0, y1, x0, x1)
            if freq_sort_idx is not None:
                q_ch = q_ch[freq_sort_idx]
            fq = np.where(np.isfinite(q_ch), q_ch, 0.0)
            noise_sum  += fq.sum(axis=(1, 2))
            noise_sum2 += (fq ** 2).sum(axis=(1, 2))
            n_pix_total += (y1 - y0) * (x1 - x0)

    mean_q      = noise_sum / n_pix_total
    noise_per_chan = np.sqrt(np.maximum(noise_sum2 / n_pix_total - mean_q ** 2, 0.0))
    with np.errstate(divide="ignore", invalid="ignore"):
        weights = np.where(noise_per_chan > 0, 1.0 / noise_per_chan ** 2, 0.0)
    w_max = weights.max()
    if w_max > 0:
        weights /= w_max
    good_chans = weights > 0
    noise_q    = noise_per_chan.astype(np.float32)
    logger.info(f"Weights: {int((weights == 0).sum())}/{n_freq} channels flagged")

    # Same for I if provided
    noise_i = None
    if i_cube is not None:
        logger.info("Pass 1a (I): estimating per-channel noise for Stokes I...")
        ns_i  = np.zeros(n_freq, dtype=np.float64)
        ns2_i = np.zeros(n_freq, dtype=np.float64)
        np_i  = 0
        for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1a I (rows)", unit="row"):
            y1 = min(y0 + chunk_side, n_dec)
            for x0 in range(0, n_ra, chunk_side):
                x1 = min(x0 + chunk_side, n_ra)
                i_ch = _load_chunk(i_cube, y0, y1, x0, x1)
                if freq_sort_idx is not None:
                    i_ch = i_ch[freq_sort_idx]
                fi = np.where(np.isfinite(i_ch), i_ch, 0.0)
                ns_i  += fi.sum(axis=(1, 2))
                ns2_i += (fi ** 2).sum(axis=(1, 2))
                np_i  += (y1 - y0) * (x1 - x0)
        mean_i  = ns_i / np_i
        noise_i = np.sqrt(np.maximum(ns2_i / np_i - mean_i ** 2, 0.0)).astype(np.float32)

    # -- Pass 1b: collapsed P map over good channels ------------------------
    logger.info("Pass 1b: building P map over good channels...")
    p_map = np.zeros((n_dec, n_ra), dtype=np.float64)

    for y0 in tqdm(range(0, n_dec, chunk_side), desc="Pass 1b (rows)", unit="row"):
        y1 = min(y0 + chunk_side, n_dec)
        for x0 in range(0, n_ra, chunk_side):
            x1 = min(x0 + chunk_side, n_ra)
            q_ch = _load_chunk(q_cube, y0, y1, x0, x1)
            u_ch = _load_chunk(u_cube, y0, y1, x0, x1)
            if freq_sort_idx is not None:
                q_ch = q_ch[freq_sort_idx]
                u_ch = u_ch[freq_sort_idx]
            p_ch = np.sqrt(q_ch[good_chans] ** 2 + u_ch[good_chans] ** 2)
            p_map[y0:y1, x0:x1] = np.nanmean(p_ch, axis=0)

    p_median    = np.nanmedian(p_map)
    p_threshold = args.snr_threshold * p_median
    logger.info(
        f"P map: median={p_median:.6f}, "
        f"threshold ({args.snr_threshold}× median)={p_threshold:.6f}"
    )

    # Dump diagnostic maps (same as cube-infer)
    _fits.writeto(str(output_dir / "p_map.fits"),         p_map.astype(np.float32),         overwrite=True)
    _fits.writeto(str(output_dir / "noise_per_chan.fits"), noise_per_chan.astype(np.float32), overwrite=True)
    _fits.writeto(str(output_dir / "weights.fits"),        weights.astype(np.float32),        overwrite=True)
    logger.info("Dumped p_map.fits, noise_per_chan.fits, weights.fits")

    # -- Collect active pixel spectra (re-read RAM after allocations) --------
    logger.info("Collecting active pixel spectra...")
    available_p2  = psutil.virtual_memory().available
    chunk_side_p2 = max(64, int(np.sqrt(available_p2 * 0.25 / bytes_per_pixel)))
    chunk_side_p2 = min(chunk_side_p2, n_dec, n_ra)
    logger.info(
        f"Pass 2 available RAM: {available_p2 / 2**30:.1f} GB  →  "
        f"chunk {chunk_side_p2}×{chunk_side_p2}"
    )

    pixel_args = []
    for y0 in tqdm(range(0, n_dec, chunk_side_p2), desc="Collecting (rows)", unit="row"):
        y1 = min(y0 + chunk_side_p2, n_dec)
        for x0 in range(0, n_ra, chunk_side_p2):
            x1 = min(x0 + chunk_side_p2, n_ra)
            chunk_p = p_map[y0:y1, x0:x1]
            if not np.any(chunk_p >= p_threshold):
                continue

            q_ch = _load_chunk(q_cube, y0, y1, x0, x1)
            u_ch = _load_chunk(u_cube, y0, y1, x0, x1)
            if freq_sort_idx is not None:
                q_ch = q_ch[freq_sort_idx]
                u_ch = u_ch[freq_sort_idx]

            i_ch = None
            if i_cube is not None:
                i_ch = _load_chunk(i_cube, y0, y1, x0, x1)
                if freq_sort_idx is not None:
                    i_ch = i_ch[freq_sort_idx]
                q_ch, u_ch = _normalize_qu_by_i(q_ch, u_ch, i_ch)

            for dec_local, ra_local in np.argwhere(chunk_p >= p_threshold):
                Q_pix = q_ch[:, dec_local, ra_local].copy()
                U_pix = u_ch[:, dec_local, ra_local].copy()
                I_pix = i_ch[:, dec_local, ra_local].copy() if i_ch is not None else None

                bad = ~good_chans | ~np.isfinite(Q_pix) | ~np.isfinite(U_pix)
                Q_pix[bad] = np.nan
                U_pix[bad] = np.nan
                if I_pix is not None:
                    I_pix[bad] = np.nan

                pixel_args.append((
                    int(y0 + dec_local), int(x0 + ra_local),
                    frequencies_hz,
                    Q_pix, U_pix, I_pix,
                    noise_q, noise_i, good_chans,
                    str(tmp_dir), args.nlive,
                ))

    n_active = len(pixel_args)
    logger.info(f"Running qufit on {n_active} active pixels with {args.ncores} cores...")

    # -- Pre-allocate result maps --------------------------------------------
    shape2d = (n_dec, n_ra)
    nan2d   = lambda: np.full(shape2d, np.nan, dtype=np.float32)
    maps = {
        "rm_mean":   nan2d(), "rm_std":   nan2d(),
        "rm_p16":    nan2d(), "rm_p84":   nan2d(),
        "amp_mean":  nan2d(), "amp_std":  nan2d(),
        "amp_p16":   nan2d(), "amp_p84":  nan2d(),
        "chi0_mean": nan2d(), "chi0_std": nan2d(),
        "chi0_p16":  nan2d(), "chi0_p84": nan2d(),
        "log_evidence": nan2d(),
    }

    # -- Parallel qufit ------------------------------------------------------
    with mp.Pool(processes=args.ncores) as pool:
        for dec_idx, ra_idx, params in tqdm(
            pool.imap_unordered(_run_pixel, pixel_args),
            total=n_active,
            desc="qufit",
        ):
            if params is None:
                continue
            for key, val in params.items():
                if key in maps:
                    maps[key][dec_idx, ra_idx] = val

    # -- Write FITS maps -----------------------------------------------------
    logger.info(f"Writing FITS maps to {output_dir}/...")
    header = wcs_2d.to_header()
    for name, data in maps.items():
        _fits.PrimaryHDU(data=data, header=header).writeto(
            str(output_dir / f"{name}.fits"), overwrite=True
        )
    np.savez(str(output_dir / "results.npz"), **maps)
    logger.info(f"Wrote {len(maps)} maps + results.npz")

    # -- Clean up tmp --------------------------------------------------------
    shutil.rmtree(tmp_dir, ignore_errors=True)
    logger.info("Cleaned up tmp/")


if __name__ == "__main__":
    main()
