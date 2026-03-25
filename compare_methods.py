#!/usr/bin/env python
"""
compare_methods.py — Compare RM estimates from three methods:
    1. RM Synthesis  (RMtools peak RM)
    2. VROOM-SBI     (posterior mean / std)
    3. RMtools QUfitting (model m1, bilby multinest)

For each detected pixel (above SNR threshold in polarised intensity):
  - Reads Q, U, I spectra from cubes
  - Runs `qufit -m 1` via subprocess
  - Collects RM synthesis + VROOM-SBI values from pre-computed FITS maps
  - Saves everything to a single CSV table

Usage
-----
    micromamba run -n 310data python compare_methods.py \
        --cube-q cube_Q_regrid.fits \
        --cube-u cube_U_regrid.fits \
        --cube-i cube_I_regrid.fits \
        --rm-syn-map  results_vroom/out_peak_rm.fits \
        --rm-syn-err  results_vroom/out_rm_err.fits \
        --rm-syn-pi   results_vroom/out_peak_pi.fits \
        --sbi-dir     results/ \
        --noise-map   noise_per_chan.fits \
        --pixel-csv   pixel_comparison.csv \
        --nlive 300 \
        --ncores 8 \
        --output comparison_table.csv
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import warnings as _warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.wcs import WCS
from tqdm import tqdm


# ---------------------------------------------------------------------------
# QUfit result parser
# ---------------------------------------------------------------------------

def parse_qufit_result(out_dir: Path) -> dict:
    """Parse bilby result JSON from qufit output directory."""
    jsons = list(out_dir.glob("*_result.json"))
    if not jsons:
        return {}
    try:
        data = json.loads(jsons[0].read_text())
    except Exception:
        return {}

    post = data.get("posterior", {})
    if isinstance(post, dict) and "content" in post:
        post = post["content"]

    def _get(d, *keys):
        for k in keys:
            if k in d:
                v = np.asarray(d[k], dtype=float)
                return v if len(v) else None
        return None

    rm_s  = _get(post, "RM_radm2", "RM0", "rm", "RM", "rm0")
    amp_s = _get(post, "fracPol", "amp0", "p0", "amp", "A0")
    chi_s = _get(post, "psi0_deg", "pa0", "psi0", "chi0", "PA0", "psi")

    if rm_s is None:
        return {}

    out = {
        "rm_qufit_mean": float(np.mean(rm_s)),
        "rm_qufit_std":  float(np.std(rm_s)),
        "rm_qufit_med":  float(np.median(rm_s)),
        "rm_qufit_p16":  float(np.percentile(rm_s, 16)),
        "rm_qufit_p84":  float(np.percentile(rm_s, 84)),
        "log_evidence":  float(data.get("log_evidence", np.nan)),
    }
    if amp_s is not None:
        out.update({
            "amp_qufit_mean": float(np.mean(amp_s)),
            "amp_qufit_std":  float(np.std(amp_s)),
            "amp_qufit_p16":  float(np.percentile(amp_s, 16)),
            "amp_qufit_p84":  float(np.percentile(amp_s, 84)),
        })
    if chi_s is not None:
        out.update({
            "chi0_qufit_mean": float(np.mean(chi_s)),
            "chi0_qufit_std":  float(np.std(chi_s)),
            "chi0_qufit_p16":  float(np.percentile(chi_s, 16)),
            "chi0_qufit_p84":  float(np.percentile(chi_s, 84)),
        })
    return out


# ---------------------------------------------------------------------------
# Pixel-level QUfitting
# ---------------------------------------------------------------------------

def run_qufit_pixel(
    freq_hz, Q_pix, U_pix, I_pix,
    noise_q, noise_i, good_chans,
    row, col, tmp_dir, nlive,
):
    """Write ASCII spectrum, call qufit, parse result, clean up."""
    pix_name   = f"pix_{row}_{col}"
    ascii_path = tmp_dir / f"{pix_name}.dat"
    out_dir    = tmp_dir / f"{pix_name}_m1_dynesty"

    # Pass raw Q, U (and I if available) — let qufit handle Q/I normalisation
    has_i = I_pix is not None

    sel = good_chans & np.isfinite(Q_pix) & np.isfinite(U_pix)
    if has_i:
        sel &= np.isfinite(I_pix) & (I_pix > 0)
    if sel.sum() < 10:
        return {}

    if has_i:
        rows = np.column_stack([
            freq_hz[sel], I_pix[sel], Q_pix[sel], U_pix[sel],
            noise_i[sel], noise_q[sel], noise_q[sel],
        ])
    else:
        rows = np.column_stack([
            freq_hz[sel], Q_pix[sel], U_pix[sel],
            noise_q[sel], noise_q[sel],
        ])
    np.savetxt(ascii_path, rows)

    cmd = ["qufit", str(ascii_path), "-m", "1",
           "--nlive", str(nlive), "--ncores", "1"]
    if not has_i:
        cmd.append("-i")

    env = dict(os.environ, MPLBACKEND="Agg")

    try:
        subprocess.run(cmd, capture_output=True, cwd=str(tmp_dir), env=env, check=False)
        return parse_qufit_result(out_dir)
    except Exception:
        return {}
    finally:
        if ascii_path.exists():
            ascii_path.unlink()
        if out_dir.exists():
            shutil.rmtree(out_dir, ignore_errors=True)
        # qufit also writes files next to the .dat (e.g. fig, .dat summary)
        for ext in (".dat", ".pdf", ".png"):
            for f in tmp_dir.glob(f"{pix_name}*{ext}"):
                f.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Compare RM estimates: RM Synthesis vs VROOM-SBI vs QUfitting"
    )
    # Input cubes
    parser.add_argument("--cube-q", required=True, help="Stokes Q cube FITS")
    parser.add_argument("--cube-u", required=True, help="Stokes U cube FITS")
    parser.add_argument("--cube-i", default=None,  help="Stokes I cube FITS (optional)")

    # Pre-computed maps
    parser.add_argument("--rm-syn-map", required=True, help="RM synthesis peak RM FITS")
    parser.add_argument("--rm-syn-err", default=None,  help="RM synthesis error FITS")
    parser.add_argument("--rm-syn-pi",  default=None,  help="RM synthesis peak PI FITS")
    parser.add_argument("--sbi-dir",    required=True, help="VROOM-SBI results directory")
    parser.add_argument("--noise-map",  required=True, help="Per-channel noise FITS")

    # Pixel selection: read from existing CSV
    parser.add_argument("--pixel-csv",  required=True,
                        help="CSV with 'row' and 'col' columns (e.g. pixel_comparison.csv)")

    # Parameters
    parser.add_argument("--nlive",  type=int, default=300, help="bilby nlive")
    parser.add_argument("--ncores", type=int, default=1,
                        help="Number of parallel qufit workers (serial per pixel)")

    # Output
    parser.add_argument("--output", default="comparison_table.csv", help="Output CSV")
    args = parser.parse_args()

    # -- Read pixel list from CSV --------------------------------------------
    pixel_df = pd.read_csv(args.pixel_csv)
    assert "row" in pixel_df.columns and "col" in pixel_df.columns, \
        f"pixel-csv must have 'row' and 'col' columns, got: {list(pixel_df.columns)}"
    rows_sel = pixel_df["row"].values.astype(int)
    cols_sel = pixel_df["col"].values.astype(int)
    print(f"Read {len(rows_sel)} pixels from {args.pixel_csv}")

    # -- Load 2D maps -------------------------------------------------------
    print("Loading maps...")
    rm_syn = fits.getdata(args.rm_syn_map).squeeze().astype(float)

    rm_syn_err = None
    if args.rm_syn_err:
        rm_syn_err = fits.getdata(args.rm_syn_err).squeeze().astype(float)

    rm_syn_pi = None
    if args.rm_syn_pi:
        rm_syn_pi = fits.getdata(args.rm_syn_pi).squeeze().astype(float)

    sbi_dir = Path(args.sbi_dir)

    # VROOM-SBI maps (comp1)
    sbi_maps = {}
    sbi_files = {
        "rm_sbi_mean":  "rm_mean_comp1.fits",
        "rm_sbi_std":   "rm_std_comp1.fits",
        "rm_sbi_p16":   "rm_p16_comp1.fits",
        "rm_sbi_p84":   "rm_p84_comp1.fits",
        "amp_sbi_mean": "amp_mean_comp1.fits",
        "amp_sbi_std":  "amp_std_comp1.fits",
        "amp_sbi_p16":  "amp_p16_comp1.fits",
        "amp_sbi_p84":  "amp_p84_comp1.fits",
        "chi0_sbi_mean":"chi0_mean_comp1.fits",
        "chi0_sbi_std": "chi0_std_comp1.fits",
        "chi0_sbi_p16": "chi0_p16_comp1.fits",
        "chi0_sbi_p84": "chi0_p84_comp1.fits",
        "sbi_log_evidence": "log_evidence.fits",
        "sbi_n_components":  "n_components.fits",
    }
    for key, fname in sbi_files.items():
        fpath = sbi_dir / fname
        if fpath.exists():
            sbi_maps[key] = fits.getdata(str(fpath)).squeeze().astype(float)
        else:
            print(f"  [warn] {fpath} not found — column will be NaN")

    # Use ra/dec from pixel CSV if available
    has_coords = "ra_deg" in pixel_df.columns and "dec_deg" in pixel_df.columns
    if has_coords:
        ra_deg  = pixel_df["ra_deg"].values
        dec_deg = pixel_df["dec_deg"].values
        print("Using ra_deg/dec_deg from pixel CSV")

    # -- Per-channel noise ---------------------------------------------------
    noise_q = fits.getdata(args.noise_map).squeeze().astype(np.float32)
    noise_i = noise_q.copy()  # conservative estimate
    good_chans = noise_q > 0
    print(f"Noise channels: {good_chans.sum()}/{len(good_chans)} good")

    # -- Open cubes (memmap) -------------------------------------------------
    hq = fits.open(args.cube_q, memmap=True)
    hu = fits.open(args.cube_u, memmap=True)
    hi = fits.open(args.cube_i, memmap=True) if args.cube_i else None
    q_cube = hq[0].data.squeeze()
    u_cube = hu[0].data.squeeze()
    i_cube = hi[0].data.squeeze() if hi else None

    n_freq = q_cube.shape[0]

    # Frequency axis
    try:
        import spectral_cube
        import astropy.units as u
        with _warnings.catch_warnings():
            _warnings.simplefilter("ignore")
            sc = spectral_cube.SpectralCube.read(args.cube_q)
        freq_hz = sc.spectral_axis.to(u.Hz).value
    except Exception:
        # Fallback: build from FITS header
        hdr = hq[0].header
        crval = hdr.get("CRVAL3", hdr.get("CRVAL4"))
        cdelt = hdr.get("CDELT3", hdr.get("CDELT4"))
        crpix = hdr.get("CRPIX3", hdr.get("CRPIX4"))
        freq_hz = crval + (np.arange(n_freq) - (crpix - 1)) * cdelt

    # Ensure ascending frequency order
    if freq_hz[0] > freq_hz[-1]:
        freq_hz = freq_hz[::-1]
        q_cube = q_cube[::-1]
        u_cube = u_cube[::-1]
        if i_cube is not None:
            i_cube = i_cube[::-1]
        print("Frequencies reordered to ascending.")


    # -- Temp dir for qufit --------------------------------------------------
    tmp_dir = Path(args.output).parent / "tmp_qufit"
    tmp_dir.mkdir(parents=True, exist_ok=True)

    # -- Main loop -----------------------------------------------------------
    records = []
    out_csv = Path(args.output)

    for i, (row, col) in enumerate(tqdm(
        zip(rows_sel, cols_sel), total=len(rows_sel), desc="qufit+compare"
    )):
        row, col = int(row), int(col)

        # Extract pixel spectra
        Q_pix = q_cube[:, row, col].astype(float)
        U_pix = u_cube[:, row, col].astype(float)
        I_pix = i_cube[:, row, col].astype(float) if i_cube is not None else None

        # Run QUfitting
        qufit_res = run_qufit_pixel(
            freq_hz, Q_pix, U_pix, I_pix,
            noise_q, noise_i, good_chans,
            row, col, tmp_dir, args.nlive,
        )

        # Build record
        rec = {
            "row":     row,
            "col":     col,
            "ra_deg":  float(ra_deg[i]) if has_coords else np.nan,
            "dec_deg": float(dec_deg[i]) if has_coords else np.nan,
            # RM Synthesis
            "rm_syn":  float(rm_syn[row, col]),
        }
        if rm_syn_err is not None:
            rec["rm_syn_err"] = float(rm_syn_err[row, col])
        if rm_syn_pi is not None:
            rec["pi_syn"] = float(rm_syn_pi[row, col])

        # VROOM-SBI
        for key, data in sbi_maps.items():
            rec[key] = float(data[row, col])

        # QUfitting
        rec.update(qufit_res)

        # Derived: residuals
        if "rm_qufit_mean" in rec:
            rec["delta_rm_syn_qufit"]  = rec["rm_syn"] - rec["rm_qufit_mean"]
        if "rm_sbi_mean" in rec:
            rec["delta_rm_syn_sbi"]    = rec["rm_syn"] - rec["rm_sbi_mean"]
        if "rm_qufit_mean" in rec and "rm_sbi_mean" in rec:
            rec["delta_rm_sbi_qufit"]  = rec["rm_sbi_mean"] - rec["rm_qufit_mean"]

        records.append(rec)

        # Save incrementally (don't lose progress)
        if (i + 1) % 10 == 0 or i == len(rows_sel) - 1:
            pd.DataFrame(records).to_csv(out_csv, index=False, float_format="%.6f")

    # -- Final save ----------------------------------------------------------
    df = pd.DataFrame(records)
    df.to_csv(out_csv, index=False, float_format="%.6f")

    hq.close()
    hu.close()
    if hi:
        hi.close()
    shutil.rmtree(tmp_dir, ignore_errors=True)

    # -- Summary stats -------------------------------------------------------
    print(f"\nSaved {len(df)} rows to {out_csv}")
    print(f"Columns: {list(df.columns)}")

    if "rm_qufit_mean" in df.columns and "rm_sbi_mean" in df.columns:
        valid = df.dropna(subset=["rm_qufit_mean", "rm_sbi_mean"])
        if len(valid):
            print(f"\n--- Summary (N={len(valid)} pixels with all 3 methods) ---")
            for col_name, label in [
                ("delta_rm_syn_qufit",  "RM_syn - RM_qufit"),
                ("delta_rm_syn_sbi",    "RM_syn - RM_sbi"),
                ("delta_rm_sbi_qufit",  "RM_sbi - RM_qufit"),
            ]:
                if col_name in valid.columns:
                    vals = valid[col_name].dropna()
                    print(f"  {label:25s}  mean={vals.mean():+.3f}  "
                          f"std={vals.std():.3f}  "
                          f"med={vals.median():+.3f}  "
                          f"MAD={np.median(np.abs(vals - vals.median())):.3f}")


if __name__ == "__main__":
    main()
