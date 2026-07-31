"""
Corner plot (joint posterior) for any pixel, spectrum, or sky region.

Produces a full corner plot of all posterior parameters for a given
observation, showing 1D marginals on the diagonal and 2D joint contours
off-diagonal. Useful for inspecting degeneracies and multi-component
label switching at any point in a cube.

Three input modes
-----------------
1. Raw spectrum (--q / --u): comma-separated Q and U values, one per channel.
2. Single pixel (--cube-q / --cube-u + --pixel RA DEC): extract one pixel
   from a FITS cube and infer.
3. Region average (--cube-q / --cube-u + --region RA DEC RADIUS_ARCMIN):
   average all pixels within the given radius before inferring.

Run:
    # From raw spectrum:
    pixi run -e gpu vroom-sbi corner \
        --config config.yaml --model-dir models/ \
        --freq-file freq_bands/freq_vla_l.txt \
        --q "0.1,0.2,..." --u "0.05,0.1,..." \
        --output corner.png

    # From a cube pixel:
    pixi run -e io vroom-sbi corner \
        --config config.yaml --model-dir models/ \
        --cube-q Q.fits --cube-u U.fits \
        --pixel 123.456 -30.789 \
        --output corner.png

    # From a region average:
    pixi run -e io vroom-sbi corner \
        --config config.yaml --model-dir models/ \
        --cube-q Q.fits --cube-u U.fits \
        --region 123.456 -30.789 2.0 \
        --output corner.png
"""

from __future__ import annotations

import argparse

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import Configuration
from ..inference import InferenceEngine


def _extract_pixel(q_fits: str, u_fits: str, ra: float, dec: float):
    """Return (Q, U, freq_hz, weights) for the nearest pixel to (ra, dec)."""
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(q_fits) as hq, fits.open(u_fits) as hu:
        wcs = WCS(hq[0].header, naxis=2)
        q_data = hq[0].data   # (..., n_dec, n_ra)
        u_data = hu[0].data

        # Squeeze spectral / Stokes axes down to (n_freq, n_dec, n_ra)
        while q_data.ndim > 3:
            q_data = q_data[0]
            u_data = u_data[0]

        # Pixel coordinate of requested sky position
        px, py = wcs.all_world2pix([[ra, dec]], 0)[0]
        col, row = int(round(px)), int(round(py))

        n_dec, n_ra = q_data.shape[-2], q_data.shape[-1]
        col = max(0, min(col, n_ra - 1))
        row = max(0, min(row, n_dec - 1))

        Q = q_data[:, row, col].astype(np.float64)
        U = u_data[:, row, col].astype(np.float64)

        # Frequencies from header if available
        try:
            from astropy.wcs import WCS as WCS3
            wcs3 = WCS3(hq[0].header)
            n_freq = Q.shape[0]
            pix_coords = np.column_stack([
                np.zeros(n_freq), np.zeros(n_freq), np.arange(n_freq)
            ])
            world = wcs3.all_pix2world(pix_coords, 0)
            freq_hz = world[:, 2]
        except Exception:
            freq_hz = None

    weights = (~np.isnan(Q)).astype(np.float32)
    Q = np.where(np.isnan(Q), 0.0, Q)
    U = np.where(np.isnan(U), 0.0, U)
    return Q, U, freq_hz, weights


def _extract_region(q_fits: str, u_fits: str, ra: float, dec: float, radius_arcmin: float):
    """Return (Q, U, freq_hz, weights) averaged over a circular region."""
    import astropy.units as u
    from astropy.coordinates import SkyCoord
    from astropy.io import fits
    from astropy.wcs import WCS

    with fits.open(q_fits) as hq, fits.open(u_fits) as hu:
        wcs2 = WCS(hq[0].header, naxis=2)
        q_data = hq[0].data
        u_data = hu[0].data

        while q_data.ndim > 3:
            q_data = q_data[0]
            u_data = u_data[0]

        n_freq, n_dec, n_ra = q_data.shape
        ra_grid, dec_grid = wcs2.all_pix2world(
            *np.meshgrid(np.arange(n_ra), np.arange(n_dec)), 0
        )
        center = SkyCoord(ra * u.deg, dec * u.deg)
        grid_coords = SkyCoord(ra_grid * u.deg, dec_grid * u.deg)
        sep = center.separation(grid_coords).to(u.arcmin).value
        mask = sep <= radius_arcmin

        if mask.sum() == 0:
            raise ValueError(f"No pixels found within {radius_arcmin} arcmin of ({ra}, {dec})")

        Q = np.nanmean(q_data[:, mask], axis=1)
        U = np.nanmean(u_data[:, mask], axis=1)

        try:
            from astropy.wcs import WCS as WCS3
            wcs3 = WCS3(hq[0].header)
            pix_coords = np.column_stack([
                np.zeros(n_freq), np.zeros(n_freq), np.arange(n_freq)
            ])
            world = wcs3.all_pix2world(pix_coords, 0)
            freq_hz = world[:, 2]
        except Exception:
            freq_hz = None

    weights = (~np.isnan(Q)).astype(np.float32)
    Q = np.where(np.isnan(Q), 0.0, Q)
    U = np.where(np.isnan(U), 0.0, U)
    return Q, U, freq_hz, weights


def make_corner_plot(
    config_path: str | None,
    model_dir: str,
    Q: np.ndarray,
    U: np.ndarray,
    freq_hz: np.ndarray | None = None,
    weights: np.ndarray | None = None,
    n_samples: int = 10000,
    device: str = "cuda",
    model_type: str | None = None,
    n_components: int | None = None,
    output: str = "corner.png",
    title: str | None = None,
) -> str:
    """Run inference on Q/U and produce a corner plot. Returns output path."""
    import corner

    config = Configuration.from_yaml(config_path) if config_path else None
    engine = InferenceEngine(config=config, model_dir=model_dir, device=device)
    engine.load_models()

    qu_obs = np.concatenate([Q, U]).astype(np.float32)

    use_classifier = (model_type is None and n_components is None)
    results, best_key = engine.run_inference(
        qu_obs,
        weights=weights,
        n_samples=n_samples,
        use_classifier=use_classifier,
        model_type=model_type,
    )

    # Pick the requested or best model
    if model_type is not None and n_components is not None:
        key = f"{model_type}_n{n_components}"
        result = results.get(key) or results[best_key]
    else:
        result = results[best_key]

    samples = result.all_samples  # (n_samples, n_params)

    # Build parameter labels
    _param_base = {
        "faraday_thin": ["RM", "amp", "χ₀"],
        "burn_slab": ["φ_c", "Δφ", "amp", "χ₀"],
        "external_dispersion": ["φ", "σ_φ", "amp", "χ₀"],
        "internal_dispersion": ["φ", "σ_φ", "amp", "χ₀"],
    }
    base = _param_base.get(result.model_type, [f"p{i}" for i in range(samples.shape[1])])
    n_comp = result.n_components
    if n_comp == 1:
        labels = base
    else:
        labels = [f"{p}_{j+1}" for j in range(n_comp) for p in base]

    fig = corner.corner(
        samples,
        labels=labels,
        quantiles=[0.16, 0.5, 0.84],
        show_titles=True,
        title_fmt=".2f",
        title_kwargs={"fontsize": 9},
        label_kwargs={"fontsize": 9},
        smooth=1.0,
    )
    t = title or f"{result.model_type} N={n_comp}  |  log Z={result.log_evidence:.2f}"
    fig.suptitle(t, y=1.01, fontsize=10)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"Model    : {result.model_type}  N={result.n_components}")
    print(f"log Z    : {result.log_evidence:.3f}")
    print(f"Samples  : {n_samples}")
    print(f"Corner   : {output}")
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--model-dir", default="models")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-samples", type=int, default=10000)
    ap.add_argument("--model-type", default=None,
                    choices=["faraday_thin", "burn_slab", "external_dispersion", "internal_dispersion"])
    ap.add_argument("--n-components", type=int, default=None)
    ap.add_argument("--output", default="corner.png")
    ap.add_argument("--title", default=None)

    # Input mode 1: raw spectrum
    ap.add_argument("--q", default=None, help="Comma-separated Q values (one per channel)")
    ap.add_argument("--u", default=None, help="Comma-separated U values (one per channel)")
    ap.add_argument("--freq-file", default=None, metavar="PATH",
                    help="Freq file (Hz, one per line); needed with --q/--u to set freq axis")

    # Input mode 2/3: FITS cubes
    ap.add_argument("--cube-q", default=None, metavar="PATH")
    ap.add_argument("--cube-u", default=None, metavar="PATH")
    ap.add_argument("--pixel", nargs=2, type=float, metavar=("RA", "DEC"),
                    help="Sky coordinate of a single pixel (degrees)")
    ap.add_argument("--region", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_ARCMIN"),
                    help="Sky coordinate + radius for region average (degrees, arcmin)")

    args = ap.parse_args()

    Q = U = freq_hz = weights = None

    if args.q and args.u:
        Q = np.array([float(v) for v in args.q.split(",")])
        U = np.array([float(v) for v in args.u.split(",")])
        if args.freq_file:
            from ..simulator.physics import load_frequencies
            freq_hz, _ = load_frequencies(args.freq_file)
    elif args.cube_q and args.cube_u:
        if args.pixel:
            ra, dec = args.pixel
            Q, U, freq_hz, weights = _extract_pixel(args.cube_q, args.cube_u, ra, dec)
        elif args.region:
            ra, dec, radius = args.region
            Q, U, freq_hz, weights = _extract_region(args.cube_q, args.cube_u, ra, dec, radius)
        else:
            ap.error("With --cube-q/--cube-u you must supply --pixel or --region")
    else:
        ap.error("Supply either --q/--u or --cube-q/--cube-u with --pixel or --region")

    make_corner_plot(
        config_path=args.config,
        model_dir=args.model_dir,
        Q=Q,
        U=U,
        freq_hz=freq_hz,
        weights=weights,
        n_samples=args.n_samples,
        device=args.device,
        model_type=args.model_type,
        n_components=args.n_components,
        output=args.output,
        title=args.title,
    )


if __name__ == "__main__":
    main()
