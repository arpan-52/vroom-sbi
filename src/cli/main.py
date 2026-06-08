"""VROOM-SBI CLI."""

import argparse
import logging
import sys

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def train_command(args):
    from ..config import Configuration
    from ..training import train_all_models

    config = Configuration.from_yaml(args.config)
    if args.device:
        config.training.device = args.device
    if args.freq_file:
        config.freq_file = args.freq_file
    if args.save_dir:
        config.training.save_dir = args.save_dir

    # Single-model mode: train one (model_type, N) posterior. Lets several
    # models run as separate processes (e.g. one per GPU).
    n_comp = getattr(args, "n_components", None)
    if n_comp is not None:
        from ..training import SBITrainer

        model_type = args.model_type or config.physics.model_types[0]
        trainer = SBITrainer(config)
        trainer.train_model(model_type, n_comp)
        return

    train_all_models(config, classifier_only=args.classifier_only)


def infer_command(args):
    import numpy as np

    from ..config import Configuration
    from ..inference import InferenceEngine

    Q = np.array([float(x.strip()) for x in args.q.split(",")])
    U = np.array([float(x.strip()) for x in args.u.split(",")])

    if len(Q) != len(U):
        raise ValueError(f"Q and U must have same length: {len(Q)} vs {len(U)}")

    qu_obs = np.concatenate([Q, U])

    config = None
    if args.config:
        config = Configuration.from_yaml(args.config)

    device = args.device or ("cuda" if config is None else config.training.device)
    engine = InferenceEngine(config=config, model_dir=args.model_dir, device=device)
    engine.load_models(max_components=args.max_components)

    best_result, all_results = engine.infer(qu_obs, n_samples=args.n_samples)

    print("\n" + "=" * 60)
    print("INFERENCE RESULTS")
    print("=" * 60)
    print(
        f"\nBest model: {best_result.model_type} with {best_result.n_components} component(s)"
    )
    print(f"Log evidence: {best_result.log_evidence:.2f}")

    for i, comp in enumerate(best_result.components):
        print(f"\nComponent {i + 1}:")
        print(f"  RM = {comp.rm_mean:.2f} ± {comp.rm_std:.2f} rad/m²")
        if comp.sigma_phi_mean is not None:
            print(
                f"  σ_φ = {comp.sigma_phi_mean:.2f} ± {comp.sigma_phi_std:.2f} rad/m²"
            )
        if comp.delta_phi_mean is not None:
            print(f"  Δφ = {comp.delta_phi_mean:.2f} ± {comp.delta_phi_std:.2f} rad/m²")


def validate_command(args):
    from ..validation.validator import run_validation

    # Parse prior bounds if provided
    prior_low = None
    prior_high = None
    if args.prior_low:
        prior_low = [float(x) for x in args.prior_low.split(",")]
    if args.prior_high:
        prior_high = [float(x) for x in args.prior_high.split(",")]

    print("\n" + "=" * 50)
    print("VROOM-SBI Validation")
    print("=" * 50)
    print(f"Posterior: {args.posterior}")
    if args.model:
        print(f"Model: {args.model} (user provided)")
    else:
        print("Model: (will read from checkpoint)")
    if args.n_components:
        print(f"N components: {args.n_components} (user provided)")
    else:
        print("N components: (will read from checkpoint)")
    if prior_low and prior_high:
        print(f"Prior low: {prior_low}")
        print(f"Prior high: {prior_high}")
    print(f"Output: {args.output_dir}")
    print(f"Noise: {args.noise_percent}% of signal")
    print(f"Missing: {args.missing_fraction * 100:.0f}% flagged")
    print(f"Device: {args.device}")
    print("=" * 50 + "\n")

    run_validation(
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        model_type=args.model,
        n_components=args.n_components,
        prior_low=prior_low,
        prior_high=prior_high,
        n_sweep_points=args.n_sweep_points,
        n_cases=args.n_cases,
        noise_percent=args.noise_percent,
        missing_fraction=args.missing_fraction,
        n_samples=args.n_samples,
        device=args.device,
        seed=args.seed,
    )


def cube_infer_pol(args):
    """Polarization (Q/U) cube inference — renamed from cube-infer."""
    import sys
    import warnings

    from ..config import Configuration
    from ..inference import InferenceEngine
    from ..io import open_i_cube_lazy, open_qu_cubes_lazy, write_results_maps

    if not args.cube and not args.cube_u:
        print("error: --cube-u is required when --cube-q is used", file=sys.stderr)
        sys.exit(2)

    if args.cube:
        print("error: --cube (IQUV) not yet supported in chunked mode; use --cube-q/--cube-u", file=sys.stderr)
        sys.exit(2)

    # Open lazily — no data loaded yet
    q_cube, u_cube, shape, frequencies, wcs_2d = open_qu_cubes_lazy(
        args.cube_q, args.cube_u
    )

    i_cube = None
    if args.cube_i:
        i_cube = open_i_cube_lazy(args.cube_i)
    else:
        warnings.warn(
            "No Stokes I provided — assuming Q and U are already spectrally "
            "normalised (fractional polarisation p = Q/I, U/I).",
            UserWarning,
            stacklevel=2,
        )

    config = None
    if args.config:
        config = Configuration.from_yaml(args.config)

    device = args.device or ("cuda" if config is None else config.training.device)
    engine = InferenceEngine(config=config, model_dir=args.model_dir, device=device)
    engine.load_models(max_components=args.max_components)

    results = engine.run_inference_cube_chunked(
        q_cube,
        u_cube,
        shape=shape,
        frequencies_hz=frequencies,
        snr_threshold=args.snr_threshold if args.snr_threshold is not None else 5.0,
        n_samples=args.n_samples,
        i_cube=i_cube,
        ci95=getattr(args, "ci95", False),
    )

    write_results_maps(results, wcs_2d, args.output_dir)


def cube_infer_spectra(args):
    """Spectral shape (total intensity) cube inference."""
    import astropy.units as u

    from ..config import Configuration
    from ..inference import InferenceEngine
    from ..io import open_i_cube_lazy, write_results_maps

    config = None
    if args.config:
        config = Configuration.from_yaml(args.config)

    device = args.device or ("cuda" if config is None else config.training.device)
    engine = InferenceEngine(config=config, model_dir="models", device=device)

    # Load spectral shape model
    engine.load_spectral_shape_model(args.model)

    # Open I cube lazily (returns a SpectralCube object)
    i_cube = open_i_cube_lazy(args.fits)
    shape = tuple(int(x) for x in i_cube.shape)  # (n_freq, n_dec, n_ra)
    frequencies = i_cube.spectral_axis.to(u.Hz).value

    try:
        wcs_2d = i_cube.wcs.celestial
    except Exception:
        wcs_2d = None

    results = engine.run_spectral_shape_cube_chunked(
        i_cube,
        shape=shape,
        frequencies_hz=frequencies,
        snr_threshold=args.snr_threshold if args.snr_threshold is not None else 5.0,
        n_samples=args.n_samples,
    )

    write_results_maps(results, wcs_2d, args.output_dir)


def corner_command(args):
    from ..validation.corner_plot import make_corner_plot, _extract_pixel, _extract_region
    import numpy as np

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
            raise ValueError("With --cube-q/--cube-u supply --pixel or --region")
    else:
        raise ValueError("Supply --q/--u or --cube-q/--cube-u with --pixel or --region")

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


def download_command(args):
    from ..utils import DEFAULT_HF_REPO, download_from_huggingface

    repo_id = args.repo_id or DEFAULT_HF_REPO
    print(f"Downloading models from {repo_id} → {args.model_dir}/")
    download_from_huggingface(repo_id, model_dir=args.model_dir, token=args.token)
    print("Done.")


def push_command(args):
    from ..utils import push_to_huggingface

    push_to_huggingface(
        model_dir=args.model_dir,
        repo_id=args.repo_id,
        token=args.token,
        private=args.private,
    )


def main():
    parser = argparse.ArgumentParser(description="VROOM-SBI")
    subparsers = parser.add_subparsers(dest="command")

    # Train
    train_p = subparsers.add_parser("train")
    train_p.add_argument("--config", default="config.yaml")
    train_p.add_argument("--device", default=None)
    train_p.add_argument("--classifier-only", action="store_true")
    train_p.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="train only this single N (one (model_type, N) posterior)",
    )
    train_p.add_argument(
        "--model-type",
        default=None,
        help="model type for single-model mode (default: first in config)",
    )
    train_p.add_argument(
        "--freq-file",
        default=None,
        metavar="PATH",
        help="override freq_file from config (path to frequency list in Hz)",
    )
    train_p.add_argument(
        "--save-dir",
        default=None,
        metavar="DIR",
        help="override training.save_dir from config",
    )
    train_p.set_defaults(func=train_command)

    # Infer
    infer_p = subparsers.add_parser("infer")
    infer_p.add_argument("--q", required=True)
    infer_p.add_argument("--u", required=True)
    infer_p.add_argument("--config", default=None)
    infer_p.add_argument("--model-dir", default="models")
    infer_p.add_argument("--max-components", type=int, default=5)
    infer_p.add_argument("--n-samples", type=int, default=10000)
    infer_p.add_argument("--device", default=None)
    infer_p.add_argument("--output", default=None)
    infer_p.set_defaults(func=infer_command)

    # Validate
    val_p = subparsers.add_parser(
        "validate",
        help="Run validation with parameter sweeps and case plots",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Model Types and Parameter Order:
================================
Each model type has specific parameters PER COMPONENT:

  faraday_thin (3 params per component):
    [RM, amp, chi0]
    - RM: Faraday depth (rad/m²)
    - amp: Polarized amplitude
    - chi0: Intrinsic polarization angle (rad)

  burn_slab (4 params per component):
    [phi_c, delta_phi, amp, chi0]
    - phi_c: Central Faraday depth (rad/m²)
    - delta_phi: Faraday depth extent (rad/m²)
    - amp: Polarized amplitude
    - chi0: Intrinsic polarization angle (rad)

  external_dispersion / internal_dispersion (4 params per component):
    [phi, sigma_phi, amp, chi0]
    - phi: Mean Faraday depth (rad/m²)
    - sigma_phi: Faraday dispersion (rad/m²)
    - amp: Polarized amplitude
    - chi0: Intrinsic polarization angle (rad)

For N components, parameters are concatenated:
  N=1: [p1, p2, p3, ...]
  N=2: [p1_1, p2_1, p3_1, ..., p1_2, p2_2, p3_2, ...]

Priority:
=========
1. If --model and --n-components provided, use them
2. Else, try to read from posterior checkpoint
3. If both fail, error with message

Examples:
=========
# Let validator read model info from checkpoint (recommended for new posteriors)
vroom-sbi validate --posterior model.pt --output-dir val_results

# Override with explicit model info (for old posteriors)
vroom-sbi validate --posterior model.pt --model faraday_thin --n-components 1

# With custom prior ranges
vroom-sbi validate --posterior model.pt --model faraday_thin --n-components 1 \\
    --prior-low "-100,0.001,-3.14" --prior-high "100,1.0,3.14"
""",
    )
    val_p.add_argument("--posterior", required=True, help="Path to posterior .pt file")
    val_p.add_argument(
        "--output-dir", default="validation_results", help="Output directory"
    )
    val_p.add_argument(
        "--model",
        type=str,
        default=None,
        choices=[
            "faraday_thin",
            "burn_slab",
            "external_dispersion",
            "internal_dispersion",
        ],
        help="Model type (optional, reads from checkpoint if not provided)",
    )
    val_p.add_argument(
        "--n-components",
        type=int,
        default=None,
        help="Number of Faraday components (optional, reads from checkpoint)",
    )
    val_p.add_argument(
        "--prior-low",
        type=str,
        default=None,
        help="Comma-separated lower bounds for each parameter (optional)",
    )
    val_p.add_argument(
        "--prior-high",
        type=str,
        default=None,
        help="Comma-separated upper bounds for each parameter (optional)",
    )
    val_p.add_argument(
        "--noise-percent",
        type=float,
        default=10.0,
        help="Noise as %% of signal amplitude (default: 10)",
    )
    val_p.add_argument(
        "--missing-fraction",
        type=float,
        default=0.1,
        help="Fraction of channels to flag (default: 0.1)",
    )
    val_p.add_argument(
        "--n-sweep-points",
        type=int,
        default=20,
        help="Points per parameter sweep (default: 20)",
    )
    val_p.add_argument(
        "--n-cases",
        type=int,
        default=10,
        help="Number of individual test cases (default: 10)",
    )
    val_p.add_argument(
        "--n-samples",
        type=int,
        default=5000,
        help="Posterior samples per inference (default: 5000)",
    )
    val_p.add_argument("--device", default="auto")
    val_p.add_argument("--seed", type=int, default=42)
    val_p.set_defaults(func=validate_command)

    # Cube infer (polarization)
    cube_p = subparsers.add_parser(
        "cube-infer-pol",
        help="Run polarization (Q/U) inference over all pixels of a spectral cube",
    )
    cube_input = cube_p.add_mutually_exclusive_group(required=True)
    cube_input.add_argument(
        "--cube",
        metavar="PATH",
        help="4D IQUV FITS cube (enables Q/I, U/I normalisation)",
    )
    cube_input.add_argument(
        "--cube-q",
        metavar="PATH",
        help="3D Stokes Q cube (use with --cube-u; assumed already spectrally normalised)",
    )
    cube_p.add_argument(
        "--cube-u",
        metavar="PATH",
        help="3D Stokes U cube (required with --cube-q)",
    )
    cube_p.add_argument(
        "--cube-i",
        metavar="PATH",
        default=None,
        help="3D Stokes I cube for Q/I, U/I normalisation (optional with --cube-q/--cube-u)",
    )
    cube_p.add_argument(
        "--noise-cube",
        metavar="PATH",
        default=None,
        help="Per-channel noise FITS cube for inverse-variance weighting",
    )
    cube_p.add_argument(
        "--mask",
        metavar="PATH",
        default=None,
        help="2D FITS mask; non-zero pixels are processed",
    )
    cube_p.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help="Only process pixels with mean polarised SNR above this value",
    )
    cube_p.add_argument(
        "--output-dir",
        default="cube_results",
        help="Output directory (default: cube_results/)",
    )
    cube_p.add_argument("--model-dir", default="models")
    cube_p.add_argument("--config", default=None)
    cube_p.add_argument("--device", default=None)
    cube_p.add_argument("--max-components", type=int, default=5)
    cube_p.add_argument("--n-samples", type=int, default=1000)
    cube_p.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Pixels per progress chunk (default: 1, serial)",
    )
    cube_p.add_argument(
        "--ci95",
        action="store_true",
        help="Also write 95%% CI (p2.5/p97.5) uncertainty maps per parameter",
    )
    cube_p.set_defaults(func=cube_infer_pol)

    # Cube infer (spectral shape — total intensity)
    spectra_p = subparsers.add_parser(
        "cube-infer-spectra",
        help="Run spectral shape (total intensity) inference over all pixels of an I cube",
    )
    spectra_p.add_argument(
        "--fits",
        metavar="PATH",
        required=True,
        help="3D Stokes I FITS cube",
    )
    spectra_p.add_argument(
        "--model",
        metavar="PATH",
        required=True,
        help="Path to trained spectral_shape_posterior.pt",
    )
    spectra_p.add_argument(
        "--snr-threshold",
        type=float,
        default=None,
        help="Only process pixels with mean I above this SNR (default: 5)",
    )
    spectra_p.add_argument(
        "--output-dir",
        default="spectra_results",
        help="Output directory (default: spectra_results/)",
    )
    spectra_p.add_argument("--config", default=None)
    spectra_p.add_argument("--device", default=None)
    spectra_p.add_argument("--n-samples", type=int, default=1000)
    spectra_p.set_defaults(func=cube_infer_spectra)

    # Corner plot
    corner_p = subparsers.add_parser(
        "corner",
        help="Joint posterior corner plot for a spectrum, pixel, or sky region",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    corner_p.add_argument("--config", default=None)
    corner_p.add_argument("--model-dir", default="models")
    corner_p.add_argument("--device", default="cuda")
    corner_p.add_argument("--n-samples", type=int, default=10000)
    corner_p.add_argument("--model-type", default=None,
                          choices=["faraday_thin", "burn_slab", "external_dispersion", "internal_dispersion"])
    corner_p.add_argument("--n-components", type=int, default=None)
    corner_p.add_argument("--output", default="corner.png")
    corner_p.add_argument("--title", default=None)
    # Input mode 1: raw spectrum
    corner_p.add_argument("--q", default=None, help="Comma-separated Q values")
    corner_p.add_argument("--u", default=None, help="Comma-separated U values")
    corner_p.add_argument("--freq-file", default=None, metavar="PATH")
    # Input mode 2/3: FITS cubes
    corner_p.add_argument("--cube-q", default=None, metavar="PATH")
    corner_p.add_argument("--cube-u", default=None, metavar="PATH")
    corner_p.add_argument("--pixel", nargs=2, type=float, metavar=("RA", "DEC"))
    corner_p.add_argument("--region", nargs=3, type=float, metavar=("RA", "DEC", "RADIUS_ARCMIN"))
    corner_p.set_defaults(func=corner_command)

    # Download
    dl_p = subparsers.add_parser(
        "download", help="Download pre-trained models from HuggingFace"
    )
    dl_p.add_argument(
        "--model-dir", default="models", help="Local directory to save models (default: models/)"
    )
    dl_p.add_argument(
        "--repo-id", default=None, help="HuggingFace repo ID (default: arpan-52/vroom-sbi)"
    )
    dl_p.add_argument("--token", default=None, help="HF token for private repos")
    dl_p.set_defaults(func=download_command)

    # Push
    push_p = subparsers.add_parser("push")
    push_p.add_argument("--model-dir", default="models")
    push_p.add_argument("--repo-id", required=True)
    push_p.add_argument("--token", default=None)
    push_p.add_argument("--private", action="store_true")
    push_p.set_defaults(func=push_command)

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(0)
    args.func(args)


if __name__ == "__main__":
    main()
