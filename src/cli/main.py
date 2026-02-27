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
    train_all_models(config, classifier_only=args.classifier_only)


def infer_command(args):
    import numpy as np

    from ..inference import InferenceEngine

    Q = np.array([float(x) for x in args.q.split(",")])
    U = np.array([float(x) for x in args.u.split(",")])
    engine = InferenceEngine(
        model_dir=args.model_dir,
        config_path=args.config,
        max_components=args.max_components,
        device=args.device,
    )
    result = engine.infer(Q, U, n_samples=args.n_samples)
    print(result)
    if args.output:
        result.plot(args.output)


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


def cube_infer_command(args):
    import numpy as np
    from astropy.io import fits as afits

    from ..config import Configuration
    from ..inference import InferenceEngine
    from ..io import (
        compute_weights,
        normalize_qu_by_i,
        read_iquv_cube,
        read_qu_cubes,
        write_results_maps,
    )

    # ------------------------------------------------------------------
    # Read cubes
    # ------------------------------------------------------------------
    if args.cube:
        q_data, u_data, frequencies, wcs_2d, i_data = read_iquv_cube(args.cube)
        q_data, u_data = normalize_qu_by_i(q_data, u_data, i_data)
    else:
        if not args.cube_u:
            import sys

            print("error: --cube-u is required when --cube-q is used", file=sys.stderr)
            sys.exit(2)
        # warns internally that spectra assumed already normalised
        q_data, u_data, frequencies, wcs_2d = read_qu_cubes(args.cube_q, args.cube_u)

    # ------------------------------------------------------------------
    # Noise weights
    # ------------------------------------------------------------------
    noise_cube = None
    if args.noise_cube:
        noise_cube = afits.getdata(args.noise_cube).astype(np.float64)
    weights = compute_weights(q_data, u_data, noise_cube=noise_cube)

    # ------------------------------------------------------------------
    # Spatial mask
    # ------------------------------------------------------------------
    mask = None
    if args.mask:
        mask = afits.getdata(args.mask) != 0

    # ------------------------------------------------------------------
    # Engine
    # ------------------------------------------------------------------
    config = None
    if args.config:
        config = Configuration.from_yaml(args.config)

    device = args.device or ("cuda" if config is None else config.training.device)
    engine = InferenceEngine(config=config, model_dir=args.model_dir, device=device)
    engine.load_models(max_components=args.max_components)

    # ------------------------------------------------------------------
    # Run
    # ------------------------------------------------------------------
    results = engine.run_inference_cube(
        q_data,
        u_data,
        weights=weights,
        mask=mask,
        snr_threshold=args.snr_threshold,
        n_samples=args.n_samples,
        batch_size=args.batch_size,
    )

    write_results_maps(results, wcs_2d, args.output_dir)


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

    # Cube infer
    cube_p = subparsers.add_parser(
        "cube-infer",
        help="Run inference over all pixels of a Stokes spectral cube",
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
    cube_p.set_defaults(func=cube_infer_command)

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
