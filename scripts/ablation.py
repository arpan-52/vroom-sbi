#!/usr/bin/env python3
"""
Single-model calibration ablation: train -> TARP + coverage, in one command.

Trains one cheap posterior (faraday_thin N=1 by default) from ``config_ablation.yaml``
and immediately runs the joint TARP calibration test and the marginal coverage
probe on the fresh model, printing a one-glance verdict. Built for the tight loop:

    edit config_ablation.yaml  ->  python scripts/ablation.py  ->  read ATRC gap
    ->  adjust training/noise params  ->  rerun  ->  until calibrated.

The noise-conditioning knob lives in the config (``noise.condition_on_noise``) so
training and evaluation always share the same data contract — do NOT override it
per-process, or the x = [Q, U, chan3] contract diverges between train and eval.

Target: ATRC gap ~ 0 (|gap| < 0.02) and cov68 ~ 0.68, cov90 ~ 0.90 per parameter.
gap > 0 = over-confident (too narrow); gap < 0 = under-confident (too wide).

Usage
-----
    pixi run -e gpu python scripts/ablation.py --config config_ablation.yaml
    pixi run -e gpu python scripts/ablation.py --skip-train      # re-eval only
"""

import argparse
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_ablation.yaml")
    ap.add_argument("--model", default="faraday_thin")
    ap.add_argument("--n-components", type=int, default=1)
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--skip-train", action="store_true",
                    help="skip training; only re-run TARP + coverage on the existing model")
    ap.add_argument("--output-dir", default=None,
                    help="calibration artifacts dir (default: <save_dir>/calib)")
    args = ap.parse_args()

    from src.config import Configuration
    from src.training import SBITrainer
    from src.validation.online_probe import run_probe
    from src.validation.tarp_test import run_tarp

    config = Configuration.from_yaml(args.config)
    save_dir = Path(config.training.save_dir)
    cond = getattr(config.noise, "condition_on_noise", False)
    posterior = save_dir / f"posterior_{args.model}_n{args.n_components}.pt"
    output_dir = Path(args.output_dir) if args.output_dir else save_dir / "calib"

    print(f"\n{'=' * 60}")
    print("ABLATION: single-model calibration loop")
    print(f"  config             : {args.config}")
    print(f"  model              : {args.model} N={args.n_components}")
    print(f"  condition_on_noise : {cond}")
    print(f"  lr / patience      : {config.training.learning_rate} / "
          f"{config.training.stop_after_epochs}")
    print(f"  posterior          : {posterior}")
    print(f"{'=' * 60}\n")

    if not args.skip_train:
        SBITrainer(config).train_model(args.model, args.n_components)

    if not posterior.exists():
        raise SystemExit(f"posterior not found: {posterior} (training failed?)")

    tarp = run_tarp(
        config_path=args.config, posterior_path=str(posterior),
        output_dir=output_dir, n_cases=args.n_cases, n_samples=args.n_samples,
        device=args.device,
    )
    probe = run_probe(
        config_path=args.config, posterior_path=str(posterior),
        n_cases=args.n_cases, n_samples=args.n_samples, device=args.device,
        output_dir=output_dir,
    )

    # One-glance verdict
    gap = tarp["atrc_gap"]
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    print(f"condition_on_noise : {cond}")
    print(f"TARP ATRC gap      : {gap:+.4f}  ({tarp['verdict']})   "
          f"[target |gap| < 0.02]")
    print(f"{'param':<10}{'bias':>10}{'cov68':>9}{'cov90':>9}   (ideal 0 / 0.68 / 0.90)")
    print("-" * 48)
    for j, name in enumerate(probe["labels"]):
        print(f"{name:<10}{probe['bias'][j].item():>10.4f}"
              f"{probe['cov68'][j].item():>9.3f}{probe['cov90'][j].item():>9.3f}")
    print("-" * 48)
    calibrated = abs(gap) < 0.02
    print(f"\n{'CALIBRATED' if calibrated else 'NOT YET — adjust and retrain'}"
          f"  (ATRC |gap|={abs(gap):.4f})\n")


if __name__ == "__main__":
    main()
