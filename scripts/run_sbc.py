#!/usr/bin/env python3
"""
Simulation-based calibration (SBC) matrix driver for vroom-sbi posteriors.

Runs SBC rank-statistic calibration across a grid of (model_type, N) posteriors
in the flat ``models/`` layout produced by ``vroom-sbi train`` (the full matrix
run). For each posterior it draws theta from the prior, simulates x, samples the
posterior, and records the rank of each true theta among its posterior samples;
uniform ranks => calibrated. Per-posterior it writes a rank-histogram PNG and a
summary JSON (via ``src.validation.sbc``); across the grid it writes a combined
summary JSON and a printed PASS/FAIL table.

This is the model-type analogue of ``scripts/sweep_bands.py`` (which sweeps
telescope bands for faraday_thin only). Start with one model type to validate
the harness on the wide prior, then widen ``--model`` to the full matrix:

Usage
-----
    # One model type, all N (harness probe on the wide prior):
    pixi run -e gpu python scripts/run_sbc.py --config config_a100.yaml \
        --model faraday_thin

    # Full matrix (all model types from the config), all N:
    pixi run -e gpu python scripts/run_sbc.py --config config_a100.yaml

    # Subset of N, custom case/sample counts:
    pixi run -e gpu python scripts/run_sbc.py --config config_a100.yaml \
        --model burn_slab --n-range 1-3 --n-cases 2000 --n-samples 1000

    # Both marginal (SBC) and joint (TARP) calibration in one pass:
    pixi run -e gpu python scripts/run_sbc.py --config config_a100.yaml \
        --tests sbc tarp --n-cases 5000 --n-samples 2000

Despite the name, ``--tests tarp`` runs the joint DRP/TARP coverage test too;
SBC and TARP share the simulator/posterior-rebuild contract so they see
identical models and data.
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.validation.sbc import run_sbc_for_posterior


def parse_n_range(spec: str) -> list[int]:
    """Parse an N specification: "1-5", "3", or "1,3,5" -> sorted int list."""
    ns: set[int] = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            lo, hi = part.split("-")
            ns.update(range(int(lo), int(hi) + 1))
        elif part:
            ns.add(int(part))
    return sorted(ns)


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="config_a100.yaml",
                    help="Config YAML (source of model_types, save_dir, N range)")
    ap.add_argument("--model", nargs="+", default=None,
                    help="Model type(s) to run (default: physics.model_types from config)")
    ap.add_argument("--n-range", default=None,
                    help='N components, e.g. "1-5", "3", or "1,3,5" '
                         "(default: model_selection min..max from config)")
    ap.add_argument("--save-dir", default=None,
                    help="Directory holding posterior_*.pt (default: training.save_dir)")
    ap.add_argument("--output-dir", default="validation_results/sbc",
                    help="Directory for rank histograms + summaries")
    ap.add_argument("--tests", nargs="+", default=["sbc"],
                    choices=["sbc", "tarp"],
                    help="calibration tests per posterior: sbc (marginal ranks), "
                         "tarp (joint DRP coverage), or both")
    ap.add_argument("--n-cases", type=int, default=2000,
                    help="Prior draws (simulated cases) per posterior")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Posterior draws per case")
    ap.add_argument("--freq-file", default=None,
                    help="Override freq_file from config (must match training band)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    with open(args.config) as fh:
        cfg = yaml.safe_load(fh)

    model_types = args.model or cfg.get("physics", {}).get(
        "model_types", cfg.get("physics", {}).get("model_type", ["faraday_thin"])
    )
    if isinstance(model_types, str):
        model_types = [model_types]

    if args.n_range:
        n_list = parse_n_range(args.n_range)
    else:
        ms = cfg.get("model_selection", {})
        n_list = list(range(ms.get("min_components", 1), ms.get("max_components", 5) + 1))

    save_dir = args.save_dir or cfg.get("training", {}).get("save_dir", "models")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("vroom-sbi SBC matrix")
    print(f"  config      : {args.config}")
    print(f"  models dir  : {save_dir}")
    print(f"  model types : {model_types}")
    print(f"  N range     : {n_list}")
    print(f"  tests       : {args.tests}")
    print(f"  cases/draws : {args.n_cases} / {args.n_samples}")
    print(f"  output      : {output_dir}")
    print(f"{'=' * 60}\n")

    if "tarp" in args.tests:
        from src.validation.tarp_test import run_tarp

    results = []  # one record per posterior, holding results from each test
    missing = []
    for model_type in model_types:
        for n in n_list:
            pt = Path(save_dir) / f"posterior_{model_type}_n{n}.pt"
            if not pt.exists():
                print(f"[SKIP] {pt} not found", flush=True)
                missing.append(str(pt))
                continue
            record = {"model_type": model_type, "n_components": n}
            if "sbc" in args.tests:
                print(f"\n{'#' * 60}\n# SBC: {model_type} N={n}\n{'#' * 60}", flush=True)
                out = run_sbc_for_posterior(
                    config_path=args.config, posterior_path=str(pt),
                    output_dir=output_dir, n_cases=args.n_cases,
                    n_samples=args.n_samples, device=args.device,
                    freq_file=args.freq_file,
                )
                record["sbc"] = out["summary"]
            if "tarp" in args.tests:
                print(f"\n{'#' * 60}\n# TARP: {model_type} N={n}\n{'#' * 60}", flush=True)
                out = run_tarp(
                    config_path=args.config, posterior_path=str(pt),
                    output_dir=output_dir, n_cases=args.n_cases,
                    n_samples=args.n_samples, device=args.device,
                    freq_file=args.freq_file,
                )
                record["tarp"] = out["summary"]
            results.append(record)

    # Combined matrix summary
    matrix = {
        "config": args.config,
        "save_dir": save_dir,
        "tests": args.tests,
        "n_cases": args.n_cases,
        "n_samples": args.n_samples,
        "results": results,
        "missing": missing,
    }
    matrix_path = output_dir / "calibration_matrix_summary.json"
    with matrix_path.open("w") as fh:
        json.dump(matrix, fh, indent=2)

    # Printed roll-up: one row per posterior, columns per test.
    print(f"\n{'=' * 62}")
    print("CALIBRATION MATRIX SUMMARY")
    print(f"{'=' * 62}")
    header = f"{'model':<22}{'N':>3}"
    if "sbc" in args.tests:
        header += f"{'sbc pass/fail':>16}"
    if "tarp" in args.tests:
        header += f"{'tarp gap':>12}{'verdict':>16}"
    print(header)
    print("-" * 62)
    for r in results:
        row = f"{r['model_type']:<22}{r['n_components']:>3}"
        if "sbc" in args.tests and "sbc" in r:
            row += f"{r['sbc']['n_pass']:>7}/{r['sbc']['n_fail']:<8}"
        if "tarp" in args.tests and "tarp" in r:
            row += f"{r['tarp']['atrc_gap']:>+12.4f}{r['tarp']['verdict']:>16}"
        print(row)
    print("-" * 62)
    print(f"Combined summary: {matrix_path}")
    if missing:
        print(f"({len(missing)} posterior(s) missing — see summary JSON)")
    print()

    # Non-zero exit if any SBC parameter failed OR any TARP was uncalibrated.
    any_fail = any(
        ("sbc" in r and r["sbc"]["n_fail"] > 0)
        or ("tarp" in r and not r["tarp"]["calibrated"])
        for r in results
    )
    sys.exit(1 if any_fail else 0)


if __name__ == "__main__":
    main()
