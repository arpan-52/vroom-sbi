#!/usr/bin/env python3
"""
Full-band training sweep for vroom-sbi faraday_thin posteriors.

Trains N=1..5 for each telescope/band combination, then runs the per-dimension
coverage probe (online_probe) and plots the bias/rmse/cov68/cov90 summary
(plot_coverage). SBC histograms are skipped by default; pass --sbc to enable.

Bands covered
-------------
VLA       : P, L, S, C
            Ref: NRAO VLA OSS
            https://science.nrao.edu/facilities/vla/docs/manuals/oss/performance/vla-frequency-bands-and-tunability

MeerKAT   : UHF, L, S0, S1, S2, S3, S4, S-full
            Ref: Hale et al. 2025, MNRAS 536, arXiv:2412.09314 (Table 4)
            https://skaafrica.atlassian.net/wiki/spaces/ESDKB/pages/277315585/MeerKAT+specifications

uGMRT     : Band 3, Band 4, Band 5
            Ref: NCRA GMRT System Parameters
            https://www.gmrt.ncra.tifr.res.in/doc/GMRT_specs.pdf
            GTAC Cycle 50: https://indrayani.ncra.tifr.res.in/~secr-ops/sch/c50webfiles/gtac_50_announcement.pdf

Usage
-----
    # Full sweep (train + coverage + plot):
    pixi run -e gpu python scripts/sweep_bands.py --config config_a100_lrtest.yaml

    # Selected bands only:
    pixi run -e gpu python scripts/sweep_bands.py --config config_a100_lrtest.yaml \
        --bands vla_p vla_l meerkat_uhf ugmrt_band3

    # Skip training (coverage + plot only, models already exist):
    pixi run -e gpu python scripts/sweep_bands.py --config config_a100_lrtest.yaml \
        --skip-train

    # Include SBC rank histograms:
    pixi run -e gpu python scripts/sweep_bands.py --config config_a100_lrtest.yaml --sbc
"""

import argparse
import subprocess
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Band registry
# ---------------------------------------------------------------------------
# Each entry: band_key -> (freq_file, label)
# freq_file is relative to the repo root.
BANDS: dict[str, tuple[str, str]] = {
    # VLA
    "vla_p":          ("freq_bands/freq_vla_p.txt",          "VLA P-band (200--503 MHz)"),
    "vla_l":          ("freq_bands/freq_vla_l.txt",          "VLA L-band (1000--2040 MHz)"),
    "vla_s":          ("freq_bands/freq_vla_s.txt",          "VLA S-band (2000--4000 MHz)"),
    "vla_c":          ("freq_bands/freq_vla_c.txt",          "VLA C-band (4000--8000 MHz)"),
    # MeerKAT
    "meerkat_uhf":    ("freq_bands/freq_meerkat_uhf.txt",    "MeerKAT UHF (544--1087 MHz)"),
    "meerkat_l":      ("freq_bands/freq_meerkat_l.txt",      "MeerKAT L (856--1711 MHz)"),
    "meerkat_s0":     ("freq_bands/freq_meerkat_s0.txt",     "MeerKAT S0 (1750--2625 MHz)"),
    "meerkat_s1":     ("freq_bands/freq_meerkat_s1.txt",     "MeerKAT S1 (1969--2844 MHz)"),
    "meerkat_s2":     ("freq_bands/freq_meerkat_s2.txt",     "MeerKAT S2 (2188--3063 MHz)"),
    "meerkat_s3":     ("freq_bands/freq_meerkat_s3.txt",     "MeerKAT S3 (2406--3281 MHz)"),
    "meerkat_s4":     ("freq_bands/freq_meerkat_s4.txt",     "MeerKAT S4 (2625--3500 MHz)"),
    "meerkat_s_full": ("freq_bands/freq_meerkat_s_full.txt", "MeerKAT S full (1750--3500 MHz, idealized)"),
    # uGMRT
    "ugmrt_band3":    ("freq_bands/freq_ugmrt_band3.txt",    "uGMRT Band 3 (250--500 MHz)"),
    "ugmrt_band4":    ("freq_bands/freq_ugmrt_band4.txt",    "uGMRT Band 4 (550--850 MHz)"),
    "ugmrt_band5":    ("freq_bands/freq_ugmrt_band5.txt",    "uGMRT Band 5 (1000--1460 MHz)"),
}

DEFAULT_BAND_ORDER = [
    "vla_p", "vla_l", "vla_s", "vla_c",
    "meerkat_uhf", "meerkat_l",
    "meerkat_s0", "meerkat_s1", "meerkat_s2", "meerkat_s3", "meerkat_s4",
    "meerkat_s_full",
    "ugmrt_band3", "ugmrt_band4", "ugmrt_band5",
]

MODEL_TYPE = "faraday_thin"
N_COMPONENTS_RANGE = range(1, 6)   # N = 1, 2, 3, 4, 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def run(cmd: list[str], log_path: Path | None = None) -> int:
    """Run a subprocess, optionally tee-ing stdout+stderr to a log file."""
    print(f"\n>>> {' '.join(cmd)}", flush=True)
    if log_path:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("w") as fh:
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = proc.stdout.decode(errors="replace")
            fh.write(output)
            sys.stdout.write(output)
        return proc.returncode
    else:
        return subprocess.run(cmd).returncode


def band_save_dir(base_save_dir: str, band_key: str) -> str:
    """Derive a per-band model output directory from the config's save_dir."""
    return f"{base_save_dir}/{band_key}"


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_train(
    band_key: str,
    freq_file: str,
    config_path: str,
    save_dir: str,
    device: str,
    n_components_range,
) -> bool:
    """Train N=1..max for one band. Returns True if all succeeded."""
    ok = True
    for n in n_components_range:
        log = Path(save_dir) / "train_logs" / f"train_{band_key}_n{n}.log"
        ret = run(
            [
                "vroom-sbi", "train",
                "--config", config_path,
                "--device", device,
                "--n-components", str(n),
                # Override freq_file and save_dir via env isn't supported;
                # we patch the config at runtime via the two extra flags below.
                # The CLI reads config.yaml then these flags override it.
            ]
            # NOTE: vroom-sbi train doesn't yet accept --freq-file / --save-dir
            # as CLI flags; the per-band config YAML is generated alongside this
            # script by generate_band_configs.py.  Update this call if CLI flags
            # are added later.
            ,
            log_path=log,
        )
        if ret != 0:
            print(f"  [ERROR] training failed for {band_key} N={n} (exit {ret})", flush=True)
            ok = False
    return ok


def step_coverage(
    band_key: str,
    config_path: str,
    save_dir: str,
    device: str,
    n_components_range,
    n_cases: int,
    n_samples: int,
) -> Path | None:
    """Run online_probe for all N and write a combined coverage log. Returns log path."""
    log_path = Path(save_dir) / "validation" / f"coverage_{band_key}.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as fh:
        for n in n_components_range:
            pt = Path(save_dir) / f"posterior_{MODEL_TYPE}_n{n}.pt"
            if not pt.exists():
                msg = f"[SKIP] {pt} not found — was training skipped or did it fail?\n"
                print(msg, end="", flush=True)
                fh.write(msg)
                continue
            cmd = [
                "python", "-m", "src.validation.online_probe",
                "--config", config_path,
                "--posterior", str(pt),
                "--device", device,
                "--n-cases", str(n_cases),
                "--n-samples", str(n_samples),
            ]
            print(f"\n>>> {' '.join(cmd)}", flush=True)
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            output = proc.stdout.decode(errors="replace")
            fh.write(output)
            sys.stdout.write(output)
    return log_path


def step_plot(band_key: str, save_dir: str, log_path: Path) -> Path | None:
    """Run plot_coverage on the coverage log. Returns output plot path."""
    out_png = Path(save_dir) / "validation" / f"coverage_{band_key}.png"
    ret = run([
        "python", "-m", "src.validation.plot_coverage",
        str(log_path),
        "--output", str(out_png),
    ])
    if ret != 0:
        print(f"  [ERROR] plot_coverage failed for {band_key}", flush=True)
        return None
    return out_png


def step_sbc(
    band_key: str,
    config_path: str,
    save_dir: str,
    device: str,
    n_components_range,
    n_cases: int,
    n_samples: int,
) -> None:
    """Run SBC rank histograms for all N (optional, slow)."""
    out_dir = Path(save_dir) / "validation" / "sbc"
    out_dir.mkdir(parents=True, exist_ok=True)
    for n in n_components_range:
        pt = Path(save_dir) / f"posterior_{MODEL_TYPE}_n{n}.pt"
        if not pt.exists():
            print(f"[SKIP SBC] {pt} not found", flush=True)
            continue
        run([
            "python", "-m", "src.validation.sbc",
            "--config", config_path,
            "--posterior", str(pt),
            "--device", device,
            "--n-cases", str(n_cases),
            "--n-samples", str(n_samples),
            "--output-dir", str(out_dir),
        ])


# ---------------------------------------------------------------------------
# Config generation
# ---------------------------------------------------------------------------

def write_band_config(
    band_key: str,
    freq_file: str,
    base_config_path: str,
    base_save_dir: str,
    out_config_path: Path,
) -> None:
    """Write a per-band YAML config by patching freq_file and save_dir."""
    import yaml  # available in all pixi envs that have vroom-sbi

    with open(base_config_path) as fh:
        cfg = yaml.safe_load(fh)

    cfg["freq_file"] = freq_file
    cfg["training"]["save_dir"] = band_save_dir(base_save_dir, band_key)

    out_config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_config_path, "w") as fh:
        yaml.dump(cfg, fh, default_flow_style=False, sort_keys=False)

    print(f"  wrote config: {out_config_path}", flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="config_a100_lrtest.yaml",
                    help="Base config YAML (freq_file and save_dir are overridden per band)")
    ap.add_argument("--bands", nargs="+", default=None,
                    help="Band keys to run (default: all). Available: " + ", ".join(BANDS))
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-cases", type=int, default=2000,
                    help="Coverage probe: number of simulated cases")
    ap.add_argument("--n-samples", type=int, default=1000,
                    help="Coverage probe: posterior draws per case")
    ap.add_argument("--skip-train", action="store_true",
                    help="Skip training; only run coverage + plot on existing models")
    ap.add_argument("--skip-coverage", action="store_true",
                    help="Skip coverage probe and plotting")
    ap.add_argument("--sbc", action="store_true",
                    help="Also run SBC rank histograms (slow, ~20 min per band)")
    args = ap.parse_args()

    # Resolve band list
    band_keys = args.bands if args.bands else DEFAULT_BAND_ORDER
    unknown = [b for b in band_keys if b not in BANDS]
    if unknown:
        ap.error(f"Unknown bands: {unknown}. Available: {list(BANDS)}")

    # Read base config to find base save_dir
    import yaml
    with open(args.config) as fh:
        base_cfg = yaml.safe_load(fh)
    base_save_dir = base_cfg.get("training", {}).get("save_dir", "models_sweep")

    # Per-band configs go next to the base config
    config_dir = Path(args.config).parent / "band_configs"

    print(f"\n{'='*60}")
    print(f"vroom-sbi band sweep")
    print(f"  base config : {args.config}")
    print(f"  base save   : {base_save_dir}")
    print(f"  bands       : {band_keys}")
    print(f"  train       : {not args.skip_train}")
    print(f"  coverage    : {not args.skip_coverage}")
    print(f"  sbc         : {args.sbc}")
    print(f"{'='*60}\n")

    for band_key in band_keys:
        freq_file, label = BANDS[band_key]
        save_dir = band_save_dir(base_save_dir, band_key)
        band_config_path = config_dir / f"config_{band_key}.yaml"

        print(f"\n{'='*60}")
        print(f"BAND: {label}")
        print(f"  freq_file : {freq_file}")
        print(f"  save_dir  : {save_dir}")
        print(f"{'='*60}")

        # Generate per-band config
        write_band_config(band_key, freq_file, args.config, base_save_dir, band_config_path)
        band_config = str(band_config_path)

        # Train
        if not args.skip_train:
            step_train(
                band_key, freq_file, band_config, save_dir,
                args.device, N_COMPONENTS_RANGE,
            )

        # Coverage probe + plot
        if not args.skip_coverage:
            log = step_coverage(
                band_key, band_config, save_dir, args.device,
                N_COMPONENTS_RANGE, args.n_cases, args.n_samples,
            )
            if log and log.stat().st_size > 0:
                step_plot(band_key, save_dir, log)

        # SBC (optional)
        if args.sbc:
            step_sbc(
                band_key, band_config, save_dir, args.device,
                N_COMPONENTS_RANGE, args.n_cases, args.n_samples,
            )

    print(f"\n{'='*60}")
    print("Sweep complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
