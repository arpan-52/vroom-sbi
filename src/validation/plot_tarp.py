"""Ingestible TARP figures from saved ``tarp_<stem>.json`` summaries.

Two products, both readable at a glance without knowing the ATRC gap number:

  1. per-model banded ECP curve  (``plot_ecp_band``)
       observed coverage with its per-α sampling band, drawn against the
       perfectly-calibrated null envelope. Deviation is real only where the
       observed band clears the null envelope. Uses the tarp bootstrap band
       (``ecp_std``) when present, else the analytic binomial error
       sqrt(ECP(1-ECP)/n_cases) — free, no re-run.

  2. calibration matrix heatmap  (``plot_area_matrix``)
       unsigned miscalibration area over the (model_type, N) grid, colored by
       how far each sits above its null threshold; the honest analogue of a
       gap-matrix that cannot be fooled by cancellation.

Run:
    pixi run python -m src.validation.plot_tarp --tarp-dir validation_results/tarp
"""

import argparse
import glob
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .tarp_test import (
    _coverage_stats, _null_thresholds, null_band, verdict_from_stats,
)


def _per_bin_sigma(ecp: np.ndarray, n_cases: int, ecp_std) -> np.ndarray:
    """Per-α 1σ: bootstrap std if available, else binomial approximation."""
    if ecp_std is not None:
        return np.asarray(ecp_std)
    return np.sqrt(np.clip(ecp * (1.0 - ecp), 0.0, None) / max(n_cases, 1))


def plot_ecp_band(json_path: str | Path, out_path: str | Path | None = None) -> Path:
    """Banded ECP curve for one posterior; returns the PNG path written."""
    d = json.load(open(json_path))
    alphas = np.asarray(d["alphas"])
    ecp = np.asarray(d["ecp"])
    n_cases = int(d["n_cases"])
    sigma = _per_bin_sigma(ecp, n_cases, d.get("ecp_std"))
    stats = _coverage_stats(ecp, alphas)
    nb = null_band(n_cases, alphas)
    verdict = verdict_from_stats(stats, _null_thresholds(n_cases, alphas)["area_thresh"])
    src = "bootstrap" if d.get("ecp_std") is not None else "binomial approx"

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.fill_between(nb["alphas"], nb["ecp_lo"], nb["ecp_hi"], color="0.7",
                    alpha=0.35, label=f"null {nb['level']:.0%} envelope")
    ax.fill_between(alphas, ecp - 2 * sigma, ecp + 2 * sigma, color="C0",
                    alpha=0.25, label=f"±2σ ({src})")
    ax.plot(alphas, ecp, color="C0", lw=2, label="observed ECP")
    ax.set_xlabel("Credible level α")
    ax.set_ylabel("Expected coverage probability")
    ax.set_title(f"{d['model_type']} N={d['n_components']} — {verdict}")
    ax.text(0.03, 0.97,
            f"signed gap {stats['signed_gap']:+.4f}\n"
            f"|area| {stats['unsigned_area']:.4f}\n"
            f"max dev {stats['max_dev']:+.3f}@α={stats['max_dev_alpha']:.2f}\n"
            f"crosses: {stats['crosses']}",
            transform=ax.transAxes, va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round", fc="white", alpha=0.8))
    ax.legend(fontsize=8, loc="lower right")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else Path(json_path).with_name(
        f"band_{Path(json_path).stem.replace('tarp_', '')}.png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_area_matrix(tarp_dir: str | Path, out_path: str | Path | None = None) -> Path:
    """Heatmap of unsigned miscalibration area over the (model, N) grid."""
    tarp_dir = Path(tarp_dir)
    recs = []
    for f in sorted(glob.glob(str(tarp_dir / "tarp_*.json"))):
        d = json.load(open(f))
        ecp = np.asarray(d["ecp"]); a = np.asarray(d["alphas"])
        s = _coverage_stats(ecp, a)
        nl = _null_thresholds(int(d["n_cases"]), a)
        recs.append({
            "model": d["model_type"], "n": int(d["n_components"]),
            "area": s["unsigned_area"], "thresh": nl["area_thresh"],
            "verdict": verdict_from_stats(s, nl["area_thresh"]),
        })
    models = sorted({r["model"] for r in recs})
    ns = sorted({r["n"] for r in recs})
    # excess = area / null_threshold; >1 means distinguishable from calibrated
    grid = np.full((len(models), len(ns)), np.nan)
    ann = np.empty((len(models), len(ns)), dtype=object)
    vinit = {"calibrated": "ok", "over-confident": "over",
             "under-confident": "und", "mixed": "MIX"}
    for r in recs:
        i, j = models.index(r["model"]), ns.index(r["n"])
        grid[i, j] = r["area"] / r["thresh"]
        ann[i, j] = f"{r['area']:.3f}\n{vinit.get(r['verdict'], '')}"

    fig, ax = plt.subplots(figsize=(1.6 * len(ns) + 2, 0.9 * len(models) + 2))
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0.0, vmax=2.0, aspect="auto")
    ax.set_xticks(range(len(ns))); ax.set_xticklabels([f"N={n}" for n in ns])
    ax.set_yticks(range(len(models))); ax.set_yticklabels(models)
    for i in range(len(models)):
        for j in range(len(ns)):
            if not np.isnan(grid[i, j]):
                ax.text(j, i, ann[i, j], ha="center", va="center", fontsize=8)
    ax.set_title("TARP miscalibration:  |area| / null threshold\n"
                 "(<1 green = calibrated;  >1 red = distinguishable)")
    fig.colorbar(im, ax=ax, label="area / threshold")
    fig.tight_layout()

    out_path = Path(out_path) if out_path else tarp_dir / "calibration_area_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tarp-dir", default="validation_results/tarp",
                    help="directory holding tarp_*.json summaries")
    ap.add_argument("--matrix-only", action="store_true",
                    help="only the matrix heatmap, skip per-model band plots")
    args = ap.parse_args()

    tarp_dir = Path(args.tarp_dir)
    if not args.matrix_only:
        for f in sorted(glob.glob(str(tarp_dir / "tarp_*.json"))):
            print("band:", plot_ecp_band(f))
    print("matrix:", plot_area_matrix(tarp_dir))


if __name__ == "__main__":
    main()
