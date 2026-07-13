"""Histogram plot for ``ppc_flag_<stem>.json`` summaries.

A well-specified model gives an (approximately) uniform p-value distribution
on the null (in-family) cases. Deviation from uniform is calibration signal,
not misspecification signal — so before trusting an OOD detection rate, look
at whether the null histogram is flat. A null piled up near 0 means the flag
is firing on posterior calibration quality, not out-of-family structure.

Run:
    pixi run python -m src.validation.plot_ppc --json validation_results/ppc_probe/ppc_flag_faraday_thin_n1.json
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_ppc_ecdf(json_path: str | Path, out_path: str | Path | None = None) -> Path:
    """ECDF of null/OOD p-values against the uniform diagonal — clearer than a
    histogram when most mass sits near 0."""
    d = json.load(open(json_path))
    null = d["null"]
    ood = d.get("ood")
    stats = [("p_chi2", "chi2 (residual amplitude)"), ("p_acorr", "lag-1 autocorr (residual structure)")]

    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4.2))
    if len(stats) == 1:
        axes = [axes]

    for ax, (key, label) in zip(axes, stats):
        p_null = np.sort(np.asarray(null[key]))
        x = np.arange(1, len(p_null) + 1) / len(p_null)
        ax.plot(p_null, x, color="C0", lw=2,
                label=f"null ({null['label']}, n={null['n_cases']})")
        if ood is not None:
            p_ood = np.sort(np.asarray(ood[key]))
            xo = np.arange(1, len(p_ood) + 1) / len(p_ood)
            ax.plot(p_ood, xo, color="C3", lw=2,
                    label=f"OOD ({ood['label']}, n={ood['n_cases']})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="uniform null (ideal)")
        ax.set_xlabel(f"p-value  [{label}]")
        ax.set_ylabel("cumulative fraction of cases")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, loc="lower right")
        ax.grid(alpha=0.3)

    model_desc = f"{d['model_type']} N={d['n_components']}"
    fig.suptitle(f"PPC p-value ECDF — {model_desc}  "
                 f"(curve above the diagonal = piled near 0)", fontsize=10)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else Path(json_path).with_name(
        Path(json_path).stem.replace("ppc_flag_", "ppc_ecdf_") + ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_ppc_hist(json_path: str | Path, out_path: str | Path | None = None) -> Path:
    d = json.load(open(json_path))
    null = d["null"]
    ood = d.get("ood")

    stats = [("p_chi2", "chi2 (residual amplitude)"), ("p_acorr", "lag-1 autocorr (residual structure)")]
    fig, axes = plt.subplots(1, len(stats), figsize=(5 * len(stats), 4.2))
    if len(stats) == 1:
        axes = [axes]

    bins = np.linspace(0, 1, 21)
    for ax, (key, label) in zip(axes, stats):
        p_null = np.asarray(null[key])
        ax.hist(p_null, bins=bins, density=True, alpha=0.55, color="C0",
                label=f"null ({null['label']}, n={null['n_cases']})")
        if ood is not None:
            p_ood = np.asarray(ood[key])
            ax.hist(p_ood, bins=bins, density=True, alpha=0.55, color="C3",
                    label=f"OOD ({ood['label']}, n={ood['n_cases']})")
        ax.axhline(1.0, color="k", ls="--", lw=1, label="uniform null (ideal)")
        ax.set_xlabel(f"p-value  [{label}]")
        ax.set_ylabel("density")
        ax.set_xlim(0, 1)
        ax.legend(fontsize=7, loc="upper center")
        ax.grid(alpha=0.3)

    model_desc = f"{d['model_type']} N={d['n_components']}"
    null_flag = null.get("flag_either@0.05")
    ood_flag = ood.get("flag_either@0.05") if ood is not None else None
    title = f"PPC p-value distributions — {model_desc}\n" \
            f"flag rate @ p<0.05 (either stat): null={null_flag:.2f}"
    if ood_flag is not None:
        title += f"  OOD={ood_flag:.2f}"
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()

    out_path = Path(out_path) if out_path else Path(json_path).with_name(
        Path(json_path).stem.replace("ppc_flag_", "ppc_hist_") + ".png")
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--json", required=True, help="ppc_flag_<stem>.json path")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    out = plot_ppc_hist(args.json, args.out)
    print("wrote:", out)
    out2 = plot_ppc_ecdf(args.json, None if args.out is None else args.out.replace("hist", "ecdf"))
    print("wrote:", out2)


if __name__ == "__main__":
    main()
