"""
Diagnose the train/val gap against the learning-rate trajectory.

On-the-fly training draws fresh sims each step, so a rising val loss is not
classic overfitting to a finite set — it points at the optimizer leaving the
basin. This overlays, per run:

  - train vs val loss on a shared axis (the gap and the val turn-up)
  - the learning-rate trajectory (twin axis), with ReduceLROnPlateau drops
    marked, and the best-val epoch marked

If val turns up while lr is still high and only drops afterwards, the schedule
reacted too late; if val keeps rising even after lr drops, the basin itself is
unstable at that lr.

Run:
    pixi run -e gpu python -m src.validation.lr_diagnostic \
        models/posterior_faraday_thin_n{1,2,3,4,5}.pt \
        --output models/lr_diagnostic.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _load(path: str | Path) -> dict:
    c = torch.load(path, map_location="cpu", weights_only=False)
    h = c.get("training_history", {})
    return {
        "label": f"{c['model_type']} N={c['n_components']}",
        "train": np.asarray(h.get("train_loss", []), float),
        "val": np.asarray(h.get("val_loss", []), float),
        "lr": np.asarray(h.get("learning_rates", []), float),
    }


def plot_lr(paths: list[str | Path], output: str | Path) -> Path:
    runs = [_load(p) for p in paths]
    n = len(runs)
    fig, axes = plt.subplots(n, 1, figsize=(11, 2.8 * n), squeeze=False)

    for r, run in enumerate(runs):
        ax = axes[r, 0]
        ep = np.arange(1, len(run["val"]) + 1)
        ax.plot(ep, run["train"], color="tab:blue", lw=1.2, label="train")
        ax.plot(ep, run["val"], color="tab:orange", lw=1.2, label="val")

        b = int(run["val"].argmin())
        ax.scatter(b + 1, run["val"][b], color="k", zorder=5, s=28, label="best val")
        # Shade the post-best region where val is allowed to drift up.
        ax.axvspan(b + 1, len(ep), color="red", alpha=0.05)
        ax.set_ylabel(f"{run['label']}\nloss", fontsize=9)
        ax.grid(alpha=0.3)

        lr = run["lr"]
        if lr.size:
            axr = ax.twinx()
            axr.plot(ep, lr, color="tab:green", lw=1.0, ls="--", label="lr")
            axr.set_yscale("log")
            axr.set_ylabel("lr (log)", fontsize=8, color="tab:green")
            axr.tick_params(axis="y", labelcolor="tab:green", labelsize=7)
            # Mark lr drops (ReduceLROnPlateau halving).
            drops = np.where(np.diff(lr) < 0)[0] + 1
            for d in drops:
                axr.axvline(d + 1, color="tab:green", alpha=0.25, lw=0.8)

        if r == 0:
            ax.legend(fontsize=7, loc="upper right")
    axes[-1, 0].set_xlabel("epoch")
    fig.suptitle("Train/val gap vs learning-rate schedule", y=1.0)
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("posteriors", nargs="+")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()
    out = plot_lr(args.posteriors, args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
