"""
Overlay training/validation loss curves for a set of saved posteriors.

Each checkpoint stores ``training_history`` with ``train_loss``/``val_loss``
lists (one entry per epoch). This reads several checkpoints and draws them on
shared axes so runs that differ only in component count can be compared at a
glance. Loss is the NPE objective (negative log-prob), so it shifts upward with
parameter dimension; the useful signal here is the *shape* of each descent and
whether a run was still improving when early stopping fired.

Run:
    pixi run -e gpu python -m src.validation.plot_training_curves \
        models/posterior_faraday_thin_n1.pt ... \
        --output models/training_curves_faraday_thin.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch


def _load_history(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    hist = ckpt.get("training_history", {})
    val = hist.get("val_loss", [])
    return {
        "label": f"{ckpt['model_type']} N={ckpt['n_components']}",
        "train": hist.get("train_loss", []),
        "val": val,
        "best_epoch": (int(min(range(len(val)), key=val.__getitem__)) if val else None),
        "best_val": (min(val) if val else None),
    }


def plot_curves(paths: list[str | Path], output: str | Path) -> Path:
    runs = [_load_history(p) for p in paths]

    fig, (ax_t, ax_v) = plt.subplots(1, 2, figsize=(13, 5), sharex=True)
    cmap = plt.get_cmap("viridis")
    for i, r in enumerate(runs):
        color = cmap(i / max(len(runs) - 1, 1))
        epochs = range(1, len(r["val"]) + 1)
        ax_t.plot(range(1, len(r["train"]) + 1), r["train"], color=color, lw=1.2)
        ax_v.plot(epochs, r["val"], color=color, lw=1.2, label=r["label"])
        if r["best_epoch"] is not None:
            ax_v.scatter(
                r["best_epoch"] + 1, r["best_val"], color=color, s=30, zorder=5
            )

    ax_t.set_title("Train loss")
    ax_v.set_title("Validation loss (● = best / restored)")
    for ax in (ax_t, ax_v):
        ax.set_xlabel("epoch")
        ax.set_ylabel("NPE loss (neg log-prob)")
        ax.grid(alpha=0.3)
    ax_v.legend(fontsize=8, loc="upper right")
    fig.suptitle("VROOM-SBI online training curves")
    fig.tight_layout()

    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("posteriors", nargs="+", help="Posterior .pt checkpoints")
    ap.add_argument("--output", required=True, help="Output image path")
    args = ap.parse_args()
    out = plot_curves(args.posteriors, args.output)
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
