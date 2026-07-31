"""
Diagnose early-stopping behaviour from recorded val-loss histories.

The online trainer stops when val loss fails to beat its running best for
``stop_after_epochs`` epochs (strict ``<``, no smoothing). When the per-epoch
trend improvement falls below the val-loss noise amplitude, the counter trips
even though the mean is still descending. This replays that rule on the saved
``val_loss`` arrays under different smoothing windows and shows:

  - left  : raw val loss + moving-average smoothing + running best
  - right : the "epochs without improvement" counter each scheme produces,
            against the actual patience threshold

Important: each recorded curve ends at the *actual* stop epoch, so this shows
the mechanism on observed data, not a counterfactual past the stop.

Run:
    pixi run -e gpu python -m src.validation.early_stop_diagnostic \
        models/posterior_faraday_thin_n{1,2,3,4,5}.pt \
        --patience 20 --windows 1 5 10 \
        --output models/early_stop_diagnostic.png
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1:
        return x
    # Trailing average (causal): only past epochs, matching how a live trainer
    # would see the signal. Edges shrink the window so length is preserved.
    out = np.empty_like(x)
    for i in range(len(x)):
        out[i] = x[max(0, i - window + 1) : i + 1].mean()
    return out


def _improvement_counter(signal: np.ndarray) -> np.ndarray:
    """Replay strict best-so-far: counter of consecutive non-improvements."""
    best = np.inf
    cnt = 0
    counters = np.empty(len(signal), dtype=int)
    for i, v in enumerate(signal):
        if v < best:
            best = v
            cnt = 0
        else:
            cnt += 1
        counters[i] = cnt
    return counters


def _load_val(path: str | Path) -> dict:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    val = np.asarray(ckpt.get("training_history", {}).get("val_loss", []), float)
    return {
        "label": f"{ckpt['model_type']} N={ckpt['n_components']}",
        "val": val,
    }


def plot_diagnostic(
    paths: list[str | Path],
    output: str | Path,
    patience: int = 20,
    windows: tuple[int, ...] = (1, 5, 10),
) -> Path:
    runs = [_load_val(p) for p in paths]
    n = len(runs)
    fig, axes = plt.subplots(n, 2, figsize=(13, 3.0 * n), squeeze=False)
    colors = plt.get_cmap("plasma")(np.linspace(0.1, 0.8, len(windows)))

    for r, run in enumerate(runs):
        val = run["val"]
        epochs = np.arange(1, len(val) + 1)
        ax_l, ax_r = axes[r, 0], axes[r, 1]

        ax_l.plot(epochs, val, color="0.7", lw=0.9, label="raw val")
        for w, c in zip(windows, colors):
            sm = _moving_average(val, w)
            if w > 1:
                ax_l.plot(epochs, sm, color=c, lw=1.4, label=f"MA(w={w})")
            counters = _improvement_counter(sm)
            ax_r.plot(epochs, counters, color=c, lw=1.4, label=f"w={w}")

        # Running best on raw signal + actual best marker.
        best_idx = int(val.argmin())
        ax_l.scatter(best_idx + 1, val[best_idx], color="k", zorder=5, s=25)
        ax_l.set_ylabel(f"{run['label']}\nval loss", fontsize=9)
        ax_l.grid(alpha=0.3)

        ax_r.axhline(patience, color="r", ls="--", lw=1, label=f"patience={patience}")
        ax_r.set_ylabel("epochs w/o improve", fontsize=9)
        ax_r.grid(alpha=0.3)
        if r == 0:
            ax_l.legend(fontsize=7, loc="upper right")
            ax_r.legend(fontsize=7, loc="upper left")
    axes[-1, 0].set_xlabel("epoch")
    axes[-1, 1].set_xlabel("epoch")
    fig.suptitle(
        "Early-stop diagnostic: val smoothing vs patience counter", y=1.0
    )
    fig.tight_layout()
    output = Path(output)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return output


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("posteriors", nargs="+")
    ap.add_argument("--output", required=True)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 5, 10])
    args = ap.parse_args()
    out = plot_diagnostic(
        args.posteriors, args.output, args.patience, tuple(args.windows)
    )
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
