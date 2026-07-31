"""
Paper-figure generator for the wide-prior vroom-sbi posterior matrix.

Regenerates the publication figures from the trained posteriors and the
machine-readable artifacts written by the validation runners. Wired for the
full matrix (all model types x N=1..5) from the start; ``--model`` narrows it.

Design
------
The validation runners (``sbc``, ``online_probe``, ``ppc``, ``tarp``,
``corner_plot``) own the GPU compute and each persist a small artifact
(rank tensors, coverage JSON, PPC summaries). This module is the *composition*
layer: it reads those artifacts and lays them out as paper figures. Figures
whose artifacts are missing are (optionally) computed on demand with ``--compute``
and cached, so restyling never re-runs GPU work.

Figures
-------
  F1  sbc-grid       : SBC rank-histogram grid (model type x N)      [artifact: sbc_*_ranks.pt]
  F2  coverage       : cov68/cov90 heatmap over the matrix           [artifact: probe_*.json]
  F3  rm-recovery    : true vs recovered Faraday depth per model     [GPU compute + cache]
  F4  corner         : representative joint posterior per model      [GPU compute]
  F5  ppc            : posterior-predictive Q/U panel per model      [GPU compute]
  F6  confusion      : classifier confusion matrix (model selection) [GPU compute + cache]

Usage
-----
    # All figures, full matrix, composing from existing artifacts where possible:
    pixi run -e gpu python -m src.validation.paper_figures \
        --config config_a100.yaml --models-dir models \
        --sbc-dir validation_results/sbc --output-dir paper_figures --figures all

    # One figure, compute artifacts if missing:
    pixi run -e gpu python -m src.validation.paper_figures \
        --config config_a100.yaml --figures coverage --compute

    # One model type end-to-end (harness probe):
    pixi run -e gpu python -m src.validation.paper_figures \
        --config config_a100.yaml --model faraday_thin --figures all --compute
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

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


MODEL_TYPES = [
    "faraday_thin",
    "burn_slab",
    "external_dispersion",
    "internal_dispersion",
]

# Pretty labels for figure titles/axes.
MODEL_PRETTY = {
    "faraday_thin": "Faraday thin",
    "burn_slab": "Burn slab",
    "external_dispersion": "External disp.",
    "internal_dispersion": "Internal disp.",
}


# ---------------------------------------------------------------------------
# Discovery helpers
# ---------------------------------------------------------------------------

def posterior_path(models_dir: Path, model_type: str, n: int) -> Path:
    return models_dir / f"posterior_{model_type}_n{n}.pt"


def available_cells(
    models_dir: Path, model_types: list[str], n_list: list[int]
) -> list[tuple[str, int]]:
    """Return (model_type, N) cells whose posterior file exists on disk."""
    cells = []
    for mt in model_types:
        for n in n_list:
            if posterior_path(models_dir, mt, n).exists():
                cells.append((mt, n))
    return cells


# ---------------------------------------------------------------------------
# F1 — SBC rank-histogram grid  (composition from sbc_*_ranks.pt)
# ---------------------------------------------------------------------------

def fig_sbc_grid(
    sbc_dir: Path,
    model_types: list[str],
    n_list: list[int],
    output_dir: Path,
) -> Path | None:
    """Grid of SBC rank histograms; rows=model type, cols=N.

    Each cell aggregates ranks across all parameters of that posterior into one
    histogram, with the expected-uniform band shaded. Reads the ranks tensors
    persisted by ``sbc.run_sbc_for_posterior``; cells without an artifact are
    left blank (run SBC to fill them).
    """
    import torch

    nrows, ncols = len(model_types), len(n_list)
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(2.6 * ncols, 2.2 * nrows), squeeze=False
    )
    n_bins = 20
    found_any = False
    for r, mt in enumerate(model_types):
        for c, n in enumerate(n_list):
            ax = axes[r][c]
            ranks_file = sbc_dir / f"sbc_{mt}_n{n}_ranks.pt"
            if not ranks_file.exists():
                ax.set_axis_off()
                ax.text(0.5, 0.5, "—", ha="center", va="center", color="0.7")
            else:
                found_any = True
                blob = torch.load(ranks_file, map_location="cpu", weights_only=False)
                ranks = np.asarray(blob["ranks"]).reshape(-1)
                nps = blob["num_posterior_samples"]
                # Expected count per bin under uniformity + Poisson 99% band.
                expected = ranks.size / n_bins
                band = 2.576 * np.sqrt(expected)
                ax.axhspan(expected - band, expected + band, color="0.85", zorder=0)
                ax.axhline(expected, color="0.6", lw=0.8, ls="--", zorder=1)
                ax.hist(ranks, bins=n_bins, range=(0, nps),
                        color="tab:blue", alpha=0.8, zorder=2)
                ax.set_yticks([])
            if r == 0:
                ax.set_title(f"N={n}", fontsize=10)
            if c == 0:
                ax.set_ylabel(MODEL_PRETTY.get(mt, mt), fontsize=9)
    if not found_any:
        plt.close(fig)
        print(f"[F1] no SBC rank artifacts in {sbc_dir} — run scripts/run_sbc.py first")
        return None
    fig.suptitle("SBC rank calibration (aggregated over parameters)", y=1.005)
    fig.tight_layout()
    out = output_dir / "F1_sbc_grid.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[F1] {out}")
    return out


# ---------------------------------------------------------------------------
# F2 — coverage heatmap  (composition from probe_*.json, optional compute)
# ---------------------------------------------------------------------------

def fig_coverage(
    probe_dir: Path,
    models_dir: Path,
    model_types: list[str],
    n_list: list[int],
    output_dir: Path,
    config_path: str,
    compute: bool,
    device: str,
    freq_file: str | None,
    n_cases: int,
    n_samples: int,
) -> Path | None:
    """Heatmap of mean central-interval coverage (cov68, cov90) over the matrix.

    Prefers cached ``probe_<stem>.json`` from ``online_probe.run_probe``. With
    ``--compute`` it runs the probe for any missing cell (writing the JSON).
    """
    cov68 = np.full((len(model_types), len(n_list)), np.nan)
    cov90 = np.full((len(model_types), len(n_list)), np.nan)
    for r, mt in enumerate(model_types):
        for c, n in enumerate(n_list):
            jp = probe_dir / f"probe_{mt}_n{n}.json"
            if not jp.exists() and compute and posterior_path(models_dir, mt, n).exists():
                from .online_probe import run_probe
                run_probe(
                    config_path=config_path,
                    posterior_path=str(posterior_path(models_dir, mt, n)),
                    n_cases=n_cases, n_samples=n_samples,
                    device=device, freq_file=freq_file, output_dir=probe_dir,
                )
            if jp.exists():
                d = json.loads(jp.read_text())
                cov68[r, c] = float(np.mean(d["cov68"]))
                cov90[r, c] = float(np.mean(d["cov90"]))

    if np.all(np.isnan(cov68)):
        print(f"[F2] no probe artifacts in {probe_dir} — run online_probe or pass --compute")
        return None

    fig, axes = plt.subplots(1, 2, figsize=(3.2 * len(n_list) / 2 + 4, 0.7 * len(model_types) + 2))
    for ax, mat, ideal, name in [
        (axes[0], cov68, 0.68, "central 68%"),
        (axes[1], cov90, 0.90, "central 90%"),
    ]:
        # Diverging around the ideal coverage: white = calibrated.
        im = ax.imshow(mat - ideal, cmap="RdBu_r", vmin=-0.25, vmax=0.25, aspect="auto")
        ax.set_xticks(range(len(n_list)), [f"N={n}" for n in n_list])
        ax.set_yticks(range(len(model_types)),
                      [MODEL_PRETTY.get(m, m) for m in model_types])
        ax.set_title(f"{name} coverage − ideal")
        for r in range(len(model_types)):
            for c in range(len(n_list)):
                if not np.isnan(mat[r, c]):
                    ax.text(c, r, f"{mat[r, c]:.2f}", ha="center", va="center",
                            fontsize=7, color="0.1")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Marginal coverage vs. nominal (blue=over-, red=under-confident)")
    fig.tight_layout()
    out = output_dir / "F2_coverage.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[F2] {out}")
    return out


# ---------------------------------------------------------------------------
# F3 — Faraday-depth recovery  (GPU compute + npz cache)
# ---------------------------------------------------------------------------

def fig_rm_recovery(
    models_dir: Path,
    model_types: list[str],
    n_list: list[int],
    output_dir: Path,
    cache_dir: Path,
    config_path: str,
    device: str,
    freq_file: str | None,
    n_cases: int,
    n_samples: int,
) -> Path | None:
    """True vs recovered Faraday depth (parameter 0, first component) per model.

    One panel per model type; points are posterior-mean vs truth across n_cases
    prior draws, coloured by the highest available N. Cached to npz per cell.
    """
    import torch

    from ..config import Configuration
    from ..simulator.gpu_simulator import GPUSimulator
    from .online_probe import VAL_SEED, _rebuild_posterior

    dev = device if torch.cuda.is_available() else "cpu"
    config = Configuration.from_yaml(config_path)
    if freq_file:
        config.freq_file = freq_file

    def recover(mt: str, n: int) -> tuple[np.ndarray, np.ndarray] | None:
        pt = posterior_path(models_dir, mt, n)
        if not pt.exists():
            return None
        cache = cache_dir / f"recovery_{mt}_n{n}.npz"
        if cache.exists():
            z = np.load(cache)
            return z["truth"], z["mean"]
        ckpt = torch.load(pt, map_location="cpu", weights_only=False)
        sim = GPUSimulator(
            config=config, model_type=mt, n_components=n, device=dev,
            sampling_method=getattr(config.training, "sampling_method", "uniform"),
        )
        gen = torch.Generator(device=dev)
        gen.manual_seed(VAL_SEED)
        post = _rebuild_posterior(ckpt, sim, dev)
        theta, x = sim.generate_batch(n_cases, generator=gen)
        truth = theta[:, 0].cpu().numpy()
        means = np.empty(n_cases)
        with torch.no_grad():
            for i in range(n_cases):
                s = post.sample((n_samples,), x=x[i:i + 1], show_progress_bars=False)
                means[i] = s[:, 0].mean().item()
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, truth=truth, mean=means)
        return truth, means

    fig, axes = plt.subplots(1, len(model_types),
                             figsize=(3.4 * len(model_types), 3.4), squeeze=False)
    plotted = False
    for c, mt in enumerate(model_types):
        ax = axes[0][c]
        n = max(n_list)  # highest-N posterior is the hardest case
        res = recover(mt, n)
        while res is None and n > min(n_list):
            n -= 1
            res = recover(mt, n)
        if res is None:
            ax.set_axis_off()
            continue
        plotted = True
        truth, means = res
        lo, hi = truth.min(), truth.max()
        ax.plot([lo, hi], [lo, hi], color="0.5", lw=1, ls="--", zorder=1)
        ax.scatter(truth, means, s=4, alpha=0.3, color="tab:blue", zorder=2)
        ax.set_title(f"{MODEL_PRETTY.get(mt, mt)} (N={n})", fontsize=10)
        ax.set_xlabel("true φ (rad/m²)")
        if c == 0:
            ax.set_ylabel("recovered φ (rad/m²)")
    if not plotted:
        plt.close(fig)
        print("[F3] no posteriors found for recovery")
        return None
    fig.suptitle("Faraday-depth recovery across the wide prior")
    fig.tight_layout()
    out = output_dir / "F3_rm_recovery.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[F3] {out}")
    return out


# ---------------------------------------------------------------------------
# F4 — representative corner per model type  (delegates to corner_plot)
# ---------------------------------------------------------------------------

def fig_corner(
    models_dir: Path,
    model_types: list[str],
    output_dir: Path,
    config_path: str,
    device: str,
    freq_file: str | None,
    n_components: int,
) -> list[Path]:
    """One joint-posterior corner per model type, from a simulated case.

    Draws a single theta from the prior, forward-simulates a clean Q/U spectrum,
    and runs the matching posterior. Produces one PNG per model type (corner
    plots do not compose into a single grid cleanly).
    """
    import torch

    from ..config import Configuration
    from ..simulator.gpu_simulator import GPUSimulator
    from ..simulator.torch_physics import polarization_signal
    from .corner_plot import make_corner_plot

    dev = device if torch.cuda.is_available() else "cpu"
    config = Configuration.from_yaml(config_path)
    if freq_file:
        config.freq_file = freq_file
    outs = []
    for mt in model_types:
        pt = posterior_path(models_dir, mt, n_components)
        if not pt.exists():
            continue
        sim = GPUSimulator(
            config=config, model_type=mt, n_components=n_components, device=dev,
            sampling_method=getattr(config.training, "sampling_method", "uniform"),
        )
        gen = torch.Generator(device=dev)
        gen.manual_seed(VAL_SEED_CORNER)
        theta, _ = sim.generate_batch(1, generator=gen)
        P = polarization_signal(theta.to(dev), sim.lambda_sq, mt, n_components)
        Q = P.real[0].cpu().numpy()
        U = P.imag[0].cpu().numpy()
        out = output_dir / f"F4_corner_{mt}.png"
        make_corner_plot(
            config_path=config_path, model_dir=str(models_dir),
            Q=Q, U=U, freq_hz=sim.freq.cpu().numpy(),
            n_samples=10000, device=dev,
            model_type=mt, n_components=n_components,
            output=str(out),
            title=f"{MODEL_PRETTY.get(mt, mt)} N={n_components} (simulated)",
        )
        outs.append(out)
        print(f"[F4] {out}")
    return outs


VAL_SEED_CORNER = 20260707  # fixed representative-case seed for reproducible corners


# ---------------------------------------------------------------------------
# F5 — posterior-predictive panel per model type  (delegates to ppc)
# ---------------------------------------------------------------------------

def fig_ppc(
    models_dir: Path,
    model_types: list[str],
    output_dir: Path,
    config_path: str,
    device: str,
    freq_file: str | None,
    n_components: int,
    n_cases: int,
    n_samples: int,
) -> list[Path]:
    """Posterior-predictive check per model type (delegates to ppc.run_ppc)."""
    from .ppc import run_ppc

    outs = []
    ppc_dir = output_dir / "F5_ppc"
    for mt in model_types:
        pt = posterior_path(models_dir, mt, n_components)
        if not pt.exists():
            continue
        res = run_ppc(
            config_path=config_path, posterior_path=str(pt),
            output_dir=ppc_dir, n_cases=n_cases, n_samples=n_samples,
            device=device, freq_file=freq_file,
        )
        outs.append(Path(res["summary_png"]))
        print(f"[F5] {res['summary_png']}")
    return outs


# ---------------------------------------------------------------------------
# F6 — classifier confusion matrix  (GPU compute + npz cache)
# ---------------------------------------------------------------------------

def fig_confusion(
    models_dir: Path,
    output_dir: Path,
    cache_dir: Path,
    config_path: str,
    classifier_path: Path,
    device: str,
    freq_file: str | None,
    n_per_class: int,
) -> Path | None:
    """Confusion matrix for the model-selection classifier.

    Generates ``n_per_class`` labelled spectra for each class the classifier
    knows (from its ``class_to_label`` map), predicts, and plots the row-
    normalised confusion matrix. Cached to npz.
    """
    import torch

    from ..config import Configuration
    from ..simulator.gpu_simulator import GPUSimulator
    from ..training.classifier_trainer import ClassifierTrainer

    if not classifier_path.exists():
        print(f"[F6] classifier not found at {classifier_path}")
        return None

    dev = device if torch.cuda.is_available() else "cpu"
    config = Configuration.from_yaml(config_path)
    if freq_file:
        config.freq_file = freq_file

    # load() rebuilds the model from the checkpoint, so the ctor shapes are
    # placeholders overwritten immediately below.
    clf = ClassifierTrainer(n_freq=1, n_classes=1, config=None, device=dev)
    clf.load(str(classifier_path))
    mapping = clf.class_to_label or {}
    if not mapping:
        print("[F6] classifier has no class_to_label map — cannot label axes")
        return None
    classes = sorted(mapping.keys())
    labels = [mapping[c] for c in classes]  # (model_type, n_components)

    cache = cache_dir / "confusion.npz"
    if cache.exists():
        conf = np.load(cache)["conf"]
    else:
        conf = np.zeros((len(classes), len(classes)), dtype=float)
        gen = torch.Generator(device=dev)
        gen.manual_seed(VAL_SEED_CORNER)
        for true_idx, cls in enumerate(classes):
            mt, n = mapping[cls]
            sim = GPUSimulator(
                config=config, model_type=mt, n_components=n, device=dev,
                sampling_method=getattr(config.training, "sampling_method", "uniform"),
            )
            _, x = sim.generate_batch(n_per_class, generator=gen)
            for i in range(n_per_class):
                pred_mt, pred_n, _, _ = clf.predict_label(x[i])
                # Map predicted (model_type, n) back to a class column.
                for col_idx, c2 in enumerate(classes):
                    if mapping[c2] == (pred_mt, pred_n):
                        conf[true_idx, col_idx] += 1
                        break
        cache_dir.mkdir(parents=True, exist_ok=True)
        np.savez(cache, conf=conf)

    row_sums = conf.sum(axis=1, keepdims=True)
    conf_norm = np.divide(conf, row_sums, out=np.zeros_like(conf), where=row_sums > 0)

    tick_labels = [f"{MODEL_PRETTY.get(m, m)[:4]} N{n}" for (m, n) in labels]
    fig, ax = plt.subplots(figsize=(0.5 * len(classes) + 3, 0.5 * len(classes) + 3))
    im = ax.imshow(conf_norm, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(len(classes)), tick_labels, rotation=90, fontsize=6)
    ax.set_yticks(range(len(classes)), tick_labels, fontsize=6)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Model-selection confusion (row-normalised)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    out = output_dir / "F6_confusion.png"
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"[F6] {out}")
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

ALL_FIGURES = ["sbc-grid", "coverage", "rm-recovery", "corner", "ppc", "confusion"]


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", default="config_a100.yaml")
    ap.add_argument("--models-dir", default="models")
    ap.add_argument("--sbc-dir", default="validation_results/sbc",
                    help="where sbc_*_ranks.pt live (for F1)")
    ap.add_argument("--probe-dir", default="validation_results/probe",
                    help="where probe_*.json live / are written (for F2)")
    ap.add_argument("--output-dir", default="paper_figures")
    ap.add_argument("--cache-dir", default="paper_figures/cache")
    ap.add_argument("--figures", nargs="+", default=["all"],
                    help=f"subset of {ALL_FIGURES} or 'all'")
    ap.add_argument("--model", nargs="+", default=None,
                    help="model type subset (default: all four)")
    ap.add_argument("--n-range", default="1-5", help='e.g. "1-5", "1,3,5"')
    ap.add_argument("--corner-n", type=int, default=2,
                    help="N to use for the representative corner/PPC figures")
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--compute", action="store_true",
                    help="compute missing artifacts on demand (GPU) rather than skip")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freq-file", default=None)
    ap.add_argument("--classifier", default=None,
                    help="classifier .pt (default: <models-dir>/classifier.pt)")
    args = ap.parse_args()

    figures = ALL_FIGURES if "all" in args.figures else args.figures
    model_types = args.model or MODEL_TYPES
    n_list = parse_n_range(args.n_range)
    models_dir = Path(args.models_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    sbc_dir = Path(args.sbc_dir)
    probe_dir = Path(args.probe_dir)
    classifier_path = Path(args.classifier) if args.classifier else models_dir / "classifier.pt"

    print(f"Figures     : {figures}")
    print(f"Model types : {model_types}")
    print(f"N range     : {n_list}")
    print(f"Compute     : {args.compute}\n")

    if "sbc-grid" in figures:
        fig_sbc_grid(sbc_dir, model_types, n_list, output_dir)
    if "coverage" in figures:
        fig_coverage(probe_dir, models_dir, model_types, n_list, output_dir,
                     args.config, args.compute, args.device, args.freq_file,
                     args.n_cases, args.n_samples)
    if "rm-recovery" in figures:
        fig_rm_recovery(models_dir, model_types, n_list, output_dir, cache_dir,
                        args.config, args.device, args.freq_file,
                        args.n_cases, args.n_samples)
    if "corner" in figures:
        fig_corner(models_dir, model_types, output_dir, args.config,
                   args.device, args.freq_file, args.corner_n)
    if "ppc" in figures:
        fig_ppc(models_dir, model_types, output_dir, args.config, args.device,
                args.freq_file, args.corner_n, min(20, args.n_cases), 500)
    if "confusion" in figures:
        fig_confusion(models_dir, output_dir, cache_dir, args.config,
                      classifier_path, args.device, args.freq_file,
                      n_per_class=max(200, args.n_cases // 4))

    print(f"\nDone. Figures in {output_dir}/")


if __name__ == "__main__":
    main()
