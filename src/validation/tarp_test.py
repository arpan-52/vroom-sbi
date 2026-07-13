"""
TARP (Test of Amortized Posteriors using Regression) for online-trained NPE.

TARP tests *joint* calibration via a single expected-coverage curve, without
marginalizing to individual parameters. It is complementary to SBC (which is
marginal) and coverage probe (which reports only 68%/90% levels).

Reference: Lemos et al. 2023, "Sampling-Based Accuracy Testing of Posterior
Estimators for General Inference" -- https://arxiv.org/abs/2302.03026

The ATRC (Amortized Test of Ranks and Coverage) curve plots the fraction of
test cases where the true theta falls within the alpha-credible HPD region, as
a function of alpha. Perfect calibration lies on the diagonal. The area between
the curve and the diagonal (ATRC gap) is the summary statistic: 0 = perfect,
positive = over-confident, negative = under-confident.

Run:
    pixi run -e gpu python -m src.validation.tarp_test \
        --config config_a100_lrtest.yaml \
        --posterior models/posterior_faraday_thin_n1.pt \
        --n-cases 2000 --n-samples 1000 \
        --output-dir validation_results
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# numpy>=2.0 renamed np.trapz -> np.trapezoid; keep working on both.
_trapz = getattr(np, "trapezoid", None) or np.trapz

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from .online_probe import VAL_SEED, _param_labels, _rebuild_posterior


def _coverage_stats(ecp: np.ndarray, alphas: np.ndarray) -> dict:
    """Alpha-resolved summary of an ECP curve, not just the signed gap.

    The signed ATRC gap ``∫(ECP-α)dα`` collapses the whole curve to one scalar
    and *cancels*: a posterior over-confident at some α and under-confident at
    others can integrate to ~0 and masquerade as calibrated. We report, instead:

      - signed_gap    : net direction (as before; >0 over-, <0 under-confident)
      - unsigned_area : ∫|ECP-α|dα — total miscalibration magnitude, never cancels
      - max_dev       : worst signed deviation and the α where it occurs
      - crosses       : whether the curve crosses the diagonal (shape, not scale)
      - mixed         : significant cancellation — a scalar temperature can't fix it
    """
    dev = ecp - alphas
    signed_gap = float(_trapz(dev, alphas))
    unsigned_area = float(_trapz(np.abs(dev), alphas))
    imax = int(np.argmax(np.abs(dev)))
    nz = dev[np.abs(dev) > 1e-12]
    crosses = bool(np.any(np.diff(np.sign(nz)) != 0)) if nz.size else False
    mixed = bool(crosses and abs(signed_gap) < 0.5 * unsigned_area)
    return {
        "signed_gap": signed_gap,
        "unsigned_area": unsigned_area,
        "max_dev": float(dev[imax]),
        "max_dev_alpha": float(alphas[imax]),
        "crosses": crosses,
        "mixed": mixed,
    }


def verdict_from_stats(stats: dict, area_thresh: float) -> str:
    """Map coverage stats + null threshold to a verdict, in one place.

    Gates on the *unsigned* area (never cancels); a curve that crosses the
    diagonal with little net bias is "mixed" (shape error, not a scale error).
    Shared by ``run_tarp`` and the plotting module so color and label always agree.
    """
    if stats["unsigned_area"] <= area_thresh:
        return "calibrated"
    if stats["mixed"]:
        return "mixed"
    return "over-confident" if stats["signed_gap"] > 0 else "under-confident"


def _null_thresholds(
    n_cases: int, alphas: np.ndarray,
    level: float = 0.95, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Monte-Carlo null for the coverage statistics — a *derived* threshold.

    Under perfect calibration the per-case credibility levels are iid U(0,1), so
    ECP(α) is the empirical CDF of ``n_cases`` uniforms and ECP-α is a standard
    empirical process (Kolmogorov-Smirnov family). Simulating that null gives
    finite-sample critical values for the unsigned area and sup deviation at this
    exact ``n_cases`` and α grid, replacing the hard-coded 0.02 with a threshold
    tied to the estimator's own sampling noise.

    NOTE: with ``get_tarp_coverage(norm=True)`` the random reference point adds
    mild correlation beyond iid-uniform, so this null slightly understates the
    true variance (thresholds run a touch tight). Upgrade path: parametric
    bootstrap through get_tarp_coverage on known-calibrated mock samples.
    """
    alphas = np.asarray(alphas)
    rng = np.random.default_rng(seed)
    area = np.empty(n_boot)
    sup = np.empty(n_boot)
    for b in range(n_boot):
        u = np.sort(rng.random(n_cases))
        ecp0 = np.searchsorted(u, alphas, side="right") / n_cases
        dev = ecp0 - alphas
        area[b] = _trapz(np.abs(dev), alphas)
        sup[b] = float(np.max(np.abs(dev)))
    return {
        "area_thresh": float(np.quantile(area, level)),
        "sup_thresh": float(np.quantile(sup, level)),
        "level": level,
        "n_boot": n_boot,
        "n_cases": n_cases,
    }


def null_band(
    n_cases: int, alphas: np.ndarray,
    level: float = 0.95, n_boot: int = 2000, seed: int = 0,
) -> dict:
    """Per-α envelope of ECP under perfect calibration (for plotting).

    Same iid-uniform null as ``_null_thresholds`` but keeps the per-bin structure:
    returns the central ``level`` band of ECP(α) at each α, so a plot can show
    where a perfectly-calibrated curve would lie given ``n_cases``. A curve that
    stays inside this envelope is indistinguishable from calibrated.
    """
    alphas = np.asarray(alphas)
    rng = np.random.default_rng(seed)
    ecp0 = np.empty((n_boot, alphas.size))
    for b in range(n_boot):
        u = np.sort(rng.random(n_cases))
        ecp0[b] = np.searchsorted(u, alphas, side="right") / n_cases
    lo = (1.0 - level) / 2.0
    return {
        "alphas": alphas,
        "ecp_lo": np.quantile(ecp0, lo, axis=0),
        "ecp_hi": np.quantile(ecp0, 1.0 - lo, axis=0),
        "level": level,
    }


def run_tarp(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cuda",
    batch_size: int = 8192,
    freq_file: str | None = None,
    num_bootstrap: int = 0,
) -> dict:
    """Run TARP joint calibration test and save the ATRC curve plot.

    ``num_bootstrap`` > 0 turns on tarp's resample-over-cases bootstrap: the ECP
    curve is then the *mean* over bootstraps and we additionally save its per-α
    standard deviation (``ecp_std``) and the bootstrap spread of the scalar
    gap/area — i.e. the sampling uncertainty of the coverage estimate itself.
    """
    try:
        from tarp import get_tarp_coverage
    except ImportError as exc:
        raise ImportError(
            "tarp is required: pip install tarp  (or add to pyproject.toml)"
        ) from exc

    device = device if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Configuration.from_yaml(config_path)
    if freq_file:
        config.freq_file = freq_file

    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_components = ckpt["n_components"]
    labels = _param_labels(model_type, n_components)

    sim = GPUSimulator(
        config=config,
        model_type=model_type,
        n_components=n_components,
        device=device,
        sampling_method=getattr(config.training, "sampling_method", "uniform"),
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(VAL_SEED)
    posterior = _rebuild_posterior(ckpt, sim, device)

    # Generate fixed validation set (same seed as coverage probe / SBC)
    thetas, xs = [], []
    remaining = n_cases
    while remaining > 0:
        b = min(batch_size, remaining)
        theta, x = sim.generate_batch(b, generator=gen)
        thetas.append(theta)
        xs.append(x)
        remaining -= b
    theta_true = torch.cat(thetas)[:n_cases]   # (N, n_params)
    x_val = torch.cat(xs)[:n_cases]             # (N, n_data)

    # Draw posterior samples for every test case
    samples_list = []
    with torch.no_grad():
        for i in range(n_cases):
            s = posterior.sample(
                (n_samples,), x=x_val[i : i + 1], show_progress_bars=False
            )
            samples_list.append(s.cpu())
    # (N, n_samples, n_params)
    samples = torch.stack(samples_list, dim=0).numpy()
    theta_np = theta_true.cpu().numpy()  # (N, n_params)

    # tarp.get_tarp_coverage expects:
    #   samples: (n_samples, n_cases, n_params)  -- note the axis order
    #   theta:   (n_cases, n_params)
    # and returns (ecp, alpha) in that order.
    samples_tarp = np.transpose(samples, (1, 0, 2))  # (n_samples, N, n_params)

    # Point estimate with a fixed reference seed so the curve is reproducible.
    ecp, alphas = get_tarp_coverage(samples_tarp, theta_np, norm=True, seed=0)
    ecp = np.asarray(ecp)
    alphas = np.asarray(alphas)

    ecp_std = None
    gap_bs_std = area_bs_std = None
    if num_bootstrap and num_bootstrap > 0:
        # Proper bootstrap: resample cases WITH REPLACEMENT and redraw the random
        # DRP reference each iteration, capturing both the case-sampling and the
        # reference-point variance. Runs on the GPU via a torch reimplementation
        # of the exact DRP algorithm (validated bit-for-bit against the numpy
        # library, max|dECP| ~ 6e-8), so 50k resamples take seconds instead of the
        # hours the numpy library needs single-threaded. The band is evaluated on
        # the point-estimate ``alphas`` grid so every bootstrap curve shares it.
        # (tarp's own bootstrap=True is unusable: it mutates the array cumulatively
        # and, with a fixed seed, freezes the references, collapsing the spread.)
        from .tarp_gpu import bootstrap_bands_torch

        bands = bootstrap_bands_torch(
            samples_tarp, theta_np, alphas, num_bootstrap=num_bootstrap,
            norm=True, device=device, seed=0,
            chunk_report=max(0, num_bootstrap // 10),
        )
        ecp_std = bands["ecp_std"]
        gap_bs_std = bands["gap_bs_std"]
        area_bs_std = bands["area_bs_std"]

    stats = _coverage_stats(ecp, alphas)
    null = _null_thresholds(n_cases, alphas)
    atrc_gap = stats["signed_gap"]  # kept for backward-compat (readers/exit code)

    stem = Path(posterior_path).stem.replace("posterior_", "")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(alphas, ecp, lw=2,
            label=f"TARP  (gap={atrc_gap:+.4f}, area={stats['unsigned_area']:.4f})")
    ax.fill_between(alphas, alphas, ecp, alpha=0.15)
    if ecp_std is not None:
        ax.fill_between(alphas, ecp - 2 * ecp_std, ecp + 2 * ecp_std,
                        color="C0", alpha=0.20, label="±2σ bootstrap")
    ax.set_xlabel("Credible level α")
    ax.set_ylabel("Expected coverage probability")
    ax.set_title(f"TARP ATRC curve: {model_type} N={n_components}")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out_png = output_dir / f"tarp_{stem}.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # Verdict gates on the *unsigned* area against the MC-null threshold — a real
    # "distinguishable from calibrated at this n_cases" test, not the old |gap|<0.02
    # convention. Direction comes from the signed gap; a curve that crosses the
    # diagonal (cancellation) is flagged "mixed" — a scalar rescale can't fix it.
    verdict = verdict_from_stats(stats, null["area_thresh"])
    calibrated = verdict == "calibrated"

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"TARP verdict  : {verdict}")
    print(f"  signed gap    : {atrc_gap:+.4f}   (>0 over-, <0 under-confident)")
    print(f"  unsigned area : {stats['unsigned_area']:.4f}   "
          f"(null {null['level']:.0%} threshold {null['area_thresh']:.4f})")
    print(f"  max deviation : {stats['max_dev']:+.4f} @ α={stats['max_dev_alpha']:.2f}"
          f"   crosses diagonal: {stats['crosses']}")
    print(f"ATRC curve: {out_png}\n")

    # Machine-readable summary for the paper TARP figure (ATRC-gap matrix) and
    # for cross-checking against SBC/coverage without re-sampling.
    summary = {
        "model_type": model_type,
        "n_components": n_components,
        "n_cases": n_cases,
        "n_samples": n_samples,
        "atrc_gap": atrc_gap,
        "signed_gap": stats["signed_gap"],
        "unsigned_area": stats["unsigned_area"],
        "max_dev": stats["max_dev"],
        "max_dev_alpha": stats["max_dev_alpha"],
        "crosses": stats["crosses"],
        "mixed": stats["mixed"],
        "null_area_thresh": null["area_thresh"],
        "null_sup_thresh": null["sup_thresh"],
        "null_level": null["level"],
        "verdict": verdict,
        "calibrated": calibrated,
        "alphas": np.asarray(alphas).tolist(),
        "ecp": np.asarray(ecp).tolist(),
        "ecp_std": None if ecp_std is None else np.asarray(ecp_std).tolist(),
        "num_bootstrap": int(num_bootstrap),
        "gap_bs_std": gap_bs_std,
        "area_bs_std": area_bs_std,
        "out_png": str(out_png),
    }
    json_path = output_dir / f"tarp_{stem}.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON : {json_path}\n")

    return {
        "alphas": alphas,
        "ecp": ecp,
        "atrc_gap": atrc_gap,
        "verdict": verdict,
        "calibrated": calibrated,
        "out_png": str(out_png),
        "json_path": str(json_path),
        "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--output-dir", default="validation_results")
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freq-file", default=None, metavar="PATH",
                    help="override freq_file from config")
    ap.add_argument("--num-bootstrap", type=int, default=0,
                    help="bootstrap iterations for per-α coverage error bands (0 = off)")
    args = ap.parse_args()
    run_tarp(
        config_path=args.config,
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        n_samples=args.n_samples,
        device=args.device,
        freq_file=args.freq_file,
        num_bootstrap=args.num_bootstrap,
    )


if __name__ == "__main__":
    main()
