"""
Simulation-based calibration (SBC) for online-trained NPE posteriors.

SBC is the rank-statistic calibration test: draw theta from the prior, simulate
x, sample the posterior at x, and record the rank of each true theta component
among its posterior samples. For a perfectly calibrated posterior those ranks
are uniform on [0, num_samples]; systematic departures diagnose miscalibration
(U-shape = over-confident, ∩-shape = under-confident, slope = bias).

This complements ``online_probe`` (which reports coverage at fixed credible
levels): SBC tests the *whole* marginal, not just the 68/90% intervals, and
gives a per-parameter KS p-value against uniformity.

It reuses the exact posterior-rebuild and simulator contract from
``online_probe`` so the two diagnostics see identical models and data.

Run:
    pixi run -e gpu python -m src.validation.sbc \
        --config config_a100.yaml \
        --posterior models/posterior_faraday_thin_n3.pt \
        --n-cases 2000 --n-samples 1000 \
        --output-dir validation_results
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from sbi.analysis import sbc_rank_plot
from sbi.diagnostics import check_sbc, run_sbc

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from .online_probe import VAL_SEED, _param_labels, _rebuild_posterior


def run_sbc_for_posterior(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cuda",
    batch_size: int = 8192,
) -> dict:
    device = device if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Configuration.from_yaml(config_path)
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
    # Same deterministic draws as the coverage probe (prior theta -> simulated x).
    gen = torch.Generator(device=device)
    gen.manual_seed(VAL_SEED)
    posterior = _rebuild_posterior(ckpt, sim, device)

    thetas, xs = [], []
    remaining = n_cases
    while remaining > 0:
        b = min(batch_size, remaining)
        theta, x = sim.generate_batch(b, generator=gen)
        thetas.append(theta)
        xs.append(x)
        remaining -= b
    theta_true = torch.cat(thetas)[:n_cases]
    x_val = torch.cat(xs)[:n_cases]

    ranks, dap_samples = run_sbc(
        theta_true,
        x_val,
        posterior,
        num_posterior_samples=n_samples,
        reduce_fns="marginals",
        show_progress_bar=False,
    )
    stats = check_sbc(
        ranks, theta_true, dap_samples, num_posterior_samples=n_samples
    )

    ks_pvals = stats["ks_pvals"].cpu()
    c2st = stats["c2st_ranks"].cpu()

    stem = Path(posterior_path).stem.replace("posterior_", "")
    fig, _ = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=n_samples,
        plot_type="hist",
        parameter_labels=labels,
    )
    fig.suptitle(f"SBC rank histograms: {model_type} N={n_components}", y=1.02)
    hist_path = output_dir / f"sbc_hist_{stem}.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"Rank histogram: {hist_path}\n")
    print(f"{'param':<10}{'KS p-value':>12}{'C2ST':>10}{'flag':>8}")
    print("-" * 40)
    for j, name in enumerate(labels):
        # p < 0.05 rejects uniformity; C2ST ~0.5 is ideal (no classifier signal).
        flag = "FAIL" if ks_pvals[j].item() < 0.05 else "ok"
        print(
            f"{name:<10}{ks_pvals[j].item():>12.4f}{c2st[j].item():>10.3f}{flag:>8}"
        )
    print("-" * 40)
    print("(KS p>=0.05 => ranks consistent with uniform; C2ST ideal ~0.5)\n")

    return {
        "labels": labels,
        "ks_pvals": ks_pvals,
        "c2st_ranks": c2st,
        "hist_path": str(hist_path),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--output-dir", default="validation_results")
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_sbc_for_posterior(
        config_path=args.config,
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
