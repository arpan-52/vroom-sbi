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

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from .online_probe import VAL_SEED, _param_labels, _rebuild_posterior


def run_tarp(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cuda",
    batch_size: int = 8192,
    freq_file: str | None = None,
) -> dict:
    """Run TARP joint calibration test and save the ATRC curve plot."""
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

    ecp, alphas = get_tarp_coverage(samples_tarp, theta_np, norm=True)

    atrc_gap = float(np.trapz(ecp - alphas, alphas))

    stem = Path(posterior_path).stem.replace("posterior_", "")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(alphas, ecp, lw=2, label=f"TARP  (gap={atrc_gap:+.4f})")
    ax.fill_between(alphas, alphas, ecp, alpha=0.15)
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

    calibrated = abs(atrc_gap) < 0.02
    verdict = "calibrated" if calibrated else ("over-confident" if atrc_gap > 0 else "under-confident")

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"TARP ATRC gap : {atrc_gap:+.4f}  ({verdict})")
    print(f"  >0 = over-confident (posterior too narrow)")
    print(f"  <0 = under-confident (posterior too broad)")
    print(f"ATRC curve: {out_png}\n")

    # Machine-readable summary for the paper TARP figure (ATRC-gap matrix) and
    # for cross-checking against SBC/coverage without re-sampling.
    summary = {
        "model_type": model_type,
        "n_components": n_components,
        "n_cases": n_cases,
        "n_samples": n_samples,
        "atrc_gap": atrc_gap,
        "verdict": verdict,
        "calibrated": calibrated,
        "alphas": np.asarray(alphas).tolist(),
        "ecp": np.asarray(ecp).tolist(),
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
    args = ap.parse_args()
    run_tarp(
        config_path=args.config,
        posterior_path=args.posterior,
        output_dir=args.output_dir,
        n_cases=args.n_cases,
        n_samples=args.n_samples,
        device=args.device,
        freq_file=args.freq_file,
    )


if __name__ == "__main__":
    main()
