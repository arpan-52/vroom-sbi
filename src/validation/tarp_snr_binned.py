"""SNR-stratified TARP: is under-confidence uniform across SNR, or concentrated?

Marginal/joint TARP (``tarp_test.py``) pools all cases into one ECP curve. If a
posterior is trained across a wide noise-augmentation range (see
``augment_weights_continuous_batch`` / ``noise_ratio_min``/``max``) but learns
one shared hedge level, the pooled curve can show under-confidence that is
really concentrated in the high-SNR cases (where the data supports a much
sharper posterior than the network delivers) while low-SNR cases are fine.
Splitting by SNR distinguishes an architecture/training-recipe artifact from a
genuine, SNR-independent capacity ceiling.

Run:
    pixi run -e gpu python -m src.validation.tarp_snr_binned \
        --config config_a100.yaml --posterior models/posterior_faraday_thin_n1.pt \
        --n-cases 2000 --n-samples 1000 --n-bins 4
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
from .online_probe import VAL_SEED, _rebuild_posterior
from .ppc_flag import _generate_cases
from .tarp_test import _coverage_stats, _null_thresholds


def run_tarp_snr_binned(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results/tarp_snr",
    n_cases: int = 2000,
    n_samples: int = 1000,
    n_bins: int = 4,
    device: str = "cuda",
) -> dict:
    try:
        from tarp import get_tarp_coverage
    except ImportError as exc:
        raise ImportError("tarp is required: pip install tarp") from exc

    device = device if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = Configuration.from_yaml(config_path)
    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_components = ckpt["n_components"]

    sim = GPUSimulator(
        config=config, model_type=model_type, n_components=n_components,
        device=device, sampling_method=getattr(config.training, "sampling_method", "uniform"),
    )
    posterior = _rebuild_posterior(ckpt, sim, device)

    # _generate_cases mirrors generate_batch's observable assembly but also
    # returns sigma_per_chan, which generate_batch folds into x and discards.
    # Same (model_type, n_components) as the posterior's own family -- this is
    # the same in-distribution null tarp_test.py uses, just with sigma exposed.
    x, q_obs, u_obs, sigma_per_chan = _generate_cases(
        sim, model_type, n_components, n_cases, seed=VAL_SEED
    )

    # Reconstruct theta the same way generate_batch does: _generate_cases
    # doesn't return it, so redraw with the identical seed/generator sequence.
    from ..simulator.torch_prior import sample_prior
    gen = torch.Generator(device=device)
    gen.manual_seed(VAL_SEED)
    theta_true = sample_prior(
        n_cases, sim.low, sim.high, n_components, model_type,
        method=sim.sampling_method, generator=gen,
    )

    # SNR proxy: median 1/sigma over good (unflagged) channels per case.
    good = sigma_per_chan > 0
    inv_sigma = torch.where(good, 1.0 / sigma_per_chan.clamp_min(1e-30), torch.zeros_like(sigma_per_chan))
    snr_proxy = torch.stack([
        inv_sigma[i][good[i]].median() if good[i].any() else torch.tensor(0.0, device=device)
        for i in range(n_cases)
    ])

    # Posterior draws for every case (shared across all bins).
    samples_list = []
    with torch.no_grad():
        for i in range(n_cases):
            s = posterior.sample((n_samples,), x=x[i : i + 1], show_progress_bars=False)
            samples_list.append(s.cpu())
    samples = torch.stack(samples_list, dim=0).numpy()  # (N, n_samples, n_params)
    theta_np = theta_true.cpu().numpy()
    snr_np = snr_proxy.cpu().numpy()

    edges = np.quantile(snr_np, np.linspace(0, 1, n_bins + 1))
    edges[-1] += 1e-6  # include the max in the last bin
    bin_idx = np.digitize(snr_np, edges[1:-1])

    per_bin = []
    for b in range(n_bins):
        mask = bin_idx == b
        n_b = int(mask.sum())
        if n_b < 20:
            per_bin.append({"bin": b, "n_cases": n_b, "snr_lo": float(edges[b]),
                            "snr_hi": float(edges[b + 1]), "unsigned_area": None})
            continue
        s_bin = np.transpose(samples[mask], (1, 0, 2))  # (n_samples, n_b, n_params)
        ecp, alphas = get_tarp_coverage(s_bin, theta_np[mask], norm=True, seed=0)
        stats = _coverage_stats(np.asarray(ecp), np.asarray(alphas))
        null = _null_thresholds(n_b, np.asarray(alphas))
        per_bin.append({
            "bin": b, "n_cases": n_b,
            "snr_lo": float(edges[b]), "snr_hi": float(edges[b + 1]),
            "signed_gap": stats["signed_gap"], "unsigned_area": stats["unsigned_area"],
            "null_area_thresh": null["area_thresh"],
        })
        print(f"  bin {b}  SNR [{edges[b]:.2f}, {edges[b+1]:.2f})  n={n_b}  "
              f"signed_gap={stats['signed_gap']:+.4f}  area={stats['unsigned_area']:.4f}  "
              f"(null thresh {null['area_thresh']:.4f})")

    result = {
        "model_type": model_type, "n_components": n_components,
        "n_cases": n_cases, "n_samples": n_samples, "n_bins": n_bins,
        "bins": per_bin,
    }

    stem = Path(posterior_path).stem.replace("posterior_", "")
    json_path = output_dir / f"tarp_snr_{stem}.json"
    with json_path.open("w") as fh:
        json.dump(result, fh, indent=2)
    print(f"\nSNR-binned TARP JSON : {json_path}")

    # Plot: signed gap and unsigned area vs. SNR bin midpoint.
    valid = [b for b in per_bin if b["unsigned_area"] is not None]
    mids = [(b["snr_lo"] + b["snr_hi"]) / 2 for b in valid]
    gaps = [b["signed_gap"] for b in valid]
    areas = [b["unsigned_area"] for b in valid]
    threshs = [b["null_area_thresh"] for b in valid]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.plot(mids, areas, "o-", color="C0", label="unsigned area")
    ax.plot(mids, threshs, "k--", lw=1, label="null threshold")
    ax2 = ax.twinx()
    ax2.plot(mids, gaps, "s--", color="C3", alpha=0.6, label="signed gap")
    ax2.axhline(0, color="C3", lw=0.5, alpha=0.4)
    ax.set_xlabel("SNR proxy (median 1/sigma, good channels)")
    ax.set_ylabel("unsigned TARP area", color="C0")
    ax2.set_ylabel("signed gap (>0 over-, <0 under-confident)", color="C3")
    ax.set_title(f"SNR-stratified TARP — {model_type} N={n_components}")
    fig.legend(loc="upper center", ncol=3, fontsize=8, bbox_to_anchor=(0.5, 1.02))
    fig.tight_layout()
    png_path = output_dir / f"tarp_snr_{stem}.png"
    fig.savefig(png_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"SNR-binned TARP plot : {png_path}")

    return result


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--n-bins", type=int, default=4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output-dir", default="validation_results/tarp_snr")
    args = ap.parse_args()
    run_tarp_snr_binned(
        config_path=args.config, posterior_path=args.posterior,
        n_cases=args.n_cases, n_samples=args.n_samples, n_bins=args.n_bins,
        device=args.device, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
