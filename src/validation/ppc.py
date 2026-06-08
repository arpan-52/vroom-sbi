"""
Posterior predictive check (PPC) for online-trained NPE posteriors.

For each validation case: draw theta from the posterior given the observed x,
forward-simulate Q and U via the GPU physics model, and compare the predictive
envelope to the original observation. A well-specified model will have the
observed Q/U trace pass through the central predictive interval.

Diagnostic plots per posterior:
  - One panel per case (default 10): observed Q, U vs freq overlaid with the
    posterior predictive median and 68%/95% envelopes.
  - A summary panel showing the fraction of observed channels covered by the
    68%/95% predictive intervals across all cases and frequencies.

Run:
    pixi run -e gpu python -m src.validation.ppc \
        --config config_a100_lrtest.yaml \
        --posterior models/posterior_faraday_thin_n1.pt \
        --n-cases 20 --n-samples 500 \
        --output-dir validation_results
"""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from ..simulator.torch_physics import polarization_signal
from .online_probe import VAL_SEED, _param_labels, _rebuild_posterior


def run_ppc(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 20,
    n_samples: int = 500,
    device: str = "cuda",
    batch_size: int = 8192,
    freq_file: str | None = None,
) -> dict:
    """Run posterior predictive check and save diagnostic plots."""
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

    # Fixed validation set (same seed as all other probes)
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

    # x layout: [Q_0..Q_F, U_0..U_F, w_0..w_F]  (3-channel online contract)
    n_freq = sim.n_freq
    freq_mhz = sim.freq.cpu().numpy() / 1e6
    lambda_sq = sim.lambda_sq  # (n_freq,) on device

    # Covered channel mask from weights channel
    weights_obs = x_val[:, 2 * n_freq:].cpu().numpy()   # (N, F)

    # Coverage trackers across all cases and freq channels
    covered_68 = []
    covered_95 = []

    stem = Path(posterior_path).stem.replace("posterior_", "")
    case_dir = output_dir / f"ppc_cases_{stem}"
    case_dir.mkdir(exist_ok=True)

    with torch.no_grad():
        for i in range(n_cases):
            obs_x = x_val[i : i + 1]   # (1, n_data)
            obs_q = x_val[i, :n_freq].cpu().numpy()
            obs_u = x_val[i, n_freq : 2 * n_freq].cpu().numpy()
            wt = weights_obs[i]

            # Posterior samples -> predictive Q/U
            post_samples = posterior.sample(
                (n_samples,), x=obs_x, show_progress_bars=False
            )  # (n_samples, n_params)

            # Forward simulate noise-free Q/U for each sample
            P_pred = polarization_signal(
                post_samples.to(device), lambda_sq, model_type, n_components
            )  # (n_samples, n_freq) complex
            Q_pred = P_pred.real.cpu().numpy()  # (n_samples, n_freq)
            U_pred = P_pred.imag.cpu().numpy()

            q_p2  = np.percentile(Q_pred, 2.5,  axis=0)
            q_p16 = np.percentile(Q_pred, 16,   axis=0)
            q_p50 = np.percentile(Q_pred, 50,   axis=0)
            q_p84 = np.percentile(Q_pred, 84,   axis=0)
            q_p97 = np.percentile(Q_pred, 97.5, axis=0)

            u_p2  = np.percentile(U_pred, 2.5,  axis=0)
            u_p16 = np.percentile(U_pred, 16,   axis=0)
            u_p50 = np.percentile(U_pred, 50,   axis=0)
            u_p84 = np.percentile(U_pred, 84,   axis=0)
            u_p97 = np.percentile(U_pred, 97.5, axis=0)

            # Per-channel coverage (masked channels excluded)
            good = wt > 0
            if good.any():
                covered_68.append(
                    ((obs_q[good] >= q_p16[good]) & (obs_q[good] <= q_p84[good])).mean() * 0.5
                    + ((obs_u[good] >= u_p16[good]) & (obs_u[good] <= u_p84[good])).mean() * 0.5
                )
                covered_95.append(
                    ((obs_q[good] >= q_p2[good]) & (obs_q[good] <= q_p97[good])).mean() * 0.5
                    + ((obs_u[good] >= u_p2[good]) & (obs_u[good] <= u_p97[good])).mean() * 0.5
                )

            # Per-case plot
            fig, (ax_q, ax_u) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
            for ax, obs, p2, p16, p50, p84, p97, lbl in [
                (ax_q, obs_q, q_p2, q_p16, q_p50, q_p84, q_p97, "Q"),
                (ax_u, obs_u, u_p2, u_p16, u_p50, u_p84, u_p97, "U"),
            ]:
                ax.fill_between(freq_mhz, p2, p97, alpha=0.2, color="tab:blue", label="95% interval")
                ax.fill_between(freq_mhz, p16, p84, alpha=0.4, color="tab:blue", label="68% interval")
                ax.plot(freq_mhz, p50, lw=1, color="tab:blue", label="median")
                ax.scatter(freq_mhz[good], obs[good], s=4, color="tab:red", zorder=3, label="observed")
                ax.scatter(freq_mhz[~good], obs[~good], s=4, color="0.7", zorder=2, label="flagged")
                ax.set_ylabel(lbl)
                ax.grid(alpha=0.25)
            ax_q.legend(fontsize=7, ncol=4, loc="upper right")
            ax_u.set_xlabel("Frequency (MHz)")
            fig.suptitle(
                f"PPC case {i + 1}/{n_cases}  |  {model_type} N={n_components}  |  "
                f"true RM={theta_true[i, 0].item():.1f}"
            )
            fig.tight_layout()
            fig.savefig(case_dir / f"case_{i:04d}.png", dpi=120, bbox_inches="tight")
            plt.close(fig)

    # Summary coverage plot
    mean_cov68 = float(np.mean(covered_68)) if covered_68 else float("nan")
    mean_cov95 = float(np.mean(covered_95)) if covered_95 else float("nan")

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.axhline(0.68, color="tab:blue", ls="--", lw=1, label="ideal 68%")
    ax.axhline(0.95, color="tab:orange", ls="--", lw=1, label="ideal 95%")
    ax.plot(covered_68, "o", color="tab:blue", ms=4, label=f"cov68 mean={mean_cov68:.3f}")
    ax.plot(covered_95, "s", color="tab:orange", ms=4, label=f"cov95 mean={mean_cov95:.3f}")
    ax.set_xlabel("Validation case index")
    ax.set_ylabel("Channel coverage fraction")
    ax.set_title(f"PPC coverage summary: {model_type} N={n_components}")
    ax.legend(fontsize=8, ncol=2)
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    summary_png = output_dir / f"ppc_summary_{stem}.png"
    fig.savefig(summary_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"Mean channel coverage  68%: {mean_cov68:.3f}  (ideal 0.68)")
    print(f"Mean channel coverage  95%: {mean_cov95:.3f}  (ideal 0.95)")
    print(f"Case plots : {case_dir}/")
    print(f"Summary    : {summary_png}\n")

    return {
        "mean_cov68": mean_cov68,
        "mean_cov95": mean_cov95,
        "summary_png": str(summary_png),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--output-dir", default="validation_results")
    ap.add_argument("--n-cases", type=int, default=20)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freq-file", default=None, metavar="PATH",
                    help="override freq_file from config")
    args = ap.parse_args()
    run_ppc(
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
