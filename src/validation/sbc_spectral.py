"""
Calibration validation for the spectral-shape posterior.

The spectral-shape model infers the total-intensity SED shape
``log F(nu) = alpha*x + beta*x^2 + gamma*x^3`` (x = log(nu/nu0), F(nu0)=1), so
its physics, parameter set (``[alpha, beta, gamma]``) and data vector (flux
F(nu) over frequency) differ from the Faraday-depth posteriors. It therefore
does not fit the ``(model_type, n_components)`` + ``GPUSimulator`` contract that
``src.validation.sbc`` / ``src.validation.tarp_test`` assume, and is validated
by this dedicated driver instead.

Both tests draw theta iid from the *prior* (a uniform box — NOT the Sobol
quasi-random draws used for training, which are not iid and would bias the
rank-uniformity guarantee), simulate x through the exact training pipeline, and
sample the stored posterior:

  * SBC  — marginal rank statistics (per-parameter KS p-value / C2ST).
  * TARP — joint DRP coverage (signed gap, unsigned area, verdict).

The simulator is built on the *checkpoint's own* frequency grid (``freq_hz``),
not on any config ``freq_file``, so validation always matches the grid the
posterior was trained on regardless of which config is passed.

Emitted JSON matches the schema of the Faraday drivers (with
``model_type="spectral_shape"``, ``n_components=1``) so the spectral result
folds into the same calibration matrix and paper figures.

Run:
    pixi run python -m src.validation.sbc_spectral \
        --config config_spectra.yaml \
        --posterior models/spectral_shape_posterior.pt \
        --tests sbc tarp --n-cases 2000 --n-samples 1000
"""

import json
import tempfile
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from ..config import Configuration
from ..simulator.augmentation import augment_weights_combined
from ..simulator.prior import build_spectral_shape_prior
from ..simulator.spectral_simulator import SpectralShapeSimulator
from .online_probe import VAL_SEED
from .tarp_test import _coverage_stats, _null_thresholds, verdict_from_stats

SPECTRAL_LABELS = ["alpha", "beta", "gamma"]


def _build_spectral_simulator(ckpt: dict) -> SpectralShapeSimulator:
    """Build a simulator on the checkpoint's exact training frequency grid.

    The training ``freq_file`` may not be present (or may hold a different
    channel count) at validation time, so we reconstruct the grid from the
    ``freq_hz`` stored in the checkpoint via a throwaway temp file (the only
    constructor ``SpectralShapeSimulator`` exposes).
    """
    freq_hz = ckpt["freq_hz"]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as fh:
        fh.write("\n".join(repr(float(f)) for f in freq_hz))
        tmp = fh.name
    try:
        sim = SpectralShapeSimulator(tmp)
    finally:
        Path(tmp).unlink(missing_ok=True)
    if sim.n_freq != ckpt["n_freq"]:
        raise ValueError(
            f"Reconstructed grid has {sim.n_freq} channels, checkpoint expects "
            f"{ckpt['n_freq']}"
        )
    return sim


def _spectral_validation_set(
    config: Configuration,
    ckpt: dict,
    sim: SpectralShapeSimulator,
    n_cases: int,
    seed: int = VAL_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw a deterministic (theta, x) validation set matching training's joint.

    theta ~ prior (iid uniform box). x is simulated exactly as in
    ``SpectralShapeTrainer._simulate_chunk``: per-case augmented channel weights,
    additive Gaussian noise with a per-case sigma drawn from the training range,
    and flagged channels zeroed. Sigma is drawn *per case* (not per chunk as in
    training) so each case is an iid draw from the joint p(theta, x) — the
    requirement SBC/TARP rest on.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    spec = config.spectral_shape
    noise = config.noise
    prior = build_spectral_shape_prior(spec, device="cpu")

    theta = prior.sample((n_cases,)).cpu().numpy().astype(np.float64)  # (N, 3)

    # Per-case additive-noise sigma over the same range the trainer samples.
    if noise.augmentation_enable:
        lo = spec.sigma_min * noise.augmentation_min_factor
        hi = spec.sigma_max * noise.augmentation_max_factor
        sigmas = np.random.uniform(lo, hi, size=n_cases)
    else:
        sigmas = np.full(n_cases, (spec.sigma_min + spec.sigma_max) / 2.0)

    xs = np.empty((n_cases, sim.n_freq), dtype=np.float32)
    for i in range(n_cases):
        w = augment_weights_combined(
            sim.weights, noise_variation=config.weight_augmentation.noise_variation
        )[None, :]
        xi = sim.simulate_batch(theta[i : i + 1], w, noise_sigma=float(sigmas[i]))
        xs[i] = np.where(w > 0, xi, 0.0).astype(np.float32)

    theta_t = torch.tensor(theta, dtype=torch.float32)
    x_t = torch.tensor(xs, dtype=torch.float32)
    return theta_t, x_t


def _gpu_validation_set(
    config: Configuration,
    n_cases: int,
    device: str,
    seed: int = VAL_SEED,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Draw (theta, x) from the SAME torch ``GPUSpectralSimulator`` used for
    online training.

    This removes any distribution gap between validation and training: it is the
    exact forward path the online model saw, so if SBC still fails here the
    miscalibration is a property of the posterior, not of a validation-vs-training
    simulator mismatch. Uses ``sampling_method="uniform"`` for iid prior draws
    (SBC requirement), not Sobol.
    """
    from ..simulator.gpu_simulator import GPUSpectralSimulator

    torch.manual_seed(seed)
    sim = GPUSpectralSimulator(config, device=device, sampling_method="uniform")
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    theta, x = sim.generate_batch(n_cases, generator=gen)
    return theta.cpu(), x.cpu()


def run_sbc_spectral(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cpu",
    seed: int = VAL_SEED,
    gen_backend: str = "numpy",
) -> dict:
    """Marginal SBC rank calibration for the spectral-shape posterior."""
    from sbi.analysis import sbc_rank_plot
    from sbi.diagnostics import check_sbc, run_sbc

    device = device if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Configuration.from_yaml(config_path)
    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    posterior = ckpt["posterior"]
    posterior.to(device)  # in place (returns None); moves the density estimator
    labels = list(ckpt.get("param_names") or SPECTRAL_LABELS)

    if gen_backend == "gpu":
        theta_true, x_val = _gpu_validation_set(config, n_cases, device, seed)
    else:
        sim = _build_spectral_simulator(ckpt)
        theta_true, x_val = _spectral_validation_set(config, ckpt, sim, n_cases, seed)
    theta_true = theta_true.to(device)
    x_val = x_val.to(device)

    ranks, dap_samples = run_sbc(
        theta_true,
        x_val,
        posterior,
        num_posterior_samples=n_samples,
        reduce_fns="marginals",
        show_progress_bar=False,
    )
    stats = check_sbc(ranks, theta_true, dap_samples, num_posterior_samples=n_samples)
    ks_pvals = stats["ks_pvals"].cpu()
    c2st = stats["c2st_ranks"].cpu()

    fig, _ = sbc_rank_plot(
        ranks=ranks,
        num_posterior_samples=n_samples,
        plot_type="hist",
        parameter_labels=labels,
    )
    fig.suptitle("SBC rank histograms: spectral_shape", y=1.02)
    hist_path = output_dir / "sbc_hist_spectral_shape.png"
    fig.savefig(hist_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPosterior : {posterior_path}")
    print("Model     : spectral_shape")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"Rank histogram: {hist_path}\n")
    print(f"{'param':<10}{'KS p-value':>12}{'C2ST':>10}{'flag':>8}")
    print("-" * 40)
    for j, name in enumerate(labels):
        flag = "FAIL" if ks_pvals[j].item() < 0.05 else "ok"
        print(f"{name:<10}{ks_pvals[j].item():>12.4f}{c2st[j].item():>10.3f}{flag:>8}")
    print("-" * 40)
    print("(KS p>=0.05 => ranks consistent with uniform; C2ST ideal ~0.5)\n")

    params = [
        {
            "name": name,
            "ks_pval": ks_pvals[j].item(),
            "c2st": c2st[j].item(),
            "pass": ks_pvals[j].item() >= 0.05,
        }
        for j, name in enumerate(labels)
    ]
    summary = {
        "model_type": "spectral_shape",
        "n_components": 1,
        "n_cases": n_cases,
        "n_samples": n_samples,
        "freq_file": None,
        "params": params,
        "n_pass": sum(p["pass"] for p in params),
        "n_fail": sum(not p["pass"] for p in params),
        "hist_path": str(hist_path),
    }
    json_path = output_dir / "sbc_spectral_shape.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON  : {json_path}\n")

    return {"labels": labels, "summary": summary, "json_path": str(json_path)}


def run_tarp_spectral(
    config_path: str,
    posterior_path: str,
    output_dir: str | Path = "validation_results",
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cpu",
    seed: int = VAL_SEED,
    gen_backend: str = "numpy",
) -> dict:
    """Joint TARP/DRP coverage calibration for the spectral-shape posterior."""
    from tarp import get_tarp_coverage

    device = device if torch.cuda.is_available() else "cpu"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    config = Configuration.from_yaml(config_path)
    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    posterior = ckpt["posterior"]
    posterior.to(device)  # in place (returns None); moves the density estimator

    if gen_backend == "gpu":
        theta_true, x_val = _gpu_validation_set(config, n_cases, device, seed)
    else:
        sim = _build_spectral_simulator(ckpt)
        theta_true, x_val = _spectral_validation_set(config, ckpt, sim, n_cases, seed)
    x_val = x_val.to(device)

    samples_list = []
    with torch.no_grad():
        for i in range(n_cases):
            s = posterior.sample(
                (n_samples,), x=x_val[i : i + 1], show_progress_bars=False
            )
            samples_list.append(s.cpu())
    samples = torch.stack(samples_list, dim=0).numpy()  # (N, n_samples, 3)
    theta_np = theta_true.cpu().numpy()

    # tarp expects samples as (n_samples, n_cases, n_params).
    samples_tarp = np.transpose(samples, (1, 0, 2))
    ecp, alphas = get_tarp_coverage(samples_tarp, theta_np, norm=True, seed=0)
    ecp = np.asarray(ecp)
    alphas = np.asarray(alphas)

    stats = _coverage_stats(ecp, alphas)
    null = _null_thresholds(n_cases, alphas)
    verdict = verdict_from_stats(stats, null["area_thresh"])
    calibrated = verdict == "calibrated"
    atrc_gap = stats["signed_gap"]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="ideal")
    ax.plot(
        alphas,
        ecp,
        lw=2,
        label=f"TARP  (gap={atrc_gap:+.4f}, area={stats['unsigned_area']:.4f})",
    )
    ax.fill_between(alphas, alphas, ecp, alpha=0.15)
    ax.set_xlabel("Credible level α")
    ax.set_ylabel("Expected coverage probability")
    ax.set_title("TARP ATRC curve: spectral_shape")
    ax.legend(fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    out_png = output_dir / "tarp_spectral_shape.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    print(f"\nPosterior : {posterior_path}")
    print("Model     : spectral_shape")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"TARP verdict  : {verdict}")
    print(f"  signed gap    : {atrc_gap:+.4f}   (>0 over-, <0 under-confident)")
    print(
        f"  unsigned area : {stats['unsigned_area']:.4f}   "
        f"(null {null['level']:.0%} threshold {null['area_thresh']:.4f})"
    )
    print(f"ATRC curve: {out_png}\n")

    summary = {
        "model_type": "spectral_shape",
        "n_components": 1,
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
        "alphas": alphas.tolist(),
        "ecp": ecp.tolist(),
        "png": str(out_png),
    }
    json_path = output_dir / "tarp_spectral_shape.json"
    with json_path.open("w") as fh:
        json.dump(summary, fh, indent=2)
    print(f"Summary JSON  : {json_path}\n")

    return {"summary": summary, "json_path": str(json_path)}


def main():
    import argparse

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config_spectra.yaml")
    ap.add_argument("--posterior", default="models/spectral_shape_posterior.pt")
    ap.add_argument("--output-dir", default="validation_results")
    ap.add_argument("--tests", nargs="+", default=["sbc", "tarp"],
                    choices=["sbc", "tarp"])
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--gen", default="numpy", choices=["numpy", "gpu"],
                    help="validation-set generator: numpy (reproduction) or gpu "
                         "(the exact GPUSpectralSimulator used for online training)")
    args = ap.parse_args()

    if "sbc" in args.tests:
        run_sbc_spectral(
            args.config, args.posterior, args.output_dir,
            args.n_cases, args.n_samples, args.device, gen_backend=args.gen,
        )
    if "tarp" in args.tests:
        run_tarp_spectral(
            args.config, args.posterior, args.output_dir,
            args.n_cases, args.n_samples, args.device, gen_backend=args.gen,
        )


if __name__ == "__main__":
    main()
