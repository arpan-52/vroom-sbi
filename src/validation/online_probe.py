"""
Per-dimension recovery + coverage probe for online-trained NPE posteriors.

A normalizing flow yields a *joint* log-probability, so the training
``val_loss`` cannot be attributed to individual parameters. This probe gives
the per-dimension view instead: it regenerates the *exact* fixed validation
set the online trainer used (same deterministic seed, same GPUSimulator data
contract ``x = [Q, U, weights]``), samples the saved posterior for each case,
and reports, per parameter:

  - bias    : mean(posterior_mean - truth), in prior-width units
  - rmse    : RMS(posterior_mean - truth), in prior-width units
  - cov68   : fraction of truths inside the central 68% credible interval
  - cov90   : fraction of truths inside the central 90% credible interval

A well-calibrated dimension has cov68 ~= 0.68 and cov90 ~= 0.90. Uniformly
mediocre numbers point to global optimization issues (e.g. learning rate);
one bad dimension points to a parametrization problem (angle wrapping, a
prior edge, label switching).

Run:
    pixi run -e gpu python -m src.validation.online_probe \
        --config config_a100.yaml --posterior models/posterior_faraday_thin_n3.pt \
        --n-cases 2000 --n-samples 1000
"""

import argparse

import torch
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_maf, build_nsf
from sbi.utils import BoxUniform

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from ..training.networks import SpectralEmbedding

# Per-component parameter names (mirrors validator.ValidationRunner.PARAM_DEFS).
PARAM_NAMES = {
    "faraday_thin": ["RM", "amp", "chi0"],
    "burn_slab": ["phi_c", "delta_phi", "amp", "chi0"],
    "external_dispersion": ["phi", "sigma_phi", "amp", "chi0"],
    "internal_dispersion": ["phi", "sigma_phi", "amp", "chi0"],
}

# Matches OnlineNPETrainer._generate_val_set.
VAL_SEED = 12345


def _param_labels(model_type: str, n_components: int) -> list[str]:
    base = PARAM_NAMES[model_type]
    return [f"{name}_{i + 1}" for i in range(n_components) for name in base]


def _rebuild_posterior(ckpt: dict, sim: GPUSimulator, device: str) -> DirectPosterior:
    """Reconstruct DirectPosterior from saved state-dicts.

    The flow's z-scoring stats are stored as buffers in the state-dict, so the
    sample batch used by build_nsf only needs correct shapes — its statistics
    are overwritten by load_state_dict.
    """
    arch = ckpt["architecture"]
    embedding_net = SpectralEmbedding(
        input_dim=arch["input_dim"], output_dim=arch["embedding_dim"]
    ).to(device)
    embedding_net.load_state_dict(ckpt["embedding_net_state"])

    shape_gen = torch.Generator(device=device)
    shape_gen.manual_seed(0)
    theta0, x0 = sim.generate_batch(256, generator=shape_gen)

    flow = arch.get("sbi_model", "nsf").lower()
    kwargs = {
        "hidden_features": arch["hidden_features"],
        "num_transforms": arch["num_transforms"],
        "embedding_net": embedding_net,
    }
    if flow == "nsf":
        kwargs["num_bins"] = arch["num_bins"]
        estimator = build_nsf(theta0, x0, **kwargs)
    else:
        estimator = build_maf(theta0, x0, **kwargs)
    estimator.load_state_dict(ckpt["density_estimator_state"])
    estimator = estimator.to(device)

    prior = BoxUniform(low=sim.low, high=sim.high, device=device)
    return DirectPosterior(posterior_estimator=estimator, prior=prior)


def run_probe(
    config_path: str,
    posterior_path: str,
    n_cases: int = 2000,
    n_samples: int = 1000,
    device: str = "cuda",
    batch_size: int = 8192,
) -> dict:
    device = device if torch.cuda.is_available() else "cpu"

    config = Configuration.from_yaml(config_path)
    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_components = ckpt["n_components"]
    labels = _param_labels(model_type, n_components)

    # Regenerate the *exact* fixed validation set the trainer used: same seed,
    # same GPUSimulator, same data contract.  We only need the first n_cases.
    sim = GPUSimulator(
        config=config,
        model_type=model_type,
        n_components=n_components,
        device=device,
        sampling_method=getattr(config.training, "sampling_method", "uniform"),
    )
    gen = torch.Generator(device=device)
    gen.manual_seed(VAL_SEED)

    # Rebuild the density estimator from saved state-dicts (online checkpoints
    # store states, not a pickled posterior object), then wrap in DirectPosterior.
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

    prior_width = (sim.high - sim.low).clamp_min(1e-12)  # (n_params,)

    means = torch.empty_like(theta_true)
    q16 = torch.empty_like(theta_true)
    q84 = torch.empty_like(theta_true)
    q05 = torch.empty_like(theta_true)
    q95 = torch.empty_like(theta_true)

    with torch.no_grad():
        for i in range(n_cases):
            s = posterior.sample(
                (n_samples,), x=x_val[i : i + 1], show_progress_bars=False
            )
            means[i] = s.mean(dim=0)
            qs = torch.quantile(
                s, torch.tensor([0.05, 0.16, 0.84, 0.95], device=s.device), dim=0
            )
            q05[i], q16[i], q84[i], q95[i] = qs[0], qs[1], qs[2], qs[3]

    err = (means - theta_true) / prior_width
    bias = err.mean(dim=0)
    rmse = err.pow(2).mean(dim=0).sqrt()
    cov68 = ((theta_true >= q16) & (theta_true <= q84)).float().mean(dim=0)
    cov90 = ((theta_true >= q05) & (theta_true <= q95)).float().mean(dim=0)

    hist = ckpt.get("training_history", {})
    val_hist = hist.get("val_loss", [])
    best_val = min(val_hist) if val_hist else float("nan")

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    print(f"Best val_loss (history): {best_val:.4f}\n")
    print(f"{'param':<10}{'bias':>10}{'rmse':>10}{'cov68':>9}{'cov90':>9}")
    print("-" * 48)
    for j, name in enumerate(labels):
        print(
            f"{name:<10}{bias[j].item():>10.4f}{rmse[j].item():>10.4f}"
            f"{cov68[j].item():>9.3f}{cov90[j].item():>9.3f}"
        )
    print("-" * 48)
    print("(bias/rmse in prior-width units; ideal cov68~0.68, cov90~0.90)\n")

    return {
        "labels": labels,
        "bias": bias.cpu(),
        "rmse": rmse.cpu(),
        "cov68": cov68.cpu(),
        "cov90": cov90.cpu(),
        "best_val_loss": best_val,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--n-cases", type=int, default=2000)
    ap.add_argument("--n-samples", type=int, default=1000)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()
    run_probe(
        config_path=args.config,
        posterior_path=args.posterior,
        n_cases=args.n_cases,
        n_samples=args.n_samples,
        device=args.device,
    )


if __name__ == "__main__":
    main()
