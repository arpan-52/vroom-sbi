"""Posterior predictive p-values: per-source misspecification / OOD flagging.

TARP and SBC are *closed-world* diagnostics — they certify calibration when the
data really come from the training simulator. They cannot say whether a given
observation is adequately represented by the model family at all. This module
supplies that open-world axis (the visual envelope diagnostic in ``ppc.py``
shows the same information qualitatively; this one turns it into a number a
survey pipeline can gate on).

For each posterior draw theta_k we simulate a replicate spectrum y_rep_k under
the observation's own channel noise and flagging, and compare a discrepancy
T(y, theta_k) between replicate and observation:

    p = mean_k[ T(y_rep_k, theta_k) >= T(y_obs, theta_k) ]

Two complementary discrepancies:

  - chi2  : weighted chi^2 of the QU residual — residual *amplitude*.
  - acorr : lag-1 autocorrelation of the standardized QU residual across good
            channels — residual *structure*. A missed Faraday component leaves
            a coherent oscillation in Q/U vs lambda^2 that chi^2 can average
            away; this statistic catches it.

A well-specified model gives roughly uniform p-values (in practice slightly
conservative — PPC double-uses the data — so operational thresholds are read
off the empirical null this module prints, not taken at nominal face value).

Run (in-distribution null + cross-family OOD detection rates):
    pixi run -e gpu python -m src.validation.ppc_flag \
        --config config_a100.yaml --posterior models/posterior_faraday_thin_n1.pt \
        --n-cases 1000 --n-samples 500 --ood-model burn_slab
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ..config import Configuration
from ..simulator.gpu_simulator import GPUSimulator
from ..simulator.torch_augmentation import augment_weights_continuous_batch
from ..simulator.torch_physics import compute_polarization
from ..simulator.torch_prior import sample_prior
from .online_probe import _rebuild_posterior

# Held-out PPC cases must not reuse the trainer's fixed validation set (VAL_SEED).
PPC_SEED = 67890


def ppc_pvalues(
    q_obs: torch.Tensor,
    u_obs: torch.Tensor,
    sigma_per_chan: torch.Tensor,
    theta_samples: torch.Tensor,
    lambda_sq: torch.Tensor,
    n_components: int,
    model_type: str,
    generator: torch.Generator | None = None,
) -> dict:
    """Predictive p-values for one observed spectrum.

    q_obs, u_obs, sigma_per_chan: (F,); flagged channels have sigma == 0.
    theta_samples: (K, D) posterior draws for this observation. The residual
    physics uses the *posterior's* (model_type, n_components) — the hypothesis
    under test — regardless of what generated the data.
    """
    good = sigma_per_chan > 0
    sig = sigma_per_chan[good]  # (G,)
    P = compute_polarization(theta_samples, lambda_sq, n_components, model_type)
    q_mod = P.real[:, good]  # (K, G)
    u_mod = P.imag[:, good]

    def _stats(rq: torch.Tensor, ru: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """chi2 and lag-1 autocorrelation of standardized residuals, (K,) each."""
        chi2 = (rq**2 + ru**2).sum(dim=1)
        # Lag-1 over the good-channel sequence, Q and U jointly, normalized so
        # white residuals give ~0 regardless of chi2 amplitude.
        num = (rq[:, :-1] * rq[:, 1:] + ru[:, :-1] * ru[:, 1:]).sum(dim=1)
        return chi2, num / chi2.clamp_min(1e-30)

    rq_obs = (q_obs[good][None, :] - q_mod) / sig
    ru_obs = (u_obs[good][None, :] - u_mod) / sig
    chi2_obs, ac_obs = _stats(rq_obs, ru_obs)

    # Replicates: y_rep = model(theta_k) + noise, so the standardized replicate
    # residual is exactly white N(0,1) — no need to re-run the physics.
    K, G = q_mod.shape
    dev = theta_samples.device
    chi2_rep, ac_rep = _stats(
        torch.randn(K, G, device=dev, generator=generator),
        torch.randn(K, G, device=dev, generator=generator),
    )

    return {
        # One-sided: misspecification inflates chi2 and produces positively
        # correlated residuals; p near 0 on either statistic flags the source.
        "p_chi2": float((chi2_rep >= chi2_obs).float().mean()),
        "p_acorr": float((ac_rep >= ac_obs).float().mean()),
        "chi2_obs_mean": float(chi2_obs.mean()),
        "acorr_obs_mean": float(ac_obs.mean()),
        "n_good": int(good.sum()),
    }


def flag_from_pvalues(p_chi2: float, p_acorr: float, threshold: float) -> bool:
    """Decision rule: is this source flagged as misspecified/OOD?

    TODO(user): pick the combination rule. Candidates, with trade-offs:
      (a) either-fails:  p_chi2 < threshold or p_acorr < threshold
          — most sensitive; false-positive rate ~2x threshold.
      (b) Bonferroni:    min(p_chi2, p_acorr) < threshold / 2
          — controls the combined false-positive rate at ~threshold.
      (c) Fisher:        -2*(ln p_chi2 + ln p_acorr) vs chi2(df=4) tail
          — pools weak evidence from both statistics; assumes independence,
            which is only approximate here (both use the same residuals).
    Whichever rule: the number for the paper is its *empirical* false-flag
    rate on the in-distribution null printed by run_ppc_flag, not the nominal
    threshold.
    """
    raise NotImplementedError("decision rule pending — see options in docstring")


def _generate_cases(
    sim: GPUSimulator,
    gen_model: str,
    gen_n: int,
    n_cases: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Simulate held-out cases, keeping the per-channel sigma PPC needs.

    Mirrors the observable assembly of ``GPUSimulator.generate_batch`` (same
    augmentation, noise model, and x data contract) but generates from an
    arbitrary (gen_model, gen_n) — possibly not the posterior's own family —
    and returns sigma_per_chan, which generate_batch folds into x and discards.
    Returns (x, q_obs, u_obs, sigma_per_chan).
    """
    device = sim.device
    config = sim.config
    wa = config.weight_augmentation
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    gen_sim = GPUSimulator(
        config=config, model_type=gen_model, n_components=gen_n,
        device=device, sampling_method=sim.sampling_method,
    )
    theta = sample_prior(
        n_cases, gen_sim.low, gen_sim.high, gen_n, gen_model,
        method=sim.sampling_method, generator=gen,
    )

    weights = augment_weights_continuous_batch(
        sim.base_weights, n_cases,
        noise_ratio_min=getattr(wa, "noise_ratio_min", 2.0),
        noise_ratio_max=getattr(wa, "noise_ratio_max", 300.0),
        scattered_prob=wa.scattered_prob,
        gap_prob=wa.gap_prob,
        large_block_prob=wa.large_block_prob,
        generator=gen,
    )
    good = weights > 0
    sigma_base = sim._sigma_base(n_cases, gen)
    sigma_per_chan = torch.where(
        good, sigma_base[:, None] / torch.sqrt(weights + 1e-12),
        torch.zeros_like(weights),
    )

    P = compute_polarization(theta, sim.lambda_sq, gen_n, gen_model)
    noise_q = torch.randn(n_cases, sim.n_freq, device=device, generator=gen) * sigma_per_chan
    noise_u = torch.randn(n_cases, sim.n_freq, device=device, generator=gen) * sigma_per_chan
    zeros = torch.zeros_like(weights)
    q_obs = torch.where(good, P.real + noise_q, zeros)
    u_obs = torch.where(good, P.imag + noise_u, zeros)

    if getattr(config.noise, "condition_on_noise", False):
        log_prec = -torch.log10(sigma_per_chan + 1e-30)
        chan3 = torch.where(good, (log_prec + 1.0) / 4.0, zeros)
    else:
        chan3 = weights
    x = torch.cat([q_obs, u_obs, chan3], dim=1).to(torch.float32)
    return x, q_obs, u_obs, sigma_per_chan


def run_ppc_flag(
    config_path: str,
    posterior_path: str,
    n_cases: int = 1000,
    n_samples: int = 500,
    device: str = "cuda",
    freq_file: str | None = None,
    ood_model: str | None = None,
    ood_n_components: int | None = None,
    output_dir: str | Path | None = None,
) -> dict:
    """In-distribution PPC null (+ optional OOD detection) for one posterior.

    Generates held-out cases from the posterior's own (model, N) — the null,
    where flag rates are the empirical false-positive rates — and, if
    ``ood_model`` is given, cases from that generator pushed through the *same*
    posterior, where the flag rates are detection rates.
    """
    device = device if torch.cuda.is_available() else "cpu"
    config = Configuration.from_yaml(config_path)
    if freq_file:
        config.freq_file = freq_file

    ckpt = torch.load(posterior_path, map_location="cpu", weights_only=False)
    model_type = ckpt["model_type"]
    n_components = ckpt["n_components"]

    sim = GPUSimulator(
        config=config, model_type=model_type, n_components=n_components,
        device=device,
        sampling_method=getattr(config.training, "sampling_method", "uniform"),
    )
    posterior = _rebuild_posterior(ckpt, sim, device)

    def _pvalues_for(gen_model: str, gen_n: int, seed: int) -> list[dict]:
        x, q_obs, u_obs, sigma = _generate_cases(sim, gen_model, gen_n, n_cases, seed)
        g = torch.Generator(device=device)
        g.manual_seed(seed + 1)
        out = []
        with torch.no_grad():
            for i in range(n_cases):
                s = posterior.sample(
                    (n_samples,), x=x[i : i + 1], show_progress_bars=False
                )
                out.append(ppc_pvalues(
                    q_obs[i], u_obs[i], sigma[i], s, sim.lambda_sq,
                    n_components, model_type, generator=g,
                ))
        return out

    null_res = _pvalues_for(model_type, n_components, PPC_SEED)
    ood_res = None
    if ood_model is not None:
        ood_res = _pvalues_for(
            ood_model, ood_n_components or n_components, PPC_SEED + 1000
        )

    def _summarize(res: list[dict], label: str) -> dict:
        p_chi2 = np.array([r["p_chi2"] for r in res])
        p_ac = np.array([r["p_acorr"] for r in res])
        summary = {"label": label, "n_cases": len(res)}
        print(f"\n[{label}]  n_cases={len(res)}")
        print(f"{'threshold':>10}{'flag% chi2':>12}{'flag% acorr':>13}{'flag% either':>14}")
        for thr in (0.05, 0.01, 0.001):
            f1 = float((p_chi2 < thr).mean())
            f2 = float((p_ac < thr).mean())
            fe = float(((p_chi2 < thr) | (p_ac < thr)).mean())
            summary[f"flag_chi2@{thr}"] = f1
            summary[f"flag_acorr@{thr}"] = f2
            summary[f"flag_either@{thr}"] = fe
            print(f"{thr:>10}{f1:>12.3f}{f2:>13.3f}{fe:>14.3f}")
        summary["p_chi2"] = p_chi2.tolist()
        summary["p_acorr"] = p_ac.tolist()
        return summary

    print(f"\nPosterior : {posterior_path}")
    print(f"Model     : {model_type}  N={n_components}")
    print(f"Cases     : {n_cases}   posterior draws/case: {n_samples}")
    result = {
        "model_type": model_type,
        "n_components": n_components,
        "null": _summarize(null_res, f"null: {model_type} n{n_components}"),
    }
    if ood_res is not None:
        result["ood"] = _summarize(
            ood_res, f"OOD: {ood_model} n{ood_n_components or n_components}"
        )
    print("\n(null rows = empirical false-flag rate; OOD rows = detection rate)\n")

    if output_dir is not None:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(posterior_path).stem.replace("posterior_", "")
        json_path = output_dir / f"ppc_flag_{stem}.json"
        with json_path.open("w") as fh:
            json.dump(result, fh, indent=2)
        print(f"PPC-flag JSON : {json_path}\n")

    return result


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--config", required=True)
    ap.add_argument("--posterior", required=True)
    ap.add_argument("--n-cases", type=int, default=1000)
    ap.add_argument("--n-samples", type=int, default=500)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--freq-file", default=None, metavar="PATH",
                    help="override freq_file from config")
    ap.add_argument("--ood-model", default=None,
                    help="also generate cases from this family and report "
                         "detection rates through the same posterior")
    ap.add_argument("--ood-n-components", type=int, default=None,
                    help="component count for --ood-model (default: same as posterior)")
    ap.add_argument("--output-dir", default=None, metavar="PATH",
                    help="if set, write ppc_flag_<stem>.json here")
    args = ap.parse_args()
    run_ppc_flag(
        config_path=args.config, posterior_path=args.posterior,
        n_cases=args.n_cases, n_samples=args.n_samples, device=args.device,
        freq_file=args.freq_file, ood_model=args.ood_model,
        ood_n_components=args.ood_n_components, output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
