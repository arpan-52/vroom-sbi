"""GPU/torch TARP-DRP coverage — a fast, validated reimplementation.

``tarp.get_tarp_coverage`` is numpy-only and single-threaded, so a large
bootstrap (tens of thousands of resamples) runs for hours on one CPU core while
the GPU sits idle. This module reproduces the *exact* DRP algorithm of
``tarp._get_tarp_coverage_single`` (Lemos et al. 2023) in torch so the whole
bootstrap runs on the GPU in seconds.

Correctness is not assumed: ``validate_against_tarp`` checks this implementation
against the numpy library on identical inputs (same explicit references, so no
RNG mismatch) and asserts agreement to floating-point tolerance. Only the
bootstrap *uncertainty band* uses this path; the point-estimate curve and the
verdict still come from the numpy library, so published numbers are unchanged.

Algorithm (matches the library exactly):
    normalize samples/theta by theta's per-dim [min,max] over sims (if norm);
    references ~ U(0,1)^d;  d_s = ||ref - sample||,  d_t = ||ref - theta||;
    f_i = mean_s( d_s < d_t );  ECP = empirical CDF of f.
"""

import numpy as np
import torch


def _coverage_f_torch(
    samples: torch.Tensor, theta: torch.Tensor, references: torch.Tensor,
    norm: bool = True,
) -> torch.Tensor:
    """DRP credibility f_i per simulation. Shapes: samples (S, N, D),
    theta (N, D), references (N, D). Returns f of shape (N,)."""
    if norm:
        low = theta.min(dim=0, keepdim=True).values          # (1, D)
        high = theta.max(dim=0, keepdim=True).values
        scale = (high - low + 1e-10)
        samples = (samples - low) / scale
        theta = (theta - low) / scale
    d_samples = torch.sqrt(((references.unsqueeze(0) - samples) ** 2).sum(-1))  # (S, N)
    d_theta = torch.sqrt(((references - theta) ** 2).sum(-1))                    # (N,)
    return (d_samples < d_theta.unsqueeze(0)).to(samples.dtype).mean(dim=0)      # (N,)


def coverage_curve_torch(
    samples: torch.Tensor, theta: torch.Tensor, references: torch.Tensor,
    num_alpha_bins: int, norm: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproduce tarp's histogram-based ECP curve (for validation)."""
    f = _coverage_f_torch(samples, theta, references, norm=norm)
    n = f.numel()
    fmin, fmax = float(f.min()), float(f.max())
    counts = torch.histc(f, bins=num_alpha_bins, min=fmin, max=fmax)
    dx = (fmax - fmin) / num_alpha_bins
    h = counts / (n * dx)
    ecp = torch.cat([torch.zeros(1, device=f.device), torch.cumsum(h, 0) * dx])
    alpha = torch.linspace(fmin, fmax, num_alpha_bins + 1, device=f.device)
    return ecp, alpha


def bootstrap_bands_torch(
    samples_tarp: np.ndarray, theta_np: np.ndarray, query_alphas: np.ndarray,
    num_bootstrap: int, norm: bool = True, device: str = "cuda",
    seed: int = 0, chunk_report: int = 0,
) -> dict:
    """GPU bootstrap of the coverage band, evaluated on a fixed ``query_alphas``.

    Resamples cases with replacement and draws fresh U(0,1) references each
    iteration (capturing both case-sampling and reference-point variance), then
    evaluates the ECDF of f at ``query_alphas`` so every bootstrap curve shares
    one grid. Returns per-bin mean/std of ECP and the bootstrap spread of the
    scalar signed gap and unsigned area.
    """
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    samples = torch.as_tensor(samples_tarp, dtype=torch.float32, device=dev)  # (S,N,D)
    theta = torch.as_tensor(theta_np, dtype=torch.float32, device=dev)        # (N,D)
    q = torch.as_tensor(np.asarray(query_alphas), dtype=torch.float32, device=dev)
    n_sims, n_dims = theta.shape
    g = torch.Generator(device=dev).manual_seed(seed)

    boot = torch.empty((num_bootstrap, q.numel()), device=dev)
    for b in range(num_bootstrap):
        idx = torch.randint(0, n_sims, (n_sims,), generator=g, device=dev)
        refs = torch.rand((n_sims, n_dims), generator=g, device=dev)
        f = _coverage_f_torch(samples[:, idx, :], theta[idx], refs, norm=norm)  # (N,)
        # ECDF of f at the fixed query grid
        boot[b] = (f.unsqueeze(0) <= q.unsqueeze(1)).to(f.dtype).mean(dim=1)
        if chunk_report and (b + 1) % chunk_report == 0:
            print(f"  bootstrap {b + 1}/{num_bootstrap}", flush=True)

    dev_q = q
    dgap = boot - dev_q.unsqueeze(0)                                  # (B, nbins)
    gap_bs = torch.trapz(dgap, dev_q, dim=1)                          # (B,)
    area_bs = torch.trapz(dgap.abs(), dev_q, dim=1)                   # (B,)
    return {
        "ecp_mean": boot.mean(0).cpu().numpy(),
        "ecp_std": boot.std(0).cpu().numpy(),
        "gap_bs_std": float(gap_bs.std().cpu()),
        "area_bs_std": float(area_bs.std().cpu()),
        "num_bootstrap": num_bootstrap,
    }


def validate_against_tarp(seed: int = 0, n_samp: int = 400, n_sims: int = 600,
                          n_dims: int = 3, num_alpha_bins: int = 60,
                          tol: float = 1e-5, device: str = "cpu") -> float:
    """Assert coverage_curve_torch matches tarp on identical explicit references.

    Uses the *same* reference array for both (no RNG mismatch), so the two must
    agree to floating-point tolerance. Returns the max abs ECP difference.
    """
    from tarp import get_tarp_coverage

    rng = np.random.default_rng(seed)
    theta = rng.normal(size=(n_sims, n_dims)).astype(np.float32)
    samples = (theta[None] + rng.normal(size=(n_samp, n_sims, n_dims))).astype(np.float32)
    refs = rng.uniform(0, 1, size=(n_sims, n_dims)).astype(np.float32)

    ecp_np, alpha_np = get_tarp_coverage(
        samples.copy(), theta.copy(), references=refs.copy(), norm=True,
        num_alpha_bins=num_alpha_bins,
    )
    dev = torch.device(device)
    ecp_t, alpha_t = coverage_curve_torch(
        torch.as_tensor(samples, device=dev), torch.as_tensor(theta, device=dev),
        torch.as_tensor(refs, device=dev), num_alpha_bins=num_alpha_bins, norm=True,
    )
    d_ecp = float(np.max(np.abs(ecp_t.cpu().numpy() - np.asarray(ecp_np))))
    d_alpha = float(np.max(np.abs(alpha_t.cpu().numpy() - np.asarray(alpha_np))))
    assert d_ecp < tol and d_alpha < tol, (
        f"GPU TARP disagrees with tarp: max|dECP|={d_ecp:.2e}, "
        f"max|dalpha|={d_alpha:.2e} (tol {tol:.0e})"
    )
    return d_ecp


if __name__ == "__main__":
    d = validate_against_tarp()
    print(f"validate_against_tarp OK: max|dECP| = {d:.2e}")
