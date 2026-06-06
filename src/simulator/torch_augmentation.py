"""
GPU-resident, batch-vectorized weight augmentation.

Vectorized equivalent of ``augment_weights_continuous`` in
``simulator/augmentation.py``. The NumPy version runs one sample at a time
inside a Python list comprehension; this generates a whole ``(B, F)`` batch in
parallel on-device.

Parity note: the *structure* and *distribution* match the NumPy scheme
(same flagging probabilities, same noise-ratio law, same
``w = min(1, (sigma_med / sigma_k)^2)`` weighting), but because Torch and NumPy
use different RNGs the per-element values are not bit-identical. Parity tests
assert distributional/structural equivalence, not exact values.
"""

import torch

# Hard-coded structural constants, matching augmentation.py
_SCATTERED_MISSING_PROB = 0.1
_MIN_GAP, _MAX_GAP = 2, 8
_MIN_BLOCK, _MAX_BLOCK = 10, 30


def _randu(B, F, lo, hi, device, generator):
    return lo + (hi - lo) * torch.rand(B, F, device=device, generator=generator)


def augment_weights_continuous_batch(
    base_weights: torch.Tensor,
    batch_size: int,
    noise_ratio_min: float = 2.0,
    noise_ratio_max: float = 300.0,
    scattered_prob: float = 0.3,
    gap_prob: float = 0.3,
    large_block_prob: float = 0.1,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Return a (batch_size, F) tensor of augmented per-channel weights.

    Mirrors ``augment_weights_continuous`` applied independently per row.
    """
    device = base_weights.device
    F = base_weights.shape[0]
    B = batch_size
    g = generator
    col = torch.arange(F, device=device)[None, :]  # (1, F)

    aug = base_weights[None, :].expand(B, F).clone()

    # --- Step 1: binary RFI flagging --------------------------------------
    # Scattered missing channels
    apply_scat = torch.rand(B, device=device, generator=g) < scattered_prob
    scat_mask = torch.rand(B, F, device=device, generator=g) < _SCATTERED_MISSING_PROB
    aug[apply_scat[:, None] & scat_mask] = 0.0

    # Contiguous gap
    if F > _MAX_GAP:
        apply_gap = torch.rand(B, device=device, generator=g) < gap_prob
        gap_size = torch.randint(
            _MIN_GAP, _MAX_GAP + 1, (B,), device=device, generator=g
        )
        gap_start = (
            torch.rand(B, device=device, generator=g) * (F - gap_size + 1)
        ).long()
        gmask = (col >= gap_start[:, None]) & (col < (gap_start + gap_size)[:, None])
        aug[apply_gap[:, None] & gmask] = 0.0

    # Large RFI block
    if F > _MAX_BLOCK:
        block_hi = min(_MAX_BLOCK + 1, F // 2)  # exclusive upper bound
        apply_blk = torch.rand(B, device=device, generator=g) < large_block_prob
        blk_size = torch.randint(
            _MIN_BLOCK, block_hi, (B,), device=device, generator=g
        )
        blk_start = (
            torch.rand(B, device=device, generator=g) * (F - blk_size + 1)
        ).long()
        bmask = (col >= blk_start[:, None]) & (col < (blk_start + blk_size)[:, None])
        aug[apply_blk[:, None] & bmask] = 0.0

    # --- Steps 2 & 3: per-channel noise profile -> inverse-variance weights -
    good = aug > 0  # (B, F)

    # Per-sample noise ratio R ~ LogUniform[min, max]
    log_R = (
        torch.log(torch.tensor(noise_ratio_min, device=device))
        + (
            torch.log(torch.tensor(noise_ratio_max, device=device))
            - torch.log(torch.tensor(noise_ratio_min, device=device))
        )
        * torch.rand(B, device=device, generator=g)
    )  # (B,)

    # Per-channel sigma_k ~ LogUniform[1, R]  (sigma_min = 1)
    u = torch.rand(B, F, device=device, generator=g)
    sigma_k = torch.exp(u * log_R[:, None])  # (B, F)

    # Median of sigma_k over *good* channels only (per row)
    sigma_masked = torch.where(good, sigma_k, torch.full_like(sigma_k, float("nan")))
    sigma_med = torch.nanmedian(sigma_masked, dim=1).values  # (B,)

    w = torch.clamp((sigma_med[:, None] / sigma_k) ** 2, max=1.0)
    out = torch.where(good, w, torch.zeros_like(w))
    return out
