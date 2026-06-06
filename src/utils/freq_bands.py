"""
Receiver-band frequency-grid definitions and freq.txt generation.

Generates evenly-spaced channel-centre frequency files (one frequency in Hz
per line) for the VLA and MeerKAT bands targeted by VROOM-SBI. These files are
the ``freq_file`` input consumed at training and inference time.

Band edges are nominal receiver ranges; channel counts are the full-band
defaults below. Both can be overridden via ``write_band_file``. Flagged
explicitly because real observations rarely span the full nominal band and
often drop edge channels to RFI — regenerate with the actual edges/count of
your data when training a model meant for a specific observation.
"""

from pathlib import Path

import numpy as np

# (low_hz, high_hz, default_n_channels). Edges are nominal receiver ranges.
BANDS: dict[str, tuple[float, float, int]] = {
    # VLA
    "vla_p": (230e6, 470e6, 1024),
    "vla_l": (1.0e9, 2.0e9, 1024),
    "vla_c": (4.0e9, 8.0e9, 2048),
    "vla_x": (8.0e9, 12.0e9, 2048),
    # MeerKAT
    "meerkat_uhf": (544e6, 1088e6, 4096),
    "meerkat_l": (856e6, 1712e6, 4096),
}


def band_frequencies(name: str, n_channels: int | None = None) -> np.ndarray:
    """Return channel-centre frequencies (Hz) for a named band.

    Channel centres are evenly spaced: the band is split into ``n_channels``
    equal-width channels and the centre of each is returned.
    """
    if name not in BANDS:
        raise ValueError(f"Unknown band '{name}'. Known: {sorted(BANDS)}")
    low, high, default_n = BANDS[name]
    n = n_channels or default_n
    edges = np.linspace(low, high, n + 1)
    return 0.5 * (edges[:-1] + edges[1:])


def write_band_file(
    name: str, out_path: Path, n_channels: int | None = None
) -> Path:
    """Write a freq.txt for a named band (one frequency in Hz per line)."""
    freqs = band_frequencies(name, n_channels)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, freqs, fmt="%.6f")
    return out_path


def write_all_bands(out_dir: Path) -> list[Path]:
    """Write freq files for every known band into ``out_dir``."""
    out_dir = Path(out_dir)
    return [write_band_file(name, out_dir / f"freq_{name}.txt") for name in BANDS]
