"""
FITS I/O for VROOM-SBI cube inference.

Requires the optional 'io' dependencies:
    pip install 'vroom-sbi[io]'
"""

from .cube_io import (
    compute_weights,
    normalize_qu_by_i,
    read_iquv_cube,
    read_qu_cubes,
    write_results_maps,
)

__all__ = [
    "read_iquv_cube",
    "read_qu_cubes",
    "normalize_qu_by_i",
    "compute_weights",
    "write_results_maps",
]
