"""
FITS I/O for VROOM-SBI cube inference.

Requires the optional 'io' dependencies:
    pip install 'vroom-sbi[io]'
"""

from .cube_io import (
    compute_weights,
    load_i_chunk,
    load_spatial_chunk,
    normalize_qu_by_i,
    open_i_cube_lazy,
    open_qu_cubes_lazy,
    read_iquv_cube,
    read_qu_cubes,
    write_results_maps,
)

__all__ = [
    "read_iquv_cube",
    "read_qu_cubes",
    "open_qu_cubes_lazy",
    "open_i_cube_lazy",
    "load_spatial_chunk",
    "load_i_chunk",
    "normalize_qu_by_i",
    "compute_weights",
    "write_results_maps",
]
