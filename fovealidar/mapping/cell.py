"""Map cell definitions and structured array conversion."""

from typing import Tuple, Optional, List
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.types import MapCell


# Structured dtype for lightweight memory-compact storage of millions of cells
MAP_CELL_DTYPE = np.dtype([
    ('row', np.int32),
    ('col', np.int32),
    ('elevation_min', np.float32),
    ('elevation_max', np.float32),
    ('elevation_mean', np.float32),
    ('occupancy', np.float32),
    ('point_count', np.int32),
    ('intensity_mean', np.float32),
    ('roughness', np.float32),
    ('resolution_m', np.float32),
])


def create_empty_cell_array(size: int) -> NDArray:
    """Create a structured NumPy array for map cells."""
    return np.zeros(size, dtype=MAP_CELL_DTYPE)
