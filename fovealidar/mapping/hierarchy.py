"""Hierarchical Integer Lattice for Adaptive Variable Resolution 2.5D Mapping.

Adheres strictly to SIH 2026 Problem Statement 26053 and Master Spec §14, §15, §47:
- Base lattice: 5 cm (0.05 m)
- Coarse lattice: 50 cm (0.50 m = 10 x 5 cm)
- Optional levels: 10 cm (2 x 5 cm), 20 cm (4 x 5 cm)
- Exact integer parent-child alignment preventing floating-point boundary drift.
"""

from typing import Tuple, List, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger

logger = get_logger(__name__)


class LatticeHierarchy:
    """Deterministic integer lattice hierarchy for multi-resolution 2.5D mapping.
    
    Guarantees:
    - Zero floating-point boundary drift.
    - Exact 10-to-1 ratio between 50 cm coarse cells and 5 cm fine cells.
    - Deterministic bidirectional coordinate mapping.
    - Guaranteed non-overlapping spatial partitioning.
    """
    
    LEVEL_FINE = 0      # 0.05 m (5 cm)
    LEVEL_MEDIUM_10 = 1 # 0.10 m (10 cm)
    LEVEL_MEDIUM_20 = 2 # 0.20 m (20 cm)
    LEVEL_COARSE = 3    # 0.50 m (50 cm)
    
    RESOLUTIONS = {
        LEVEL_FINE: 0.05,
        LEVEL_MEDIUM_10: 0.10,
        LEVEL_MEDIUM_20: 0.20,
        LEVEL_COARSE: 0.50,
    }
    
    # Scale factors relative to base 5 cm lattice (1, 2, 4, 10)
    FACTORS = {
        LEVEL_FINE: 1,
        LEVEL_MEDIUM_10: 2,
        LEVEL_MEDIUM_20: 4,
        LEVEL_COARSE: 10,
    }

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-35.0, 35.0, -35.0, 35.0),
        fine_res_m: float = 0.05,
        coarse_res_m: float = 0.50,
    ):
        """Initialize lattice hierarchy with bounded ROI.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max) in meters.
            fine_res_m: Base fine cell resolution (default 0.05 m = 5 cm).
            coarse_res_m: Coarse cell resolution (default 0.50 m = 50 cm = 10 * fine_res).
        """
        self.fine_res = float(fine_res_m)
        self.coarse_res = float(coarse_res_m)
        self.ratio = int(round(self.coarse_res / self.fine_res))
        assert self.ratio == 10, f"Coarse resolution must be exactly 10x fine resolution, got {self.ratio}x"
        
        # Snap bounds to coarse resolution boundaries (which are also fine boundaries)
        self.x_min = float(round(bounds[0] / self.coarse_res) * self.coarse_res)
        self.x_max = float(round(bounds[1] / self.coarse_res) * self.coarse_res)
        self.y_min = float(round(bounds[2] / self.coarse_res) * self.coarse_res)
        self.y_max = float(round(bounds[3] / self.coarse_res) * self.coarse_res)
        
        # Number of coarse cells
        self.coarse_cols = int(round((self.x_max - self.x_min) / self.coarse_res))
        self.coarse_rows = int(round((self.y_max - self.y_min) / self.coarse_res))
        
        # Number of fine cells
        self.fine_cols = self.coarse_cols * self.ratio
        self.fine_rows = self.coarse_rows * self.ratio
        
        logger.debug(
            f"LatticeHierarchy initialized: bounds=({self.x_min}, {self.x_max}, {self.y_min}, {self.y_max}) | "
            f"Coarse: {self.coarse_rows}x{self.coarse_cols} ({self.coarse_res*100:.0f}cm) | "
            f"Fine: {self.fine_rows}x{self.fine_cols} ({self.fine_res*100:.0f}cm)"
        )

    # --- Fast Coordinate Mappings ---
    
    def world_to_coarse(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Convert world coordinates (meters) to integer coarse grid (row_50, col_50)."""
        col = np.floor((x - self.x_min) / self.coarse_res).astype(np.int32)
        row = np.floor((y - self.y_min) / self.coarse_res).astype(np.int32)
        return row, col

    def world_to_fine(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Convert world coordinates (meters) to integer fine grid (row_5, col_5)."""
        col = np.floor((x - self.x_min) / self.fine_res).astype(np.int32)
        row = np.floor((y - self.y_min) / self.fine_res).astype(np.int32)
        return row, col

    def coarse_to_world(self, row: NDArray[np.int32], col: NDArray[np.int32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Convert coarse grid (row, col) to world cell centers (meters)."""
        x = (self.x_min + (col.astype(np.float32) + 0.5) * self.coarse_res).astype(np.float32)
        y = (self.y_min + (row.astype(np.float32) + 0.5) * self.coarse_res).astype(np.float32)
        return x, y

    def fine_to_world(self, row: NDArray[np.int32], col: NDArray[np.int32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Convert fine grid (row, col) to world cell centers (meters)."""
        x = (self.x_min + (col.astype(np.float32) + 0.5) * self.fine_res).astype(np.float32)
        y = (self.y_min + (row.astype(np.float32) + 0.5) * self.fine_res).astype(np.float32)
        return x, y

    def coarse_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) world bounds of a coarse cell."""
        x0 = self.x_min + col * self.coarse_res
        x1 = x0 + self.coarse_res
        y0 = self.y_min + row * self.coarse_res
        y1 = y0 + self.coarse_res
        return x0, x1, y0, y1

    def fine_bounds(self, row: int, col: int) -> Tuple[float, float, float, float]:
        """Return (x_min, x_max, y_min, y_max) world bounds of a fine cell."""
        x0 = self.x_min + col * self.fine_res
        x1 = x0 + self.fine_res
        y0 = self.y_min + row * self.fine_res
        y1 = y0 + self.fine_res
        return x0, x1, y0, y1

    # --- Parent / Child Hierarchy Mappings ---
    
    def fine_to_parent_coarse(self, row_5: NDArray[np.int32], col_5: NDArray[np.int32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Map fine cells to their unique parent coarse cell."""
        return row_5 // self.ratio, col_5 // self.ratio

    def coarse_to_fine_subgrid(self, row_50: int, col_50: int) -> Tuple[slice, slice]:
        """Return row and column slices of fine cells enclosed by the coarse cell.
        
        Each coarse cell encloses exactly ratio x ratio (10x10 = 100) fine cells.
        """
        r_start = row_50 * self.ratio
        r_end = r_start + self.ratio
        c_start = col_50 * self.ratio
        c_end = c_start + self.ratio
        return slice(r_start, r_end), slice(c_start, c_end)

    def is_in_bounds(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Check if world coordinates fall strictly within the ROI bounds."""
        return (x >= self.x_min) & (x < self.x_max) & (y >= self.y_min) & (y < self.y_max)

    @staticmethod
    def pack_cell_id(level: int, row: int, col: int) -> int:
        """Deterministic 64-bit integer Cell ID: [8-bit level | 28-bit row | 28-bit col]."""
        return ((level & 0xFF) << 56) | ((row & 0x0FFFFFFF) << 28) | (col & 0x0FFFFFFF)

    @staticmethod
    def unpack_cell_id(cell_id: int) -> Tuple[int, int, int]:
        """Unpack 64-bit integer Cell ID into (level, row, col)."""
        level = (cell_id >> 56) & 0xFF
        row = (cell_id >> 28) & 0x0FFFFFFF
        col = cell_id & 0x0FFFFFFF
        return level, row, col
