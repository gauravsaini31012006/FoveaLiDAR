"""Uniform 2D Planar Grid on a deterministic integer lattice."""

from dataclasses import dataclass
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger


logger = get_logger(__name__)


class Uniform2DGrid:
    """Planar uniform 2D grid grounded on a deterministic 5 cm fundamental lattice.
    
    Adheres to Master Spec §13 (Uniform Baseline Mode A) and §47 (Mapping Invariants).
    Ensures zero floating-point boundary drift.
    """
    
    def __init__(
        self,
        resolution_m: float = 0.05,
        bounds: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
    ):
        """Initialize Uniform2DGrid.
        
        Args:
            resolution_m: Fundamental cell resolution in meters (nominal 0.05m = 5cm).
            bounds: (x_min, x_max, y_min, y_max) in meters.
        """
        self.resolution_m = resolution_m
        
        # Snap bounds to exact integer multiples of resolution to guarantee zero drift
        self.x_min = round(bounds[0] / resolution_m) * resolution_m
        self.x_max = round(bounds[1] / resolution_m) * resolution_m
        self.y_min = round(bounds[2] / resolution_m) * resolution_m
        self.y_max = round(bounds[3] / resolution_m) * resolution_m
        
        # Grid dimensions in integer cells
        self.num_cols = int(round((self.x_max - self.x_min) / resolution_m))
        self.num_rows = int(round((self.y_max - self.y_min) / resolution_m))
        
        logger.debug(
            f"Uniform2DGrid initialized: {self.num_rows}x{self.num_cols} cells "
            f"({self.num_rows * self.num_cols:,} total) @ {resolution_m*100:.1f} cm"
        )
    
    def world_to_grid(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> Tuple[NDArray[np.int32], NDArray[np.int32]]:
        """Convert continuous world coordinates (meters) to integer grid (row, col)."""
        col = np.floor((x - self.x_min) / self.resolution_m).astype(np.int32)
        row = np.floor((y - self.y_min) / self.resolution_m).astype(np.int32)
        return row, col
    
    def grid_to_world(self, row: NDArray[np.int32], col: NDArray[np.int32]) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
        """Convert integer grid (row, col) to continuous world cell centers (meters)."""
        x = (self.x_min + (col.astype(np.float32) + 0.5) * self.resolution_m).astype(np.float32)
        y = (self.y_min + (row.astype(np.float32) + 0.5) * self.resolution_m).astype(np.float32)
        return x, y
    
    def is_in_bounds(self, x: NDArray[np.float32], y: NDArray[np.float32]) -> NDArray[np.bool_]:
        """Check if world coordinates fall strictly within the grid bounds."""
        return (x >= self.x_min) & (x < self.x_max) & (y >= self.y_min) & (y < self.y_max)
    
    def is_valid_cell(self, row: NDArray[np.int32], col: NDArray[np.int32]) -> NDArray[np.bool_]:
        """Check if grid indices fall within array dimensions."""
        return (row >= 0) & (row < self.num_rows) & (col >= 0) & (col < self.num_cols)
