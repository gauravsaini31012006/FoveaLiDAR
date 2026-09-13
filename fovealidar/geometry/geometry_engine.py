"""Vectorized Geometry Engine for spatial feature extraction."""

from dataclasses import dataclass
from typing import Optional, Tuple, Dict, Any
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud


logger = get_logger(__name__)


@dataclass
class GeometryFeatures:
    """Cell/region-level geometry features computed from point clouds."""
    cell_indices: NDArray[np.int32]        # (M, 2) [row, col]
    cell_centers: NDArray[np.float32]      # (M, 2) [x, y]
    point_counts: NDArray[np.int32]        # (M,)
    density: NDArray[np.float32]           # (M,) points / m^2
    z_min: NDArray[np.float32]             # (M,)
    z_max: NDArray[np.float32]             # (M,)
    z_mean: NDArray[np.float32]            # (M,)
    z_variance: NDArray[np.float32]        # (M,) local height variance
    roughness: NDArray[np.float32]         # (M,) standard deviation of height
    slope_proxy: NDArray[np.float32]       # (M,) height difference / span
    complexity: NDArray[np.float32]        # (M,) geometric complexity index [0, 1]


class GeometryEngine:
    """Computes spatial geometric features using fast vectorized 2D binning.
    
    Adheres to Master Spec §11: No O(N^2) neighbor computation.
    Uses C-speed NumPy spatial reduction and histogram binning.
    """
    
    def __init__(self, bin_size_m: float = 0.50):
        """Initialize GeometryEngine.
        
        Args:
            bin_size_m: Planar bin size in meters for spatial grouping.
        """
        self.bin_size_m = bin_size_m
    
    def compute_features(
        self,
        pc: PointCloud,
        origin: Tuple[float, float] = (-50.0, -50.0),
        bounds: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
    ) -> GeometryFeatures:
        """Extract vectorized geometry features across spatial bins.
        
        Args:
            pc: PointCloud container with (N, 3) points.
            origin: (x_origin, y_origin) anchor coordinate.
            bounds: (x_min, x_max, y_min, y_max) clipping bounds.
            
        Returns:
            GeometryFeatures dataclass with per-cell metrics.
        """
        pts = pc.points
        if len(pts) == 0:
            return GeometryFeatures(
                cell_indices=np.zeros((0, 2), dtype=np.int32),
                cell_centers=np.zeros((0, 2), dtype=np.float32),
                point_counts=np.zeros(0, dtype=np.int32),
                density=np.zeros(0, dtype=np.float32),
                z_min=np.zeros(0, dtype=np.float32),
                z_max=np.zeros(0, dtype=np.float32),
                z_mean=np.zeros(0, dtype=np.float32),
                z_variance=np.zeros(0, dtype=np.float32),
                roughness=np.zeros(0, dtype=np.float32),
                slope_proxy=np.zeros(0, dtype=np.float32),
                complexity=np.zeros(0, dtype=np.float32),
            )
        
        # Spatial filtering
        x = pts[:, 0]
        y = pts[:, 1]
        z = pts[:, 2]
        
        valid = (x >= bounds[0]) & (x < bounds[1]) & (y >= bounds[2]) & (y < bounds[3])
        v_pts = pts[valid]
        v_z = z[valid]
        
        if len(v_pts) == 0:
            return self.compute_features(PointCloud(points=np.zeros((0, 3), dtype=np.float32), intensity=np.zeros(0, dtype=np.float32)))
        
        # Vectorized integer 2D binning
        col_idx = np.floor((v_pts[:, 0] - origin[0]) / self.bin_size_m).astype(np.int32)
        row_idx = np.floor((v_pts[:, 1] - origin[1]) / self.bin_size_m).astype(np.int32)
        
        # Combine row and col into unique cells
        grid_coords = np.column_stack([row_idx, col_idx])
        unique_cells, inverse, counts = np.unique(grid_coords, axis=0, return_inverse=True, return_counts=True)
        m_cells = len(unique_cells)
        
        # Compute cell centers
        centers_x = origin[0] + (unique_cells[:, 1] + 0.5) * self.bin_size_m
        centers_y = origin[1] + (unique_cells[:, 0] + 0.5) * self.bin_size_m
        cell_centers = np.column_stack([centers_x, centers_y]).astype(np.float32)
        
        # Density (points / area in m^2)
        cell_area = self.bin_size_m ** 2
        density = (counts.astype(np.float32) / cell_area)
        
        # Vectorized min, max, sum, sum_sq of Z per cell
        z_min = np.full(m_cells, np.inf, dtype=np.float32)
        z_max = np.full(m_cells, -np.inf, dtype=np.float32)
        z_sum = np.zeros(m_cells, dtype=np.float64)
        z_sq_sum = np.zeros(m_cells, dtype=np.float64)
        
        # Fast aggregations
        np.minimum.at(z_min, inverse, v_z)
        np.maximum.at(z_max, inverse, v_z)
        np.add.at(z_sum, inverse, v_z.astype(np.float64))
        np.add.at(z_sq_sum, inverse, (v_z ** 2).astype(np.float64))
        
        # Mean and Variance
        float_counts = counts.astype(np.float64)
        z_mean = (z_sum / float_counts).astype(np.float32)
        z_var = np.maximum(0.0, (z_sq_sum / float_counts) - (z_mean.astype(np.float64) ** 2)).astype(np.float32)
        roughness = np.sqrt(z_var).astype(np.float32)
        
        # Slope proxy: elevation span (z_max - z_min)
        slope_proxy = (z_max - z_min).astype(np.float32)
        
        # Complexity index [0, 1]: normalized combination of roughness, span, and point density
        norm_roughness = np.clip(roughness / 1.0, 0.0, 1.0)
        norm_span = np.clip(slope_proxy / 2.5, 0.0, 1.0)
        norm_density = np.clip(density / 200.0, 0.0, 1.0)
        complexity = (0.45 * norm_roughness + 0.35 * norm_span + 0.20 * norm_density).astype(np.float32)
        
        logger.debug(f"GeometryEngine: {len(v_pts)} points -> {m_cells} active cells")
        
        return GeometryFeatures(
            cell_indices=unique_cells.astype(np.int32),
            cell_centers=cell_centers,
            point_counts=counts.astype(np.int32),
            density=density,
            z_min=z_min,
            z_max=z_max,
            z_mean=z_mean,
            z_variance=z_var,
            roughness=roughness,
            slope_proxy=slope_proxy,
            complexity=complexity,
        )
