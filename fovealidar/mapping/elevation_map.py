"""2.5D Elevation and Occupancy Map (Mode A: Uniform 5 cm Baseline)."""

import time
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud, MapCell
from fovealidar.mapping.uniform_grid import Uniform2DGrid
from fovealidar.mapping.cell import MAP_CELL_DTYPE, create_empty_cell_array


logger = get_logger(__name__)


class ElevationMap25D:
    """Mode A: High-resolution uniform 2.5D elevation and occupancy map.
    
    Implements Master Spec §12 (2.5D Representation) and §13 (Uniform Baseline).
    Provides zero-overhead NumPy layer arrays:
    - elevation_max (surface canopy)
    - elevation_min (ground base)
    - elevation_span (obstacle vertical height)
    - elevation_mean (center of mass)
    - occupancy (occupancy probability)
    - density (point count per cell)
    - intensity (mean remission)
    - roughness (standard deviation of Z)
    """
    
    def __init__(
        self,
        resolution_m: float = 0.05,
        bounds: Tuple[float, float, float, float] = (-50.0, 50.0, -50.0, 50.0),
        mode_name: str = "uniform_5cm",
    ):
        """Initialize 2.5D elevation map.
        
        Args:
            resolution_m: Nominal cell resolution in meters (0.05m = 5cm).
            bounds: (x_min, x_max, y_min, y_max) in meters.
            mode_name: Identifier for benchmark comparison.
        """
        self.resolution_m = resolution_m
        self.grid = Uniform2DGrid(resolution_m=resolution_m, bounds=bounds)
        self.mode_name = mode_name
        
        # Grid dimensions
        self.rows = self.grid.num_rows
        self.cols = self.grid.num_cols
        
        # Allocate dense 2D layer buffers
        self._allocate_layers()
        self.last_update_time_ms = 0.0
        self.total_frames_mapped = 0
    
    def _allocate_layers(self):
        """Allocate or reset compact 2D NumPy array layers."""
        self.elevation_max = np.full((self.rows, self.cols), -np.inf, dtype=np.float32)
        self.elevation_min = np.full((self.rows, self.cols), np.inf, dtype=np.float32)
        self.elevation_sum = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.elevation_sq_sum = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.point_count = np.zeros((self.rows, self.cols), dtype=np.int32)
        self.intensity_sum = np.zeros((self.rows, self.cols), dtype=np.float64)
        self.occupancy = np.zeros((self.rows, self.cols), dtype=np.float32)
    
    def clear(self):
        """Reset the map to initial unobserved state in-place (zero allocation)."""
        if hasattr(self, 'elevation_max') and self.elevation_max is not None:
            self.elevation_max.fill(-np.inf)
            self.elevation_min.fill(np.inf)
            self.elevation_sum.fill(0)
            self.elevation_sq_sum.fill(0)
            self.point_count.fill(0)
            self.intensity_sum.fill(0)
            self.occupancy.fill(0)
        else:
            self._allocate_layers()
    
    def update(self, pc: PointCloud, clear_first: bool = True) -> Dict[str, Any]:
        """Project point cloud onto the 2.5D elevation map using C-speed vectorized aggregations.
        
        Args:
            pc: PointCloud object.
            clear_first: If True, reset layers before mapping new frame to avoid ghosting/smearing.
                         If False, accumulate points across frames into a persistent map.
            
        Returns:
            Dictionary of mapping telemetry (active cells, latency ms, memory).
        """
        if clear_first:
            self.clear()
            
        t0 = time.perf_counter()
        
        pts = pc.points
        intens = pc.intensity if pc.intensity is not None else np.zeros(len(pts), dtype=np.float32)
        n_input = len(pts)
        
        if n_input == 0:
            return {
                'mapping_time_ms': 0.0,
                'input_points': 0,
                'mapped_points': 0,
                'active_cells': 0,
                'mode': self.mode_name,
            }
        
        # 1. Fast in-bounds filtering
        in_bounds = self.grid.is_in_bounds(pts[:, 0], pts[:, 1])
        valid_pts = pts[in_bounds]
        valid_int = intens[in_bounds]
        n_mapped = len(valid_pts)
        
        if n_mapped == 0:
            return {
                'mapping_time_ms': (time.perf_counter() - t0) * 1000.0,
                'input_points': n_input,
                'mapped_points': 0,
                'active_cells': 0,
                'mode': self.mode_name,
            }
        
        # 2. Vectorized integer grid indexing
        row, col = self.grid.world_to_grid(valid_pts[:, 0], valid_pts[:, 1])
        
        # Clip to ensure safety
        row = np.clip(row, 0, self.rows - 1)
        col = np.clip(col, 0, self.cols - 1)
        
        # Flat 1D index for fast vectorized aggregation: flat = row * cols + col
        flat_indices = (row.astype(np.int64) * self.cols + col.astype(np.int64)).astype(np.int64)
        z = valid_pts[:, 2]
        
        # 3. Vectorized layer updates using np.minimum.at, np.maximum.at, np.add.at
        # Flatten layer views for in-place updates
        elev_min_flat = self.elevation_min.ravel()
        elev_max_flat = self.elevation_max.ravel()
        elev_sum_flat = self.elevation_sum.ravel()
        elev_sq_sum_flat = self.elevation_sq_sum.ravel()
        pt_cnt_flat = self.point_count.ravel()
        int_sum_flat = self.intensity_sum.ravel()
        
        np.minimum.at(elev_min_flat, flat_indices, z)
        np.maximum.at(elev_max_flat, flat_indices, z)
        np.add.at(elev_sum_flat, flat_indices, z.astype(np.float64))
        np.add.at(elev_sq_sum_flat, flat_indices, (z ** 2).astype(np.float64))
        np.add.at(pt_cnt_flat, flat_indices, 1)
        np.add.at(int_sum_flat, flat_indices, valid_int.astype(np.float64))
        
        # 4. Occupancy probability: sigmoid / saturated log-odds from point count & span
        # Active cells mask
        active_mask = self.point_count > 0
        n_active = int(np.sum(active_mask))
        
        # Update occupancy for active cells
        if n_active > 0:
            cnt_active = self.point_count[active_mask].astype(np.float32)
            span_active = self.elevation_max[active_mask] - self.elevation_min[active_mask]
            # Higher point density and larger elevation span increase occupancy confidence
            occ_val = np.clip(1.0 - np.exp(-0.25 * cnt_active) + 0.3 * np.clip(span_active / 1.5, 0.0, 1.0), 0.1, 1.0)
            self.occupancy[active_mask] = occ_val.astype(np.float32)
        
        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_update_time_ms = dt_ms
        self.total_frames_mapped += 1
        
        logger.debug(f"ElevationMap25D update: {n_mapped}/{n_input} pts -> {n_active} active cells in {dt_ms:.2f}ms")
        
        return {
            'mapping_time_ms': round(dt_ms, 3),
            'input_points': n_input,
            'mapped_points': n_mapped,
            'active_cells': n_active,
            'mode': self.mode_name,
            'resolution_m': self.resolution_m,
            'grid_shape': (self.rows, self.cols),
            'memory_mb': (self.rows * self.cols * 4 * 7) / (1024 * 1024),
        }
    
    @property
    def active_mask(self) -> NDArray[np.bool_]:
        """Boolean mask of active (observed) cells."""
        return self.point_count > 0
    
    @property
    def elevation_span(self) -> NDArray[np.float32]:
        """Elevation span (max - min) in meters (obstacle vertical height)."""
        span = np.zeros((self.rows, self.cols), dtype=np.float32)
        active = self.active_mask
        span[active] = self.elevation_max[active] - self.elevation_min[active]
        return span
    
    @property
    def elevation_mean(self) -> NDArray[np.float32]:
        """Mean elevation in meters for active cells."""
        mean_z = np.full((self.rows, self.cols), np.nan, dtype=np.float32)
        active = self.active_mask
        if np.any(active):
            mean_z[active] = (self.elevation_sum[active] / self.point_count[active].astype(np.float64)).astype(np.float32)
        return mean_z
    
    @property
    def roughness(self) -> NDArray[np.float32]:
        """Local surface roughness (height standard deviation) in meters."""
        rough = np.zeros((self.rows, self.cols), dtype=np.float32)
        active = self.active_mask
        if np.any(active):
            cnt = self.point_count[active].astype(np.float64)
            mean = self.elevation_sum[active] / cnt
            var = np.maximum(0.0, (self.elevation_sq_sum[active] / cnt) - (mean ** 2))
            rough[active] = np.sqrt(var).astype(np.float32)
        return rough
    
    @property
    def intensity_mean(self) -> NDArray[np.float32]:
        """Mean intensity / remission per active cell."""
        i_mean = np.zeros((self.rows, self.cols), dtype=np.float32)
        active = self.active_mask
        if np.any(active):
            i_mean[active] = (self.intensity_sum[active] / self.point_count[active].astype(np.float64)).astype(np.float32)
        return i_mean
    
    def get_active_cells(self) -> NDArray:
        """Extract structured array of active cells (per Master Spec §7 MapCell contract)."""
        rows, cols = np.where(self.active_mask)
        n = len(rows)
        if n == 0:
            return create_empty_cell_array(0)
        
        cells = create_empty_cell_array(n)
        cells['row'] = rows
        cells['col'] = cols
        cells['elevation_min'] = self.elevation_min[rows, cols]
        cells['elevation_max'] = self.elevation_max[rows, cols]
        cells['elevation_mean'] = (self.elevation_sum[rows, cols] / self.point_count[rows, cols].astype(np.float64)).astype(np.float32)
        cells['occupancy'] = self.occupancy[rows, cols]
        cells['point_count'] = self.point_count[rows, cols]
        cells['intensity_mean'] = (self.intensity_sum[rows, cols] / self.point_count[rows, cols].astype(np.float64)).astype(np.float32)
        cells['roughness'] = self.roughness[rows, cols]
        cells['resolution_m'] = self.resolution_m
        
        return cells
    
    def get_cell_info(self, row: int, col: int) -> Optional[Dict[str, Any]]:
        """Inspect a single cell's telemetry (for interactive click/hover inspection)."""
        if not (0 <= row < self.rows and 0 <= col < self.cols):
            return None
        
        pts_in_cell = int(self.point_count[row, col])
        if pts_in_cell == 0:
            x, y = self.grid.grid_to_world(np.array([row]), np.array([col]))
            return {
                'row': row,
                'col': col,
                'center_x': float(x[0]),
                'center_y': float(y[0]),
                'occupied': False,
                'point_count': 0,
            }
        
        x, y = self.grid.grid_to_world(np.array([row]), np.array([col]))
        z_min = float(self.elevation_min[row, col])
        z_max = float(self.elevation_max[row, col])
        z_mean = float(self.elevation_sum[row, col] / pts_in_cell)
        span = z_max - z_min
        occ = float(self.occupancy[row, col])
        intensity = float(self.intensity_sum[row, col] / pts_in_cell)
        
        return {
            'row': row,
            'col': col,
            'center_x': round(float(x[0]), 3),
            'center_y': round(float(y[0]), 3),
            'occupied': True,
            'point_count': pts_in_cell,
            'elevation_min': round(z_min, 3),
            'elevation_max': round(z_max, 3),
            'elevation_mean': round(z_mean, 3),
            'elevation_span': round(span, 3),
            'occupancy': round(occ, 3),
            'intensity_mean': round(intensity, 3),
            'resolution_m': self.resolution_m,
        }
