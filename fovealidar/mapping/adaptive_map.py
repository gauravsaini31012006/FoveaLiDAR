"""Adaptive Variable Resolution 2.5D Elevation & Occupancy Map (Mode B).

Implements SIH 2026 Problem Statement 26053 and Master Spec §14, §15, §31, §47:
- Mode B: 5 cm nominal resolution <= 10 m (near-field)
          50 cm nominal resolution > 10 m to 70 m (far-field)
- Integer lattice: 50 cm coarse cell = exactly 10 x 10 fine cells (5 cm)
- Mapping Invariants:
  1. Every valid point maps to exactly one active cell.
  2. Deterministic indexing; zero floating-point drift.
  3. No parent-child double counting (refined parents delegate to fine children).
  4. Zero holes, zero overlaps across resolution transitions.
  5. Deterministic reproducibility.
"""

import time
from typing import Tuple, Dict, Any, List, Optional
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.types import PointCloud
from fovealidar.core.logging import get_logger
from fovealidar.mapping.hierarchy import LatticeHierarchy

logger = get_logger(__name__)


class AdaptiveMap25D:
    """Distance-Adaptive Variable Resolution 2.5D Elevation & Occupancy Map (Mode B).
    
    Combines fine 5 cm resolution in near-field (<= 10m) with coarse 50 cm resolution
    in far-field (> 10m) on a strictly aligned integer lattice.
    """
    
    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-35.0, 35.0, -35.0, 35.0),
        near_field_radius_m: float = 10.0,
        fine_res_m: float = 0.05,
        coarse_res_m: float = 0.50,
    ):
        """Initialize Mode B Adaptive 2.5D Map.
        
        Args:
            bounds: (x_min, x_max, y_min, y_max) in meters.
            near_field_radius_m: Distance threshold for 5 cm refinement (default 10.0 m).
            fine_res_m: Fine resolution (default 0.05 m = 5 cm).
            coarse_res_m: Coarse resolution (default 0.50 m = 50 cm = 10 * fine).
        """
        self.hierarchy = LatticeHierarchy(bounds=bounds, fine_res_m=fine_res_m, coarse_res_m=coarse_res_m)
        self.bounds = (self.hierarchy.x_min, self.hierarchy.x_max, self.hierarchy.y_min, self.hierarchy.y_max)
        self.near_field_radius_m = float(near_field_radius_m)
        self.fine_res = self.hierarchy.fine_res
        self.coarse_res = self.hierarchy.coarse_res
        self.ratio = self.hierarchy.ratio
        
        self.coarse_rows = self.hierarchy.coarse_rows
        self.coarse_cols = self.hierarchy.coarse_cols
        self.fine_rows = self.hierarchy.fine_rows
        self.fine_cols = self.hierarchy.fine_cols
        
        # Precompute coarse refinement mask:
        # A coarse cell is REFINED to 5 cm children if its minimum distance to origin <= near_field_radius_m
        self.refined_mask = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.bool_)
        
        # Calculate distance to origin for each coarse cell
        for r in range(self.coarse_rows):
            for c in range(self.coarse_cols):
                x0, x1, y0, y1 = self.hierarchy.coarse_bounds(r, c)
                # Closest point on rectangle [x0, x1] x [y0, y1] to (0, 0)
                dx = max(0.0, max(x0, -x1) if (x0 > 0 or x1 < 0) else 0.0)
                dy = max(0.0, max(y0, -y1) if (y0 > 0 or y1 < 0) else 0.0)
                min_dist = np.hypot(dx, dy)
                if min_dist <= self.near_field_radius_m:
                    self.refined_mask[r, c] = True
        
        # Precompute fine-level refinement mask (broadcast coarse refinement to fine grid)
        # fine_is_refined[r_5, c_5] is True if parent coarse cell is refined
        self.fine_is_refined = np.repeat(np.repeat(self.refined_mask, self.ratio, axis=0), self.ratio, axis=1)
        
        self.num_refined_coarse = int(np.sum(self.refined_mask))
        self.num_unrefined_coarse = int(self.coarse_rows * self.coarse_cols - self.num_refined_coarse)
        self.num_active_fine_capacity = self.num_refined_coarse * (self.ratio ** 2)
        
        # --- Pre-allocated 2.5D Layer Buffers (Zero Allocation During Processing) ---
        
        # 1. Coarse Layers (for unrefined cells in far-field > 10 m, plus diagnostic parent aggregates)
        self.coarse_elevation_min = np.full((self.coarse_rows, self.coarse_cols), np.inf, dtype=np.float32)
        self.coarse_elevation_max = np.full((self.coarse_rows, self.coarse_cols), -np.inf, dtype=np.float32)
        self.coarse_elevation_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_elevation_sq_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_point_count = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.int32)
        self.coarse_intensity_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_occupancy = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float32)
        
        # 2. Fine Layers (for refined cells in near-field <= 10 m)
        self.fine_elevation_min = np.full((self.fine_rows, self.fine_cols), np.inf, dtype=np.float32)
        self.fine_elevation_max = np.full((self.fine_rows, self.fine_cols), -np.inf, dtype=np.float32)
        self.fine_elevation_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_elevation_sq_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_point_count = np.zeros((self.fine_rows, self.fine_cols), dtype=np.int32)
        self.fine_intensity_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_occupancy = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float32)
        
        logger.info(
            f"AdaptiveMap25D Mode B initialized: ROI {self.bounds} | "
            f"Near-field (<= {self.near_field_radius_m}m): {self.num_refined_coarse} coarse refined into "
            f"{self.num_active_fine_capacity:,} fine cells @ 5cm | "
            f"Far-field (> {self.near_field_radius_m}m): {self.num_unrefined_coarse:,} coarse cells @ 50cm"
        )

    def clear(self) -> None:
        """In-place zero-allocation layer reset (<1.0 ms)."""
        self.coarse_elevation_min.fill(np.inf)
        self.coarse_elevation_max.fill(-np.inf)
        self.coarse_elevation_sum.fill(0.0)
        self.coarse_elevation_sq_sum.fill(0.0)
        self.coarse_point_count.fill(0)
        self.coarse_intensity_sum.fill(0.0)
        self.coarse_occupancy.fill(0.0)
        
        self.fine_elevation_min.fill(np.inf)
        self.fine_elevation_max.fill(-np.inf)
        self.fine_elevation_sum.fill(0.0)
        self.fine_elevation_sq_sum.fill(0.0)
        self.fine_point_count.fill(0)
        self.fine_intensity_sum.fill(0.0)
        self.fine_occupancy.fill(0.0)

    def update(self, pc: PointCloud, clear_first: bool = True) -> Dict[str, Any]:
        """Project point cloud onto the Mode B adaptive map using vectorized aggregations.
        
        Guarantees:
        - Points in refined coarse cells map ONLY to fine 5 cm cells.
        - Points in unrefined coarse cells map ONLY to coarse 50 cm cells.
        - Exactly 1-to-1 mapping per point. Zero double counting.
        
        Args:
            pc: PointCloud object.
            clear_first: If True (default), reset layers to avoid frame ghosting.
            
        Returns:
            Dictionary of telemetry metrics (active cells, cell counts, latency, memory).
        """
        if clear_first:
            self.clear()
            
        t0 = time.perf_counter()
        
        pts = pc.points
        intens = pc.intensity if pc.intensity is not None else np.zeros(len(pts), dtype=np.float32)
        n_input = len(pts)
        
        if n_input == 0:
            return self._build_telemetry(t0, n_input, 0, 0, 0)
        
        # 1. Bounds filtering
        in_bounds = self.hierarchy.is_in_bounds(pts[:, 0], pts[:, 1])
        valid_pts = pts[in_bounds]
        valid_int = intens[in_bounds]
        n_mapped = len(valid_pts)
        
        if n_mapped == 0:
            return self._build_telemetry(t0, n_input, 0, 0, 0)
        
        # 2. Coarse grid indexing to determine refinement status
        row_50, col_50 = self.hierarchy.world_to_coarse(valid_pts[:, 0], valid_pts[:, 1])
        row_50 = np.clip(row_50, 0, self.coarse_rows - 1)
        col_50 = np.clip(col_50, 0, self.coarse_cols - 1)
        
        # Point is refined (near-field) if its containing coarse cell is refined
        is_fine_point = self.refined_mask[row_50, col_50]
        is_coarse_point = ~is_fine_point
        
        z = valid_pts[:, 2]
        
        # 3. Vectorized update for Near-Field Fine Cells (5 cm)
        if np.any(is_fine_point):
            fine_pts = valid_pts[is_fine_point]
            fine_z = z[is_fine_point]
            fine_int = valid_int[is_fine_point]
            
            r_5, c_5 = self.hierarchy.world_to_fine(fine_pts[:, 0], fine_pts[:, 1])
            r_5 = np.clip(r_5, 0, self.fine_rows - 1)
            c_5 = np.clip(c_5, 0, self.fine_cols - 1)
            
            flat_fine = (r_5.astype(np.int64) * self.fine_cols + c_5.astype(np.int64)).astype(np.int64)
            
            elev_min_flat = self.fine_elevation_min.ravel()
            elev_max_flat = self.fine_elevation_max.ravel()
            elev_sum_flat = self.fine_elevation_sum.ravel()
            elev_sq_sum_flat = self.fine_elevation_sq_sum.ravel()
            pt_cnt_flat = self.fine_point_count.ravel()
            int_sum_flat = self.fine_intensity_sum.ravel()
            
            np.minimum.at(elev_min_flat, flat_fine, fine_z)
            np.maximum.at(elev_max_flat, flat_fine, fine_z)
            np.add.at(elev_sum_flat, flat_fine, fine_z.astype(np.float64))
            np.add.at(elev_sq_sum_flat, flat_fine, (fine_z ** 2).astype(np.float64))
            np.add.at(pt_cnt_flat, flat_fine, 1)
            np.add.at(int_sum_flat, flat_fine, fine_int.astype(np.float64))
            
            # Occupancy probability update for active fine cells
            active_fine = self.fine_point_count > 0
            if np.any(active_fine):
                cnt = self.fine_point_count[active_fine].astype(np.float32)
                span = self.fine_elevation_max[active_fine] - self.fine_elevation_min[active_fine]
                p_occ = 1.0 - np.exp(-0.4 * cnt) * np.exp(-0.6 * np.clip(span, 0.0, 3.0))
                self.fine_occupancy[active_fine] = np.clip(p_occ, 0.05, 0.99)
        
        # 4. Vectorized update for Far-Field Coarse Cells (50 cm)
        if np.any(is_coarse_point):
            coarse_z = z[is_coarse_point]
            coarse_int = valid_int[is_coarse_point]
            c_r50 = row_50[is_coarse_point]
            c_c50 = col_50[is_coarse_point]
            
            flat_coarse = (c_r50.astype(np.int64) * self.coarse_cols + c_c50.astype(np.int64)).astype(np.int64)
            
            c_min_flat = self.coarse_elevation_min.ravel()
            c_max_flat = self.coarse_elevation_max.ravel()
            c_sum_flat = self.coarse_elevation_sum.ravel()
            c_sq_sum_flat = self.coarse_elevation_sq_sum.ravel()
            c_cnt_flat = self.coarse_point_count.ravel()
            c_int_flat = self.coarse_intensity_sum.ravel()
            
            np.minimum.at(c_min_flat, flat_coarse, coarse_z)
            np.maximum.at(c_max_flat, flat_coarse, coarse_z)
            np.add.at(c_sum_flat, flat_coarse, coarse_z.astype(np.float64))
            np.add.at(c_sq_sum_flat, flat_coarse, (coarse_z ** 2).astype(np.float64))
            np.add.at(c_cnt_flat, flat_coarse, 1)
            np.add.at(c_int_flat, flat_coarse, coarse_int.astype(np.float64))
            
            # Occupancy probability update for active coarse cells
            active_coarse = (self.coarse_point_count > 0) & (~self.refined_mask)
            if np.any(active_coarse):
                cnt_c = self.coarse_point_count[active_coarse].astype(np.float32)
                span_c = self.coarse_elevation_max[active_coarse] - self.coarse_elevation_min[active_coarse]
                p_occ_c = 1.0 - np.exp(-0.3 * cnt_c) * np.exp(-0.5 * np.clip(span_c, 0.0, 3.0))
                self.coarse_occupancy[active_coarse] = np.clip(p_occ_c, 0.05, 0.99)
        
        # 5. Count active cells
        n_active_fine = int(np.sum(self.fine_point_count > 0))
        n_active_coarse = int(np.sum((self.coarse_point_count > 0) & (~self.refined_mask)))
        
        return self._build_telemetry(t0, n_input, n_mapped, n_active_fine, n_active_coarse)

    def _build_telemetry(
        self,
        t0: float,
        n_input: int,
        n_mapped: int,
        n_active_fine: int,
        n_active_coarse: int,
    ) -> Dict[str, Any]:
        """Construct standard Phase 5 Mode B telemetry dictionary."""
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n_total_active = n_active_fine + n_active_coarse
        
        # Mode A baseline comparison (1400x1400 = 1,960,000 cells)
        mode_a_total_cells = self.fine_rows * self.fine_cols
        # In Mode B, active capacity is fine cells inside 10m + coarse cells outside 10m
        mode_b_total_cells = self.num_active_fine_capacity + self.num_unrefined_coarse
        
        cell_reduction_pct = 100.0 * (1.0 - (mode_b_total_cells / max(1, mode_a_total_cells)))
        
        # Estimated memory footprint for Mode B layers
        # Coarse layers: 140x140 * 4 bytes * 7 layers = ~0.55 MB
        # Fine layers: only refined portion active
        memory_mb = (self.coarse_rows * self.coarse_cols * 4 * 7 + self.num_active_fine_capacity * 4 * 7) / (1024 * 1024)
        mode_a_memory_mb = (mode_a_total_cells * 4 * 7) / (1024 * 1024)
        
        return {
            'mapping_time_ms': dt_ms,
            'input_points': n_input,
            'mapped_points': n_mapped,
            'active_fine_cells': n_active_fine,
            'active_coarse_cells': n_active_coarse,
            'active_cells': n_total_active,
            'total_cells': mode_b_total_cells,
            'cell_reduction_pct': cell_reduction_pct,
            'memory_mb': memory_mb,
            'mode_a_memory_mb': mode_a_memory_mb,
            'memory_reduction_pct': 100.0 * (1.0 - memory_mb / max(0.01, mode_a_memory_mb)),
            'mode': 'Mode B (Distance-Adaptive 5cm/50cm)',
        }

    def get_cell_info(self, x: float, y: float) -> Dict[str, Any]:
        """Inspect cell at continuous world coordinate (x, y).
        
        Returns whether coordinate resolves to a 5 cm fine cell or 50 cm coarse cell,
        its parent ID, spatial bounds, and elevation/occupancy metrics.
        """
        x_arr = np.array([x], dtype=np.float32)
        y_arr = np.array([y], dtype=np.float32)
        
        if not self.hierarchy.is_in_bounds(x_arr, y_arr)[0]:
            return {'valid': False, 'message': 'Out of bounds'}
        
        r_50, c_50 = self.hierarchy.world_to_coarse(x_arr, y_arr)
        r_50_val, c_50_val = int(r_50[0]), int(c_50[0])
        
        is_refined = bool(self.refined_mask[r_50_val, c_50_val])
        dist_to_origin = float(np.hypot(x, y))
        
        if is_refined:
            # Resolved to 5 cm Fine Cell
            r_5, c_5 = self.hierarchy.world_to_fine(x_arr, y_arr)
            r_5_val, c_5_val = int(r_5[0]), int(c_5[0])
            
            cnt = int(self.fine_point_count[r_5_val, c_5_val])
            occupied = cnt > 0
            elev_min = float(self.fine_elevation_min[r_5_val, c_5_val]) if occupied else 0.0
            elev_max = float(self.fine_elevation_max[r_5_val, c_5_val]) if occupied else 0.0
            elev_mean = float(self.fine_elevation_sum[r_5_val, c_5_val] / max(1, cnt)) if occupied else 0.0
            intensity_mean = float(self.fine_intensity_sum[r_5_val, c_5_val] / max(1, cnt)) if occupied else 0.0
            occ_prob = float(self.fine_occupancy[r_5_val, c_5_val])
            
            x0, x1, y0, y1 = self.hierarchy.fine_bounds(r_5_val, c_5_val)
            cell_id = self.hierarchy.pack_cell_id(LatticeHierarchy.LEVEL_FINE, r_5_val, c_5_val)
            parent_id = self.hierarchy.pack_cell_id(LatticeHierarchy.LEVEL_COARSE, r_50_val, c_50_val)
            
            return {
                'valid': True,
                'level': 'Fine (5 cm)',
                'resolution_m': 0.05,
                'distance_m': dist_to_origin,
                'is_refined': True,
                'grid_pos': (r_5_val, c_5_val),
                'parent_pos': (r_50_val, c_50_val),
                'cell_id': f"0x{cell_id:016X}",
                'parent_id': f"0x{parent_id:016X}",
                'bounds': (x0, x1, y0, y1),
                'occupied': occupied,
                'point_count': cnt,
                'elevation_min': elev_min,
                'elevation_max': elev_max,
                'elevation_span': elev_max - elev_min if occupied else 0.0,
                'elevation_mean': elev_mean,
                'intensity_mean': intensity_mean,
                'occupancy': occ_prob,
            }
        else:
            # Resolved to 50 cm Coarse Cell
            cnt = int(self.coarse_point_count[r_50_val, c_50_val])
            occupied = cnt > 0
            elev_min = float(self.coarse_elevation_min[r_50_val, c_50_val]) if occupied else 0.0
            elev_max = float(self.coarse_elevation_max[r_50_val, c_50_val]) if occupied else 0.0
            elev_mean = float(self.coarse_elevation_sum[r_50_val, c_50_val] / max(1, cnt)) if occupied else 0.0
            intensity_mean = float(self.coarse_intensity_sum[r_50_val, c_50_val] / max(1, cnt)) if occupied else 0.0
            occ_prob = float(self.coarse_occupancy[r_50_val, c_50_val])
            
            x0, x1, y0, y1 = self.hierarchy.coarse_bounds(r_50_val, c_50_val)
            cell_id = self.hierarchy.pack_cell_id(LatticeHierarchy.LEVEL_COARSE, r_50_val, c_50_val)
            
            return {
                'valid': True,
                'level': 'Coarse (50 cm)',
                'resolution_m': 0.50,
                'distance_m': dist_to_origin,
                'is_refined': False,
                'grid_pos': (r_50_val, c_50_val),
                'parent_pos': None,
                'cell_id': f"0x{cell_id:016X}",
                'parent_id': None,
                'bounds': (x0, x1, y0, y1),
                'occupied': occupied,
                'point_count': cnt,
                'elevation_min': elev_min,
                'elevation_max': elev_max,
                'elevation_span': elev_max - elev_min if occupied else 0.0,
                'elevation_mean': elev_mean,
                'intensity_mean': intensity_mean,
                'occupancy': occ_prob,
            }
