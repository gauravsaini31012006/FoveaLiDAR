"""Dynamic Foveal Map (Mode C) — Phase 6.

Implements SIH 2026 Master Spec §16, §31:
- Extends Mode B (Distance-Adaptive 5cm/50cm) with dynamic per-frame refinement.
- Static 10 m near-field: always 5 cm fine cells (inherited from AdaptiveMap25D).
- Dynamic far-field: coarse cells exceeding the importance threshold are upgraded
  to 5 cm via the FovealController.
- All mapping invariants (1-to-1, no double counting, no holes) are preserved.
"""

import time
from typing import Tuple, Dict, Any, Optional, List
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.types import PointCloud
from fovealidar.core.logging import get_logger
from fovealidar.mapping.hierarchy import LatticeHierarchy

logger = get_logger(__name__)


class FovealMap25D:
    """Dynamic Foveal Variable Resolution 2.5D Map (Mode C).

    Combines:
    - Static near-field 5 cm core (distance <= 10 m)
    - Dynamic importance-driven 5 cm refinement of far-field cells
    - Background far-field at 50 cm coarse resolution
    """

    def __init__(
        self,
        bounds: Tuple[float, float, float, float] = (-35.0, 35.0, -35.0, 35.0),
        near_field_radius_m: float = 10.0,
        fine_res_m: float = 0.05,
        coarse_res_m: float = 0.50,
        max_refined_regions: int = 128,
        refinement_threshold: float = 0.6,
        corridor_enabled: bool = True,
    ):
        """Initialise Mode C Dynamic Foveal Map.

        Args:
            bounds: (x_min, x_max, y_min, y_max) in metres.
            near_field_radius_m: Static near-field radius for 5 cm core.
            fine_res_m: Fine resolution (5 cm).
            coarse_res_m: Coarse resolution (50 cm).
            max_refined_regions: Maximum dynamically refined far-field tiles.
            refinement_threshold: Minimum importance to qualify for dynamic refinement.
            corridor_enabled: Whether to apply vehicle corridor boost.
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

        # --- Static near-field refinement mask (same as Mode B) ---
        self.static_refined_mask = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.bool_)
        for r in range(self.coarse_rows):
            for c in range(self.coarse_cols):
                x0, x1, y0, y1 = self.hierarchy.coarse_bounds(r, c)
                dx = max(0.0, max(x0, -x1) if (x0 > 0 or x1 < 0) else 0.0)
                dy = max(0.0, max(y0, -y1) if (y0 > 0 or y1 < 0) else 0.0)
                min_dist = np.hypot(dx, dy)
                if min_dist <= self.near_field_radius_m:
                    self.static_refined_mask[r, c] = True

        # --- Dynamic refinement mask (updated per frame by FovealController) ---
        self.dynamic_refined_mask = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.bool_)

        # Combined active refined mask = static | dynamic
        self.active_refined_mask = self.static_refined_mask.copy()

        # --- Foveal Controller (lazy import to avoid circular dep) ---
        from fovealidar.foveation.foveal_controller import FovealController
        self.controller = FovealController(
            coarse_rows=self.coarse_rows,
            coarse_cols=self.coarse_cols,
            coarse_res=self.coarse_res,
            x_min=self.hierarchy.x_min,
            y_min=self.hierarchy.y_min,
            near_field_radius_m=self.near_field_radius_m,
            max_refined_regions=max_refined_regions,
            refinement_threshold=refinement_threshold,
            corridor_enabled=corridor_enabled,
        )

        # --- Pre-allocated 2.5D Layer Buffers ---
        # Coarse layers
        self.coarse_elevation_min = np.full((self.coarse_rows, self.coarse_cols), np.inf, dtype=np.float32)
        self.coarse_elevation_max = np.full((self.coarse_rows, self.coarse_cols), -np.inf, dtype=np.float32)
        self.coarse_elevation_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_elevation_sq_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_point_count = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.int32)
        self.coarse_intensity_sum = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float64)
        self.coarse_occupancy = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.float32)

        # Fine layers
        self.fine_elevation_min = np.full((self.fine_rows, self.fine_cols), np.inf, dtype=np.float32)
        self.fine_elevation_max = np.full((self.fine_rows, self.fine_cols), -np.inf, dtype=np.float32)
        self.fine_elevation_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_elevation_sq_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_point_count = np.zeros((self.fine_rows, self.fine_cols), dtype=np.int32)
        self.fine_intensity_sum = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float64)
        self.fine_occupancy = np.zeros((self.fine_rows, self.fine_cols), dtype=np.float32)

        # For importance heatmap rendering
        self.last_importance_map: Optional[NDArray[np.float32]] = None
        self.last_update_time_ms: float = 0.0

        # Previous frame occupancy for motion scoring
        self._prev_occupancy: Optional[NDArray[np.float32]] = None

        logger.info(
            f"FovealMap25D Mode C initialized: ROI {self.bounds} | "
            f"Static near-field: {int(np.sum(self.static_refined_mask))} coarse tiles @ 5cm | "
            f"Dynamic budget: {max_refined_regions} tiles"
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
        """Project point cloud onto Mode C dynamic foveal map.

        Two-pass strategy:
        1. First pass: project ALL in-bounds points onto the COARSE grid to
           build the elevation / occupancy statistics needed by the FovealController.
        2. Run the FovealController to compute the dynamic_refined_mask.
        3. Second pass: re-project points in refined cells (static + dynamic)
           onto the FINE 5 cm grid. Unrefined coarse cells keep their coarse data.

        Args:
            pc: PointCloud object.
            clear_first: If True (default), reset layers before processing.

        Returns:
            Dict of telemetry metrics.
        """
        if clear_first:
            self.clear()

        t0 = time.perf_counter()

        pts = pc.points
        intens = pc.intensity if pc.intensity is not None else np.zeros(len(pts), dtype=np.float32)
        n_input = len(pts)

        if n_input == 0:
            return self._build_telemetry(t0, n_input, 0, 0, 0, 0)

        # --- Bounds filtering ---
        in_bounds = self.hierarchy.is_in_bounds(pts[:, 0], pts[:, 1])
        valid_pts = pts[in_bounds]
        valid_int = intens[in_bounds]
        n_mapped = len(valid_pts)

        if n_mapped == 0:
            return self._build_telemetry(t0, n_input, 0, 0, 0, 0)

        z = valid_pts[:, 2]

        # ======== PASS 1: Coarse grid aggregation (all points) ========
        row_50, col_50 = self.hierarchy.world_to_coarse(valid_pts[:, 0], valid_pts[:, 1])
        row_50 = np.clip(row_50, 0, self.coarse_rows - 1)
        col_50 = np.clip(col_50, 0, self.coarse_cols - 1)

        flat_coarse = (row_50.astype(np.int64) * self.coarse_cols + col_50.astype(np.int64)).astype(np.int64)

        c_min_flat = self.coarse_elevation_min.ravel()
        c_max_flat = self.coarse_elevation_max.ravel()
        c_sum_flat = self.coarse_elevation_sum.ravel()
        c_sq_flat = self.coarse_elevation_sq_sum.ravel()
        c_cnt_flat = self.coarse_point_count.ravel()
        c_int_flat = self.coarse_intensity_sum.ravel()

        np.minimum.at(c_min_flat, flat_coarse, z)
        np.maximum.at(c_max_flat, flat_coarse, z)
        np.add.at(c_sum_flat, flat_coarse, z.astype(np.float64))
        np.add.at(c_sq_flat, flat_coarse, (z ** 2).astype(np.float64))
        np.add.at(c_cnt_flat, flat_coarse, 1)
        np.add.at(c_int_flat, flat_coarse, valid_int.astype(np.float64))

        # Coarse occupancy
        active_coarse = self.coarse_point_count > 0
        if np.any(active_coarse):
            cnt_c = self.coarse_point_count[active_coarse].astype(np.float32)
            span_c = self.coarse_elevation_max[active_coarse] - self.coarse_elevation_min[active_coarse]
            p_occ = 1.0 - np.exp(-0.3 * cnt_c) * np.exp(-0.5 * np.clip(span_c, 0.0, 3.0))
            self.coarse_occupancy[active_coarse] = np.clip(p_occ, 0.05, 0.99)

        # ======== FOVEAL CONTROLLER ========
        importance_map, dyn_mask, ctrl_telem = self.controller.compute_importance(
            coarse_point_count=self.coarse_point_count,
            coarse_elevation_max=self.coarse_elevation_max,
            coarse_elevation_min=self.coarse_elevation_min,
            coarse_elevation_sq_sum=self.coarse_elevation_sq_sum,
            coarse_elevation_sum=self.coarse_elevation_sum,
            static_refined_mask=self.static_refined_mask,
            previous_occupancy=self._prev_occupancy,
            current_occupancy=self.coarse_occupancy,
        )

        self.dynamic_refined_mask = dyn_mask
        self.active_refined_mask = self.static_refined_mask | self.dynamic_refined_mask
        self.last_importance_map = importance_map

        # Save current occupancy for next frame's motion score
        self._prev_occupancy = self.coarse_occupancy.copy()

        # ======== PASS 2: Fine grid projection for refined cells ========
        is_refined_point = self.active_refined_mask[row_50, col_50]

        if np.any(is_refined_point):
            fine_pts = valid_pts[is_refined_point]
            fine_z = z[is_refined_point]
            fine_int = valid_int[is_refined_point]

            r_5, c_5 = self.hierarchy.world_to_fine(fine_pts[:, 0], fine_pts[:, 1])
            r_5 = np.clip(r_5, 0, self.fine_rows - 1)
            c_5 = np.clip(c_5, 0, self.fine_cols - 1)

            flat_fine = (r_5.astype(np.int64) * self.fine_cols + c_5.astype(np.int64)).astype(np.int64)

            f_min = self.fine_elevation_min.ravel()
            f_max = self.fine_elevation_max.ravel()
            f_sum = self.fine_elevation_sum.ravel()
            f_sq = self.fine_elevation_sq_sum.ravel()
            f_cnt = self.fine_point_count.ravel()
            f_int = self.fine_intensity_sum.ravel()

            np.minimum.at(f_min, flat_fine, fine_z)
            np.maximum.at(f_max, flat_fine, fine_z)
            np.add.at(f_sum, flat_fine, fine_z.astype(np.float64))
            np.add.at(f_sq, flat_fine, (fine_z ** 2).astype(np.float64))
            np.add.at(f_cnt, flat_fine, 1)
            np.add.at(f_int, flat_fine, fine_int.astype(np.float64))

            # Fine occupancy
            active_fine = self.fine_point_count > 0
            if np.any(active_fine):
                cnt_f = self.fine_point_count[active_fine].astype(np.float32)
                span_f = self.fine_elevation_max[active_fine] - self.fine_elevation_min[active_fine]
                p_occ_f = 1.0 - np.exp(-0.4 * cnt_f) * np.exp(-0.6 * np.clip(span_f, 0.0, 3.0))
                self.fine_occupancy[active_fine] = np.clip(p_occ_f, 0.05, 0.99)

        # ======== Telemetry ========
        n_active_fine = int(np.sum(self.fine_point_count > 0))
        n_active_coarse = int(np.sum((self.coarse_point_count > 0) & (~self.active_refined_mask)))
        n_dynamic = int(np.sum(self.dynamic_refined_mask))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        self.last_update_time_ms = dt_ms

        telemetry = self._build_telemetry(t0, n_input, n_mapped, n_active_fine, n_active_coarse, n_dynamic)
        telemetry.update(ctrl_telem)
        return telemetry

    def _build_telemetry(
        self,
        t0: float,
        n_input: int,
        n_mapped: int,
        n_active_fine: int,
        n_active_coarse: int,
        n_dynamic: int,
    ) -> Dict[str, Any]:
        """Construct Phase 6 Mode C telemetry dictionary."""
        dt_ms = (time.perf_counter() - t0) * 1000.0
        n_total_active = n_active_fine + n_active_coarse

        mode_a_total = self.fine_rows * self.fine_cols
        n_static = int(np.sum(self.static_refined_mask))
        # Mode C total capacity = static fine + dynamic fine + unrefined coarse
        mode_c_fine = (n_static + n_dynamic) * (self.ratio ** 2)
        mode_c_coarse = self.coarse_rows * self.coarse_cols - n_static - n_dynamic
        mode_c_total = mode_c_fine + mode_c_coarse

        cell_reduction = 100.0 * (1.0 - mode_c_total / max(1, mode_a_total))
        memory_mb = (mode_c_total * 4 * 7) / (1024 * 1024)

        return {
            'mapping_time_ms': dt_ms,
            'input_points': n_input,
            'mapped_points': n_mapped,
            'active_fine_cells': n_active_fine,
            'active_coarse_cells': n_active_coarse,
            'active_cells': n_total_active,
            'dynamic_refined_tiles': n_dynamic,
            'static_refined_tiles': n_static,
            'total_cells': mode_c_total,
            'cell_reduction_pct': cell_reduction,
            'memory_mb': memory_mb,
            'mode': 'Mode C (Dynamic Foveal)',
        }

    def get_cell_info(self, x: float, y: float) -> Dict[str, Any]:
        """Query detailed hierarchy and state information for a continuous world coordinate."""
        if not self.hierarchy.is_in_bounds(np.array([x]), np.array([y]))[0]:
            return {'valid': False, 'reason': 'Out of bounds'}

        x_arr = np.array([x], dtype=np.float32)
        y_arr = np.array([y], dtype=np.float32)

        r_50, c_50 = self.hierarchy.world_to_coarse(x_arr, y_arr)
        r_50_val, c_50_val = int(r_50[0]), int(c_50[0])

        is_refined = bool(self.active_refined_mask[r_50_val, c_50_val])
        is_static = bool(self.static_refined_mask[r_50_val, c_50_val])
        is_dynamic = bool(self.dynamic_refined_mask[r_50_val, c_50_val])
        dist_to_origin = float(np.hypot(x, y))
        importance = float(self.last_importance_map[r_50_val, c_50_val]) if self.last_importance_map is not None else 0.0

        if is_refined:
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

            level_str = "Fine (5 cm) [Dynamic Far-Field]" if is_dynamic else "Fine (5 cm) [Static Near-Field]"

            return {
                'valid': True,
                'level': level_str,
                'resolution_m': 0.05,
                'distance_m': dist_to_origin,
                'is_refined': True,
                'is_static': is_static,
                'is_dynamic': is_dynamic,
                'importance_score': importance,
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
                'level': 'Coarse (50 cm) [Background]',
                'resolution_m': 0.50,
                'distance_m': dist_to_origin,
                'is_refined': False,
                'is_static': False,
                'is_dynamic': False,
                'importance_score': importance,
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
