"""Foveal Controller — Dynamic Importance-Based Selective Refinement (Phase 6).

Implements SIH 2026 Master Spec §16:
- Aggregates 5 importance scores (distance, geometry, uncertainty, semantic, motion)
  using configurable weights into a per-cell importance map over the 50 cm coarse grid.
- Applies a Vehicle-Centric Corridor boost (§40) to prioritise the ego driving path.
- Enforces a compute budget (max_refined_regions) via fast top-K truncation.
- Outputs a boolean `dynamic_refined_mask` indicating which far-field coarse cells
  should be dynamically upgraded to 5 cm resolution.
"""

import time
from typing import Tuple, Dict, Any, Optional
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger
from fovealidar.foveation.scores import (
    distance_score,
    geometry_score,
    uncertainty_score,
    motion_score,
    semantic_score,
)
from fovealidar.perception.semantic import HeuristicSemanticBackend, SalsaNextBackend
from fovealidar.perception.tracking import DetectionBackend, SimpleTracker

logger = get_logger(__name__)


class FovealController:
    """Orchestrates per-cell importance scoring and selective far-field refinement.

    Given the coarse grid metadata from an AdaptiveMap25D (or FovealMap25D),
    this controller:
    1. Computes vectorized importance scores over ALL coarse cells.
    2. Applies the vehicle corridor boost.
    3. Selects the top-K cells (by importance) that exceed the threshold AND
       are NOT already statically refined (i.e. outside the 10 m near-field).
    4. Returns a `dynamic_refined_mask` for use by FovealMap25D.
    """

    def __init__(
        self,
        coarse_rows: int,
        coarse_cols: int,
        coarse_res: float = 0.50,
        x_min: float = -35.0,
        y_min: float = -35.0,
        near_field_radius_m: float = 10.0,
        weights: Optional[Dict[str, float]] = None,
        max_refined_regions: int = 128,
        refinement_threshold: float = 0.6,
        corridor_enabled: bool = True,
        corridor_width_m: float = 4.0,
        corridor_max_range_m: float = 50.0,
        corridor_angle_deg: float = 60.0,
        corridor_importance_boost: float = 0.3,
        semantic_enabled: bool = False,
        tracking_enabled: bool = False,
        use_mock_dl_backends: bool = False,
    ):
        """Initialise the Foveal Controller.

        Args:
            coarse_rows: Number of coarse grid rows.
            coarse_cols: Number of coarse grid columns.
            coarse_res: Coarse cell resolution in metres (0.50 m).
            x_min: World X origin of grid.
            y_min: World Y origin of grid.
            near_field_radius_m: Static near-field radius.
            weights: Dict of score name -> weight. Must sum to ~1.0.
            max_refined_regions: Hard budget cap on dynamically refined tiles.
            refinement_threshold: Minimum importance to qualify for refinement.
            corridor_enabled: Whether to apply the vehicle corridor boost.
            corridor_width_m: Half-width of the corridor (metres).
            corridor_max_range_m: Maximum forward distance of the corridor.
            corridor_angle_deg: Angular span of the corridor (degrees).
            corridor_importance_boost: Additive importance boost inside corridor.
            semantic_enabled: Enable Phase 7 semantic AI.
            tracking_enabled: Enable Phase 8 detection + tracking.
            use_mock_dl_backends: Use SalsaNext/CenterPoint mock backends instead of heuristics.
        """
        self.coarse_rows = coarse_rows
        self.coarse_cols = coarse_cols
        self.coarse_res = coarse_res
        self.x_min = x_min
        self.y_min = y_min
        self.near_field_radius_m = near_field_radius_m

        self.weights = weights or {
            "distance": 0.20,
            "semantic": 0.25,
            "uncertainty": 0.25,
            "geometry": 0.15,
            "motion": 0.15,
        }
        self.max_refined_regions = max_refined_regions
        self.refinement_threshold = refinement_threshold

        self.corridor_enabled = corridor_enabled
        self.corridor_width_m = corridor_width_m
        self.corridor_max_range_m = corridor_max_range_m
        self.corridor_angle_rad = np.radians(corridor_angle_deg / 2.0)
        self.corridor_importance_boost = corridor_importance_boost

        self.semantic_enabled = semantic_enabled
        self.tracking_enabled = tracking_enabled
        
        # Initialize perception modules
        if use_mock_dl_backends:
            self.semantic_backend = SalsaNextBackend()
            self.detection_backend = DetectionBackend()
        else:
            self.semantic_backend = HeuristicSemanticBackend()
            self.detection_backend = DetectionBackend() # Mock detection even for heuristics
            
        self.tracker = SimpleTracker()

        # Precompute coarse cell centers (row, col) -> (x, y)
        cols_arr = np.arange(self.coarse_cols, dtype=np.float32)
        rows_arr = np.arange(self.coarse_rows, dtype=np.float32)
        cc, rr = np.meshgrid(cols_arr, rows_arr)
        self.cell_center_x = (self.x_min + (cc + 0.5) * self.coarse_res).astype(np.float32)
        self.cell_center_y = (self.y_min + (rr + 0.5) * self.coarse_res).astype(np.float32)
        self.cell_distances = np.sqrt(
            self.cell_center_x ** 2 + self.cell_center_y ** 2
        ).astype(np.float32)

        # Vehicle corridor mask: cone in +X direction
        self.corridor_mask = np.zeros((self.coarse_rows, self.coarse_cols), dtype=np.bool_)
        if self.corridor_enabled:
            angle_to_cell = np.arctan2(self.cell_center_y, self.cell_center_x)
            in_angle = np.abs(angle_to_cell) <= self.corridor_angle_rad
            in_range = (self.cell_center_x > 0.0) & (self.cell_distances <= self.corridor_max_range_m)
            self.corridor_mask = in_angle & in_range

        # Last computed importance map and perception states (for rendering / inspection)
        self.last_importance_map: Optional[NDArray[np.float32]] = None
        self.last_semantic_labels: Optional[NDArray[np.int32]] = None
        self.last_tracked_bboxes: Optional[list] = None
        self.last_telemetry: Dict[str, Any] = {}

        logger.info(
            f"FovealController initialized: {coarse_rows}x{coarse_cols} grid | "
            f"Budget: {max_refined_regions} tiles | Threshold: {refinement_threshold:.2f} | "
            f"Corridor: {'ON' if corridor_enabled else 'OFF'}"
        )

    def compute_importance(
        self,
        coarse_point_count: NDArray[np.int32],
        coarse_elevation_max: NDArray[np.float32],
        coarse_elevation_min: NDArray[np.float32],
        coarse_elevation_sq_sum: NDArray[np.float64],
        coarse_elevation_sum: NDArray[np.float64],
        static_refined_mask: NDArray[np.bool_],
        previous_occupancy: Optional[NDArray[np.float32]] = None,
        current_occupancy: Optional[NDArray[np.float32]] = None,
    ) -> Tuple[NDArray[np.float32], NDArray[np.bool_], Dict[str, Any]]:
        """Compute full importance map and select cells for dynamic refinement.

        Args:
            coarse_point_count: (R, C) int32 point counts per coarse cell.
            coarse_elevation_max: (R, C) float32 max elevation per cell.
            coarse_elevation_min: (R, C) float32 min elevation per cell.
            coarse_elevation_sq_sum: (R, C) float64 sum of z^2 per cell.
            coarse_elevation_sum: (R, C) float64 sum of z per cell.
            static_refined_mask: (R, C) bool — True for cells already in the 10 m near-field.
            previous_occupancy: (R, C) float32 — previous frame occupancy (optional).
            current_occupancy: (R, C) float32 — current frame occupancy (optional).

        Returns:
            Tuple of:
                importance_map: (R, C) float32 ∈ [0, 1]
                dynamic_refined_mask: (R, C) bool — True for cells to refine dynamically
                telemetry: dict of timing and selection stats
        """
        t0 = time.perf_counter()
        R, C = self.coarse_rows, self.coarse_cols
        n_cells = R * C

        # --- Flatten inputs for vectorized scoring ---
        dists_flat = self.cell_distances.ravel()
        counts_flat = coarse_point_count.ravel()

        # Elevation span
        span_flat = (coarse_elevation_max - coarse_elevation_min).ravel()
        # Clamp invalid spans (where no points exist)
        no_pts = counts_flat == 0
        span_flat[no_pts] = 0.0

        # Elevation variance: var = E[z^2] - E[z]^2
        float_counts = np.maximum(counts_flat.astype(np.float64), 1.0)
        z_mean = (coarse_elevation_sum.ravel() / float_counts)
        z_var = np.maximum(0.0, (coarse_elevation_sq_sum.ravel() / float_counts) - z_mean ** 2).astype(np.float32)
        z_var[no_pts] = 0.0

        # --- Phase 7 & 8 Perception ---
        semantic_labels = None
        semantic_latency_ms = 0.0
        tracked_bboxes = None
        tracking_latency_ms = 0.0
        
        if self.semantic_enabled:
            semantic_labels, _, semantic_latency_ms = self.semantic_backend.infer(
                coarse_elevation_max,
                coarse_elevation_min,
                z_var.reshape(R, C),
                coarse_point_count
            )
            semantic_labels = semantic_labels.ravel()

        if self.tracking_enabled and self.semantic_enabled and semantic_labels is not None:
            # Run detection on 2D semantic grid
            bboxes, det_lat = self.detection_backend.infer(semantic_labels.reshape(R, C))
            tracked_bboxes = self.tracker.update(bboxes)
            tracking_latency_ms = det_lat
            
        self.last_semantic_labels = semantic_labels.reshape(R, C) if semantic_labels is not None else None
        self.last_tracked_bboxes = tracked_bboxes
            
        # --- Compute individual scores ---
        s_dist = distance_score(dists_flat, self.near_field_radius_m, 70.0)
        s_geom = geometry_score(span_flat.astype(np.float32), counts_flat)
        s_uncert = uncertainty_score(counts_flat, z_var)
        s_motion = motion_score(
            n_cells,
            previous_occupancy.ravel() if previous_occupancy is not None else None,
            current_occupancy.ravel() if current_occupancy is not None else None,
            tracked_bboxes,
            self.coarse_cols,
        )
        s_semantic = semantic_score(n_cells, semantic_labels)

        # --- Weighted aggregation ---
        w = self.weights
        importance_flat = (
            w.get("distance", 0.0) * s_dist +
            w.get("geometry", 0.0) * s_geom +
            w.get("uncertainty", 0.0) * s_uncert +
            w.get("motion", 0.0) * s_motion +
            w.get("semantic", 0.0) * s_semantic
        ).astype(np.float32)

        # --- Vehicle Corridor Boost ---
        if self.corridor_enabled:
            importance_flat += self.corridor_mask.ravel().astype(np.float32) * self.corridor_importance_boost

        importance_flat = np.clip(importance_flat, 0.0, 1.0)

        # --- Reshape ---
        importance_map = importance_flat.reshape(R, C)
        self.last_importance_map = importance_map.copy()

        # --- Select dynamic refinement candidates ---
        # Only consider cells that:
        #   1. Are NOT already statically refined (i.e. outside 10 m near-field)
        #   2. Have occupied points (point_count > 0)
        #   3. Exceed the importance threshold
        candidates = (
            (~static_refined_mask) &
            (coarse_point_count > 0) &
            (importance_map >= self.refinement_threshold)
        )

        candidate_indices = np.argwhere(candidates)  # (K, 2) array of (row, col)
        n_candidates = len(candidate_indices)

        # --- Budget enforcement via top-K ---
        dynamic_refined_mask = np.zeros((R, C), dtype=np.bool_)

        if n_candidates > 0:
            candidate_scores = importance_map[candidates]

            if n_candidates <= self.max_refined_regions:
                # All candidates within budget
                dynamic_refined_mask[candidates] = True
            else:
                # Top-K by importance score
                top_k_local = np.argpartition(candidate_scores, -self.max_refined_regions)[-self.max_refined_regions:]
                selected = candidate_indices[top_k_local]
                dynamic_refined_mask[selected[:, 0], selected[:, 1]] = True

        n_selected = int(np.sum(dynamic_refined_mask))

        dt_ms = (time.perf_counter() - t0) * 1000.0
        telemetry = {
            'controller_time_ms': dt_ms,
            'semantic_time_ms': semantic_latency_ms,
            'tracking_time_ms': tracking_latency_ms,
            'n_candidates': n_candidates,
            'n_selected': n_selected,
            'max_budget': self.max_refined_regions,
            'threshold': self.refinement_threshold,
            'corridor_cells': int(np.sum(self.corridor_mask)),
            'mean_importance': float(np.mean(importance_map)),
            'max_importance': float(np.max(importance_map)),
            'tracked_objects': len(tracked_bboxes) if tracked_bboxes else 0,
        }
        self.last_telemetry = telemetry

        logger.debug(
            f"FovealController: {n_candidates} candidates -> {n_selected} selected "
            f"(budget {self.max_refined_regions}) in {dt_ms:.2f} ms"
        )

        return importance_map, dynamic_refined_mask, telemetry
