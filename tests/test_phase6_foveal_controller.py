"""Phase 6: Foveal Controller & Dynamic Foveal Map Tests.

Tests cover:
- Individual scoring functions (distance, geometry, uncertainty)
- FovealController importance aggregation and budget enforcement
- Vehicle corridor geometry
- FovealMap25D (Mode C) integration with point cloud processing
- Mapping invariants (no double counting, 1-to-1 assignment)
"""

import numpy as np
import pytest

from fovealidar.core.types import PointCloud
from fovealidar.foveation.scores import (
    distance_score,
    geometry_score,
    uncertainty_score,
    motion_score,
    semantic_score,
)
from fovealidar.foveation.foveal_controller import FovealController
from fovealidar.mapping.foveal_map import FovealMap25D


# =============================================================================
# Scoring Function Tests
# =============================================================================

class TestDistanceScore:
    """Tests for the distance importance scoring function."""

    def test_origin_gets_max_score(self):
        """Points at origin should get score 1.0."""
        dists = np.array([0.0], dtype=np.float32)
        scores = distance_score(dists, near_range_m=10.0, far_range_m=70.0)
        assert scores[0] == pytest.approx(1.0)

    def test_far_range_gets_zero(self):
        """Points at or beyond far_range_m should get score 0.0."""
        dists = np.array([70.0, 100.0], dtype=np.float32)
        scores = distance_score(dists, near_range_m=10.0, far_range_m=70.0)
        assert scores[0] == pytest.approx(0.0)
        assert scores[1] == pytest.approx(0.0)

    def test_monotonic_decay(self):
        """Score should decrease monotonically with distance."""
        dists = np.array([5.0, 15.0, 35.0, 60.0], dtype=np.float32)
        scores = distance_score(dists, near_range_m=10.0, far_range_m=70.0)
        for i in range(len(scores) - 1):
            assert scores[i] > scores[i + 1]

    def test_output_shape(self):
        """Output shape should match input."""
        dists = np.random.rand(100).astype(np.float32) * 70.0
        scores = distance_score(dists)
        assert scores.shape == (100,)
        assert scores.dtype == np.float32


class TestGeometryScore:
    """Tests for the geometry importance scoring function."""

    def test_flat_ground_low_score(self):
        """Flat ground (span ~ 0, low density) should score near 0."""
        span = np.array([0.0], dtype=np.float32)
        count = np.array([1], dtype=np.int32)
        scores = geometry_score(span, count)
        assert scores[0] < 0.1

    def test_tall_obstacle_high_score(self):
        """Tall obstacle (large span, moderate density) should score high."""
        span = np.array([2.5], dtype=np.float32)
        count = np.array([50], dtype=np.int32)
        scores = geometry_score(span, count)
        assert scores[0] > 0.8

    def test_bounded_output(self):
        """All scores should be in [0, 1]."""
        span = np.random.rand(200).astype(np.float32) * 5.0
        count = np.random.randint(0, 100, 200, dtype=np.int32)
        scores = geometry_score(span, count)
        assert np.all(scores >= 0.0)
        assert np.all(scores <= 1.0)


class TestUncertaintyScore:
    """Tests for the uncertainty scoring function."""

    def test_sparse_cell_high_uncertainty(self):
        """Cells with very few points should have high uncertainty."""
        count = np.array([1], dtype=np.int32)
        var = np.array([0.0], dtype=np.float32)
        scores = uncertainty_score(count, var)
        assert scores[0] > 0.4

    def test_dense_cell_low_uncertainty(self):
        """Dense cells with low variance should have low uncertainty."""
        count = np.array([50], dtype=np.int32)
        var = np.array([0.01], dtype=np.float32)
        scores = uncertainty_score(count, var)
        assert scores[0] < 0.1

    def test_stubs_return_zeros(self):
        """Motion and semantic stubs should return all zeros."""
        m = motion_score(50)
        s = semantic_score(50)
        assert m.shape == (50,)
        assert s.shape == (50,)
        assert np.all(m == 0.0)
        assert np.all(s == 0.0)


# =============================================================================
# FovealController Tests
# =============================================================================

class TestFovealController:
    """Tests for the Foveal Controller orchestrator."""

    @pytest.fixture
    def controller(self):
        """Create a controller for a small 20x20 coarse grid."""
        return FovealController(
            coarse_rows=20,
            coarse_cols=20,
            coarse_res=0.50,
            x_min=-5.0,
            y_min=-5.0,
            near_field_radius_m=2.0,
            max_refined_regions=10,
            refinement_threshold=0.3,
            corridor_enabled=True,
            corridor_max_range_m=5.0,
        )

    def test_importance_map_shape(self, controller):
        """Importance map should match coarse grid dimensions."""
        R, C = 20, 20
        counts = np.random.randint(0, 30, (R, C), dtype=np.int32)
        elev_max = np.random.rand(R, C).astype(np.float32) * 3.0
        elev_min = -np.random.rand(R, C).astype(np.float32) * 1.0
        elev_sum = np.zeros((R, C), dtype=np.float64)
        elev_sq = np.zeros((R, C), dtype=np.float64)
        static_mask = np.zeros((R, C), dtype=np.bool_)

        imp, dyn, telem = controller.compute_importance(
            counts, elev_max, elev_min, elev_sq, elev_sum, static_mask
        )

        assert imp.shape == (R, C)
        assert dyn.shape == (R, C)
        assert imp.dtype == np.float32
        assert dyn.dtype == np.bool_

    def test_budget_enforcement(self, controller):
        """Dynamic refined count should never exceed max_refined_regions."""
        R, C = 20, 20
        # Create a scenario where many cells would qualify
        counts = np.full((R, C), 30, dtype=np.int32)
        elev_max = np.full((R, C), 2.0, dtype=np.float32)
        elev_min = np.full((R, C), -1.0, dtype=np.float32)
        elev_sum = np.zeros((R, C), dtype=np.float64)
        elev_sq = np.zeros((R, C), dtype=np.float64)
        static_mask = np.zeros((R, C), dtype=np.bool_)

        _, dyn, telem = controller.compute_importance(
            counts, elev_max, elev_min, elev_sq, elev_sum, static_mask
        )

        n_selected = int(np.sum(dyn))
        assert n_selected <= controller.max_refined_regions, (
            f"Budget violated: {n_selected} > {controller.max_refined_regions}"
        )

    def test_static_cells_excluded(self, controller):
        """Cells already statically refined should NOT be dynamically refined."""
        R, C = 20, 20
        counts = np.full((R, C), 30, dtype=np.int32)
        elev_max = np.full((R, C), 2.0, dtype=np.float32)
        elev_min = np.full((R, C), -1.0, dtype=np.float32)
        elev_sum = np.zeros((R, C), dtype=np.float64)
        elev_sq = np.zeros((R, C), dtype=np.float64)
        static_mask = np.ones((R, C), dtype=np.bool_)  # ALL cells are static

        _, dyn, _ = controller.compute_importance(
            counts, elev_max, elev_min, elev_sq, elev_sum, static_mask
        )

        assert int(np.sum(dyn)) == 0, "No cells should be dynamically refined when all are static"

    def test_empty_cells_not_refined(self, controller):
        """Cells with zero point count should not be selected for refinement."""
        R, C = 20, 20
        counts = np.zeros((R, C), dtype=np.int32)
        elev_max = np.zeros((R, C), dtype=np.float32)
        elev_min = np.zeros((R, C), dtype=np.float32)
        elev_sum = np.zeros((R, C), dtype=np.float64)
        elev_sq = np.zeros((R, C), dtype=np.float64)
        static_mask = np.zeros((R, C), dtype=np.bool_)

        _, dyn, _ = controller.compute_importance(
            counts, elev_max, elev_min, elev_sq, elev_sum, static_mask
        )

        assert int(np.sum(dyn)) == 0, "Empty cells should never be dynamically refined"

    def test_vehicle_corridor_geometry(self, controller):
        """Vehicle corridor should only include cells in the forward cone."""
        corridor = controller.corridor_mask
        # All corridor cells should have positive X (forward direction)
        corridor_x = controller.cell_center_x[corridor]
        assert np.all(corridor_x > 0.0), "Corridor cells must be in front of the vehicle (+X)"


# =============================================================================
# FovealMap25D (Mode C) Integration Tests
# =============================================================================

class TestFovealMap25D:
    """Integration tests for the Dynamic Foveal Map."""

    @pytest.fixture
    def foveal_map(self):
        """Create a small foveal map for testing."""
        return FovealMap25D(
            bounds=(-10.0, 10.0, -10.0, 10.0),
            near_field_radius_m=3.0,
            max_refined_regions=16,
            refinement_threshold=0.3,
        )

    def _make_pc(self, n: int = 500, x_range=(-9.0, 9.0), y_range=(-9.0, 9.0), z_range=(-1.5, 2.0)):
        """Generate a random point cloud within bounds."""
        rng = np.random.default_rng(42)
        pts = np.column_stack([
            rng.uniform(x_range[0], x_range[1], n),
            rng.uniform(y_range[0], y_range[1], n),
            rng.uniform(z_range[0], z_range[1], n),
        ]).astype(np.float32)
        intensity = rng.uniform(0.0, 1.0, n).astype(np.float32)
        return PointCloud(points=pts, intensity=intensity)

    def test_basic_update(self, foveal_map):
        """Map should process a point cloud and return valid telemetry."""
        pc = self._make_pc()
        telemetry = foveal_map.update(pc)

        assert telemetry['input_points'] == len(pc.points)
        assert telemetry['mapped_points'] > 0
        assert telemetry['mapping_time_ms'] > 0.0
        assert telemetry['mode'] == 'Mode C (Dynamic Foveal)'

    def test_importance_map_computed(self, foveal_map):
        """After update, the importance map should be populated."""
        pc = self._make_pc()
        foveal_map.update(pc)

        assert foveal_map.last_importance_map is not None
        assert foveal_map.last_importance_map.shape == (foveal_map.coarse_rows, foveal_map.coarse_cols)

    def test_dynamic_refinement_budget(self, foveal_map):
        """Dynamic refinement should never exceed budget."""
        pc = self._make_pc(n=2000)
        foveal_map.update(pc)

        n_dynamic = int(np.sum(foveal_map.dynamic_refined_mask))
        assert n_dynamic <= foveal_map.controller.max_refined_regions

    def test_clear_resets_all_layers(self, foveal_map):
        """Clear should reset all layer buffers."""
        pc = self._make_pc()
        foveal_map.update(pc)

        assert np.any(foveal_map.coarse_point_count > 0)
        foveal_map.clear()
        assert np.all(foveal_map.coarse_point_count == 0)
        assert np.all(foveal_map.fine_point_count == 0)

    def test_no_double_counting(self, foveal_map):
        """Every point must be assigned to exactly one resolution level.

        Points in refined coarse cells go to fine grid.
        Points in unrefined coarse cells stay in coarse grid.
        Total mapped = fine_mapped + coarse_mapped (no overlap).
        """
        pc = self._make_pc(n=1000)
        telemetry = foveal_map.update(pc)

        # Count points by coarse cell refinement status
        pts = pc.points
        in_bounds = foveal_map.hierarchy.is_in_bounds(pts[:, 0], pts[:, 1])
        valid_pts = pts[in_bounds]

        row_50, col_50 = foveal_map.hierarchy.world_to_coarse(valid_pts[:, 0], valid_pts[:, 1])
        row_50 = np.clip(row_50, 0, foveal_map.coarse_rows - 1)
        col_50 = np.clip(col_50, 0, foveal_map.coarse_cols - 1)

        is_refined = foveal_map.active_refined_mask[row_50, col_50]
        n_to_fine = int(np.sum(is_refined))
        n_to_coarse = int(np.sum(~is_refined))

        total_mapped = telemetry['mapped_points']
        assert n_to_fine + n_to_coarse == total_mapped, (
            f"Double counting detected: {n_to_fine} + {n_to_coarse} != {total_mapped}"
        )

    def test_consecutive_frames_motion_scoring(self, foveal_map):
        """Running two consecutive frames should enable motion scoring."""
        pc1 = self._make_pc(n=500)
        foveal_map.update(pc1)

        # Second frame should have previous occupancy available
        pc2 = self._make_pc(n=500)
        telemetry2 = foveal_map.update(pc2)

        assert foveal_map._prev_occupancy is not None
        assert telemetry2['mapping_time_ms'] > 0.0

    def test_get_cell_info(self, foveal_map):
        """Test continuous world coordinate querying on FovealMap25D."""
        pc = self._make_pc(n=1000)
        foveal_map.update(pc)

        # Near field origin (should be static fine)
        near_info = foveal_map.get_cell_info(1.0, 1.0)
        assert near_info['valid'] is True
        assert near_info['is_refined'] is True
        assert near_info['resolution_m'] == 0.05
        assert "Fine" in near_info['level']

        # Far field cell within bounds
        far_info = foveal_map.get_cell_info(7.0, 7.0)
        assert far_info['valid'] is True
        assert far_info['resolution_m'] in (0.05, 0.50)

        # Out of bounds
        oob_info = foveal_map.get_cell_info(25.0, 25.0)
        assert oob_info['valid'] is False
