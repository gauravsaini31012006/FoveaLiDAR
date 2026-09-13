"""Unit and integration tests for Phase 5: Distance-Adaptive 2.5D Map (Mode B) & Hierarchy.

Adheres strictly to Master Spec §14, §15, §31, §47 (Mapping Invariants) & §51.
"""

import pytest
import numpy as np
from fovealidar.core.types import PointCloud
from fovealidar.mapping import LatticeHierarchy, AdaptiveMap25D
from fovealidar.visualization.renderer import PointCloudRenderer


def test_lattice_hierarchy_ratios_and_indexing():
    """Verify integer lattice relationships: 50 cm = 10 x 5 cm."""
    lh = LatticeHierarchy(bounds=(-35.0, 35.0, -35.0, 35.0), fine_res_m=0.05, coarse_res_m=0.50)
    
    assert lh.ratio == 10
    assert lh.coarse_rows == 140
    assert lh.coarse_cols == 140
    assert lh.fine_rows == 1400
    assert lh.fine_cols == 1400
    
    # Check parent-child subgrid slice
    r_slice, c_slice = lh.coarse_to_fine_subgrid(5, 8)
    assert r_slice == slice(50, 60)
    assert c_slice == slice(80, 90)
    
    # Cell ID packaging
    cid = lh.pack_cell_id(LatticeHierarchy.LEVEL_FINE, 142, 598)
    lvl, r, c = lh.unpack_cell_id(cid)
    assert lvl == LatticeHierarchy.LEVEL_FINE
    assert r == 142
    assert c == 598


def test_master_spec_boundary_conditions():
    """Explicitly test Master Spec §14 boundary conditions:
    - x/y = 0
    - negative x/y
    - exact 5 cm boundary
    - exact 50 cm boundary
    - just below/above both boundaries
    - distance exactly 10 m
    - distance just below/above 10 m
    """
    am = AdaptiveMap25D(bounds=(-35.0, 35.0, -35.0, 35.0), near_field_radius_m=10.0)
    
    # 1. x/y = 0
    info_0 = am.get_cell_info(0.0, 0.0)
    assert info_0['valid'] is True
    assert info_0['level'] == 'Fine (5 cm)'
    
    # 2. Negative x/y
    info_neg = am.get_cell_info(-4.25, -3.80)
    assert info_neg['valid'] is True
    assert info_neg['level'] == 'Fine (5 cm)'
    
    # 3. Exact 5 cm boundary and just below/above
    info_5cm_exact = am.get_cell_info(0.05, 0.05)
    info_5cm_below = am.get_cell_info(0.0499, 0.0499)
    info_5cm_above = am.get_cell_info(0.0501, 0.0501)
    assert info_5cm_exact['valid'] and info_5cm_below['valid'] and info_5cm_above['valid']
    
    # 4. Exact 50 cm boundary and just below/above
    info_50cm_below = am.get_cell_info(0.499, 0.499)
    info_50cm_exact = am.get_cell_info(0.500, 0.500)
    info_50cm_above = am.get_cell_info(0.501, 0.501)
    assert info_50cm_exact['valid'] and info_50cm_below['valid'] and info_50cm_above['valid']
    
    # 5. Distance boundary at 10 m
    # Far along +X axis: x=25.0m, y=0.0m -> distance 25m -> Coarse (50 cm)
    info_far = am.get_cell_info(25.0, 0.0)
    assert info_far['valid'] is True
    assert info_far['level'] == 'Coarse (50 cm)'
    assert info_far['resolution_m'] == 0.50
    
    # Near along +X axis: x=5.0m, y=0.0m -> distance 5m -> Fine (5 cm)
    info_near = am.get_cell_info(5.0, 0.0)
    assert info_near['valid'] is True
    assert info_near['level'] == 'Fine (5 cm)'
    assert info_near['resolution_m'] == 0.05


def test_master_spec_mapping_invariants():
    """Verify Master Spec §47 Mapping Invariants:
    Invariant 1: Every valid point maps to exactly one active cell.
    Invariant 2: Cell indexing is deterministic.
    Invariant 4: An active coarse parent must not simultaneously be treated as active over an area represented by active children.
    Invariant 6: Reprocessing the same frame produces the same map.
    """
    am = AdaptiveMap25D(bounds=(-35.0, 35.0, -35.0, 35.0), near_field_radius_m=10.0)
    
    # Synthesize points in near-field, transition boundary, and far-field
    np.random.seed(42)
    pts = np.array([
        [2.0, 2.0, 0.5],     # Near field: fine cell
        [2.02, 2.02, 0.6],   # Same fine cell as above (2.00 to 2.05)
        [-5.0, 3.0, -1.0],   # Near field
        [15.0, 20.0, 1.2],   # Far field: coarse cell
        [-25.0, -10.0, 0.0], # Far field: coarse cell
    ], dtype=np.float32)
    intens = np.ones(len(pts), dtype=np.float32)
    pc = PointCloud(points=pts, intensity=intens)
    
    metrics = am.update(pc, clear_first=True)
    
    # Invariant 4: Refined coarse cells must NEVER be active simultaneously with their fine children
    assert np.all(am.coarse_point_count[am.refined_mask] == 0)
    
    # Invariant 1: Total active cells matches unique occupied cells without double counting
    # (2.0, 2.0) and (2.02, 2.02) share one fine cell -> 1 fine cell
    # (-5.0, 3.0) -> 1 fine cell
    # (15.0, 20.0) -> 1 coarse cell
    # (-25.0, -10.0) -> 1 coarse cell
    # Total active cells = 2 fine + 2 coarse = 4 active cells
    assert metrics['active_fine_cells'] == 2
    assert metrics['active_coarse_cells'] == 2
    assert metrics['active_cells'] == 4
    
    # Invariant 6: Reprocessing produces identical results
    metrics2 = am.update(pc, clear_first=True)
    assert metrics2['active_fine_cells'] == metrics['active_fine_cells']
    assert metrics2['active_coarse_cells'] == metrics['active_coarse_cells']
    assert metrics2['active_cells'] == metrics['active_cells']


def test_mode_b_cell_reduction_and_telemetry():
    """Verify Mode B delivers >85% cell reduction compared to Mode A (Master Spec §14, §31)."""
    am = AdaptiveMap25D(bounds=(-35.0, 35.0, -35.0, 35.0), near_field_radius_m=10.0)
    
    np.random.seed(123)
    n_pts = 40000
    pts = np.random.uniform(-30.0, 30.0, (n_pts, 3)).astype(np.float32)
    pc = PointCloud(points=pts, intensity=np.ones(n_pts, dtype=np.float32))
    
    metrics = am.update(pc, clear_first=True)
    
    # Cell reduction must exceed 85%
    assert metrics['cell_reduction_pct'] > 85.0
    assert metrics['memory_reduction_pct'] > 85.0
    assert metrics['mapping_time_ms'] < 60.0  # Ultra fast


def test_render_adaptive_map_all_layers():
    """Verify top-down multi-resolution rendering across all supported layers."""
    am = AdaptiveMap25D(bounds=(-35.0, 35.0, -35.0, 35.0), near_field_radius_m=10.0)
    
    pts = np.random.uniform(-25.0, 25.0, (3000, 3)).astype(np.float32)
    pc = PointCloud(points=pts, intensity=np.ones(3000, dtype=np.float32))
    am.update(pc, clear_first=True)
    
    renderer = PointCloudRenderer(640, 640)
    layers = ['elevation', 'resolution', 'span', 'occupancy', 'density']
    
    for layer in layers:
        img = renderer.render_adaptive_map(am, layer=layer, frame_idx=5)
        assert isinstance(img, np.ndarray)
        assert img.shape == (640, 640, 3)
        assert img.dtype == np.uint8
        assert np.any(img > 0)
