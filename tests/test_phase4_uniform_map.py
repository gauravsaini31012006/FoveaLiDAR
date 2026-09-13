"""Unit and integration tests for Phase 4: Uniform 2.5D Baseline Map (Mode A) & Geometry Engine."""

import pytest
import numpy as np
from fovealidar.core.types import PointCloud
from fovealidar.geometry import GeometryEngine, GeometryFeatures
from fovealidar.mapping import Uniform2DGrid, ElevationMap25D, MAP_CELL_DTYPE
from fovealidar.visualization.renderer import PointCloudRenderer


def test_uniform_grid_lattice_and_bounds():
    """Verify deterministic 5 cm integer lattice and bounds alignment per Master Spec §14 & §47."""
    grid = Uniform2DGrid(resolution_m=0.05, bounds=(-20.0, 20.0, -20.0, 20.0))
    
    assert grid.resolution_m == 0.05
    assert grid.num_cols == 800
    assert grid.num_rows == 800
    
    # Coordinate round-trip test
    x_in = np.array([0.0, 5.25, -12.35, 19.95], dtype=np.float32)
    y_in = np.array([0.0, -8.10, 15.65, -19.95], dtype=np.float32)
    
    rows, cols = grid.world_to_grid(x_in, y_in)
    x_out, y_out = grid.grid_to_world(rows, cols)
    
    # Should snap to cell centers within half cell size (0.025m)
    assert np.all(np.abs(x_in - x_out) <= 0.026)
    assert np.all(np.abs(y_in - y_out) <= 0.026)
    
    # Check bounds
    assert np.all(grid.is_in_bounds(x_in, y_in))
    assert not grid.is_in_bounds(np.array([25.0]), np.array([0.0]))[0]


def test_geometry_engine_features():
    """Verify GeometryEngine extracts correct spatial statistics."""
    np.random.seed(42)
    pts = np.random.uniform(-5, 5, (1000, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-1.5, 2.5, 1000)
    pc = PointCloud(points=pts, intensity=np.ones(1000, dtype=np.float32))
    
    engine = GeometryEngine(bin_size_m=0.5)
    feats = engine.compute_features(pc, bounds=(-10.0, 10.0, -10.0, 10.0))
    
    assert len(feats.point_counts) > 0
    assert len(feats.density) == len(feats.point_counts)
    assert np.all(feats.z_max >= feats.z_min)
    assert np.all(feats.roughness >= 0.0)
    assert np.all((feats.complexity >= 0.0) & (feats.complexity <= 1.0))


def test_elevation_map_mode_a_layers():
    """Verify Mode A 2.5D Elevation Map layer updates and statistics."""
    # Create controlled synthetic points in a single cell
    # Cell at x=1.02, y=2.02 (both within 5cm of (1.0, 2.0))
    pts = np.array([
        [1.01, 2.01, -1.5],
        [1.02, 2.02, 0.5],
        [1.03, 2.03, 1.5],
    ], dtype=np.float32)
    intens = np.array([0.2, 0.5, 0.8], dtype=np.float32)
    pc = PointCloud(points=pts, intensity=intens)
    
    em = ElevationMap25D(resolution_m=0.05, bounds=(-10.0, 10.0, -10.0, 10.0))
    metrics = em.update(pc)
    
    assert metrics['active_cells'] == 1
    assert metrics['mapped_points'] == 3
    assert metrics['mapping_time_ms'] < 100.0  # Fast execution
    
    # Query the cell
    cells = em.get_active_cells()
    assert len(cells) == 1
    
    c = cells[0]
    assert c['point_count'] == 3
    assert pytest.approx(float(c['elevation_min']), 0.01) == -1.5
    assert pytest.approx(float(c['elevation_max']), 0.01) == 1.5
    assert pytest.approx(float(c['elevation_mean']), 0.01) == 0.167
    assert pytest.approx(float(c['intensity_mean']), 0.01) == 0.50
    assert c['occupancy'] > 0.0
    
    # Cell info inspector
    info = em.get_cell_info(int(c['row']), int(c['col']))
    assert info['occupied'] is True
    assert info['point_count'] == 3
    assert pytest.approx(info['elevation_span'], 0.01) == 3.0


def test_elevation_map_performance_on_large_cloud():
    """Verify Mode A 5cm mapping throughput on 100,000+ points."""
    np.random.seed(42)
    n = 100000
    pts = np.random.uniform(-30, 30, (n, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-2.0, 4.0, n)
    intens = np.random.uniform(0.0, 1.0, n).astype(np.float32)
    pc = PointCloud(points=pts, intensity=intens)
    
    em = ElevationMap25D(resolution_m=0.05, bounds=(-35.0, 35.0, -35.0, 35.0))
    metrics = em.update(pc)
    
    # 100k points should map in under 120ms
    assert metrics['mapped_points'] == n
    assert metrics['mapping_time_ms'] < 120.0
    assert metrics['active_cells'] > 10000


def test_render_elevation_map_layers():
    """Verify top-down rendering of 2.5D elevation map across all supported layers."""
    pts = np.random.uniform(-15, 15, (2000, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-2, 3, 2000)
    pc = PointCloud(points=pts, intensity=np.ones(2000, dtype=np.float32))
    
    em = ElevationMap25D(resolution_m=0.05, bounds=(-20.0, 20.0, -20.0, 20.0))
    em.update(pc)
    
    renderer = PointCloudRenderer(640, 640)
    for layer in ['elevation', 'span', 'occupancy', 'density', 'intensity']:
        img = renderer.render_elevation_map(em, layer=layer, frame_idx=1)
        assert isinstance(img, np.ndarray)
        assert img.shape == (640, 640, 3)
        assert img.dtype == np.uint8
        assert np.any(img > 0)


def test_elevation_map_clear_vs_accumulate():
    """Verify clean single-frame refresh (zero ghosting) vs accumulation mode."""
    pc1 = PointCloud(points=np.array([[1.0, 1.0, 0.5]], dtype=np.float32), intensity=np.array([1.0], dtype=np.float32))
    pc2 = PointCloud(points=np.array([[2.0, 2.0, 0.5]], dtype=np.float32), intensity=np.array([1.0], dtype=np.float32))
    
    em = ElevationMap25D(resolution_m=0.05, bounds=(-10.0, 10.0, -10.0, 10.0))
    
    # 1. Update frame 1 with clear_first=True
    m1 = em.update(pc1, clear_first=True)
    assert m1['active_cells'] == 1
    
    # 2. Update frame 2 with clear_first=True (Default: No ghosting/shadows from previous frame)
    m2 = em.update(pc2, clear_first=True)
    assert m2['active_cells'] == 1
    
    # Verify only (2.0, 2.0) is active, (1.0, 1.0) was cleared
    r1, c1 = em.grid.world_to_grid(np.array([1.0]), np.array([1.0]))
    r2, c2 = em.grid.world_to_grid(np.array([2.0]), np.array([2.0]))
    assert em.point_count[r1[0], c1[0]] == 0
    assert em.point_count[r2[0], c2[0]] == 1
    
    # 3. Test accumulate mode (clear_first=False)
    em.update(pc1, clear_first=False)
    assert em.point_count[r1[0], c1[0]] == 1
    assert em.point_count[r2[0], c2[0]] == 1
    assert np.sum(em.point_count > 0) == 2

