"""Phase 3 Tests - Foveated ROI Detection, Compression, and Visualization."""

import pytest
import numpy as np
import tempfile
from pathlib import Path

from fovealidar.core.types import PointCloud
from fovealidar.foveation.gaze_simulator import GazeSimulator
from fovealidar.foveation.roi_definition import ROIDefinition, FovealROI
from fovealidar.foveation.adaptive_compression import AdaptiveCompression
from fovealidar.visualization.renderer import PointCloudRenderer
from fovealidar.visualization.frame_generator import FrameGenerator


@pytest.fixture
def sample_point_cloud():
    """Create sample point cloud for testing."""
    np.random.seed(42)
    n_points = 1000
    x = np.random.uniform(-50, 50, n_points).astype(np.float32)
    y = np.random.uniform(-50, 50, n_points).astype(np.float32)
    z = np.random.uniform(0, 5, n_points).astype(np.float32)
    points = np.column_stack([x, y, z]).astype(np.float32)
    intensity = np.random.uniform(0, 1, n_points).astype(np.float32)
    return PointCloud(points=points, intensity=intensity)


@pytest.fixture
def temp_output_dir():
    """Create temporary output directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


# Gaze Simulation Tests
def test_gaze_simulator_init():
    sim = GazeSimulator(fov_angle_deg=60.0)
    assert sim.fov_angle_deg == 60.0


def test_simulate_driving_gaze(sample_point_cloud):
    sim = GazeSimulator()
    gaze_points = sim.simulate_driving_gaze(sample_point_cloud, num_samples=5)
    assert gaze_points.shape == (5, 3)
    assert np.all(gaze_points[:, 0] > 0)


def test_simulate_saliency_gaze(sample_point_cloud):
    sim = GazeSimulator()
    gaze_points = sim.simulate_saliency_gaze(sample_point_cloud, num_samples=3)
    assert gaze_points.shape[0] <= 3


def test_generate_saliency_map(sample_point_cloud):
    sim = GazeSimulator()
    saliency_map = sim.generate_saliency_map(sample_point_cloud, grid_size=64)
    assert saliency_map.shape == (64, 64)
    assert np.all(saliency_map >= 0) and np.all(saliency_map <= 1)


def test_attention_weight(sample_point_cloud):
    sim = GazeSimulator()
    point = np.array([0, 0, 0], dtype=np.float32)
    gaze = np.array([0, 0, 0], dtype=np.float32)
    assert sim.get_attention_weight(point, gaze, fovea_radius_m=2.0) == 1.0



# ROI Definition Tests
def test_foveal_roi_init():
    center = np.array([0, 0, 1.5], dtype=np.float32)
    roi = FovealROI(center=center, fovea_radius_m=10.0, periphery_radius_m=100.0)
    assert np.array_equal(roi.center, center)


def test_fovea_cells_count():
    roi = FovealROI(center=np.array([0, 0, 0], dtype=np.float32), fovea_radius_m=10.0)
    assert roi.fovea_cells_count() > 100000


def test_roi_definition_init():
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
    assert len(roi_def.rois) == 0


def test_classify_points(sample_point_cloud):
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
    roi_def.add_roi((0, 0, 1.5))
    region_ids, _ = roi_def.classify_points(sample_point_cloud)
    assert np.all(np.isin(region_ids, [0, 1, 2]))


# Compression Tests
def test_adaptive_compression_init():
    roi_def = ROIDefinition()
    compression = AdaptiveCompression(roi_def)
    assert compression.roi_def is roi_def


def test_compress_uniform(sample_point_cloud):
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
    roi_def.add_roi((0, 0, 1.5))
    compression = AdaptiveCompression(roi_def)
    compressed = compression.compress_adaptive(sample_point_cloud, strategy='uniform')
    assert len(compressed.points) <= len(sample_point_cloud.points)


def test_compress_voxel(sample_point_cloud):
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
    roi_def.add_roi((0, 0, 1.5))
    compression = AdaptiveCompression(roi_def)
    compressed = compression.compress_adaptive(sample_point_cloud, strategy='voxel')
    assert len(compressed.points) <= len(sample_point_cloud.points)


# Visualization Tests
def test_renderer_init():
    renderer = PointCloudRenderer(1280, 720)
    assert renderer.width == 1280


def test_render_bev(sample_point_cloud):
    renderer = PointCloudRenderer(1280, 720)
    image = renderer.render_bev(sample_point_cloud, color_mode='intensity')
    assert image.shape == (720, 1280, 3)


def test_frame_generator_init(temp_output_dir):
    gen = FrameGenerator(temp_output_dir)
    assert gen.output_dir.exists()


def test_generate_frame(sample_point_cloud, temp_output_dir):
    gen = FrameGenerator(temp_output_dir)
    frame_path = gen.generate_frame(sample_point_cloud, frame_id=0)
    assert Path(frame_path).exists()


# Integration Tests
def test_full_phase3_pipeline(sample_point_cloud, temp_output_dir):
    sim = GazeSimulator()
    gaze_points = sim.simulate_driving_gaze(sample_point_cloud, num_samples=3)
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
    roi_def.add_roi(tuple(gaze_points[0]))
    compression = AdaptiveCompression(roi_def)
    compressed = compression.compress_adaptive(sample_point_cloud)
    region_ids, _ = roi_def.classify_points(sample_point_cloud)
    gen = FrameGenerator(temp_output_dir)
    frame_path = gen.generate_frame_with_roi(sample_point_cloud, region_ids, frame_id=0)
    assert Path(frame_path).exists()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])


def test_gaze_history(sample_point_cloud):
    sim = GazeSimulator()
    sim.simulate_driving_gaze(sample_point_cloud, num_samples=3)
    assert sim.get_gaze_history().shape[0] >= 3
    sim.clear_history()
    assert sim.get_gaze_history().shape[0] == 0
