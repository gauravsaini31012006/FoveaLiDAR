"""Tests for Phase 3 Autonomous 3D Visualization and Multi-tier Foveation."""

import pytest
import numpy as np
from fovealidar.core.types import PointCloud
from fovealidar.foveation.roi_definition import ROIDefinition
from fovealidar.foveation.adaptive_compression import AdaptiveCompression
from fovealidar.visualization.renderer import PointCloudRenderer
from fovealidar.visualization.frame_generator import FrameGenerator, VideoComposer


def test_multi_tier_foveation_preserves_background():
    """Verify multi-tier compression preserves far-field points instead of deleting them."""
    np.random.seed(42)
    # Generate points across near (0-10m), mid (10-50m), and far (50-100m)
    near_pts = np.random.uniform(-8, 8, (1000, 3)).astype(np.float32)
    near_pts[:, 2] = np.random.uniform(-1.8, 1.0, 1000)
    
    mid_pts = np.random.uniform(-40, 40, (3000, 3)).astype(np.float32)
    mid_pts[:, 2] = np.random.uniform(-1.8, 4.0, 3000)
    
    far_pts = np.random.uniform(-80, 80, (2000, 3)).astype(np.float32)
    far_pts[:, 2] = np.random.uniform(-1.8, 6.0, 2000)
    
    all_pts = np.vstack([near_pts, mid_pts, far_pts])
    intens = np.random.uniform(0.1, 0.9, len(all_pts)).astype(np.float32)
    pc = PointCloud(points=all_pts, intensity=intens)
    
    roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=50.0)
    roi_def.add_roi((0.0, 0.0, 0.0))
    
    comp = AdaptiveCompression(roi_def, fovea_voxel_m=0.1, periphery_voxel_m=0.5, outside_voxel_m=1.0)
    compressed = comp.compress_adaptive(pc, strategy="voxel", preserve_background=True)
    
    # Should reduce point count while keeping points across all regions
    assert len(compressed.points) < len(pc.points)
    assert len(compressed.points) > 0
    assert compressed.intensity is not None
    assert len(compressed.intensity) == len(compressed.points)
    
    # Check that far points (>50m) still exist
    dists = np.linalg.norm(compressed.points[:, :2], axis=1)
    assert np.any(dists > 50.0)


def test_detect_obstacles():
    """Verify obstacle cluster detection creates 3D bounding boxes."""
    np.random.seed(42)
    # Synthetic obstacle cluster at x=15m, y=1m, z=0m
    cluster = np.random.uniform(-1.0, 1.0, (80, 3)).astype(np.float32)
    cluster[:, 0] += 15.0
    cluster[:, 1] += 1.0
    cluster[:, 2] += -0.5
    
    renderer = PointCloudRenderer(640, 360)
    boxes = renderer.detect_obstacles(cluster)
    assert len(boxes) >= 1
    assert 'center' in boxes[0]
    assert 'extent' in boxes[0]
    assert 'distance' in boxes[0]
    assert 'label' in boxes[0]
    assert 10.0 < boxes[0]['distance'] < 20.0


def test_render_autonomous_3d_modes(tmp_path):
    """Verify render_autonomous_3d executes across all color modes and produces valid RGB images."""
    np.random.seed(42)
    n = 2000
    pts = np.random.uniform(-30, 30, (n, 3)).astype(np.float32)
    pts[:, 2] = np.random.uniform(-2.0, 5.0, n)
    intens = np.random.uniform(0.0, 1.0, n).astype(np.float32)
    pc = PointCloud(points=pts, intensity=intens)
    
    renderer = PointCloudRenderer(640, 360)
    gaze = np.array([20.0, 0.0, 0.0], dtype=np.float32)
    
    for mode in ['autonomous', 'height', 'intensity', 'foveal']:
        img = renderer.render_autonomous_3d(
            pc,
            gaze_pos=gaze,
            frame_idx=1,
            color_mode=mode,
            raw_point_count=n,
        )
        assert isinstance(img, np.ndarray)
        assert img.shape == (360, 640, 3)
        assert img.dtype == np.uint8
        # Make sure it's not all empty black
        assert np.any(img > 0)


def test_frame_generator_and_video_composer(tmp_path):
    """Test generating frames and composing video."""
    frames_dir = tmp_path / "frames"
    fg = FrameGenerator(frames_dir, 320, 180)
    
    # Generate 3 dummy frames
    for i in range(3):
        dummy_img = np.full((180, 320, 3), (i * 50, 100, 150), dtype=np.uint8)
        fg.generate_frame(dummy_img, i)
    
    assert fg.get_frame_count() == 3
    assert len(list(frames_dir.glob("frame_*.png"))) == 3
    
    # Compose video
    vc = VideoComposer(fps=10)
    video_out = tmp_path / "test.mp4"
    success = vc.compose_video(frames_dir, video_out, fps=10)
    assert success
    assert video_out.exists()
    assert video_out.stat().st_size > 0
