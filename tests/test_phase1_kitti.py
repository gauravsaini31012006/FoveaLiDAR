"""
Unit tests for Phase 1: KITTI Dataset Engine.
"""

import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

from fovealidar.core.types import PointCloud
from fovealidar.data import KITTIDataset, KITTIFrame
from fovealidar.data.kitti import utils


@pytest.fixture
def mock_kitti_dataset() -> Tuple[Path, str]:
    """Create a temporary mock KITTI dataset for testing."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sequence = "00"
        
        seq_dir = root / "sequences" / sequence
        seq_dir.mkdir(parents=True, exist_ok=True)
        velodyne_dir = seq_dir / "velodyne"
        velodyne_dir.mkdir(exist_ok=True)
        
        num_frames = 5
        for frame_id in range(num_frames):
            points = np.random.randn(1000, 3).astype(np.float32)
            intensity = np.random.rand(1000).astype(np.float32) * 255
            data = np.column_stack([points, intensity]).astype(np.float32)
            bin_path = velodyne_dir / f"{frame_id:06d}.bin"
            data.tofile(bin_path)
        
        times = [i * 0.1 for i in range(num_frames)]
        times_path = seq_dir / "times.txt"
        with open(times_path, 'w') as f:
            for t in times:
                f.write(f"{t:.6f}\n")
        
        poses_path = seq_dir / "poses.txt"
        with open(poses_path, 'w') as f:
            for frame_id in range(num_frames):
                vals = [1, 0, 0, frame_id * 0.1, 0, 1, 0, 0, 0, 0, 1, 0]
                f.write(" ".join(map(str, vals)) + "\n")
        
        yield root, sequence


def test_kitti_init(mock_kitti_dataset):
    """Test dataset initialization."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    assert len(dataset) == 5


def test_kitti_invalid_root():
    """Test invalid root."""
    with pytest.raises(FileNotFoundError):
        KITTIDataset(Path("/nonexistent"))


def test_kitti_invalid_sequence(mock_kitti_dataset):
    """Test invalid sequence."""
    root, _ = mock_kitti_dataset
    with pytest.raises(ValueError):
        KITTIDataset(root, sequence="99")


def test_frame_loading(mock_kitti_dataset):
    """Test frame loading."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    frame = dataset[0]
    assert frame.num_points == 1000


def test_frame_iteration(mock_kitti_dataset):
    """Test frame iteration."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    frames = list(dataset.iterate_frames())
    assert len(frames) == 5


def test_frame_out_of_range(mock_kitti_dataset):
    """Test out of range access."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    with pytest.raises(IndexError):
        _ = dataset[10]


def test_point_cloud_structure(mock_kitti_dataset):
    """Test point cloud structure."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    frame = dataset[0]
    assert frame.point_cloud.points.shape == (1000, 3)
    assert frame.point_cloud.intensity.shape == (1000,)


def test_timestamps(mock_kitti_dataset):
    """Test timestamps."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    frame = dataset[2]
    assert abs(frame.timestamp - 0.2) < 1e-6


def test_poses(mock_kitti_dataset):
    """Test poses."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    frame = dataset[0]
    assert frame.has_pose
    assert frame.pose.shape == (4, 4)


def test_dataset_stats(mock_kitti_dataset):
    """Test dataset stats."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    stats = dataset.get_stats()
    assert stats.num_frames == 5


def test_utils_load_bin(mock_kitti_dataset):
    """Test binary loading."""
    root, sequence = mock_kitti_dataset
    bin_path = root / "sequences" / sequence / "velodyne" / "000000.bin"
    points, intensity = utils.load_point_cloud_bin(bin_path)
    assert points.shape == (1000, 3)


def test_utils_validate(mock_kitti_dataset):
    """Test validation."""
    root, sequence = mock_kitti_dataset
    seq_dir = root / "sequences" / sequence
    is_valid, _ = utils.validate_sequence_directory(seq_dir)
    assert is_valid


def test_full_workflow(mock_kitti_dataset):
    """Test complete workflow."""
    root, sequence = mock_kitti_dataset
    dataset = KITTIDataset(root, sequence=sequence)
    assert dataset.validate()
    assert dataset.get_stats().num_frames == 5
