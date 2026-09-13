"""Tests for Phase 1 Raw KITTI support and Phase 2 Preprocessing."""

import pytest
import tempfile
import numpy as np
from pathlib import Path

from fovealidar.data.kitti.raw_loader import RawKITTIDataset
from fovealidar.preprocessing import (
    PreprocessingConfig,
    PreprocessingPipeline,
    RangeFilter,
    HeightFilter,
    VoxelDownsamplingFilter,
    IntensityNormalizationFilter,
)
from fovealidar.core.types import PointCloud


@pytest.fixture
def mock_kitti_raw_data():
    """Create mock KITTI raw dataset structure."""
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        
        # Create date folder
        date_dir = root / "2011_09_26"
        date_dir.mkdir()
        
        # Create drive folder
        drive_dir = date_dir / "2011_09_26_drive_0001_sync"
        drive_dir.mkdir()
        
        # Create velodyne_points/data
        velo_dir = drive_dir / "velodyne_points" / "data"
        velo_dir.mkdir(parents=True)
        
        # Create oxts folder
        oxts_dir = drive_dir / "oxts"
        oxts_dir.mkdir()
        
        # Create mock point cloud files
        for i in range(5):
            points = np.random.randn(1000, 4).astype(np.float32)
            points[:, 2] = np.abs(points[:, 2])  # Positive Z
            points.tofile(velo_dir / f"{i:010d}.bin")
            
            # Create oxts file
            oxts_data = "48.39 8.43 90.0 0.0 0.0 0.0 0 0 0 0 0 0 0 0 0 0 0"
            (oxts_dir / f"{i:010d}.txt").write_text(oxts_data)
        
        # Create times.txt
        times = "\n".join(str(i * 0.1) for i in range(5))
        (drive_dir / "times.txt").write_text(times)
        
        yield root


class TestRawKITTIDataset:
    """Tests for Raw KITTI dataset loader."""
    
    def test_raw_kitti_init(self, mock_kitti_raw_data):
        """Test Raw KITTI dataset initialization."""
        dataset = RawKITTIDataset(mock_kitti_raw_data)
        assert len(dataset) == 5
        # current_drive is now a full path, check it ends with the drive name
        assert dataset.current_drive.endswith("2011_09_26_drive_0001_sync")
    
    def test_raw_kitti_frame_loading(self, mock_kitti_raw_data):
        """Test loading individual frames."""
        dataset = RawKITTIDataset(mock_kitti_raw_data)
        frame = dataset[0]
        
        assert frame.frame_id == 0
        assert frame.num_points == 1000
        assert frame.timestamp == 0.0


class TestPhase2Preprocessing:
    """Tests for Phase 2 preprocessing pipeline."""
    
    @pytest.fixture
    def sample_point_cloud(self):
        """Create sample point cloud."""
        points = np.array([
            [0, 0, 0],
            [10, 0, 0],
            [50, 0, 0],
            [120, 0, 0],
            [0, 0, -5],
            [0, 0, 5],
        ], dtype=np.float32)
        
        intensity = np.array([100, 150, 200, 50, 75, 180], dtype=np.float32)
        return PointCloud(points=points, intensity=intensity)
    
    def test_range_filter(self, sample_point_cloud):
        """Test range filtering."""
        filter_obj = RangeFilter(range_min=0, range_max=100)
        result = filter_obj.apply(sample_point_cloud)
        assert len(result.points) == 5
    
    def test_height_filter(self, sample_point_cloud):
        """Test height filtering."""
        filter_obj = HeightFilter(height_min=-3, height_max=10)
        result = filter_obj.apply(sample_point_cloud)
        assert len(result.points) == 5
    
    def test_intensity_normalization(self, sample_point_cloud):
        """Test intensity normalization."""
        filter_obj = IntensityNormalizationFilter(intensity_min=0, intensity_max=255)
        result = filter_obj.apply(sample_point_cloud)
        
        assert result.intensity.min() >= 0
        assert result.intensity.max() <= 1
    
    def test_voxel_downsampling(self, sample_point_cloud):
        """Test voxel downsampling."""
        filter_obj = VoxelDownsamplingFilter(voxel_size=10)
        result = filter_obj.apply(sample_point_cloud)
        assert len(result.points) <= len(sample_point_cloud.points)
    
    def test_preprocessing_pipeline(self, sample_point_cloud):
        """Test full preprocessing pipeline."""
        config = PreprocessingConfig(
            range_min=0,
            range_max=100,
            height_min=-3,
            height_max=10,
            normalize_intensity=True,
            remove_outliers=False,
            voxel_size=None,
        )
        
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process(sample_point_cloud)
        
        assert len(result.points) > 0
        assert result.intensity.max() <= 1


class TestIntegration:
    """Integration tests for raw data + preprocessing."""
    
    def test_raw_kitti_to_preprocessing(self, mock_kitti_raw_data):
        """Test loading raw KITTI data and preprocessing."""
        dataset = RawKITTIDataset(mock_kitti_raw_data)
        frame = dataset[0]
        
        config = PreprocessingConfig(
            range_max=100,
            height_min=-5,
            height_max=10,
            normalize_intensity=True,
        )
        pipeline = PreprocessingPipeline(config)
        processed = pipeline.process(frame.point_cloud)
        
        assert len(processed.points) > 0
        assert processed.intensity is not None

        assert frame.has_pose
    
    def test_raw_kitti_iteration(self, mock_kitti_raw_data):
        """Test frame iteration."""
        dataset = RawKITTIDataset(mock_kitti_raw_data)
        
        frames = list(dataset.iterate_frames(start=0, end=3))
        assert len(frames) == 3
        assert frames[0].frame_id == 0
        assert frames[2].frame_id == 2
    
    def test_raw_kitti_stats(self, mock_kitti_raw_data):
        """Test dataset statistics."""
        dataset = RawKITTIDataset(mock_kitti_raw_data)
        stats = dataset.get_stats()
        
        assert stats.num_frames == 5
        assert stats.avg_points_per_frame > 0
        assert stats.dataset_type == "KITTI Raw"
