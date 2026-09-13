"""
KITTI dataset loader implementation.

Provides complete dataset loading functionality for KITTI sequences with
frame-level access, caching, and metadata management.
"""

from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud
from fovealidar.data.dataset import DatasetBase, DatasetStats

from . import utils
from .frame import KITTIFrame

logger = get_logger(__name__)


class KITTIDataset(DatasetBase):
    """
    KITTI dataset loader for autonomous driving sequences.
    
    Supports loading LiDAR point clouds from 22 sequences (00-21) with
    optional pose information.
    """
    
    def __init__(self, root: Path, sequence: str = "00", cache_size: int = 100):
        """
        Initialize KITTI dataset loader.
        
        Args:
            root: Root directory containing KITTI sequences/
            sequence: Sequence to load (default "00")
            cache_size: Number of frames to cache in memory (default 100)
        """
        super().__init__(root, name="KITTI")
        self.sequence = sequence
        self.cache_size = cache_size
        self._cache: Dict[int, KITTIFrame] = {}
        self._sequence_dir: Optional[Path] = None
        self._frame_count = 0
        self._timestamps: List[float] = []
        self._poses: Optional[List[np.ndarray]] = None
        
        if not self.root.exists():
            raise FileNotFoundError(f"KITTI root directory not found: {self.root}")
        
        self._load_sequence(sequence)
    
    def _load_sequence(self, sequence: str) -> None:
        """Load a KITTI sequence."""
        possible_paths = [
            self.root / "sequences" / sequence,
            self.root / sequence,
            self.root / f"sequence_{sequence}",
        ]
        
        sequence_dir = None
        for path in possible_paths:
            if path.exists():
                sequence_dir = path
                break
        
        if sequence_dir is None:
            raise ValueError(f"Sequence {sequence} not found in {self.root}")
        
        is_valid, warnings = utils.validate_sequence_directory(sequence_dir)
        if not is_valid:
            raise ValueError(f"Invalid KITTI sequence: {warnings}")
        
        for warning in warnings:
            logger.warning(f"KITTI sequence {sequence}: {warning}")
        
        self._sequence_dir = sequence_dir
        self.sequence = sequence
        self._frame_count = utils.get_frame_count(sequence_dir)
        self._timestamps = utils.parse_timestamps(sequence_dir / "times.txt")
        self._poses = utils.parse_poses(sequence_dir / "poses.txt")
        self._cache.clear()
        
        logger.info(f"Loaded sequence {sequence}: {self._frame_count} frames")
    
    def __len__(self) -> int:
        """Return total number of frames in current sequence."""
        return self._frame_count
    
    def __getitem__(self, idx: int) -> KITTIFrame:
        """Get a frame by index."""
        if idx < 0 or idx >= self._frame_count:
            raise IndexError(f"Frame {idx} out of range [0, {self._frame_count - 1}]")
        
        if idx in self._cache:
            return self._cache[idx]
        
        frame = self._load_frame(idx)
        
        if len(self._cache) >= self.cache_size:
            oldest_key = min(self._cache.keys())
            del self._cache[oldest_key]
        
        self._cache[idx] = frame
        return frame
    
    def _load_frame(self, frame_id: int) -> KITTIFrame:
        """Load a single frame from disk."""
        bin_path = utils.get_frame_path(self._sequence_dir, frame_id)
        points, intensity = utils.load_point_cloud_bin(bin_path)
        
        timestamp = 0.0
        if frame_id < len(self._timestamps):
            timestamp = self._timestamps[frame_id]
        
        point_cloud = PointCloud(
            points=points,
            intensity=intensity,
            frame_id=frame_id,
            timestamp=timestamp,
        )
        
        pose = None
        if self._poses and frame_id < len(self._poses):
            pose = self._poses[frame_id]
        
        frame = KITTIFrame(
            frame_id=frame_id,
            sequence=self.sequence,
            timestamp=timestamp,
            point_cloud=point_cloud,
            pose=pose,
        )
        
        return frame
    
    
    def get_stats(self) -> DatasetStats:
        """Get or compute dataset statistics."""
        if self._stats is not None:
            return self._stats
        
        logger.info("Computing dataset statistics...")
        
        sample_size = min(10, self._frame_count)
        sample_indices = np.linspace(0, self._frame_count - 1, sample_size, dtype=int)
        
        total_points = 0
        points_per_frame = []
        spatial_bounds = {"x_min": float("inf"), "x_max": float("-inf"),
                         "y_min": float("inf"), "y_max": float("-inf"),
                         "z_min": float("inf"), "z_max": float("-inf")}
        
        for idx in sample_indices:
            frame = self[int(idx)]
            points = frame.point_cloud.points
            
            total_points += len(points)
            points_per_frame.append(len(points))
            
            spatial_bounds["x_min"] = min(spatial_bounds["x_min"], points[:, 0].min())
            spatial_bounds["x_max"] = max(spatial_bounds["x_max"], points[:, 0].max())
            spatial_bounds["y_min"] = min(spatial_bounds["y_min"], points[:, 1].min())
            spatial_bounds["y_max"] = max(spatial_bounds["y_max"], points[:, 1].max())
            spatial_bounds["z_min"] = min(spatial_bounds["z_min"], points[:, 2].min())
            spatial_bounds["z_max"] = max(spatial_bounds["z_max"], points[:, 2].max())
        
        avg_points = np.mean(points_per_frame)
        total_points_estimate = int(avg_points * self._frame_count)
        
        temporal_duration = 0.0
        if self._timestamps:
            temporal_duration = self._timestamps[-1] - self._timestamps[0]
        
        self._stats = DatasetStats(
            num_frames=self._frame_count,
            sequences=[self.sequence],
            total_points=total_points_estimate,
            avg_points_per_frame=avg_points,
            min_points_in_frame=int(min(points_per_frame)),
            max_points_in_frame=int(max(points_per_frame)),
            spatial_bounds=spatial_bounds,
            temporal_duration_s=temporal_duration,
            dataset_type="KITTI",
        )
        
        return self._stats
    
    def list_sequences(self) -> List[str]:
        """List all available sequences."""
        sequences = []
        
        for variant in ["sequences", ""]:
            base_dir = self.root / variant if variant else self.root
            if not base_dir.exists():
                continue
            
            for item in base_dir.iterdir():
                if not item.is_dir():
                    continue
                
                name = item.name
                if name.isdigit() and len(name) == 2:
                    is_valid, _ = utils.validate_sequence_directory(item)
                    if is_valid:
                        sequences.append(name)
        
        return sorted(list(set(sequences)))
    
    def get_sequence_length(self, sequence: str) -> int:
        """Get number of frames in a sequence."""
        if sequence == self.sequence:
            return self._frame_count
        
        for variant in ["sequences", ""]:
            base_dir = self.root / variant if variant else self.root
            seq_dir = base_dir / sequence
            
            if seq_dir.exists():
                is_valid, _ = utils.validate_sequence_directory(seq_dir)
                if is_valid:
                    return utils.get_frame_count(seq_dir)
        
        raise ValueError(f"Sequence {sequence} not found")
    
    def switch_sequence(self, sequence: str) -> None:
        """Switch to a different sequence."""
        if sequence == self.sequence:
            return
        
        logger.info(f"Switching to sequence {sequence}")
        self._load_sequence(sequence)
    
    def iterate_frames(self, start: int = 0, end: Optional[int] = None,
                      stride: int = 1):
        """Iterate over frames in current sequence."""
        if end is None:
            end = self._frame_count
        
        for idx in range(start, min(end, self._frame_count), stride):
            yield self[idx]
    def validate(self) -> bool:
        """Validate dataset integrity."""
        try:
            if self._frame_count > 0:
                _ = self[0]
                if self._frame_count > 1:
                    _ = self[self._frame_count - 1]
            return True
        except Exception as e:
            logger.error(f"Dataset validation failed: {e}")
            return False


