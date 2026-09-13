"""KITTI Raw dataset loader (cvlibs format)."""

from pathlib import Path
from typing import Optional, List
import numpy as np

from fovealidar.core.logging import get_logger
from fovealidar.data.dataset import DatasetBase, DatasetStats
from fovealidar.data.kitti.frame import KITTIFrame
from fovealidar.core.types import PointCloud
from fovealidar.data.kitti.raw_utils import (
    load_velodyne_raw, 
    parse_oxts_data,
    validate_raw_kitti_directory
)


logger = get_logger(__name__)


class RawKITTIDataset(DatasetBase):
    """KITTI Raw dataset loader (cvlibs downloads).
    
    Supports directory structure:
        2011_09_26/
        ├── 2011_09_26_drive_0011_sync/
        │   ├── velodyne_points/data/     (*.bin files)
        │   ├── oxts/                     (GPS/IMU)
        │   ├── times.txt
        │   └── ...
    """
    
    def __init__(self, root: Path, drive: Optional[str] = None, cache_size: int = 100):
        """Initialize Raw KITTI loader.
        
        Args:
            root: Root path containing date folders
            drive: Specific drive to load (None = first available)
            cache_size: Frames to cache in memory
        """
        self.root = Path(root)
        self.cache_size = cache_size
        self.cache = {}
        
        # Validate directory
        is_valid, warnings = validate_raw_kitti_directory(self.root)
        for w in warnings:
            logger.warning(w)
        if not is_valid:
            raise ValueError(f"Invalid KITTI raw directory: {self.root}")
        
        # Find drives
        self.drives = self._find_drives()
        if not self.drives:
            raise ValueError("No valid KITTI drives found")
        
        # Select drive
        if drive:
            if drive not in self.drives:
                raise ValueError(f"Drive {drive} not found")
            self.current_drive = drive
        else:
            self.current_drive = self.drives[0]
        
        self._load_drive()
        logger.info(f"Loaded Raw KITTI: {self.current_drive} ({len(self)} frames)")
    
    def _find_drives(self) -> List[str]:
        """Find all valid drive folders recursively, return full paths as strings."""
        drives = []
        
        def search_drives(path: Path, depth: int = 0, max_depth: int = 5) -> list:
            """Recursively search for drive folders."""
            found = []
            if depth > max_depth:
                return found
            
            try:
                for item in path.iterdir():
                    if not item.is_dir():
                        continue
                    
                    # Check if this is a valid drive folder
                    if 'drive' in item.name and 'sync' in item.name:
                        velo_dir = item / 'velodyne_points' / 'data'
                        if velo_dir.exists() and list(velo_dir.glob('*.bin')):
                            found.append(item)
                    
                    # Recurse deeper
                    found.extend(search_drives(item, depth + 1, max_depth))
            except PermissionError:
                pass
            
            return found
        
        # Search from root
        drive_paths = search_drives(self.root)
        
        # Convert to full path strings and sort
        drives = sorted([str(p) for p in drive_paths])
        
        return drives

    
    def _load_drive(self):
        """Load current drive metadata."""
        self.drive_path = None
        
        # current_drive is now a full path string
        drive_path_candidate = Path(self.current_drive)
        if drive_path_candidate.exists() and drive_path_candidate.is_dir():
            self.drive_path = drive_path_candidate
        
        if not self.drive_path:
            raise ValueError(f"Drive path not found: {self.current_drive}")
        
        # Load frames
        velo_dir = self.drive_path / 'velodyne_points' / 'data'
        self.frames = sorted(velo_dir.glob('*.bin'))
        
        if not self.frames:
            raise ValueError(f"No velodyne files in {velo_dir}")
        
        # Load timestamps
        self.timestamps = {}
        times_file = self.drive_path / 'times.txt'
        if times_file.exists():
            with open(times_file) as f:
                self.timestamps = {i: float(line.strip()) 
                                  for i, line in enumerate(f)}
        
        # Load poses
        self.poses = {}
        oxts_dir = self.drive_path / 'oxts'
        if oxts_dir.exists():
            self.poses, ts = parse_oxts_data(oxts_dir)
            if not self.timestamps:
                self.timestamps = ts
        
        self.cache = {}
    
    def __len__(self) -> int:
        """Get number of frames."""
        return len(self.frames)
    
    def __getitem__(self, frame_id: int) -> KITTIFrame:
        """Get frame by ID."""
        if not 0 <= frame_id < len(self):
            raise IndexError(f"Frame {frame_id} out of range")
        
        if frame_id in self.cache:
            return self.cache[frame_id]
        
        # Load from disk
        velo_dir = self.drive_path / 'velodyne_points' / 'data'
        points = load_velodyne_raw(velo_dir, frame_id)
        
        pc = PointCloud(
            points=points[:, :3].astype(np.float32),
            intensity=points[:, 3].astype(np.float32)
        )
        
        timestamp = self.timestamps.get(frame_id, frame_id * 0.1)
        pose = self.poses.get(frame_id, np.eye(4, dtype=np.float64))
        
        frame = KITTIFrame(
            frame_id=frame_id,
            sequence=self.current_drive,
            timestamp=timestamp,
            point_cloud=pc,
            pose=pose
        )
        
        self.cache[frame_id] = frame
        if len(self.cache) > self.cache_size:
            oldest = min(self.cache.keys())
            del self.cache[oldest]
        
        return frame
    
    def iterate_frames(self, start: int = 0, end: Optional[int] = None, 
                      stride: int = 1):
        """Iterate through frames."""
        end = end or len(self)
        for frame_id in range(start, min(end, len(self)), stride):
            yield self[frame_id]
    
    def list_sequences(self) -> List[str]:
        """Get available drives."""
        return self.drives
    
    def get_sequence_length(self, sequence: str) -> int:
        """Get number of frames in a sequence."""
        if sequence == self.current_drive:
            return len(self)
        else:
            old_drive = self.current_drive
            try:
                self.switch_drive(sequence)
                length = len(self)
                self.switch_drive(old_drive)
                return length
            except:
                self.switch_drive(old_drive)
                raise
    
    def switch_drive(self, drive: str):
        """Switch to different drive."""
        if drive not in self.drives:
            raise ValueError(f"Drive {drive} not available")
        self.current_drive = drive
        self._load_drive()
        logger.info(f"Switched to drive: {drive}")

    
    def get_stats(self) -> DatasetStats:
        """Get dataset statistics."""
        sample_size = min(10, len(self))
        sample_indices = np.linspace(0, len(self)-1, sample_size, dtype=int)
        
        all_points = []
        for idx in sample_indices:
            frame = self[idx]
            all_points.append(frame.point_cloud.points)
        
        points = np.vstack(all_points) if all_points else np.empty((0, 3))
        
        stats = DatasetStats(
            dataset_type="KITTI Raw",
            sequences=[self.current_drive],
            num_frames=len(self),
            total_points=int(np.sum([len(self[i].point_cloud.points) 
                                    for i in sample_indices])),
            avg_points_per_frame=int(np.mean([len(self[i].point_cloud.points) 
                                             for i in sample_indices])),
            min_points_in_frame=int(np.min([len(self[i].point_cloud.points) 
                                           for i in sample_indices])),
            max_points_in_frame=int(np.max([len(self[i].point_cloud.points) 
                                           for i in sample_indices])),
            spatial_bounds={
                'x_min': float(points[:, 0].min()) if len(points) > 0 else 0,
                'x_max': float(points[:, 0].max()) if len(points) > 0 else 0,
                'y_min': float(points[:, 1].min()) if len(points) > 0 else 0,
                'y_max': float(points[:, 1].max()) if len(points) > 0 else 0,
                'z_min': float(points[:, 2].min()) if len(points) > 0 else 0,
                'z_max': float(points[:, 2].max()) if len(points) > 0 else 0,
            },
            temporal_duration_s=float(list(self.timestamps.values())[-1] 
                                     if self.timestamps else len(self) * 0.1)
        )
        return stats
    
    def validate(self) -> bool:
        """Validate dataset integrity."""
        try:
            self[0]
            self[len(self) - 1]
            return True
        except Exception as e:
            logger.error(f"Validation failed: {e}")
            return False

