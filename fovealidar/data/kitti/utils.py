"""
KITTI dataset utilities for parsing files and handling data formats.

Provides functions for:
- Loading binary point cloud files (.bin format)
- Parsing calibration files
- Parsing poses and odometry
- Parsing timestamps
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger

logger = get_logger(__name__)


def load_point_cloud_bin(bin_path: Path) -> Tuple[NDArray[np.float32], NDArray[np.float32]]:
    """
    Load point cloud from binary file (.bin format used by KITTI).
    
    Each point is 4 float32 values (x, y, z, intensity).
    
    Args:
        bin_path: Path to .bin file
        
    Returns:
        Tuple of (points, intensity) arrays
        - points: (N, 3) array of [x, y, z]
        - intensity: (N,) array of intensity values
    """
    if not bin_path.exists():
        raise FileNotFoundError(f"Point cloud file not found: {bin_path}")
    
    try:
        scan = np.fromfile(bin_path, dtype=np.float32)
    except Exception as e:
        raise ValueError(f"Failed to read binary file {bin_path}: {e}")
    
    if scan.size % 4 != 0:
        raise ValueError(f"File size must be multiple of 4, got {scan.size} floats")
    
    scan = scan.reshape((-1, 4))
    points = scan[:, :3]
    intensity = scan[:, 3]
    
    return points.astype(np.float32), intensity.astype(np.float32)


def parse_timestamps(times_path: Path) -> List[float]:
    """
    Parse timestamps from times.txt file.
    
    Format: one float per line (seconds)
    
    Args:
        times_path: Path to times.txt
        
    Returns:
        List of timestamps in seconds
    """
    if not times_path.exists():
        logger.warning(f"Timestamps file not found: {times_path}")
        return []
    
    try:
        timestamps = []
        with open(times_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    timestamps.append(float(line))
        return timestamps
    except ValueError as e:
        raise ValueError(f"Invalid timestamp format in {times_path}: {e}")


def parse_poses(poses_path: Path) -> Optional[List[NDArray[np.float64]]]:
    """
    Parse poses from poses.txt file.
    
    Format: 12 values per line representing 3x4 pose matrix
    
    Args:
        poses_path: Path to poses.txt
        
    Returns:
        List of 4x4 transformation matrices, or None if file doesn't exist
    """
    if not poses_path.exists():
        logger.debug(f"Poses file not found: {poses_path}")
        return None
    
    try:
        poses = []
        with open(poses_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    values = [float(x) for x in line.split()]
                    if len(values) != 12:
                        raise ValueError(f"Expected 12 values, got {len(values)}")
                    
                    pose_matrix = np.eye(4, dtype=np.float64)
                    pose_matrix[:3, :] = np.array(values).reshape(3, 4)
                    poses.append(pose_matrix)
        
        return poses
    except ValueError as e:
        raise ValueError(f"Invalid pose format in {poses_path}: {e}")


def validate_sequence_directory(sequence_dir: Path) -> Tuple[bool, List[str]]:
    """
    Validate that a sequence directory has all required files.
    
    Args:
        sequence_dir: Path to sequence directory
        
    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    
    velodyne_dir = sequence_dir / 'velodyne'
    if not velodyne_dir.exists():
        return False, [f"Missing velodyne directory"]
    
    bin_files = list(velodyne_dir.glob('*.bin'))
    if not bin_files:
        return False, [f"No .bin files in velodyne"]
    
    times_file = sequence_dir / 'times.txt'
    if not times_file.exists():
        return False, [f"Missing times.txt"]
    
    poses_file = sequence_dir / 'poses.txt'
    if not poses_file.exists():
        warnings.append("Optional poses.txt not found")
    
    return True, warnings


def get_frame_count(sequence_dir: Path) -> int:
    """Get number of frames in sequence."""
    velodyne_dir = sequence_dir / 'velodyne'
    return len(list(velodyne_dir.glob('*.bin')))


def get_frame_path(sequence_dir: Path, frame_id: int) -> Path:
    """Get path to specific frame's binary file."""
    return sequence_dir / 'velodyne' / f'{frame_id:06d}.bin'

