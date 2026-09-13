"""KITTI Raw format support (for cvlibs raw downloads)."""

import struct
from pathlib import Path
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger


logger = get_logger(__name__)


def parse_oxts_data(oxts_dir: Path) -> Tuple[dict, dict]:
    """Parse KITTI raw oxts (GPS/IMU) data to extract poses.
    
    Converts GPS/IMU data to approximate vehicle poses.
    Returns dict of frame_id -> pose matrix.
    
    Args:
        oxts_dir: Path to oxts/ directory
        
    Returns:
        (poses_dict, timestamps_dict) where poses are 4x4 SE(3) matrices
    """
    poses = {}
    timestamps = {}
    
    # List all .txt files in oxts, excluding metadata files
    all_txts = sorted(oxts_dir.glob("*.txt"))
    oxts_files = [f for f in all_txts 
                  if f.name not in ['dataformat.txt', 'timestamps.txt', 'timestamps_start.txt', 'timestamps_end.txt']]
    
    if not oxts_files:
        logger.warning(f"No OXTS data files found in {oxts_dir}")
        return poses, timestamps
    
    # Parse timestamps from timestamps.txt if exists (in parent, not oxts)
    times_file = oxts_dir.parent / "times.txt"
    if times_file.exists():
        try:
            with open(times_file) as f:
                timestamps = {i: float(line.strip()) for i, line in enumerate(f) if line.strip()}
        except Exception as e:
            logger.warning(f"Failed to parse times.txt: {e}")
    
    # Parse GPS/IMU data
    for frame_id, oxts_file in enumerate(oxts_files):
        try:
            with open(oxts_file) as f:
                # OXTS format: lat, lon, alt, roll, pitch, yaw, vf, vl, vu, af, al, au, wf, wl, wu, pos_accuracy, ori_accuracy
                data = list(map(float, f.read().split()))
            
            if len(data) >= 6:
                # Extract position and rotation (simplified)
                # In real scenario, would do proper GPS to local coordinate conversion
                lat, lon, alt = data[0:3]
                roll, pitch, yaw = data[3:6]
                
                # Create simple 4x4 pose matrix from yaw (simplified - real conversion needed)
                pose = _create_pose_from_angles(roll, pitch, yaw, [0, 0, alt])
                poses[frame_id] = pose
                
                if frame_id not in timestamps:
                    timestamps[frame_id] = frame_id * 0.1  # Default 10 Hz
                    
        except Exception as e:
            logger.debug(f"Error parsing OXTS file {oxts_file}: {e}")
            # Create identity pose as fallback
            poses[frame_id] = np.eye(4)
    
    return poses, timestamps


def _create_pose_from_angles(roll: float, pitch: float, yaw: float, 
                            position: list) -> NDArray[np.float64]:
    """Create 4x4 SE(3) pose matrix from Euler angles and position."""
    # Rotation matrix from Euler angles (ZYX order)
    cy = np.cos(yaw)
    sy = np.sin(yaw)
    cp = np.cos(pitch)
    sp = np.sin(pitch)
    cr = np.cos(roll)
    sr = np.sin(roll)
    
    R = np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp,   cp*sr,            cp*cr           ]
    ], dtype=np.float64)
    
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = R
    pose[:3, 3] = position
    return pose


def load_velodyne_raw(velodyne_dir: Path, frame_id: int) -> NDArray[np.float32]:
    """Load point cloud from KITTI raw velodyne_points/data/.
    
    Args:
        velodyne_dir: Path to velodyne_points/data/ directory
        frame_id: Frame number (zero-padded to 10 digits)
        
    Returns:
        Point cloud array (N, 4) with x, y, z, intensity
    """
    bin_file = velodyne_dir / f"{frame_id:010d}.bin"
    
    if not bin_file.exists():
        raise FileNotFoundError(f"Point cloud file not found: {bin_file}")
    
    # KITTI format: 4 float32 values per point (x, y, z, intensity)
    points = np.fromfile(bin_file, dtype=np.float32).reshape(-1, 4)
    return points


def validate_raw_kitti_directory(root: Path) -> Tuple[bool, list]:
    """Validate KITTI raw dataset directory structure.
    
    Handles nested structures at any depth.
    
    Args:
        root: Root path to KITTI raw dataset
        
    Returns:
        (is_valid, warnings) tuple
    """
    warnings = []
    
    def find_drive_dirs(path: Path, depth: int = 0, max_depth: int = 5) -> list:
        """Recursively search for valid drive directories."""
        drive_dirs = []
        if depth > max_depth:
            return drive_dirs
        
        try:
            for item in path.iterdir():
                if not item.is_dir():
                    continue
                
                # Check if this is a drive folder
                if 'drive' in item.name and 'sync' in item.name:
                    velo_path = item / 'velodyne_points' / 'data'
                    if velo_path.exists() and list(velo_path.glob('*.bin')):
                        drive_dirs.append(item)
                
                # Recurse into subdirectories
                drive_dirs.extend(find_drive_dirs(item, depth + 1, max_depth))
        except PermissionError:
            pass
        
        return drive_dirs
    
    # Find all valid drive directories recursively
    drive_dirs = find_drive_dirs(root)
    
    if not drive_dirs:
        warnings.append(f"No valid drive folders found in {root}")
        return False, warnings
    
    # Check required subdirectories in first drive
    first_drive = drive_dirs[0]
    required = ['velodyne_points', 'oxts']
    optional = ['image_00', 'image_01', 'image_02', 'image_03']
    
    for req in required:
        if not (first_drive / req).exists():
            warnings.append(f"Missing required directory: {req}")
            return False, warnings
    
    for opt in optional:
        if not (first_drive / opt).exists():
            warnings.append(f"Missing optional camera directory: {opt}")
    
    # Check velodyne data
    velo_data = first_drive / 'velodyne_points' / 'data'
    if not velo_data.exists():
        warnings.append("Missing velodyne_points/data directory")
        return False, warnings
    
    bin_files = list(velo_data.glob('*.bin'))
    if not bin_files:
        warnings.append("No .bin files in velodyne_points/data")
        return False, warnings
    
    logger.info(f"KITTI Raw validation: Found {len(drive_dirs)} drive(s), "
                f"{len(bin_files)} frames in first drive at {first_drive}")
    
    return True, warnings
