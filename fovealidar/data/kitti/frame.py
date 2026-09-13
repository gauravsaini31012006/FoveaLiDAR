"""
KITTI frame container.

Represents a single frame from KITTI dataset with point cloud and metadata.
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from fovealidar.core.types import PointCloud


@dataclass
class KITTIFrame:
    """
    A single frame from KITTI dataset.
    
    Attributes:
        frame_id: Frame number in sequence (0-indexed)
        sequence: Sequence identifier (e.g., "00")
        timestamp: Frame timestamp in seconds (from times.txt)
        point_cloud: Point cloud data (PointCloud object)
        pose: 4x4 transformation matrix (SE(3)), if available
        velocity: Linear velocity in m/s, if available
    """
    frame_id: int
    sequence: str
    timestamp: float
    point_cloud: PointCloud
    pose: Optional[NDArray[np.float64]] = None
    velocity: Optional[NDArray[np.float64]] = None
    
    def __post_init__(self) -> None:
        """Validate frame data."""
        if self.frame_id < 0:
            raise ValueError(f"frame_id must be non-negative, got {self.frame_id}")
        if self.timestamp < 0:
            raise ValueError(f"timestamp must be non-negative, got {self.timestamp}")
        if not self.sequence:
            raise ValueError("sequence must not be empty")
        if self.point_cloud is None:
            raise ValueError("point_cloud must not be None")
        
        # Validate pose if provided
        if self.pose is not None:
            if self.pose.shape != (4, 4):
                raise ValueError(f"pose must be (4, 4), got {self.pose.shape}")
        
        # Validate velocity if provided
        if self.velocity is not None:
            if self.velocity.shape != (3,):
                raise ValueError(f"velocity must be (3,), got {self.velocity.shape}")
    
    @property
    def num_points(self) -> int:
        """Return number of points in this frame."""
        return len(self.point_cloud.points)
    
    @property
    def has_pose(self) -> bool:
        """Check if pose information is available."""
        return self.pose is not None
    
    @property
    def has_velocity(self) -> bool:
        """Check if velocity information is available."""
        return self.velocity is not None
    
    def get_position(self) -> np.ndarray:
        """
        Get camera/LiDAR position from pose matrix.
        
        Returns:
            3D position vector, or zeros if no pose available
        """
        if self.pose is not None:
            return self.pose[:3, 3]
        return np.zeros(3, dtype=np.float64)
    
    def get_rotation(self) -> np.ndarray:
        """
        Get rotation matrix from pose.
        
        Returns:
            3x3 rotation matrix, or identity if no pose available
        """
        if self.pose is not None:
            return self.pose[:3, :3]
        return np.eye(3, dtype=np.float64)
