"""
KITTI dataset loader module.

Provides access to KITTI autonomous driving dataset with support for
LiDAR point clouds, odometry, and calibration information.
"""

from .frame import KITTIFrame
from .loader import KITTIDataset
from . import utils

__all__ = [
    'KITTIFrame',
    'KITTIDataset',
    'utils',
]
