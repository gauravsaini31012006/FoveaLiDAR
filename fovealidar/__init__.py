"""
FoveaLiDAR — Adaptive Variable Resolution 2.5D LiDAR Mapping for Dynamic Environment Perception

A modular desktop application for processing sequential KITTI Velodyne LiDAR data
with distance-adaptive 2.5D mapping and foveated perception.
"""

__version__ = "0.1.0"
__author__ = "FoveaLiDAR Team"
__license__ = "MIT"

from fovealidar.core.config import Config, load_config
from fovealidar.core.logging import setup_logging, get_logger

__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "get_logger",
]