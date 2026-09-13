"""
Core package initialization for FoveaLiDAR.
"""

from fovealidar.core.config import Config, load_config
from fovealidar.core.logging import setup_logging, get_logger
from fovealidar.core.types import (
    PointCloud,
    PointCloudArray,
    BoundingBox2D,
    BoundingBox3D,
    MapCell,
    MapResolution,
    ProcessingMetrics,
    ComputeMode,
    MapMode,
)

__all__ = [
    "Config",
    "load_config",
    "setup_logging",
    "get_logger",
    "PointCloud",
    "PointCloudArray",
    "BoundingBox2D",
    "BoundingBox3D",
    "MapCell",
    "MapResolution",
    "ProcessingMetrics",
    "ComputeMode",
    "MapMode",
]