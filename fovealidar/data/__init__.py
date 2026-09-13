"""
Data loading and dataset management for FoveaLiDAR.

Provides interfaces and implementations for loading various LiDAR datasets,
with built-in support for KITTI and extensibility for custom formats.
"""

from .dataset import DatasetBase, DatasetStats
from .kitti.loader import KITTIDataset
from .kitti.frame import KITTIFrame
from .loaders import DatasetFactory, get_dataset

__all__ = [
    'DatasetBase',
    'DatasetStats',
    'KITTIDataset',
    'KITTIFrame',
    'DatasetFactory',
    'get_dataset',
]
