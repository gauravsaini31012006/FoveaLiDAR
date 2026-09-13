"""Phase 3+6: Foveated ROI Detection, Adaptive Compression, and Foveal Controller.

This module implements foveal vision - mimicking human eye's high-res center (fovea)
and low-res periphery. Supports:
- Gaze-driven attention modeling
- Dual-resolution grids (10m @ 5cm cells, 100m @ 50cm cells)
- 2.5D semantic-aware compression
- Real-time ROI detection
- Phase 6: Multi-dimensional importance scoring and selective refinement
"""

from .gaze_simulator import GazeSimulator
from .roi_definition import FovealROI, ROIDefinition
from .adaptive_compression import AdaptiveCompression
from .foveal_controller import FovealController
from .compute_scheduler import ComputeScheduler, ComputeProfile
from . import scores

__all__ = [
    'GazeSimulator',
    'FovealROI',
    'ROIDefinition',
    'AdaptiveCompression',
    'FovealController',
    'ComputeScheduler',
    'ComputeProfile',
    'scores',
]
