"""
Type definitions and contracts for FoveaLiDAR.

Defines all core data structures, enums, and type aliases used throughout
the application to ensure type safety and clear interfaces.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from numpy.typing import NDArray


# ==============================================================================
# Type Aliases
# ==============================================================================

# Point cloud: N x 4 array of [x, y, z, intensity]
PointCloudArray = NDArray[np.float32]  # Shape: (N, 4)

# 3D point array: N x 3
PointArray3D = NDArray[np.float32]  # Shape: (N, 3)

# 2D point array: N x 2
PointArray2D = NDArray[np.float32]  # Shape: (N, 2)

# Intensity array: N
IntensityArray = NDArray[np.float32]  # Shape: (N,)

# Label array: N
LabelArray = NDArray[np.int32]  # Shape: (N,)

# Cell index array: N x 2 (row, col)
CellIndexArray = NDArray[np.int32]  # Shape: (N, 2)

# Boolean mask: N
BoolMask = NDArray[np.bool_]  # Shape: (N,)

# Float mask/weight: N
FloatMask = NDArray[np.float32]  # Shape: (N,)


# ==============================================================================
# Enums
# ==============================================================================

class ComputeMode(str, Enum):
    """Compute operation modes."""
    MAX_QUALITY = "max_quality"
    BALANCED = "balanced"
    LOW_POWER = "low_power"
    CPU_ONLY = "cpu_only"


class MapMode(str, Enum):
    """Mapping modes."""
    UNIFORM_5CM = "uniform_5cm"
    DISTANCE_ADAPTIVE = "distance_adaptive"
    FOVEAL = "foveal"


class SemanticBackend(str, Enum):
    """Semantic segmentation backends."""
    NONE = "none"
    HEURISTIC = "heuristic"
    SALSA_NEXT = "salsanext"


class DetectorBackend(str, Enum):
    """Object detection backends."""
    NONE = "none"
    CENTER_POINT = "centerpoint"
    POINT_PILLARS = "pointpillars"


class ResolutionLevel(int, Enum):
    """Map resolution levels (in cm)."""
    COARSE_50CM = 50
    MEDIUM_20CM = 20
    FINE_10CM = 10
    FINEST_5CM = 5


# ==============================================================================
# Data Structures
# ==============================================================================

@dataclass(frozen=True)
class PointCloud:
    """
    Immutable point cloud container.
    
    Attributes:
        points: N x 3 array of [x, y, z] coordinates
        intensity: N array of intensity/remission values
        labels: Optional N array of semantic labels
        timestamp: Optional timestamp
        frame_id: Optional frame identifier
    """
    points: PointArray3D
    intensity: IntensityArray
    labels: Optional[LabelArray] = None
    timestamp: Optional[float] = None
    frame_id: Optional[int] = None
    
    def __post_init__(self) -> None:
        """Validate array shapes."""
        if self.points.ndim != 2 or self.points.shape[1] != 3:
            raise ValueError(f"points must be (N, 3), got {self.points.shape}")
        if self.intensity.ndim != 1 or len(self.intensity) != len(self.points):
            raise ValueError(f"intensity must be (N,), got {self.intensity.shape}")
        if self.labels is not None:
            if self.labels.ndim != 1 or len(self.labels) != len(self.points):
                raise ValueError(f"labels must be (N,), got {self.labels.shape}")
    
    @property
    def num_points(self) -> int:
        """Number of points in the cloud."""
        return len(self.points)
    
    @property
    def xy(self) -> PointArray2D:
        """Get XY coordinates only."""
        return self.points[:, :2]
    
    @property
    def z(self) -> NDArray[np.float32]:
        """Get Z coordinates only."""
        return self.points[:, 2]
    
    def to_array(self) -> PointCloudArray:
        """Convert to Nx4 array [x, y, z, intensity]."""
        return np.column_stack([self.points, self.intensity]).astype(np.float32)
    
    @classmethod
    def from_array(cls, array: PointCloudArray, **kwargs) -> PointCloud:
        """Create PointCloud from Nx4 array."""
        if array.ndim != 2 or array.shape[1] < 4:
            raise ValueError(f"array must be (N, >=4), got {array.shape}")
        return cls(
            points=array[:, :3].astype(np.float32),
            intensity=array[:, 3].astype(np.float32),
            **kwargs
        )


@dataclass(frozen=True)
class BoundingBox2D:
    """2D axis-aligned bounding box."""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    
    @property
    def width(self) -> float:
        return self.x_max - self.x_min
    
    @property
    def height(self) -> float:
        return self.y_max - self.y_min
    
    @property
    def center(self) -> Tuple[float, float]:
        return ((self.x_min + self.x_max) / 2, (self.y_min + self.y_max) / 2)
    
    def contains(self, x: float, y: float) -> bool:
        return self.x_min <= x <= self.x_max and self.y_min <= y <= self.y_max


@dataclass(frozen=True)
class BoundingBox3D:
    """3D axis-aligned bounding box."""
    x_min: float
    y_min: float
    z_min: float
    x_max: float
    y_max: float
    z_max: float
    
    @property
    def center(self) -> Tuple[float, float, float]:
        return (
            (self.x_min + self.x_max) / 2,
            (self.y_min + self.y_max) / 2,
            (self.z_min + self.z_max) / 2
        )
    
    @property
    def dimensions(self) -> Tuple[float, float, float]:
        return (
            self.x_max - self.x_min,
            self.y_max - self.y_min,
            self.z_max - self.z_min
        )


@dataclass
class MapCell:
    """
    A single cell in the 2.5D elevation/occupancy map.
    
    Attributes:
        row: Grid row index
        col: Grid column index
        resolution: Cell resolution in meters
        elevation_min: Minimum elevation in cell
        elevation_max: Maximum elevation in cell
        elevation_mean: Mean elevation in cell
        occupancy: Occupancy probability [0, 1]
        point_count: Number of points in cell
        semantic_label: Dominant semantic label
        semantic_confidence: Confidence of semantic label [0, 1]
        uncertainty: Uncertainty estimate [0, 1]
        importance: Foveal importance score [0, 1]
        resolution_level: Resolution level index
        parent_cell: Optional parent cell (row, col) if refined
        children: List of child cell indices if this cell is refined
    """
    row: int
    col: int
    resolution: float
    elevation_min: float = 0.0
    elevation_max: float = 0.0
    elevation_mean: float = 0.0
    occupancy: float = 0.0
    point_count: int = 0
    semantic_label: int = 0
    semantic_confidence: float = 0.0
    uncertainty: float = 0.0
    importance: float = 0.0
    resolution_level: int = 0
    parent_cell: Optional[Tuple[int, int]] = None
    children: List[Tuple[int, int]] = field(default_factory=list)
    
    @property
    def elevation_range(self) -> float:
        return self.elevation_max - self.elevation_min
    
    @property
    def is_refined(self) -> bool:
        return len(self.children) > 0
    
    @property
    def is_active(self) -> bool:
        """A cell is active if it's not refined (children are active instead)."""
        return not self.is_refined


@dataclass(frozen=True)
class MapResolution:
    """
    Resolution configuration for a map region.
    
    Attributes:
        resolution_m: Resolution in meters
        level: Resolution level index
        distance_range: (min_dist, max_dist) in meters where this resolution applies
    """
    resolution_m: float
    level: int
    distance_range: Tuple[float, float]


@dataclass
class ProcessingMetrics:
    """
    Timing and performance metrics for a processing frame.
    
    All times in milliseconds.
    """
    # Stage timings
    io_time_ms: float = 0.0
    preprocessing_time_ms: float = 0.0
    geometry_time_ms: float = 0.0
    perception_time_ms: float = 0.0
    foveation_time_ms: float = 0.0
    mapping_time_ms: float = 0.0
    rendering_time_ms: float = 0.0
    total_time_ms: float = 0.0
    
    # Counters
    input_points: int = 0
    filtered_points: int = 0
    active_cells: int = 0
    fine_cells: int = 0
    medium_cells: int = 0
    coarse_cells: int = 0
    refined_regions: int = 0
    
    # Memory
    ram_usage_mb: float = 0.0
    vram_usage_mb: float = 0.0
    
    # GPU
    gpu_utilization: float = 0.0
    
    @property
    def processing_fps(self) -> float:
        if self.total_time_ms > 0:
            return 1000.0 / self.total_time_ms
        return 0.0
    
    @property
    def cell_reduction_percent(self) -> float:
        """Percentage reduction vs uniform 5cm baseline."""
        total_uniform = self.fine_cells + self.medium_cells + self.coarse_cells
        if total_uniform > 0:
            # Estimate uniform cells at 5cm for same area
            uniform_estimate = total_uniform * 100  # Rough estimate
            return (1 - total_uniform / uniform_estimate) * 100
        return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'io_time_ms': self.io_time_ms,
            'preprocessing_time_ms': self.preprocessing_time_ms,
            'geometry_time_ms': self.geometry_time_ms,
            'perception_time_ms': self.perception_time_ms,
            'foveation_time_ms': self.foveation_time_ms,
            'mapping_time_ms': self.mapping_time_ms,
            'rendering_time_ms': self.rendering_time_ms,
            'total_time_ms': self.total_time_ms,
            'input_points': self.input_points,
            'filtered_points': self.filtered_points,
            'active_cells': self.active_cells,
            'fine_cells': self.fine_cells,
            'medium_cells': self.medium_cells,
            'coarse_cells': self.coarse_cells,
            'refined_regions': self.refined_regions,
            'ram_usage_mb': self.ram_usage_mb,
            'vram_usage_mb': self.vram_usage_mb,
            'gpu_utilization': self.gpu_utilization,
            'processing_fps': self.processing_fps,
            'cell_reduction_percent': self.cell_reduction_percent,
        }


# ==============================================================================
# Configuration Validation
# ==============================================================================

def validate_config(config: Any) -> List[str]:
    """
    Validate configuration and return list of errors.
    
    Args:
        config: Config object to validate
    
    Returns:
        List of error messages (empty if valid)
    """
    errors = []
    
    # Validate weights sum to 1.0
    weights = config.foveation.weights
    total_weight = sum(weights.values())
    if abs(total_weight - 1.0) > 1e-6:
        errors.append(f"Foveation weights sum to {total_weight}, must be 1.0")
    
    # Validate resolutions
    if config.mapping.near_resolution_m <= 0:
        errors.append("near_resolution_m must be positive")
    if config.mapping.far_resolution_m <= 0:
        errors.append("far_resolution_m must be positive")
    if config.mapping.near_resolution_m >= config.mapping.far_resolution_m:
        errors.append("near_resolution_m must be less than far_resolution_m")
    
    # Validate ranges
    if config.mapping.near_range_m <= 0:
        errors.append("near_range_m must be positive")
    if config.mapping.far_range_m <= config.mapping.near_range_m:
        errors.append("far_range_m must be greater than near_range_m")
    
    # Validate ROI
    roi = config.mapping.roi
    if roi[0] >= roi[1]:
        errors.append("ROI x_min must be less than x_max")
    if roi[2] >= roi[3]:
        errors.append("ROI y_min must be less than y_max")
    
    # Validate compute mode
    if config.compute.mode not in [m.value for m in ComputeMode]:
        errors.append(f"Invalid compute mode: {config.compute.mode}")
    
    # Validate map mode
    if config.mapping.mode not in [m.value for m in MapMode]:
        errors.append(f"Invalid map mode: {config.mapping.mode}")
    
    return errors