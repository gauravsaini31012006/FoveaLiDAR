"""Phase 2: Point Cloud Preprocessing Pipeline."""

from dataclasses import dataclass
from typing import Optional
import numpy as np

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud


logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Configuration for preprocessing pipeline."""
    
    # Range filtering
    range_min: float = 0.0
    range_max: float = 100.0
    
    # Height filtering
    height_min: float = -3.0
    height_max: float = 10.0
    
    # Intensity normalization
    normalize_intensity: bool = True
    intensity_min: Optional[float] = None  # None = auto-detect from data
    intensity_max: Optional[float] = None  # None = auto-detect from data
    
    # Statistical outlier removal
    remove_outliers: bool = True
    outlier_neighbors: int = 20
    outlier_std_ratio: float = 2.0
    
    # Voxel downsampling
    voxel_size: Optional[float] = None
    
    # Ground removal
    remove_ground: bool = False
    ground_threshold: float = -1.5


class RangeFilter:
    """Filter points by distance from origin."""
    
    def __init__(self, range_min: float = 0.0, range_max: float = 100.0):
        self.range_min = range_min
        self.range_max = range_max
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Filter points outside range."""
        distances = np.linalg.norm(pc.points, axis=1)
        mask = (distances >= self.range_min) & (distances <= self.range_max)
        
        logger.debug(f"Range filter: {mask.sum()}/{len(mask)} points kept")
        
        return PointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask] if pc.intensity is not None else None
        )


class HeightFilter:
    """Filter points by Z coordinate (height)."""
    
    def __init__(self, height_min: float = -3.0, height_max: float = 10.0):
        self.height_min = height_min
        self.height_max = height_max
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Filter points outside height range."""
        mask = (pc.points[:, 2] >= self.height_min) & (pc.points[:, 2] <= self.height_max)
        
        logger.debug(f"Height filter: {mask.sum()}/{len(mask)} points kept")
        
        return PointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask] if pc.intensity is not None else None
        )


class IntensityNormalizationFilter:
    """Normalize intensity values to [0, 1] range."""
    
    def __init__(self, intensity_min: Optional[float] = None, intensity_max: Optional[float] = None):
        """Initialize filter.
        
        Args:
            intensity_min: Minimum intensity value. If None, auto-detected from data.
            intensity_max: Maximum intensity value. If None, auto-detected from data.
        """
        self.intensity_min = intensity_min
        self.intensity_max = intensity_max
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Normalize intensity."""
        if pc.intensity is None:
            return pc
        
        intensity = pc.intensity.copy().astype(np.float32)
        
        # Auto-detect range if not specified
        i_min = self.intensity_min if self.intensity_min is not None else intensity.min()
        i_max = self.intensity_max if self.intensity_max is not None else intensity.max()
        
        # Avoid division by zero
        if i_max - i_min < 1e-6:
            logger.warning("Intensity range too small, skipping normalization")
            return pc
        
        intensity = (intensity - i_min) / (i_max - i_min)
        intensity = np.clip(intensity, 0, 1)
        
        logger.debug(f"Intensity normalization: [{i_min:.4f}, {i_max:.4f}] -> [0, 1]")
        
        return PointCloud(points=pc.points, intensity=intensity)


class VoxelDownsamplingFilter:
    """Downsample point cloud using voxel grid (vectorized)."""
    
    def __init__(self, voxel_size: float = 0.1):
        self.voxel_size = voxel_size
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Downsample using voxel grid — fully vectorized via np.add.at."""
        if self.voxel_size <= 0:
            return pc
        
        indices = np.floor(pc.points / self.voxel_size).astype(np.int32)
        _, inverse = np.unique(indices, axis=0, return_inverse=True)
        n_voxels = int(inverse.max()) + 1
        
        # Vectorized mean aggregation — no Python loop
        sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(sum_pts, inverse, pc.points.astype(np.float64))
        counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
        downsampled_points = (sum_pts / counts[:, None]).astype(np.float32)
        
        downsampled_intensity = None
        if pc.intensity is not None:
            sum_int = np.zeros(n_voxels, dtype=np.float64)
            np.add.at(sum_int, inverse, pc.intensity.astype(np.float64))
            downsampled_intensity = (sum_int / counts).astype(np.float32)
        
        logger.debug(f"Voxel filter: {len(downsampled_points)}/{len(pc.points)} points")
        
        return PointCloud(points=downsampled_points, intensity=downsampled_intensity)


class StatisticalOutlierFilter:
    """Remove outliers using statistical analysis."""
    
    def __init__(self, neighbors: int = 20, std_ratio: float = 2.0):
        self.neighbors = neighbors
        self.std_ratio = std_ratio
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Remove statistical outliers."""
        if len(pc.points) < self.neighbors:
            return pc
        
        center = pc.points.mean(axis=0)
        distances = np.linalg.norm(pc.points - center, axis=1)
        mean_dist = distances.mean()
        std_dist = distances.std()
        
        threshold = mean_dist + self.std_ratio * std_dist
        mask = distances <= threshold
        
        logger.debug(f"Outlier filter: {mask.sum()}/{len(mask)} points kept")
        
        return PointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask] if pc.intensity is not None else None
        )


class GroundRemovalFilter:
    """Remove points likely to be ground."""
    
    def __init__(self, ground_threshold: float = -1.5):
        self.ground_threshold = ground_threshold
    
    def apply(self, pc: PointCloud) -> PointCloud:
        """Remove ground points (very low Z)."""
        mask = pc.points[:, 2] > self.ground_threshold
        
        logger.debug(f"Ground filter: {mask.sum()}/{len(mask)} points kept")
        
        return PointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask] if pc.intensity is not None else None
        )


class PreprocessingPipeline:
    """Chainable preprocessing pipeline."""
    
    def __init__(self, config: Optional[PreprocessingConfig] = None):
        """Initialize pipeline.
        
        Args:
            config: Preprocessing configuration
        """
        self.config = config or PreprocessingConfig()
        self.filters = []
        self._build_filters()
    
    def _build_filters(self):
        """Build filter chain based on config."""
        self.filters = []
        
        # Range filtering
        self.filters.append(RangeFilter(self.config.range_min, self.config.range_max))
        
        # Height filtering
        self.filters.append(HeightFilter(self.config.height_min, self.config.height_max))
        
        # Ground removal
        if self.config.remove_ground:
            self.filters.append(GroundRemovalFilter(self.config.ground_threshold))
        
        # Intensity normalization
        if self.config.normalize_intensity:
            self.filters.append(IntensityNormalizationFilter(
                self.config.intensity_min,
                self.config.intensity_max
            ))
        
        # Outlier removal
        if self.config.remove_outliers:
            self.filters.append(StatisticalOutlierFilter(
                self.config.outlier_neighbors,
                self.config.outlier_std_ratio
            ))
        
        # Voxel downsampling (last to reduce data)
        if self.config.voxel_size is not None and self.config.voxel_size > 0:
            self.filters.append(VoxelDownsamplingFilter(self.config.voxel_size))
    
    def process(self, pc: PointCloud) -> PointCloud:
        """Process point cloud through pipeline.
        
        Args:
            pc: Input point cloud
            
        Returns:
            Processed point cloud
        """
        result = pc
        for filter_obj in self.filters:
            result = filter_obj.apply(result)
        
        logger.info(f"Pipeline: {len(pc.points)} → {len(result.points)} points")
        
        return result
    
    def add_filter(self, filter_obj):
        """Add custom filter to pipeline."""
        self.filters.append(filter_obj)
        return self
    
    def clear_filters(self):
        """Clear all filters."""
        self.filters = []
        return self

        logger.debug(f"Ground filter: {mask.sum()}/{len(mask)} points kept")
        
        return PointCloud(
            points=pc.points[mask],
            intensity=pc.intensity[mask] if pc.intensity is not None else None
        )
