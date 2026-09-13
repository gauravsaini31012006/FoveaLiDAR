"""Foveal ROI Definition - Dual-resolution grid system."""

from dataclasses import dataclass
from typing import Tuple
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud


logger = get_logger(__name__)


@dataclass
class FovealROI:
    """Represents a single foveal ROI region."""
    center: NDArray[np.float32]
    fovea_radius_m: float = 10.0
    periphery_radius_m: float = 100.0
    cell_size_fovea_m: float = 0.05
    cell_size_periphery_m: float = 0.50
    
    def fovea_cells_count(self) -> int:
        """Calculate expected cell count in fovea region."""
        area = np.pi * (self.fovea_radius_m ** 2)
        cell_area = self.cell_size_fovea_m ** 2
        return int(area / cell_area)
    
    def periphery_cells_count(self) -> int:
        """Calculate expected cell count in periphery region."""
        area = np.pi * (self.periphery_radius_m ** 2 - self.fovea_radius_m ** 2)
        cell_area = self.cell_size_periphery_m ** 2
        return int(area / cell_area)
    
    def point_region_type(self, point: NDArray[np.float32]) -> str:
        """Determine which region a point belongs to."""
        distance = np.linalg.norm(point[:2] - self.center[:2])
        
        if distance <= self.fovea_radius_m:
            return 'fovea'
        elif distance <= self.periphery_radius_m:
            return 'periphery'
        else:
            return 'outside'
    
    def get_cell_size(self, distance_from_center: float) -> float:
        """Get appropriate cell size based on distance from ROI center."""
        if distance_from_center <= self.fovea_radius_m:
            return self.cell_size_fovea_m
        elif distance_from_center <= self.periphery_radius_m:
            return self.cell_size_periphery_m
        else:
            return self.cell_size_periphery_m * 2


class ROIDefinition:
    """Manages foveal ROI regions and adaptive grid assignment."""
    
    def __init__(self, fovea_radius_m: float = 10.0, periphery_radius_m: float = 100.0):
        """Initialize ROI definition system."""
        self.fovea_radius_m = fovea_radius_m
        self.periphery_radius_m = periphery_radius_m
        self.rois: list = []
    
    def add_roi(self, center: Tuple[float, float, float]) -> FovealROI:
        """Add a new ROI at the specified center."""
        roi = FovealROI(
            center=np.array(center, dtype=np.float32),
            fovea_radius_m=self.fovea_radius_m,
            periphery_radius_m=self.periphery_radius_m,
        )
        self.rois.append(roi)
        logger.debug(f"Added ROI at {center}")
        return roi
    
    def classify_points(self, pc: PointCloud) -> Tuple[NDArray[np.int32], NDArray[np.float32]]:
        """Classify each point into resolution level (fully vectorized, no Python loops)."""
        pts = pc.points
        n = len(pts)

        if not self.rois:
            logger.warning("No ROIs defined, returning default classification")
            return (
                np.full(n, 2, dtype=np.int32),
                np.ones(n, dtype=np.float32),
            )

        # Build arrays of ROI properties  (R,)
        centers = np.array([roi.center[:2] for roi in self.rois], dtype=np.float32)   # (R, 2)
        fovea_radii = np.array([roi.fovea_radius_m for roi in self.rois], dtype=np.float32)
        periph_radii = np.array([roi.periphery_radius_m for roi in self.rois], dtype=np.float32)

        # Distance matrix: (N, R) — fully vectorized
        pts_xy = pts[:, :2]  # (N, 2)
        dists = np.sqrt(
            (pts_xy[:, 0:1] - centers[np.newaxis, :, 0]) ** 2 +
            (pts_xy[:, 1:2] - centers[np.newaxis, :, 1]) ** 2
        )  # (N, R)

        # Find nearest ROI per point
        best_roi_idx = np.argmin(dists, axis=1)           # (N,)
        best_dist = dists[np.arange(n), best_roi_idx]     # (N,)
        best_fovea_r = fovea_radii[best_roi_idx]          # (N,)
        best_periph_r = periph_radii[best_roi_idx]        # (N,)

        # Assign region IDs
        region_ids = np.full(n, 2, dtype=np.int32)
        region_ids[best_dist <= best_fovea_r] = 0
        region_ids[(best_dist > best_fovea_r) & (best_dist <= best_periph_r)] = 1

        compression_factors = np.ones(n, dtype=np.float32)
        compression_factors[region_ids == 1] = 0.01

        logger.debug(
            f"Classified {n} points: "
            f"fovea={int((region_ids == 0).sum())}, "
            f"periphery={int((region_ids == 1).sum())}, "
            f"outside={int((region_ids == 2).sum())}"
        )
        return region_ids, compression_factors
    
    def get_expected_reduction(self) -> float:
        """Calculate expected point reduction vs uniform 5cm grid."""
        fovea_area = np.pi * (self.fovea_radius_m ** 2)
        periphery_area = np.pi * (self.periphery_radius_m ** 2) - fovea_area
        
        fovea_cells = fovea_area / (0.05 ** 2)
        periphery_cells = periphery_area / (0.50 ** 2)
        total_adaptive = fovea_cells + periphery_cells
        
        total_uniform = np.pi * (self.periphery_radius_m ** 2) / (0.05 ** 2)
        
        reduction = (1 - total_adaptive / total_uniform) * 100
        return reduction
    
    def get_stats(self) -> dict:
        """Get statistics about ROI system."""
        if not self.rois:
            return {'num_rois': 0}
        
        roi = self.rois[0]
        return {
            'num_rois': len(self.rois),
            'fovea_radius_m': self.fovea_radius_m,
            'periphery_radius_m': self.periphery_radius_m,
            'fovea_cells': roi.fovea_cells_count(),
            'periphery_cells': roi.periphery_cells_count(),
            'expected_reduction_percent': self.get_expected_reduction(),
        }
