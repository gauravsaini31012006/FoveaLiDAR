"""Adaptive Compression - ROI-based multi-resolution point filtering."""

import numpy as np
from numpy.typing import NDArray
from typing import Optional, Tuple

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud
from fovealidar.foveation.roi_definition import ROIDefinition


logger = get_logger(__name__)


class AdaptiveCompression:
    """Adaptive variable-resolution compression based on foveal ROI regions."""
    
    def __init__(
        self,
        roi_def: ROIDefinition,
        fovea_voxel_m: float = 0.05,
        periphery_voxel_m: float = 0.25,
        outside_voxel_m: float = 0.60,
    ):
        self.roi_def = roi_def
        self.fovea_voxel_m = fovea_voxel_m
        self.periphery_voxel_m = periphery_voxel_m
        self.outside_voxel_m = outside_voxel_m
    
    def compress_adaptive(
        self,
        pc: PointCloud,
        strategy: str = 'voxel',
        preserve_background: bool = True,
    ) -> PointCloud:
        """Apply adaptive compression to point cloud.
        
        Args:
            pc: Input point cloud.
            strategy: 'voxel' (variable grid) or 'uniform' (random subsampling).
            preserve_background: If True, keep far-field outside points at coarse resolution
                                instead of completely discarding them.
        """
        if len(pc.points) == 0:
            return pc
        
        region_ids, _ = self.roi_def.classify_points(pc)
        
        if strategy == 'uniform':
            return self._compress_uniform(pc, region_ids, preserve_background)
        else:
            return self._compress_voxel(pc, region_ids, preserve_background)
    
    def _compress_uniform(
        self,
        pc: PointCloud,
        region_ids: NDArray[np.int32],
        preserve_background: bool = True,
    ) -> PointCloud:
        """Uniform random downsampling with preserved scene structure."""
        mask = np.zeros(len(pc.points), dtype=bool)
        
        # Keep 100% of fovea
        fovea_mask = region_ids == 0
        mask[fovea_mask] = True
        
        # Keep 15% of periphery (or 1% if background not preserved)
        periphery_mask = region_ids == 1
        if np.any(periphery_mask):
            periphery_indices = np.where(periphery_mask)[0]
            fraction = 0.15 if preserve_background else 0.01
            keep_count = max(1, int(len(periphery_indices) * fraction))
            keep_indices = np.random.choice(periphery_indices, size=keep_count, replace=False)
            mask[keep_indices] = True
        
        # Far-field background
        outside_mask = region_ids == 2
        if preserve_background and np.any(outside_mask):
            outside_indices = np.where(outside_mask)[0]
            keep_count = max(1, int(len(outside_indices) * 0.05))
            keep_indices = np.random.choice(outside_indices, size=keep_count, replace=False)
            mask[keep_indices] = True
        
        logger.debug(f"Uniform compression: {len(pc.points)} -> {mask.sum()} points")
        
        out_pts = pc.points[mask]
        out_int = pc.intensity[mask] if pc.intensity is not None else np.ones(mask.sum(), dtype=np.float32)
        return PointCloud(points=out_pts, intensity=out_int)
    
    def _compress_voxel(
        self,
        pc: PointCloud,
        region_ids: NDArray[np.int32],
        preserve_background: bool = True,
    ) -> PointCloud:
        """Vectorized multi-tier variable voxel grid downsampling."""
        pts = pc.points
        intens = pc.intensity
        
        pts_list = []
        int_list = []
        
        # Tier 0: Fovea (0-10m or gaze focus) - fine resolution (nominal 5cm)
        fov_mask = region_ids == 0
        if np.any(fov_mask):
            f_pts = pts[fov_mask]
            f_int = intens[fov_mask] if intens is not None else None
            f_down_pts, f_down_int = self._voxel_downsample_fast(f_pts, f_int, self.fovea_voxel_m)
            pts_list.append(f_down_pts)
            if f_down_int is not None:
                int_list.append(f_down_int)
        
        # Tier 1: Periphery (10-50m) - medium resolution (nominal 25cm)
        per_mask = region_ids == 1
        if np.any(per_mask):
            p_pts = pts[per_mask]
            p_int = intens[per_mask] if intens is not None else None
            p_down_pts, p_down_int = self._voxel_downsample_fast(p_pts, p_int, self.periphery_voxel_m)
            pts_list.append(p_down_pts)
            if p_down_int is not None:
                int_list.append(p_down_int)
        
        # Tier 2: Outside / Far-field (50-100m) - coarse resolution (nominal 50-60cm)
        out_mask = region_ids == 2
        if preserve_background and np.any(out_mask):
            o_pts = pts[out_mask]
            o_int = intens[out_mask] if intens is not None else None
            o_down_pts, o_down_int = self._voxel_downsample_fast(o_pts, o_int, self.outside_voxel_m)
            pts_list.append(o_down_pts)
            if o_down_int is not None:
                int_list.append(o_down_int)
        
        if pts_list:
            compressed_points = np.vstack(pts_list).astype(np.float32)
            if int_list and len(int_list) == len(pts_list):
                compressed_intensity = np.concatenate(int_list).astype(np.float32)
            else:
                compressed_intensity = np.ones(len(compressed_points), dtype=np.float32)
        else:
            compressed_points = np.zeros((0, 3), dtype=np.float32)
            compressed_intensity = np.zeros(0, dtype=np.float32)
        
        logger.debug(f"Voxel compression: {len(pc.points)} -> {len(compressed_points)} points")
        return PointCloud(points=compressed_points, intensity=compressed_intensity)
    
    def _voxel_downsample(
        self,
        points: NDArray[np.float32],
        cell_size: float,
    ) -> NDArray[np.float32]:
        """Voxel grid downsampling (backwards compatible signature)."""
        pts, _ = self._voxel_downsample_fast(points, None, cell_size)
        return pts
    
    def _voxel_downsample_fast(
        self,
        points: NDArray[np.float32],
        intensity: Optional[NDArray[np.float32]],
        cell_size: float,
    ) -> Tuple[NDArray[np.float32], Optional[NDArray[np.float32]]]:
        """Vectorized voxel downsampling using np.add.at for lightning speed."""
        if len(points) == 0:
            return np.zeros((0, 3), dtype=np.float32), None if intensity is None else np.zeros(0, dtype=np.float32)
        
        voxel_coords = np.floor(points / cell_size).astype(np.int32)
        _, inverse = np.unique(voxel_coords, axis=0, return_inverse=True)
        n_voxels = int(inverse.max()) + 1
        
        # Fast vectorized sum and count
        sum_pts = np.zeros((n_voxels, 3), dtype=np.float64)
        np.add.at(sum_pts, inverse, points.astype(np.float64))
        counts = np.bincount(inverse, minlength=n_voxels).astype(np.float64)
        downsampled_pts = (sum_pts / counts[:, None]).astype(np.float32)
        
        downsampled_int = None
        if intensity is not None and len(intensity) == len(points):
            sum_int = np.zeros(n_voxels, dtype=np.float64)
            np.add.at(sum_int, inverse, intensity.astype(np.float64))
            downsampled_int = (sum_int / counts).astype(np.float32)
        
        return downsampled_pts, downsampled_int
    
    def get_compression_stats(self, pc: PointCloud, compressed: PointCloud) -> dict:
        """Calculate compression statistics."""
        region_ids, _ = self.roi_def.classify_points(pc)
        
        total_before = len(pc.points)
        total_after = len(compressed.points)
        compression_ratio = total_before / total_after if total_after > 0 else 0
        reduction_percent = (1 - total_after / total_before) * 100 if total_before > 0 else 0
        
        return {
            'input_points': total_before,
            'output_points': total_after,
            'fovea_points': int(np.sum(region_ids == 0)),
            'periphery_points': int(np.sum(region_ids == 1)),
            'outside_points': int(np.sum(region_ids == 2)),
            'compression_ratio': float(compression_ratio),
            'reduction_percent': float(reduction_percent),
        }
