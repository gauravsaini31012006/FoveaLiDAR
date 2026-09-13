"""Eye-gaze simulation and saliency map generation for foveal attention."""

from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud


logger = get_logger(__name__)


@dataclass
class GazePoint:
    """Represents a single gaze point in 3D space."""
    x: float
    y: float
    z: float
    timestamp: Optional[float] = None
    confidence: float = 1.0
    
    def to_array(self) -> NDArray[np.float32]:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)


class GazeSimulator:
    """Simulates eye-gaze patterns for foveal attention."""
    
    def __init__(self, fov_angle_deg: float = 60.0):
        """Initialize gaze simulator."""
        self.fov_angle_deg = fov_angle_deg
        self.fov_angle_rad = np.radians(fov_angle_deg)
        self._gaze_history = []
    
    def simulate_driving_gaze(
        self, pc: PointCloud, ego_position=(0.0, 0.0, 1.5),
        lane_center_y=0.0, num_samples=5
    ) -> NDArray[np.float32]:
        """Simulate gaze patterns for autonomous driving."""
        ego_x, ego_y, ego_z = ego_position
        gaze_points = []
        
        for i in range(num_samples):
            distance = 5.0 + i * 5.0
            lateral_offset = np.sin(i * 0.5) * 0.3
            height_variation = np.cos(i * 0.7) * 0.5
            
            gaze_point = GazePoint(
                x=ego_x + distance,
                y=ego_y + lane_center_y + lateral_offset,
                z=ego_z + height_variation,
                confidence=1.0 - i * 0.05
            )
            gaze_points.append(gaze_point.to_array())
        
        self._gaze_history.extend(gaze_points)
        logger.debug(f"Generated {num_samples} driving gaze points")
        return np.array(gaze_points, dtype=np.float32)
    
    def simulate_saliency_gaze(
        self, pc: PointCloud, intensity_threshold=0.5, num_samples=3
    ) -> NDArray[np.float32]:
        """Generate gaze toward high-saliency regions."""
        if pc.intensity is None or len(pc.points) == 0:
            logger.warning("No intensity data for saliency gaze")
            return np.zeros((num_samples, 3), dtype=np.float32)
        
        salient_mask = pc.intensity >= intensity_threshold
        if not salient_mask.any():
            return np.zeros((num_samples, 3), dtype=np.float32)
        
        salient_points = pc.points[salient_mask]
        salient_intensities = pc.intensity[salient_mask]
        top_indices = np.argsort(salient_intensities)[-num_samples:]
        
        gaze_points = salient_points[top_indices]
        return gaze_points.astype(np.float32)
    
    def generate_saliency_map(
        self, pc: PointCloud, grid_size=64, sigma=2.0
    ) -> NDArray[np.float32]:
        """Generate 2D saliency map from point intensities."""
        if pc.intensity is None or len(pc.points) == 0:
            return np.zeros((grid_size, grid_size), dtype=np.float32)
        
        x_min, x_max = pc.points[:, 0].min(), pc.points[:, 0].max()
        y_min, y_max = pc.points[:, 1].min(), pc.points[:, 1].max()
        
        x_range = x_max - x_min if x_max > x_min else 1.0
        y_range = y_max - y_min if y_max > y_min else 1.0
        
        x_normalized = ((pc.points[:, 0] - x_min) / x_range * (grid_size - 1)).astype(int)
        y_normalized = ((pc.points[:, 1] - y_min) / y_range * (grid_size - 1)).astype(int)
        
        x_normalized = np.clip(x_normalized, 0, grid_size - 1)
        y_normalized = np.clip(y_normalized, 0, grid_size - 1)
        
        saliency_map = np.zeros((grid_size, grid_size), dtype=np.float32)
        np.add.at(saliency_map, (y_normalized, x_normalized), pc.intensity)
        
        saliency_max = saliency_map.max()
        if saliency_max > 0:
            saliency_map = saliency_map / saliency_max
        
        try:
            from scipy import ndimage
            saliency_map = ndimage.gaussian_filter(saliency_map, sigma=sigma)
            saliency_map = saliency_map / saliency_map.max()
        except ImportError:
            logger.warning("scipy not available, skipping Gaussian blur")
        
        return saliency_map
    
    def get_attention_weight(
        self, point, gaze_point, fovea_radius_m=2.0, periphery_radius_m=20.0
    ) -> float:
        """Calculate attention weight based on gaze direction."""
        distance = np.linalg.norm(point - gaze_point)
        
        if distance <= fovea_radius_m:
            return 1.0
        elif distance <= periphery_radius_m:
            sigma = (periphery_radius_m - fovea_radius_m) / 3.0
            weight = np.exp(-((distance - fovea_radius_m) ** 2) / (2 * sigma ** 2))
            return float(weight)
        else:
            return 0.0
    
    def get_gaze_history(self) -> NDArray[np.float32]:
        """Get all recorded gaze points."""
        if not self._gaze_history:
            return np.zeros((0, 3), dtype=np.float32)
        return np.array(self._gaze_history, dtype=np.float32)
    
    def clear_history(self):
        """Clear gaze history."""
        self._gaze_history = []
