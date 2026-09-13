"""Phase 7: Semantic AI Module.

Defines the semantic backends (heuristic and DL-based mocks) for the fovealidar pipeline.
"""

from abc import ABC, abstractmethod
import time
from typing import Tuple, Optional
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger

logger = get_logger(__name__)

# Class Definitions
# 0: UNKNOWN / UNLABELLED
# 1: GROUND
# 2: VEGETATION
# 3: OBSTACLE / VEHICLE / PEDESTRIAN


class SemanticBackend(ABC):
    """Abstract interface for semantic segmentation backends."""

    @abstractmethod
    def infer(
        self,
        coarse_elevation_max: NDArray[np.float32],
        coarse_elevation_min: NDArray[np.float32],
        coarse_elevation_variance: NDArray[np.float32],
        coarse_point_count: NDArray[np.int32],
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float]:
        """Perform semantic inference over the coarse grid.

        Returns:
            semantic_labels: (R, C) int32 map of class IDs.
            uncertainty_map: (R, C) float32 map of inference uncertainty [0, 1].
            inference_time_ms: Latency in milliseconds.
        """
        pass


class HeuristicSemanticBackend(SemanticBackend):
    """Rule-based semantic backend using geometry proxies."""

    def __init__(self, ground_height_max: float = 0.5, vegetation_variance_min: float = 0.8):
        self.ground_height_max = ground_height_max
        self.vegetation_variance_min = vegetation_variance_min
        logger.info("Initialized Heuristic Semantic Backend.")

    def infer(
        self,
        coarse_elevation_max: NDArray[np.float32],
        coarse_elevation_min: NDArray[np.float32],
        coarse_elevation_variance: NDArray[np.float32],
        coarse_point_count: NDArray[np.int32],
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float]:
        t0 = time.perf_counter()
        
        # Output buffers
        labels = np.zeros_like(coarse_point_count, dtype=np.int32)
        uncertainty = np.zeros_like(coarse_elevation_max, dtype=np.float32)
        
        height_diff = coarse_elevation_max - coarse_elevation_min
        valid_mask = coarse_point_count > 0
        
        # Rule 1: Ground
        ground_mask = valid_mask & (height_diff <= self.ground_height_max)
        labels[ground_mask] = 1
        uncertainty[ground_mask] = 0.1
        
        # Rule 2: Vegetation (high variance)
        veg_mask = valid_mask & (~ground_mask) & (coarse_elevation_variance >= self.vegetation_variance_min)
        labels[veg_mask] = 2
        uncertainty[veg_mask] = 0.5 # higher uncertainty for heuristic vegetation
        
        # Rule 3: Obstacle (everything else that has significant height and low variance)
        obs_mask = valid_mask & (~ground_mask) & (~veg_mask)
        labels[obs_mask] = 3
        uncertainty[obs_mask] = 0.2
        
        t1 = time.perf_counter()
        latency_ms = (t1 - t0) * 1000.0
        
        return labels, uncertainty, latency_ms


class SalsaNextBackend(SemanticBackend):
    """Stub implementation of a deep learning backend (SalsaNext)."""

    def __init__(self, latency_ms: float = 25.0):
        self.simulated_latency = latency_ms
        logger.info(f"Initialized SalsaNext Mock Backend (simulated latency: {latency_ms}ms).")

    def infer(
        self,
        coarse_elevation_max: NDArray[np.float32],
        coarse_elevation_min: NDArray[np.float32],
        coarse_elevation_variance: NDArray[np.float32],
        coarse_point_count: NDArray[np.int32],
    ) -> Tuple[NDArray[np.int32], NDArray[np.float32], float]:
        t0 = time.perf_counter()
        
        # Simulate inference time
        time.sleep(self.simulated_latency / 1000.0)
        
        # Output buffers (fallback to heuristic just to have data)
        labels = np.zeros_like(coarse_point_count, dtype=np.int32)
        uncertainty = np.full_like(coarse_elevation_max, 0.5, dtype=np.float32)
        
        # Assign obstacles randomly where points exist for mock effect
        valid_mask = coarse_point_count > 0
        rand_vals = np.random.rand(*labels.shape)
        
        labels[valid_mask & (rand_vals < 0.6)] = 1 # Ground
        labels[valid_mask & (rand_vals >= 0.6) & (rand_vals < 0.8)] = 2 # Veg
        labels[valid_mask & (rand_vals >= 0.8)] = 3 # Obs
        
        t1 = time.perf_counter()
        actual_latency_ms = (t1 - t0) * 1000.0
        
        return labels, uncertainty, actual_latency_ms
