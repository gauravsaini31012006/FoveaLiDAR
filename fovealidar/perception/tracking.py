"""Phase 8: Detection & Tracking Modules.

Implements detection backend interfaces and a simple bounding box tracker.
"""

import time
from typing import Tuple, List, Dict, Any
import numpy as np
from numpy.typing import NDArray

from fovealidar.core.logging import get_logger

logger = get_logger(__name__)


class DetectionBackend:
    """Mock interface for a DL detection model like CenterPoint."""

    def __init__(self, latency_ms: float = 30.0):
        self.simulated_latency = latency_ms
        logger.info(f"Initialized Detection Backend (simulated latency: {latency_ms}ms).")

    def infer(self, semantic_labels: NDArray[np.int32]) -> Tuple[List[Dict[str, Any]], float]:
        """Generate bounding boxes for obstacles.

        Args:
            semantic_labels: (R, C) semantic mask where 3 = OBSTACLE.
            
        Returns:
            bboxes: List of bounding boxes dicts (x, y, w, h).
            latency_ms: Inference time.
        """
        t0 = time.perf_counter()
        if self.simulated_latency > 0:
            time.sleep(self.simulated_latency / 1000.0)
            
        # For mock purposes, find clusters of label 3 (Obstacle)
        # Using a simple naive approach since this is a stub
        bboxes = []
        rows, cols = np.where(semantic_labels == 3)
        if len(rows) > 0:
            # Just create one large bounding box around the mean if any obstacles exist
            # Real implementation would use connected components or CenterPoint outputs
            mean_r, mean_c = int(np.mean(rows)), int(np.mean(cols))
            min_r, max_r = max(0, mean_r - 2), min(semantic_labels.shape[0] - 1, mean_r + 2)
            min_c, max_c = max(0, mean_c - 2), min(semantic_labels.shape[1] - 1, mean_c + 2)
            
            bboxes.append({
                "id": -1, # Unassigned
                "class": "vehicle",
                "r": mean_r,
                "c": mean_c,
                "min_r": min_r,
                "max_r": max_r,
                "min_c": min_c,
                "max_c": max_c,
            })
            
        t1 = time.perf_counter()
        return bboxes, (t1 - t0) * 1000.0


class SimpleTracker:
    """Simple IoU-based Tracker."""
    
    def __init__(self):
        self.tracks = {}
        self.next_id = 1
        logger.info("Initialized SimpleTracker.")

    def update(self, bboxes: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Update tracks given new detections."""
        updated_bboxes = []
        # In a real IoU tracker, we match current bboxes with self.tracks.
        # For simplicity, assign IDs based on proximity or just increment.
        for bbox in bboxes:
            assigned_id = self.next_id
            self.next_id += 1
            bbox["id"] = assigned_id
            updated_bboxes.append(bbox)
        return updated_bboxes
