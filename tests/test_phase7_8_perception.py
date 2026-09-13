import pytest
import numpy as np

from fovealidar.perception.semantic import HeuristicSemanticBackend
from fovealidar.perception.tracking import DetectionBackend, SimpleTracker

def test_heuristic_semantic_backend():
    backend = HeuristicSemanticBackend(ground_height_max=0.5, vegetation_variance_min=0.8)
    
    # 2x2 grid
    coarse_point_count = np.array([[10, 10], [10, 0]], dtype=np.int32)
    coarse_elevation_max = np.array([[0.1, 3.0], [5.0, 0.0]], dtype=np.float32)
    coarse_elevation_min = np.array([[0.0, 0.0], [4.8, 0.0]], dtype=np.float32)
    coarse_elevation_var = np.array([[0.0, 1.2], [0.1, 0.0]], dtype=np.float32)
    
    labels, uncertainty, latency = backend.infer(
        coarse_elevation_max,
        coarse_elevation_min,
        coarse_elevation_var,
        coarse_point_count
    )
    
    # Cell 0,0: diff=0.1 <= 0.5 -> GROUND (1)
    # Cell 0,1: diff=3.0 > 0.5, var=1.2 >= 0.8 -> VEGETATION (2)
    # Cell 1,0: diff=0.2 <= 0.5 -> GROUND (1)
    # Cell 1,1: count=0 -> UNLABELLED (0)
    
    assert labels[0, 0] == 1
    assert labels[0, 1] == 2
    assert labels[1, 0] == 1
    assert labels[1, 1] == 0
    assert latency >= 0.0

def test_detection_and_tracking():
    det = DetectionBackend(latency_ms=0)
    tracker = SimpleTracker()
    
    labels = np.array([[1, 3, 3], [1, 3, 3], [1, 1, 1]], dtype=np.int32)
    bboxes, lat = det.infer(labels)
    
    assert len(bboxes) == 1
    bbox = bboxes[0]
    assert bbox['class'] == 'vehicle'
    assert bbox['id'] == -1
    
    tracked = tracker.update(bboxes)
    assert len(tracked) == 1
    assert tracked[0]['id'] == 1
