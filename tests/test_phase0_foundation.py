"""Test suite for FoveaLiDAR Phase 0 - Foundation."""

import pytest
import numpy as np

from fovealidar.core.config import Config, get_config
from fovealidar.core.types import PointCloud, ComputeMode, MapMode, validate_config
from fovealidar.core.diagnostics import run_diagnostics


def test_config_defaults():
    """Test default configuration."""
    config = Config()
    assert config.mapping.near_resolution_m == 0.05
    assert config.mapping.far_resolution_m == 0.50


def test_point_cloud_creation():
    """Test creating a point cloud."""
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    intensity = np.array([100, 200], dtype=np.float32)
    
    pc = PointCloud(points=points, intensity=intensity)
    assert pc.num_points == 2


def test_point_cloud_to_array():
    """Test conversion to Nx4 array."""
    points = np.array([[0, 0, 0], [1, 1, 1]], dtype=np.float32)
    intensity = np.array([100, 200], dtype=np.float32)
    
    pc = PointCloud(points=points, intensity=intensity)
    array = pc.to_array()
    assert array.shape == (2, 4)


def test_compute_mode_enum():
    """Test compute mode enum."""
    assert ComputeMode.MAX_QUALITY.value == "max_quality"
    assert ComputeMode.BALANCED.value == "balanced"


def test_config_validation():
    """Test configuration validation."""
    config = get_config()
    errors = validate_config(config)
    assert len(errors) == 0


def test_diagnostics():
    """Test environment diagnostics."""
    info = run_diagnostics()
    assert info.os is not None
    assert info.cpu is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])