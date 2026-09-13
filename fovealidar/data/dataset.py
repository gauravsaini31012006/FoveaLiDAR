"""
Abstract base class for dataset loaders.

Defines the interface that all dataset implementations must follow,
enabling consistent access to different data sources.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class DatasetStats:
    """Statistics about a loaded dataset."""
    num_frames: int
    sequences: List[str]
    total_points: int
    avg_points_per_frame: float
    min_points_in_frame: int
    max_points_in_frame: int
    spatial_bounds: Dict[str, float]  # x_min, x_max, y_min, y_max, z_min, z_max
    temporal_duration_s: float
    dataset_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            'num_frames': self.num_frames,
            'sequences': self.sequences,
            'total_points': self.total_points,
            'avg_points_per_frame': self.avg_points_per_frame,
            'min_points_in_frame': self.min_points_in_frame,
            'max_points_in_frame': self.max_points_in_frame,
            'spatial_bounds': self.spatial_bounds,
            'temporal_duration_s': self.temporal_duration_s,
            'dataset_type': self.dataset_type,
        }


class DatasetBase(ABC):
    """Abstract base class for all dataset implementations."""
    
    def __init__(self, root: Path, name: str = "unknown"):
        """
        Initialize dataset.
        
        Args:
            root: Root directory containing dataset
            name: Human-readable dataset name
        """
        self.root = Path(root)
        self.name = name
        self._stats: Optional[DatasetStats] = None
    
    @abstractmethod
    def __len__(self) -> int:
        """Return total number of frames across all sequences."""
        pass
    
    @abstractmethod
    def __getitem__(self, idx: int) -> Any:
        """
        Get a frame by index.
        
        Args:
            idx: Frame index
            
        Returns:
            Frame object (type depends on implementation)
        """
        pass
    
    @abstractmethod
    def validate(self) -> bool:
        """
        Validate dataset integrity.
        
        Returns:
            True if dataset is valid and complete
        """
        pass
    
    @abstractmethod
    def get_stats(self) -> DatasetStats:
        """
        Get or compute dataset statistics.
        
        Returns:
            DatasetStats object
        """
        pass
    
    @abstractmethod
    def list_sequences(self) -> List[str]:
        """
        List all available sequences in dataset.
        
        Returns:
            List of sequence identifiers
        """
        pass
    
    @abstractmethod
    def get_sequence_length(self, sequence: str) -> int:
        """
        Get number of frames in a sequence.
        
        Args:
            sequence: Sequence identifier
            
        Returns:
            Number of frames
        """
        pass
