from pathlib import Path
from typing import Optional, Union

from fovealidar.core.logging import get_logger

from .dataset import DatasetBase
from .kitti.loader import KITTIDataset
from .kitti.raw_loader import RawKITTIDataset

logger = get_logger(__name__)


class DatasetFactory:
    """Factory for creating dataset loaders."""
    
    _loaders = {
        'kitti': KITTIDataset,
        'kitti_raw': RawKITTIDataset,
        'raw': RawKITTIDataset,
    }
    
    @classmethod
    def register(cls, name: str, loader_class):
        """Register a new dataset loader."""
        cls._loaders[name.lower()] = loader_class
        logger.info(f"Registered dataset loader: {name}")
    
    @classmethod
    def create(cls, dataset_type: str, root: Path, **kwargs) -> DatasetBase:
        """Create a dataset loader.
        
        Args:
            dataset_type: Type of dataset ('kitti', 'kitti_raw', 'raw', etc.)
            root: Root directory of dataset
            **kwargs: Additional arguments for dataset loader
            
        Returns:
            Dataset loader instance
        """
        dataset_type = dataset_type.lower()
        
        if dataset_type not in cls._loaders:
            raise ValueError(
                f"Unknown dataset type: {dataset_type}. "
                f"Supported: {list(cls._loaders.keys())}"
            )
        
        loader_class = cls._loaders[dataset_type]
        logger.info(f"Creating {dataset_type} dataset from {root}")
        
        return loader_class(Path(root), **kwargs)
    
    @classmethod
    def auto_detect(cls, root: Path) -> Optional[DatasetBase]:
        """Auto-detect dataset type and create loader.
        
        Handles nested directory structures and searches recursively.
        
        Args:
            root: Root directory of dataset
            
        Returns:
            Dataset loader instance, or None if not recognized
        """
        root = Path(root)
        
        if not root.exists():
            logger.error(f"Root directory does not exist: {root}")
            return None
        
        def has_valid_drive(path: Path, depth: int = 0, max_depth: int = 5) -> bool:
            """Check if path contains valid KITTI raw drive folders."""
            if depth > max_depth:
                return False
            
            try:
                for item in path.iterdir():
                    if not item.is_dir():
                        continue
                    
                    # Check for valid drive folder
                    if 'drive' in item.name and 'sync' in item.name:
                        velo_dir = item / 'velodyne_points' / 'data'
                        if velo_dir.exists() and list(velo_dir.glob('*.bin')):
                            return True
                    
                    # Recurse
                    if has_valid_drive(item, depth + 1, max_depth):
                        return True
            except PermissionError:
                pass
            
            return False
        
        # Try KITTI Raw first (cvlibs format)
        # Look for drive folders recursively
        if has_valid_drive(root):
            try:
                logger.info("Auto-detected KITTI Raw dataset")
                return cls.create('kitti_raw', root)
            except Exception as e:
                logger.warning(f"Failed to load as KITTI Raw: {e}")
        
        # Try KITTI odometry format (sequences/)
        sequences_dir = root / 'sequences'
        if sequences_dir.exists() or (root / '00').exists():
            try:
                logger.info("Auto-detected KITTI Odometry dataset")
                return cls.create('kitti', root)
            except Exception as e:
                logger.warning(f"Failed to load as KITTI: {e}")
        
        logger.warning(f"Could not auto-detect dataset type in {root}")
        return None


def get_dataset(dataset_type: str, root: Union[str, Path], 
                **kwargs) -> DatasetBase:
    """Convenience function to create a dataset loader.
    
    Args:
        dataset_type: Type of dataset ('kitti', 'kitti_raw', 'raw', etc.)
        root: Root directory of dataset
        **kwargs: Additional arguments for dataset loader
        
    Returns:
        Dataset loader instance
    """
    return DatasetFactory.create(dataset_type, Path(root), **kwargs)
