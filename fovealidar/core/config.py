"""
Configuration management for FoveaLiDAR.

Loads YAML configuration files, supports environment variable overrides,
and provides typed access to all configuration parameters.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


@dataclass
class PerceptionConfig:
    """Perception-related configuration."""
    semantic_backend: str = "none"
    detector_backend: str = "none"
    device: str = "auto"
    ai_inference_stride: int = 1
    semantic_model_path: str = ""
    detector_model_path: str = ""


@dataclass
class MappingConfig:
    """Mapping-related configuration."""
    roi: List[float] = field(default_factory=lambda: [-50.0, 50.0, -50.0, 50.0])
    near_range_m: float = 10.0
    far_range_m: float = 100.0
    near_resolution_m: float = 0.05
    far_resolution_m: float = 0.50
    refinement_levels: List[float] = field(default_factory=lambda: [0.50, 0.20, 0.10, 0.05])
    mode: str = "distance_adaptive"
    height_range: List[float] = field(default_factory=lambda: [-3.0, 5.0])
    max_cells: int = 1000000
    ground_removal_enabled: bool = True
    ground_removal_max_slope_deg: float = 10.0
    ground_removal_max_height_m: float = 0.3
    ground_removal_min_points: int = 10


@dataclass
class FoveationConfig:
    """Foveation-related configuration."""
    enabled: bool = True
    weights: Dict[str, float] = field(default_factory=lambda: {
        "distance": 0.20,
        "semantic": 0.25,
        "uncertainty": 0.25,
        "geometry": 0.15,
        "motion": 0.15,
    })
    max_refined_regions: int = 128
    refinement_threshold: float = 0.6
    vehicle_corridor_enabled: bool = True
    vehicle_corridor_width_m: float = 4.0
    vehicle_corridor_max_range_m: float = 50.0
    vehicle_corridor_angle_deg: float = 60.0
    vehicle_corridor_importance_boost: float = 0.3


@dataclass
class ComputeConfig:
    """Compute-related configuration."""
    mode: str = "balanced"
    max_refined_regions: int = 128
    point_downsample_factor: int = 1
    max_render_fps: int = 30
    ai_frequency_divider: int = 1
    prefer_gpu: bool = True
    vram_limit_gb: float = 0.0
    cpu_threads: int = 0


@dataclass
class PreprocessingConfig:
    """Preprocessing-related configuration."""
    voxel_downsample_enabled: bool = True
    voxel_downsample_size_m: float = 0.05
    range_filter_enabled: bool = True
    range_filter_min_m: float = 1.0
    range_filter_max_m: float = 100.0
    intensity_normalization_enabled: bool = True
    intensity_normalization_method: str = "linear"
    intensity_normalization_min: float = 0.0
    intensity_normalization_max: float = 255.0
    statistical_outlier_removal_enabled: bool = False
    statistical_outlier_removal_nb_neighbors: int = 20
    statistical_outlier_removal_std_ratio: float = 2.0


@dataclass
class GeometryConfig:
    """Geometry-related configuration."""
    compute_density: bool = True
    compute_height_stats: bool = True
    compute_roughness: bool = True
    compute_slope: bool = True
    compute_density_gradient: bool = True
    compute_complexity: bool = True
    bin_size_m: float = 0.5
    neighborhood_radius_m: float = 1.0


@dataclass
class VisualizationConfig:
    """Visualization-related configuration."""
    window_width: int = 1600
    window_height: int = 900
    view3d_point_size: int = 2
    view3d_background_color: List[float] = field(default_factory=lambda: [0.1, 0.1, 0.1])
    view3d_show_axes: bool = True
    view3d_show_grid: bool = True
    map2d_colormap: str = "viridis"
    map2d_show_cell_boundaries: bool = False
    map2d_show_importance: bool = False
    map2d_show_uncertainty: bool = False
    map2d_show_resolution: bool = False
    map2d_opacity: float = 0.8
    debug_show_points: bool = False
    debug_show_cell_ids: bool = False
    debug_show_cell_boundaries: bool = False
    debug_show_importance: bool = False
    debug_show_uncertainty: bool = False
    debug_show_resolution: bool = False
    debug_show_point_assignment: bool = False
    debug_show_filtered_points: bool = False


@dataclass
class DataConfig:
    """Data-related configuration."""
    kitti_root: str = ""
    sequence: str = "00"
    frame_start: int = 0
    frame_end: int = -1
    playback_auto_play: bool = False
    playback_loop: bool = True
    playback_target_fps: int = 10


@dataclass
class BenchmarkConfig:
    """Benchmark-related configuration."""
    warmup_frames: int = 5
    benchmark_frames: int = 100
    output_dir: str = "benchmarks/results"
    export_json: bool = True
    export_csv: bool = True
    include_system_info: bool = True


@dataclass
class LoggingConfig:
    """Logging-related configuration."""
    level: str = "INFO"
    file_enabled: bool = True
    file_path: str = "logs/fovealidar.log"
    max_file_size_mb: int = 10
    backup_count: int = 5
    console_enabled: bool = True
    format: str = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    date_format: str = "%Y-%m-%d %H:%M:%S"


@dataclass
class Config:
    """Main configuration class for FoveaLiDAR."""
    perception: PerceptionConfig = field(default_factory=PerceptionConfig)
    mapping: MappingConfig = field(default_factory=MappingConfig)
    foveation: FoveationConfig = field(default_factory=FoveationConfig)
    compute: ComputeConfig = field(default_factory=ComputeConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    geometry: GeometryConfig = field(default_factory=GeometryConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    data: DataConfig = field(default_factory=DataConfig)
    benchmark: BenchmarkConfig = field(default_factory=BenchmarkConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Internal: path to the config file that was loaded
    _config_path: Optional[Path] = field(default=None, repr=False)
    
    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> Config:
        """Load configuration from a YAML file."""
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'r') as f:
            data = yaml.safe_load(f)
        
        if data is None:
            data = {}
        
        config = cls()
        config._config_path = path
        
        # Update from loaded data
        if 'perception' in data:
            config.perception = PerceptionConfig(**data['perception'])
        if 'mapping' in data:
            config.mapping = MappingConfig(**data['mapping'])
        if 'foveation' in data:
            config.foveation = FoveationConfig(**data['foveation'])
        if 'compute' in data:
            config.compute = ComputeConfig(**data['compute'])
        if 'preprocessing' in data:
            config.preprocessing = PreprocessingConfig(**data['preprocessing'])
        if 'geometry' in data:
            config.geometry = GeometryConfig(**data['geometry'])
        if 'visualization' in data:
            config.visualization = VisualizationConfig(**data['visualization'])
        if 'data' in data:
            config.data = DataConfig(**data['data'])
        if 'benchmark' in data:
            config.benchmark = BenchmarkConfig(**data['benchmark'])
        if 'logging' in data:
            config.logging = LoggingConfig(**data['logging'])
        
        # Apply environment variable overrides
        config._apply_env_overrides()
        
        return config
    
    def _apply_env_overrides(self) -> None:
        """Apply environment variable overrides to configuration."""
        # Environment variables follow the pattern: FOVEALIDAR_<SECTION>_<KEY>
        # e.g., FOVEALIDAR_MAPPING_NEAR_RANGE_M=15.0
        
        env_mappings = {
            'FOVEALIDAR_KITTI_ROOT': ('data', 'kitti_root', str),
            'FOVEALIDAR_SEQUENCE': ('data', 'sequence', str),
            'FOVEALIDAR_DEVICE': ('perception', 'device', str),
            'FOVEALIDAR_COMPUTE_MODE': ('compute', 'mode', str),
            'FOVEALIDAR_LOG_LEVEL': ('logging', 'level', str),
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.environ.get(env_var)
            if value is not None:
                section_obj = getattr(self, section)
                try:
                    converted = converter(value)
                    setattr(section_obj, key, converted)
                except (ValueError, TypeError):
                    pass  # Ignore invalid values
    
    def to_yaml(self, path: Union[str, Path]) -> None:
        """Save configuration to a YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        data = {
            'perception': self.perception.__dict__,
            'mapping': self.mapping.__dict__,
            'foveation': self.foveation.__dict__,
            'compute': self.compute.__dict__,
            'preprocessing': self.preprocessing.__dict__,
            'geometry': self.geometry.__dict__,
            'visualization': self.visualization.__dict__,
            'data': self.data.__dict__,
            'benchmark': self.benchmark.__dict__,
            'logging': self.logging.__dict__,
        }
        
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False, sort_keys=False)
    
    def get_config_path(self) -> Optional[Path]:
        """Get the path of the loaded configuration file."""
        return self._config_path


# Global configuration instance
_config: Optional[Config] = None


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from file or create default."""
    global _config
    
    if config_path is None:
        # Try default locations
        default_paths = [
            Path("config/default.yaml"),
            Path("fovealidar/config/default.yaml"),
            Path(__file__).parent.parent.parent / "config" / "default.yaml",
        ]
        
        for p in default_paths:
            if p.exists():
                config_path = p
                break
    
    if config_path and Path(config_path).exists():
        _config = Config.from_yaml(config_path)
    else:
        _config = Config()
    
    return _config


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = load_config()
    return _config