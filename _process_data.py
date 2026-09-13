#!/usr/bin/env python
"""FoveaLiDAR - Process KITTI Data"""

from pathlib import Path
from fovealidar.data.loaders import DatasetFactory
from fovealidar.preprocessing import PreprocessingConfig, PreprocessingPipeline

KITTI_ROOT = Path("D:/CLG/SIH_2026/kitti_raw_data of lidar")
NUM_FRAMES = 5

def main():
    print("\n" + "="*70)
    print("FoveaLiDAR - KITTI Data Processing")
    print("="*70 + "\n")
    
    if not KITTI_ROOT.exists():
        print(f"❌ ERROR: KITTI path not found: {KITTI_ROOT}")
        print("\nEdit KITTI_ROOT in _process_data.py to point to your data")
        return
    
    print(f"📁 Loading from: {KITTI_ROOT}\n")
    
    try:
        dataset = DatasetFactory.auto_detect(KITTI_ROOT)
        if dataset is None:
            print("❌ Could not auto-detect dataset format")
            return
        print(f"✅ Dataset loaded: {type(dataset).__name__}")
    except Exception as e:
        print(f"❌ Error loading dataset: {e}")
        return
    
    print(f"   Frames: {len(dataset)}")
    print(f"   Sequences: {dataset.list_sequences()}\n")
    
    try:
        stats = dataset.get_stats()
        print("📊 Dataset Statistics:")
        print(f"   Type: {stats.dataset_type}")
        print(f"   Total frames: {stats.num_frames}")
        print(f"   Avg points/frame: {stats.avg_points_per_frame:,}")
        print(f"   Points range: {stats.min_points_in_frame:,} - {stats.max_points_in_frame:,}")
        print(f"   Spatial bounds:")
        print(f"     X: [{stats.spatial_bounds['x_min']:7.2f}, {stats.spatial_bounds['x_max']:7.2f}] m")
        print(f"     Y: [{stats.spatial_bounds['y_min']:7.2f}, {stats.spatial_bounds['y_max']:7.2f}] m")
        print(f"     Z: [{stats.spatial_bounds['z_min']:7.2f}, {stats.spatial_bounds['z_max']:7.2f}] m")
        print(f"   Duration: {stats.temporal_duration_s:.2f}s\n")
    except Exception as e:
        print(f"⚠️  Warning computing stats: {e}\n")
    
    config = PreprocessingConfig(
        range_min=0.0, range_max=100.0,
        height_min=-3.0, height_max=10.0,
        normalize_intensity=True, remove_outliers=True
    )
    
    pipeline = PreprocessingPipeline(config)
    print("🔧 Preprocessing Pipeline:")
    print(f"   Range: [{config.range_min}, {config.range_max}] m")
    print(f"   Height: [{config.height_min}, {config.height_max}] m")
    print(f"   Normalize intensity: {config.normalize_intensity}")
    print(f"   Remove outliers: {config.remove_outliers}\n")
    
    print("⚙️  Processing frames...\n")
    print(f"{'Frame':<8} {'Original':<15} {'Processed':<15} {'Reduction':<12} {'Intensity Range'}")
    print("-" * 70)
    
    total_original = 0
    total_processed = 0
    
    for frame_idx in range(min(NUM_FRAMES, len(dataset))):
        try:
            frame = dataset[frame_idx]
            original_pc = frame.point_cloud
            processed_pc = pipeline.process(original_pc)
            
            original_count = len(original_pc.points)
            processed_count = len(processed_pc.points)
            reduction = 100 * (1 - processed_count / original_count)
            
            intensity_min = processed_pc.intensity.min() if processed_pc.intensity is not None else 0
            intensity_max = processed_pc.intensity.max() if processed_pc.intensity is not None else 0
            
            print(f"{frame_idx:<8} {original_count:<15,} {processed_count:<15,} "
                  f"{reduction:<11.1f}% [{intensity_min:.3f}, {intensity_max:.3f}]")
            
            total_original += original_count
            total_processed += processed_count
            
        except Exception as e:
            print(f"{frame_idx:<8} ❌ Error: {e}")
    
    print("-" * 70)
    total_reduction = 100 * (1 - total_processed / total_original) if total_original > 0 else 0
    print(f"{'TOTAL':<8} {total_original:<15,} {total_processed:<15,} {total_reduction:<11.1f}%\n")
    
    print("✅ Processing complete!\n")

if __name__ == "__main__":
    main()
