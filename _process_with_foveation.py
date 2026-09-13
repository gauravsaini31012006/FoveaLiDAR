#!/usr/bin/env python
"""FoveaLiDAR - Full Pipeline with Phase 3 (Foveation) Processing and Autonomous 3D Video."""

import sys
import os
import json
import time
import io
from pathlib import Path
import numpy as np

# Configure UTF-8 safe stdout for Windows
if sys.platform == "win32":
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
    except Exception:
        pass

from fovealidar.data.loaders import DatasetFactory
from fovealidar.preprocessing import PreprocessingConfig, PreprocessingPipeline
from fovealidar.foveation.gaze_simulator import GazeSimulator
from fovealidar.foveation.roi_definition import ROIDefinition
from fovealidar.foveation.adaptive_compression import AdaptiveCompression
from fovealidar.visualization.renderer import PointCloudRenderer
from fovealidar.visualization.frame_generator import FrameGenerator, VideoComposer
from fovealidar.core.logging import setup_logging, get_logger

logger = get_logger(__name__)

KITTI_ROOT = Path("D:/CLG/SIH_2026/kitti_raw_data of lidar")
OUTPUT_DIR = Path("D:/CLG/SIH_2026/AI_code/output")
NUM_FRAMES = 233


def setup_output_directory() -> Path:
    """Create output directory structure."""
    output_dir = OUTPUT_DIR / "foveation_output"
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "frames").mkdir(exist_ok=True)
    (output_dir / "stats").mkdir(exist_ok=True)
    return output_dir


def main():
    """Main execution pipeline."""
    setup_logging()
    
    print("\n" + "=" * 80)
    print("  FOVEALIDAR -- PHASE 3: ADAPTIVE VARIABLE RESOLUTION 2.5D PERCEPTION")
    print("  Smart India Hackathon 2026 | Problem Statement 26053")
    print("=" * 80 + "\n")
    
    # Step 1: Load KITTI Dataset
    print("[STEP 1] Loading KITTI Dataset...")
    print("-" * 80)
    
    if not KITTI_ROOT.exists():
        print(f"[ERROR] KITTI path not found: {KITTI_ROOT}")
        return 1
    
    try:
        dataset = DatasetFactory.auto_detect(KITTI_ROOT)
        if dataset is None:
            print("[ERROR] Could not auto-detect dataset format")
            return 1
        print(f"[OK] Dataset loaded: {type(dataset).__name__} ({len(dataset)} total frames)")
    except Exception as e:
        print(f"[ERROR] Loading dataset: {e}")
        return 1
    
    num_frames = min(NUM_FRAMES, len(dataset))
    print(f"[INFO] Processing sequence: {num_frames} frames\n")
    
    # Step 2: Setup Preprocessing
    print("[STEP 2] Setting up Preprocessing Pipeline...")
    print("-" * 80)
    
    config = PreprocessingConfig(
        range_min=0.0, range_max=100.0,
        height_min=-2.8, height_max=8.0,
        normalize_intensity=True, remove_outliers=True
    )
    pipeline = PreprocessingPipeline(config)
    print("[OK] Preprocessing configured: [range: 0-100m, height: -2.8m to 8.0m, outliers: True]\n")
    
    # Step 3: Setup Foveation
    print("[STEP 3] Setting up Foveation & Multi-Tier Resolution Controller...")
    print("-" * 80)
    
    gaze_sim = GazeSimulator()
    roi_def = ROIDefinition(
        fovea_radius_m=10.0,
        periphery_radius_m=50.0,
    )
    compression = AdaptiveCompression(
        roi_def=roi_def,
        fovea_voxel_m=0.05,        # 5 cm nominal near-field resolution (PS 26053)
        periphery_voxel_m=0.25,    # 25 cm nominal periphery resolution
        outside_voxel_m=0.60,      # 60 cm nominal far-field resolution
    )
    print("[OK] Multi-tier foveation active: [0-10m @ 5cm | 10-50m @ 25cm | 50-100m @ 60cm]\n")
    
    # Step 4: Setup Visualization
    print("[STEP 4] Setting up Autonomous 3D Perspective Renderer (img2 style)...")
    print("-" * 80)
    
    output_dir = setup_output_directory()
    renderer = PointCloudRenderer(image_width=1280, image_height=720)
    frame_gen = FrameGenerator(output_dir / "frames", image_width=1280, image_height=720)
    print("[OK] Renderer ready: 1280x720 3D Chase Camera with 3D Bounding Boxes & Ground Road Rings\n")
    
    # Step 5: Process Frames
    print("[STEP 5] Processing Frames & Generating Visualizations...")
    print("-" * 80)
    print(f"{'Frame':<8} {'Original':<12} {'Preproc':<12} {'Foveated':<12} {'Reduction':<12} {'Time':<8}")
    print("-" * 80)
    
    total_original = 0
    total_preprocessed = 0
    total_compressed = 0
    start_time = time.time()
    
    frame_stats = []
    
    for frame_idx in range(num_frames):
        t0 = time.time()
        try:
            frame = dataset[frame_idx]
            original_pc = frame.point_cloud
            preprocessed_pc = pipeline.process(original_pc)
            
            # Simulated forward driving gaze / saliency gaze
            if frame_idx % 2 == 0:
                gaze_points = gaze_sim.simulate_driving_gaze(preprocessed_pc)
            else:
                gaze_points = gaze_sim.simulate_saliency_gaze(preprocessed_pc)
            
            roi_def.rois.clear()
            if len(gaze_points) > 0:
                primary_gaze = gaze_points[0]
                roi_def.add_roi((float(primary_gaze[0]), float(primary_gaze[1]), float(primary_gaze[2])))
            else:
                primary_gaze = np.array([20.0, 0.0, 0.0], dtype=np.float32)
                roi_def.add_roi((20.0, 0.0, 0.0))
            
            # Multi-tier foveated variable resolution compression
            compressed_pc = compression.compress_adaptive(preprocessed_pc, strategy="voxel", preserve_background=True)
            roi_ids, _ = roi_def.classify_points(compressed_pc)
            
            orig_cnt = len(original_pc.points)
            prep_cnt = len(preprocessed_pc.points)
            comp_cnt = len(compressed_pc.points)
            comp_ratio = 100.0 * (1.0 - comp_cnt / max(1, prep_cnt))
            
            # Render realistic 3D autonomous driving view matching img2
            render_image = renderer.render_autonomous_3d(
                pc=compressed_pc,
                region_ids=roi_ids,
                gaze_pos=primary_gaze,
                frame_idx=frame_idx,
                color_mode='autonomous',
                raw_point_count=orig_cnt,
            )
            
            # Save frame image
            frame_gen.generate_frame(render_image, frame_idx)
            
            dt = time.time() - t0
            total_original += orig_cnt
            total_preprocessed += prep_cnt
            total_compressed += comp_cnt
            
            frame_stat = {
                'frame_idx': frame_idx,
                'raw_points': orig_cnt,
                'preprocessed_points': prep_cnt,
                'foveated_points': comp_cnt,
                'reduction_percent': round(comp_ratio, 2),
                'gaze_target': [float(primary_gaze[0]), float(primary_gaze[1]), float(primary_gaze[2])],
                'processing_time_s': round(dt, 4),
            }
            frame_stats.append(frame_stat)
            
            if frame_idx % 15 == 0 or frame_idx == num_frames - 1:
                print(f"{frame_idx:<8} {orig_cnt:<12,} {prep_cnt:<12,} {comp_cnt:<12,} {comp_ratio:<11.1f}% {dt*1000:<6.0f}ms")
            
        except Exception as e:
            print(f"{frame_idx:<8} [ERROR]: {str(e)[:50]}")
    
    elapsed_total = time.time() - start_time
    total_ratio = 100.0 * (1.0 - total_compressed / max(1, total_preprocessed))
    
    print("-" * 80)
    print(f"{'TOTAL':<8} {total_original:<12,} {total_preprocessed:<12,} {total_compressed:<12,} {total_ratio:<11.1f}% {elapsed_total:.1f}s\n")
    
    # Save JSON summary statistics
    stats_file = output_dir / "stats" / "foveation_summary.json"
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_frames': num_frames,
            'total_original_points': total_original,
            'total_foveated_points': total_compressed,
            'overall_reduction_percent': round(total_ratio, 2),
            'avg_reduction_percent': round(float(np.mean([s['reduction_percent'] for s in frame_stats])), 2),
            'avg_fps': round(num_frames / max(0.1, elapsed_total), 2),
            'frames': frame_stats,
        }, f, indent=2)
    print(f"[OK] Telemetry saved: {stats_file}")
    
    # Step 6: Generate MP4 Video
    print("\n[STEP 6] Generating MP4 Video at 15 FPS...")
    print("-" * 80)
    
    video_composer = VideoComposer(fps=15)
    frames_dir = output_dir / "frames"
    video_path = output_dir / "foveation_visualization.mp4"
    
    try:
        success = video_composer.compose_video(frames_dir, output_path=video_path, fps=15)
        if success and video_path.exists():
            size_mb = video_path.stat().st_size / (1024 * 1024)
            print(f"[OK] Video created successfully: {video_path} ({size_mb:.2f} MB)\n")
        else:
            print(f"[WARNING] Video composition returned {success}\n")
    except Exception as e:
        print(f"[ERROR] Video composition error: {e}\n")
    
    # Summary
    print("=" * 80)
    print("  FOVEALIDAR PHASE 3 -- ALL STEPS COMPLETE!")
    print("=" * 80)
    print(f"Output Directory: {output_dir}")
    print(f"Frames Generated: {len(list(frames_dir.glob('frame_*.png')))} PNGs")
    print(f"MP4 Video:        {video_path}")
    print(f"Compression:      {total_ratio:.1f}% bandwidth/memory reduction")
    print(f"Processing Speed: {num_frames / elapsed_total:.1f} FPS\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
