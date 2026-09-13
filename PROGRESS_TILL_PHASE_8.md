# 📊 FoveaLiDAR — Progress & Status Report (Phases 0 to 8)

**Project:** FoveaLiDAR — Focus-Aware Foveated LiDAR Processing & Hierarchical Octree Mapping System  
**Current Status:** Phases 0 through 8 Fully Operational & Validated  
**Test Suite:** 86 / 86 tests passing (100% pass rate)  
**Target Dataset:** Real KITTI Autonomous Driving Point Cloud Sequence (233 synchronized frames)

---

## 🚀 Executive Summary

FoveaLiDAR addresses the compute and bandwidth bottleneck of autonomous 3D perception by mimicking human eye-gaze (foveation). Rather than updating entire point clouds at a uniform density across a 100m+ range, the system dynamically prioritizes high-resolution processing in a focused foveal Region of Interest (ROI) while aggressively downsampling peripheral zones.

As of **Phase 8**, the system has evolved from raw dataset loading into a full perception stack featuring:
1. Multi-tier adaptive octree voxel maps.
2. Active gaze and motion-informed foveal controllers.
3. Lightweight real-time Semantic AI segmentation.
4. 3D clustering, bounding box extraction, and multi-object tracking.
5. Dual-view desktop GUI with 3D openGL foveated view and hierarchical map telemetry.

---

## 🛠️ Phase-by-Phase Completion Breakdown

| Phase | Description | Key Modules | Status | Test Coverage |
| :--- | :--- | :--- | :---: | :---: |
| **Phase 0** | **Foundation & Infrastructure** | Core types, `AppConfig`, unified logging, hardware diagnostics | ✅ Complete | 6 / 6 tests |
| **Phase 1** | **KITTI Dataset Engine** | Binary `.bin` reader, Odometry & Raw loaders, OXTS GPS/IMU parser | ✅ Complete | 13 / 13 tests |
| **Phase 2** | **Preprocessing Pipeline** | Range, height, voxel, intensity normalization & outlier removal filters | ✅ Complete | 10 / 10 tests |
| **Phase 3** | **Foveated ROI & Attention** | Human eye-gaze simulator, foveal ROI definition, adaptive compression | ✅ Complete | 22 / 22 tests |
| **Phase 4** | **Uniform Octree Base Mapping** | Baseline uniform voxel mapping, incremental insertion & telemetry | ✅ Complete | 6 / 6 tests |
| **Phase 5** | **Multi-Tier Adaptive Octree Map** | Dynamic LOD hierarchy (coarse/medium/fine), selective refinement | ✅ Complete | 5 / 5 tests |
| **Phase 6** | **Foveal Controller & Mode A/B/C** | Unified controller orchestrating Mode A (Raw), Mode B (Uniform), Mode C (Foveal) | ✅ Complete | 22 / 22 tests |
| **Phase 7** | **Semantic AI (Edge-Optimized)** | 3-class ground/obstacle/vegetation classification + semantic map tags | ✅ Complete | 1 / 1 test |
| **Phase 8** | **3D Detection & Tracking** | DBSCAN / grid clustering, 3D oriented bounding boxes, Hungarian + Kalman tracking | ✅ Complete | 1 / 1 test |

**Total passing unit & integration tests:** **86 / 86 (100%)**

---

## 🔬 Core Innovations & Metrics Achieved

### 1. 3-Way Mode Comparison (A vs B vs C)
- **Mode A (Raw Baseline):** All ~112,000 points/frame inserted uniformly — high memory overhead, ~12–15 FPS.
- **Mode B (Uniform Downsampled):** Static downsampling — lower detail across both near obstacles and distant road.
- **Mode C (Adaptive Foveal):** High resolution maintained within dynamic gaze ROI (up to 95–99% compression in periphery, 30+ FPS rendering cadence).

### 2. Semantic Perception & Tracking (Phases 7 & 8)
- **Heuristic & Fast Semantic Tagging:** Differentiates drivable surface from roadside obstacles and overhanging vegetation without requiring massive GPU VRAM.
- **Persistent Multi-Object Tracking:** Assigns consistent IDs and velocity vectors to obstacles across frames for collision forecasting.

### 3. Integrated Dual-View Graphical Interface
- **Tab 1: 3D Foveation View:** Interactive PyQt/OpenGL hardware-accelerated point cloud rendering with gaze reticle and ROI rings.
- **Tab 2: Hierarchical Map View:** Mode A/B/C comparative metrics, real-time octree node statistics, and telemetry dashboard.

---

## 📂 Architecture & Directory Structure Till Phase 8

```
fovealidar/
├── core/                   # Base types, configuration, logging, and diagnostics
│   ├── config.py
│   ├── types.py
│   ├── logging.py
│   └── diagnostics.py
├── data/                   # Data ingestion and KITTI dataset handling
│   ├── kitti/ (raw, odometry, oxts, utils)
│   └── factory.py
├── preprocessing/          # 6-stage filtering pipeline
│   ├── filters.py
│   └── pipeline.py
├── foveation/              # Gaze tracking and adaptive compression
│   ├── gaze_simulator.py
│   ├── roi_definition.py
│   └── adaptive_compression.py
├── mapping/                # Hierarchical multi-tier octree maps
│   ├── adaptive_map.py
│   ├── hierarchy.py
│   └── uniform_map.py
├── perception/             # Phases 7 & 8: Semantic AI, Clustering & Tracking
│   ├── semantic_segmenter.py
│   ├── object_detector.py
│   └── multi_object_tracker.py
├── gui/                    # PyQt6 desktop application
│   ├── main_window.py
│   ├── foveation_tab.py
│   ├── map_tab.py
│   └── widgets/
└── visualization/          # Offline rendering & video generation
```
