# FoveaLiDAR — Master Build Specification for Claude Code

## 0. Purpose

This file is the authoritative engineering specification for the FoveaLiDAR project. Claude Code must read this file before making changes and must follow its modular architecture, phased implementation process, testing rules, and performance constraints.

Build a **Windows desktop application** that processes sequential KITTI Velodyne LiDAR data and demonstrates the Smart India Hackathon 2026 Problem Statement 26053:

> Adaptive Variable Resolution 2.5D Lidar Mapping for Dynamic Environment Perception

The prototype target machine is:
- Intel Core i7-13620H
- 16 GB RAM
- NVIDIA RTX 3050 6 GB VRAM
- Windows 10/11 64-bit

The long-term goal is graceful operation on lower-end/CPU-only systems without redesigning the application.

This is a prototype/research demonstrator, not a safety-certified autonomous-driving system.

---

# 1. Core Product Goal

Create **FoveaLiDAR**, a modular, efficient, compute-aware, foveated LiDAR perception and mapping engine.

Core pipeline:

KITTI LiDAR frame
-> preprocessing
-> geometry + optional AI perception
-> semantic/object/uncertainty information
-> foveal importance calculation
-> adaptive spatial resolution
-> 2.5D elevation/occupancy/semantic map
-> visualization
-> benchmark

The key idea is that **not all space receives equal representation or compute**.

Near-field space is inherently important and must receive high detail. Far-field space starts coarse. Regions that are semantically important, uncertain, geometrically complex, or dynamically changing can be selectively refined.

The project must therefore demonstrate both:

1. **spatial adaptation** — variable map resolution;
2. **computational adaptation** — variable AI/refinement effort.

---

# 2. Problem Statement Requirements to Implement

The implementation must directly cover:

- terrain analysis / drivable vs non-drivable estimation;
- static obstacles;
- dynamic objects;
- non-uniform spatial representation;
- resolution increasing toward the vehicle;
- correct 3D-to-2.5D projection without gaps/overlaps/data loss;
- memory/compute reduction compared with a uniform high-resolution representation;
- real-time/low-latency visualization;
- useful perception quality across distances.

Mandatory prototype baseline:

- **0–10 m: nominal 5 cm resolution**
- **10–100 m: nominal 50 cm resolution**

FoveaLiDAR must treat these as the PS-aligned baseline, not as an optional suggestion.

---

# 3. Research Conclusions and Open-Source Foundations

The project should reuse established open-source software and research concepts rather than reinventing everything.

## 3.1 KITTI / SemanticKITTI

Use the user's KITTI raw Velodyne sequence as the primary data source.

Current data is organized approximately as:

    <drive_root>/
        velodyne_points/
            data/
                0000000000.bin
                0000000001.bin
                ...

Each frame is an array of float32 values representing:

    X, Y, Z, intensity/remission

Use streaming/incremental loading. Never load the entire sequence into RAM.

SemanticKITTI's API is an important reference for KITTI point-cloud loading, sequence organization, labels, and evaluation concepts.

Reference:
https://github.com/PRBonn/semantic-kitti-api

## 3.2 Apollo

Apollo is a full autonomous-driving platform, but **do not install Apollo/Cyber RT for this project**.

Use Apollo as an architecture reference only. Its perception organization demonstrates useful separation between:

- preprocessing
- ROI
- ground processing
- LiDAR detection
- filtering
- tracking

Apollo also exposes pluggable LiDAR detection implementations including CenterPoint and PointPillars in its configuration.

Reference:
https://github.com/ApolloAuto/apollo

## 3.3 SalsaNext

SalsaNext is the primary semantic-perception candidate because it was designed for real-time full-3D-LiDAR semantic segmentation and explicitly estimates epistemic and aleatoric uncertainty.

Reference:
https://github.com/TiagoCortinhal/SalsaNext

Use it behind a backend interface. Do not make it a hard dependency of the core mapper. The upstream project uses older environments, so prefer a compatible inference path/adaptor rather than making the whole application depend on an obsolete stack.

## 3.4 CenterPoint / PointPillars

For object-centric perception, use a standalone CenterPoint-compatible implementation when practical. OpenPCDet is a reference for KITTI-compatible 3D detection and provides multiple LiDAR detectors.

References:
https://github.com/open-mmlab/OpenPCDet
https://github.com/ApolloAuto/apollo

CenterPoint/PointPillars must be optional. Detection must never be required for the basic mapping demo.

## 3.5 Open3D

Use Open3D for point-cloud structures, useful processing, and visualization. Prefer NumPy/vectorized arrays for hot-path numerical operations.

Reference:
https://github.com/isl-org/Open3D

## 3.6 PySide6

Use PySide6 for the desktop application. Keep heavy processing outside the Qt GUI thread.

Reference:
https://doc.qt.io/qtforpython-6/

## 3.7 ANYbotics Elevation Mapping

Use as a conceptual reference for elevation maps, terrain representation, and uncertainty. Do not require ROS or the full package in the Windows prototype.

Reference:
https://github.com/ANYbotics/elevation_mapping

## 3.8 OctoMap

Use OctoMap as a conceptual reference for hierarchical spatial refinement. Do not use a giant 3D octree as the core map for this project; we need a lightweight 2.5D representation.

Reference:
https://github.com/OctoMap/octomap

## 3.9 RangeNet++

RangeNet++/RangeNet library is a reference for range-image LiDAR semantic segmentation. It can be considered for a later optional backend, but its old TensorRT assumptions make it less attractive for the first Windows implementation.

Reference:
https://github.com/PRBonn/rangenet_lib

## 3.10 RoadRunner M&M

RoadRunner M&M demonstrates multi-range, multi-resolution traversability/elevation prediction and is an important research reference for the idea of high detail nearby and lower detail farther away.

Reference:
https://leggedrobotics.github.io/roadrunner_mm/

Do not claim adaptive LiDAR itself is novel. The proposed contribution is the integrated, lightweight, vehicle-oriented, semantic/uncertainty/geometry/motion-aware and compute-aware foveated 2.5D pipeline.

---

# 4. Novelty Positioning

Never say:

> We invented adaptive LiDAR.

Use language such as:

> FoveaLiDAR integrates distance-adaptive 2.5D mapping with semantic importance, uncertainty, geometry and temporal change to perform compute-aware foveated perception. The system protects detail where it matters and reduces unnecessary representation and inference elsewhere.

Innovation goals:

- 5 cm near / 50 cm far PS-aligned mapping;
- region-based foveation;
- semantic-aware refinement;
- uncertainty-aware refinement;
- geometry-complexity refinement;
- temporal-change/motion refinement;
- compute-budget-aware refinement;
- adaptive AI inference frequency;
- low-end graceful degradation;
- explicit A/B/C benchmarking.

---

# 5. High-Level Architecture

    KITTI LiDAR
         |
         v
    Dataset Adapter
         |
         v
    LiDARFrame
         |
         v
    Preprocessing
         |
      +--+-------------------+
      |                      |
      v                      v
    Geometry            AI Perception
      |               +--------+--------+
      |               |                 |
      |               v                 v
      |           Semantic          Object Detector
      |           Backend            optional
      |               |                 |
      +---------------+-----------------+
                      |
                      v
                   Tracker
                      |
                      v
               FOVEAL CONTROLLER
                      |
        +-------------+-------------+
        |       |       |      |    |
     Distance Semantic Uncert. Geometry Motion
        |       |       |      |    |
        +-------+-------+------+------+ 
                      |
                Compute Budget
                      |
                      v
               Adaptive Grid
                /          \
             coarse       refined
             50 cm       20/10/5 cm
                \          /
                      |
                      v
                   2.5D Map
                      |
             +--------+--------+
             |        |        |
         Elevation Occupancy Semantics
             |        |        |
             +--------+--------+
                      |
                      v
                Visualization
                      |
                      v
                   Benchmark

---

# 6. Repository Structure

Use this structure, or an equivalent structure with the same strict module boundaries:

    FoveaLiDAR/
    |
    +-- app/
    |   +-- main.py
    |   +-- application.py
    |   +-- app_state.py
    |   +-- services.py
    |
    +-- config/
    |   +-- default.yaml
    |   +-- semantic_labels.yaml
    |   +-- profiles/
    |       +-- rtx3050.yaml
    |       +-- balanced.yaml
    |       +-- low_power.yaml
    |       +-- cpu.yaml
    |
    +-- core/
    |   +-- frame.py
    |   +-- perception_result.py
    |   +-- map_result.py
    |   +-- events.py
    |   +-- enums.py
    |   +-- interfaces.py
    |
    +-- data/
    |   +-- base_loader.py
    |   +-- kitti_loader.py
    |   +-- sequence_index.py
    |   +-- playback.py
    |   +-- calibration.py
    |
    +-- preprocessing/
    |   +-- pipeline.py
    |   +-- filters.py
    |   +-- roi_filter.py
    |   +-- self_filter.py
    |   +-- downsampler.py
    |   +-- ground_filter.py
    |
    +-- geometry/
    |   +-- height_features.py
    |   +-- density.py
    |   +-- roughness.py
    |   +-- slope.py
    |   +-- local_complexity.py
    |
    +-- perception/
    |   +-- base_semantic.py
    |   +-- heuristic_semantic.py
    |   +-- semantic_manager.py
    |   +-- salsanext.py
    |   +-- input_adapter.py
    |   +-- uncertainty.py
    |   +-- detection/
    |       +-- base_detector.py
    |       +-- heuristic_detector.py
    |       +-- centerpoint.py
    |       +-- pointpillars.py
    |       +-- detector_manager.py
    |
    +-- tracking/
    |   +-- base_tracker.py
    |   +-- simple_tracker.py
    |   +-- motion_features.py
    |
    +-- foveation/
    |   +-- importance.py
    |   +-- distance_score.py
    |   +-- semantic_score.py
    |   +-- uncertainty_score.py
    |   +-- geometry_score.py
    |   +-- motion_score.py
    |   +-- compute_budget.py
    |   +-- refinement_policy.py
    |
    +-- mapping/
    |   +-- cell.py
    |   +-- hierarchy.py
    |   +-- uniform_grid.py
    |   +-- adaptive_grid.py
    |   +-- projection.py
    |   +-- elevation_map.py
    |   +-- map_store.py
    |   +-- boundary_alignment.py
    |
    +-- benchmark/
    |   +-- metrics.py
    |   +-- profiler.py
    |   +-- comparison.py
    |   +-- recorder.py
    |   +-- reports.py
    |
    +-- visualization/
    |   +-- pointcloud_view.py
    |   +-- map_view.py
    |   +-- foveal_view.py
    |   +-- object_view.py
    |   +-- metrics_view.py
    |   +-- legends.py
    |
    +-- ui/
    |   +-- main_window.py
    |   +-- dashboard.py
    |   +-- workers.py
    |   +-- panels/
    |       +-- dataset_panel.py
    |       +-- perception_panel.py
    |       +-- foveation_panel.py
    |       +-- performance_panel.py
    |       +-- map_panel.py
    |       +-- log_panel.py
    |
    +-- tests/
    |   +-- unit/
    |   +-- integration/
    |   +-- performance/
    |
    +-- tools/
    |   +-- inspect_kitti.py
    |   +-- benchmark_sequence.py
    |   +-- validate_environment.py
    |
    +-- models/
    |   +-- README.md
    |   +-- manifests/
    |
    +-- docs/
    +-- logs/
    +-- outputs/
    +-- THIRD_PARTY.md
    +-- MODEL_SOURCES.md
    +-- README.md
    +-- pyproject.toml
    +-- CLAUDE.md
    +-- main.py

No UI module may contain mapping/perception algorithms.

---

# 7. Stable Data Contracts

Use typed dataclasses or equivalent schemas.

## LiDARFrame

Must contain at least:
- frame_id
- timestamp
- points: Nx4 float array or equivalent x/y/z/intensity representation
- source_path
- coordinate_frame
- metadata

## PerceptionResult

Must contain:
- semantic_labels
- semantic_confidence
- point_uncertainty
- detections
- inference_time_ms
- backend_name
- uncertainty_source

## GeometryFeatures

Must contain region/cell-level:
- height_mean
- height_min
- height_max
- roughness
- slope
- density
- density_gradient
- complexity

## FovealScoreMap

Must contain:
- spatial region IDs
- importance score
- refinement level
- contributor scores
- refinement reason

## MapCell

Must contain:
- cell ID
- center x/y
- cell size
- level
- point count
- elevation min/max/mean
- occupancy
- semantic class
- semantic confidence
- uncertainty
- importance
- timestamp/frame ID

## MapResult

Must contain:
- active cells
- resolution distribution
- bounds
- mapping time
- cell statistics

Modules should exchange these contracts rather than undocumented Python dictionaries.

---

# 8. KITTI Loader

The loader must:
- discover `.bin` files automatically;
- sort frame IDs numerically;
- validate file structure;
- load one frame at a time;
- map columns to x/y/z/intensity;
- report malformed files clearly;
- expose frame count and current frame;
- support play/pause/next/previous;
- never preload a whole sequence into memory.

The raw read will normally be based on:

    np.fromfile(path, dtype=np.float32).reshape(-1, 4)

but must include validation and a clean API.

The dataset replay target is approximately KITTI's frame cadence; default playback should be ~10 Hz, with user controls for slower/faster replay.

---

# 9. Processing Pipeline

The pipeline must be asynchronous relative to the Qt UI.

Recommended flow:

    UI
     -> ApplicationController
     -> ProcessingService
     -> bounded frame queue
     -> worker
     -> Pipeline
     -> result/event
     -> UI

Never do expensive LiDAR/AI processing in the GUI thread.

Do not allow an unbounded frame queue. Prefer latest-frame processing when the producer is faster than processing.

Track separately:
- source/replay FPS
- processing FPS
- rendering FPS

---

# 10. Preprocessing

Implement vectorized filters:

1. finite-value filter;
2. range/ROI filter;
3. optional height filter;
4. optional self/vehicle filter;
5. optional outlier removal;
6. optional downsampling.

All values must be configurable.

Keep raw frame data immutable from the loader's perspective.

Return preprocessing statistics:
- input point count
- output point count
- points removed by stage
- processing time

Avoid Python loops over individual points in hot paths.

---

# 11. Geometry Engine

Compute inexpensive spatial features using vectorized binning.

At minimum:
- point density
- min/max/mean Z
- local height variance
- roughness
- slope/discontinuity proxy
- density gradient
- local complexity

Do not use O(N^2) neighbor computation over the full point cloud.

Spatial bins, efficient reduction, KD-tree only where justified, are preferred.

---

# 12. 2.5D Representation

Do NOT create a huge dense 3D voxel volume.

Represent each planar X-Y cell with vertical/elevation information.

Conceptual cell:

    X/Y footprint
       +
    elevation statistics
       +
    occupancy
       +
    semantic information
       +
    uncertainty
       +
    importance
       +
    resolution level

Use compact NumPy arrays/structured arrays or other memory-efficient representations where appropriate. Avoid millions of Python objects.

---

# 13. Uniform Baseline

Implement **Mode A: Uniform 5 cm**.

Use the same configured ROI as the adaptive modes so comparison is fair.

Measure actual:
- active cells
- mapping time
- end-to-end latency
- RAM if feasible
- rendered/processing FPS

This is the high-resolution baseline.

---

# 14. Required Adaptive Map

Implement **Mode B**:

    0 <= distance <= 10 m     -> 0.05 m nominal cells
    10 < distance <= 100 m    -> 0.50 m nominal cells

Use a deterministic integer lattice based on 5 cm.

Because 50 cm = 10 x 5 cm, coarse cells should align exactly to the fine lattice.

Do NOT independently round arbitrary floating-point boundaries per point.

The mapper must guarantee:
- deterministic assignment;
- no gaps in valid map coverage;
- no accidental overlaps;
- no double-counting of points;
- stable cell IDs;
- correct boundary behavior;
- clear parent/child hierarchy when refined.

Explicitly test:
- x/y = 0;
- negative x/y;
- exact 5 cm boundary;
- exact 50 cm boundary;
- just below/above both boundaries;
- distance exactly 10 m;
- distance just below/above 10 m.

---

# 15. Hierarchical Refinement

Support at least:

- 50 cm coarse
- 20 cm optional
- 10 cm optional
- 5 cm fine

The 20/10 cm levels support foveation and are not a substitute for the required 5 cm/50 cm baseline comparison.

When a coarse cell is refined:
- parent/child relationship must be explicit;
- active visual representation must not double-count the parent;
- children must fully cover the intended parent area;
- no holes or overlaps;
- aggregate parent information may be retained for diagnostics but the active map must know which cells are authoritative.

---

# 16. Foveal Controller

The foveal controller is one of the project's key innovation components.

It must compute an importance score per spatial region, not merely per individual point.

Base formulation:

    importance =
        Wd * distance_score +
        Ws * semantic_score +
        Wu * uncertainty_score +
        Wg * geometry_score +
        Wm * motion_score

Normalize all contributors to [0,1].

Starting example weights:
- distance 0.20
- semantic 0.25
- uncertainty 0.25
- geometry 0.15
- motion 0.15

These are tunable starting values, not scientifically optimal constants.

Expose weights through configuration and UI.

---

# 17. Semantic Importance

Create `semantic_labels.yaml` so model label IDs are not hard-coded throughout the application.

Use configurable importance levels. A reasonable conceptual priority is:

Critical/high:
- pedestrian
- cyclist
- vehicle
- unknown object in vehicle corridor
- static obstacle

Medium:
- structural edges
- relevant vegetation

Low:
- distant homogeneous background

The mapping must match the actual selected model's label set.

Never assume a class exists if the model does not output it.

---

# 18. Uncertainty

If the semantic model gives uncertainty, use the model-provided value after suitable normalization.

If only confidence is available, derive a clearly labelled confidence-derived proxy.

If neither is available, use a heuristic uncertainty signal only as a fallback.

The UI and exported metrics must report the uncertainty source as one of:

- model-provided
- confidence-derived
- heuristic
- unavailable

Never call a heuristic score "epistemic uncertainty".

---

# 19. Geometry Score

Raise geometry importance for:
- high height variance;
- rough surfaces;
- large local vertical discontinuity;
- sharp density changes;
- unusual structure relative to neighboring cells.

Flat homogeneous road should normally have a low geometry score.

Curbs, potholes, barriers, irregular terrain and unusual vertical structure should normally receive higher priority.

---

# 20. Motion / Temporal Score

Use consecutive KITTI frames.

At minimum calculate a cheap region-level change signal from:
- occupancy change;
- density change;
- centroid change;
- semantic change where available.

If an object detector/tracker is available, use object motion.

A dynamic or newly changed region should become more important.

Temporal reuse must never silently pretend stale data is current. Store frame age/timestamp for reused estimates.

---

# 21. Compute-Aware Foveation

Add a `ComputeBudgetController`.

Inputs:
- processing FPS
- total latency
- CPU
- RAM
- GPU availability
- GPU utilization
- VRAM if available
- selected user mode

Outputs:
- AI inference frequency;
- point downsampling level;
- refinement threshold;
- max refined cells/regions;
- rendering frequency.

Modes:

## MAX QUALITY
- full available AI
- frequent inference
- more refinement
- high rendering rate

## BALANCED
- normal AI cadence
- moderate refinement
- normal rendering

## LOW POWER
- less frequent AI
- stronger far-field reduction
- fewer refined regions
- lower rendering rate

## CPU ONLY
- no GPU neural inference
- heuristic perception
- adaptive map remains operational

Degradation should be graceful, not catastrophic.

When compute becomes constrained, prefer reducing expensive AI/refinement/visualization before disabling the core near-field mapping.

---

# 22. AI Inference Scheduling

Do not assume expensive semantic AI must run on every frame.

Implement configurable temporal inference:

- every N frames;
- early trigger when scene change is high;
- early trigger when new/uncertain region appears;
- optional reuse of stable results.

Make the reuse policy explicit and measurable.

Record:
- actual AI frame count;
- skipped AI frames;
- reused frames;
- trigger reason.

---

# 23. Semantic Backend Interface

Create a stable interface, for example:

    class SemanticBackend(Protocol):
        def infer(self, frame: LiDARFrame) -> PerceptionResult: ...

Implement:

- `HeuristicSemanticBackend`
- `SalsaNextBackend`

The exact class/interface naming can differ, but backend isolation is mandatory.

The application must continue to function if SalsaNext is not installed, cannot load weights, or cannot initialize CUDA.

Fallback behavior:
1. log the real error;
2. switch to configured fallback;
3. display fallback state in UI;
4. continue mapping;
5. never silently claim the neural backend is active.

---

# 24. Object Detector Interface

Create a pluggable object detector interface.

Possible backends:
- heuristic detector
- CenterPoint
- PointPillars

Do not force installation of all models.

Detector output should support:
- class
- confidence
- 3D center
- dimensions
- yaw
- optional velocity
- source backend
- timestamp
- track ID when tracker attaches one

---

# 25. Tracker

Start with a lightweight tracker.

A simple center-distance/IoU-style association is acceptable for the prototype. Kalman filtering can be added later if profiling and stability justify it.

Track:
- ID
- class
- position
- age
- last seen
- optional velocity
- motion score

Tracking exists primarily to improve dynamic-region prioritization.

---

# 26. Terrain / Traversability

Provide a prototype layer:

- likely drivable
- likely non-drivable
- unknown

Heuristic inputs may include:
- elevation
- slope
- roughness
- discontinuity
- semantic class

Label heuristic results honestly.

Do not describe this as a production safety-certified traversability classifier.

---

# 27. Static vs Dynamic

Static/dynamic understanding should combine:
- detector/tracker data where available;
- temporal change fallback where detector is unavailable.

Dynamic regions should receive higher foveal priority.

---

# 28. Desktop UI

Use **PySide6**.

The appearance should be polished and resemble an engineering/industrial perception dashboard rather than a notebook.

Suggested layout:

    +---------------------------------------------------------------+
    | FOVEALiDAR | Drive | Frame | STATUS | FPS | Latency           |
    +---------------+--------------------------------+--------------+
    | DATASET       |                                | PERCEPTION   |
    |               |                                |              |
    | sequence      |         3D LiDAR VIEW          | classes      |
    | frame         |                                | confidence   |
    | ▶ Play        |                                | uncertainty  |
    | Pause         |                                | detections   |
    | step          |                                |              |
    +---------------+--------------------------------+--------------+
    | RESOLUTION    |        2.5D ADAPTIVE MAP        | FOVEATION    |
    |               |                                |              |
    | 5 cm near     | resolution/cell overlay        | importance   |
    | 50 cm far     | elevation/terrain view         | refinement   |
    +---------------+--------------------------------+--------------+
    | PERFORMANCE   |        TIMELINE / CHARTS         | CELL DETAIL  |
    | FPS           |                                | X/Y          |
    | latency       |                                | resolution   |
    | points        |                                | elevation    |
    | cells         |                                | semantics    |
    | RAM/VRAM      |                                | importance   |
    +---------------+--------------------------------+--------------+

Main visual modes:
- RAW 3D
- HEIGHT
- SEMANTIC
- IMPORTANCE
- RESOLUTION
- TERRAIN
- OBJECTS

Controls:
- play/pause
- previous/next
- replay speed
- reset camera
- processing mode
- AI backend
- Uniform / Adaptive / Foveal
- benchmark
- export

---

# 29. Interactive Cell Inspection

Clicking a map cell must show:

- cell ID
- x/y
- resolution
- hierarchy level
- point count
- elevation min/max/mean
- occupancy
- semantic class
- confidence
- uncertainty
- importance
- importance contributors
- frame/timestamp

This feature is important for debugging and for SIH demonstrations.

---

# 30. Rendering Rules

Rendering must not become the processing bottleneck.

Processing FPS and rendering FPS are separate metrics.

Keep rendering update rate configurable.

When processing is faster than rendering, render the latest result instead of building a large queue.

When processing is slower, do not freeze the UI.

---

# 31. A/B/C Modes

These must be first-class benchmark modes.

## A — Uniform 5 cm

High-resolution baseline.

## B — Distance Adaptive

    <= 10 m -> 5 cm
    >10 to 100 m -> 50 cm

## C — Foveal

Distance-adaptive baseline plus:
- semantic importance
- uncertainty
- geometry
- motion/change
- compute budget

The comparison screen must show actual measured:

- active cells
- fine/medium/coarse counts
- mapping time
- AI time
- end-to-end latency
- processing FPS
- RAM
- VRAM when available
- refinement count
- cell reduction percentage

Never enter demonstration numbers manually.

---

# 32. Benchmark Methodology

Create:

    python tools/benchmark_sequence.py --sequence <path> --frames 100 --mode all

Benchmark requirements:
- warm-up frames excluded from primary statistics;
- fixed frame range;
- mean latency;
- median latency;
- p95 latency;
- standard deviation where useful;
- processing FPS;
- stage timing;
- system information.

Separate:
- I/O
- preprocessing
- geometry
- AI inference
- foveation
- mapping
- rendering
- total end-to-end

Store results to JSON/CSV for reproducibility.

Environment metadata must include:
- CPU
- GPU
- RAM
- Python version
- package versions
- model backend
- model weights identifier
- compute mode
- sequence
- frame range

---

# 33. Important Metrics

Prioritize:

1. latency;
2. processing FPS;
3. RAM;
4. VRAM;
5. active cells;
6. resolution distribution;
7. important-region preservation.

Also record:
- input points
- output points after filtering
- AI inference count
- refined regions
- dropped frames
- fallback events.

---

# 34. Accuracy / Quality Evaluation

Where compatible labels/annotations exist, support optional evaluation.

Possible metrics:
- semantic IoU/mIoU where the model label set matches the ground truth;
- object precision/recall where annotations are comparable;
- elevation error;
- terrain/traversability agreement where a valid reference exists.

Do not invent accuracy values.

Do not silently trade away perception quality to improve FPS.

---

# 35. Low-End Compatibility Strategy

The same architecture must degrade gracefully.

At startup detect:
- CPU
- RAM
- CUDA
- GPU model
- GPU memory where available.

Do not make a hard-coded GPU whitelist.

Recommended degradation sequence:

1. lower rendering rate;
2. reduce AI inference frequency;
3. reduce far-field input density;
4. reduce max refined regions;
5. use heuristic perception fallback;
6. CPU-only mode.

Preserve the core near-field map unless the user explicitly changes policy.

---

# 36. GPU Policy

Do not send everything to CUDA.

Use GPU for operations where it materially helps, especially neural inference.

Keep cheap operations on CPU/NumPy where appropriate.

Avoid repeated CPU/GPU transfers.

If CUDA is unavailable:
- application still opens;
- KITTI still loads;
- preprocessing works;
- geometry works;
- adaptive mapping works;
- heuristic perception works;
- benchmark works.

---

# 37. Performance Engineering

Hot path requirements:
- vectorized NumPy;
- integer cell indexing;
- preallocation/reuse where beneficial;
- bounded caches;
- avoid Python object per point;
- avoid Python loops over ~100k points;
- avoid repeated model initialization;
- avoid re-reading configuration every frame;
- avoid full map recreation when an incremental update is practical.

Do not add C++/Cython/Numba before profiling proves a real bottleneck.

---

# 38. Configuration

No critical behavior should require editing source code.

Configuration must cover:
- dataset path;
- ROI;
- near/far ranges;
- resolutions;
- semantic label priorities;
- foveal weights;
- refinement thresholds;
- max refined regions;
- AI backend;
- detector backend;
- device;
- AI frequency;
- compute mode;
- rendering rate;
- benchmark settings.

Example:

    perception:
      semantic_backend: salsanext
      detector_backend: none
      device: auto

    mapping:
      near_range_m: 10.0
      far_range_m: 100.0
      near_resolution_m: 0.05
      far_resolution_m: 0.50

    foveation:
      enabled: true
      weights:
        distance: 0.20
        semantic: 0.25
        uncertainty: 0.25
        geometry: 0.15
        motion: 0.15

    compute:
      mode: balanced
      max_refined_regions: 128

---

# 39. Critical-Region Rules

Weighted importance is not the only mechanism.

Provide configurable hard/near-hard rules for critical regions, for example:
- near-field pedestrian -> strongly/always refine;
- near-field detected obstacle -> strongly/always refine;
- high-uncertainty region in vehicle corridor -> prioritize refinement.

Hard-priority rules must be visible and separately reported in diagnostics.

---

# 40. Vehicle-Centric Interest Corridor

Implement an optional forward travel corridor/cone.

Parameters:
- corridor width;
- maximum range;
- optional angular bounds.

Being in the corridor should increase importance, not override semantic/uncertainty information completely.

---

# 41. Calibration

The first version must work using only Velodyne coordinates.

Calibration support is optional but should be isolated for future:
- camera overlay;
- sensor transforms;
- image projection.

Do not mix coordinate frames silently.

---

# 42. Visualization Debug Modes

Implement developer toggles:

- SHOW_POINTS
- SHOW_CELL_IDS
- SHOW_CELL_BOUNDARIES
- SHOW_IMPORTANCE
- SHOW_UNCERTAINTY
- SHOW_RESOLUTION
- SHOW_POINT_ASSIGNMENT
- SHOW_TRACK_IDS
- SHOW_FILTERED_POINTS

Disabled by default if they hurt performance.

---

# 43. Debug Export

Allow a selected frame to be exported:

    outputs/
        frame_0000123/
            raw_points.npy
            filtered_points.npy
            map_cells.npz
            perception.json
            foveal_scores.npz
            metrics.json
            screenshot.png

Do not automatically write large output files for every frame.

---

# 44. Error Handling

Never use silent exception handling such as:

    except Exception:
        pass

Errors must be:
- logged;
- visible where relevant;
- associated with the module/stage;
- followed by a controlled fallback only when configured.

Example UI status:

    [AI WARNING]
    SalsaNext unavailable. Falling back to heuristic perception.
    Reason: <real exception message>

The program must never claim an AI model is active when it is not.

---

# 45. Logging

Use Python logging or a lightweight structured equivalent.

Log levels:
- DEBUG
- INFO
- WARNING
- ERROR
- CRITICAL

Log:
- startup;
- environment;
- dataset validation;
- frame timing;
- model loading;
- fallbacks;
- benchmark start/end;
- errors.

Use rotating log files under `logs/`.

Provide a compact live log panel in the UI.

---

# 46. Testing Requirements

## Loader tests
- valid `.bin`;
- malformed file;
- empty file;
- numeric ordering;
- expected N x 4 shape.

## Projection tests
- positive coordinates;
- negative coordinates;
- zero;
- exact grid boundaries;
- exact 10 m distance boundary;
- exact 50/5 cm alignment.

## Adaptive map tests
- <=10 m -> 5 cm;
- >10–100 m -> 50 cm baseline;
- distant important region refines;
- unimportant distant region stays coarse;
- no active parent/child overlap;
- no missing coverage inside a valid region.

## Foveation tests
- score range [0,1];
- weight changes affect result correctly;
- semantic priority works;
- uncertainty priority works;
- compute budget caps refinement.

## Perception tests
- identical backend interface;
- heuristic backend works without GPU;
- model failure triggers fallback;
- label mapping is deterministic.

## Tracking tests
- ID persistence;
- new object;
- object disappearance;
- basic motion score.

## Benchmark tests
- all stage timers work;
- percentile calculation works;
- missing GPU metrics do not crash;
- export file is valid.

## UI tests
- application starts;
- play/pause;
- next/previous;
- mode switch;
- error display;
- GUI remains responsive during processing.

---

# 47. Mapping Invariants — MUST PASS

The adaptive mapping engine is a critical component.

The following invariants must hold:

1. Every valid point maps to exactly one active cell at the selected active resolution.
2. Cell indexing is deterministic.
3. The 5 cm fundamental lattice prevents floating-point boundary drift.
4. An active coarse parent must not simultaneously be treated as active over an area represented by active children.
5. Valid map coverage contains no accidental holes.
6. Reprocessing the same frame produces the same map.
7. Resolution transitions are aligned.

These tests are mandatory before the project advances beyond adaptive mapping.

---

# 48. Data Ownership

Document ownership of arrays and maps.

Avoid accidental copies of large point clouds.

Use bounded caches for previous frames.

The raw point cloud should be read-only after load where practical.

---

# 49. AI Integration Strategy

Do NOT begin the project by integrating all neural networks.

Sequence:

A. heuristic geometry/perception;
B. semantic backend interface;
C. SalsaNext-compatible backend;
D. object detector interface;
E. CenterPoint/PointPillars when practical;
F. tracker;
G. semantic/uncertainty-driven foveation;
H. compute-aware scheduling.

The mapper must work before neural AI is added.

---

# 50. Model Choice Principle

Do not choose the largest model simply because it scores well in a paper.

Optimize for the combination of:
- semantic usefulness;
- uncertainty availability;
- inference speed;
- Windows compatibility;
- RTX 3050 6 GB fit;
- low-end fallback;
- maintainability;
- license compatibility.

Primary semantic candidate: SalsaNext-compatible backend.

Primary object-detection candidate: CenterPoint-compatible backend; use PointPillars or another lighter backend if it is more practical and sufficiently capable.

---

# 51. Phased Implementation — DO NOT SKIP PHASES

## PHASE 0 — FOUNDATION

Create:
- repository;
- project structure;
- pyproject.toml;
- configuration system;
- logging;
- typed contracts;
- basic PySide6 dashboard shell;
- environment diagnostics;
- tests.

Dependencies should initially be minimal:
- numpy
- pyside6
- open3d
- pyyaml
- psutil
- pytest

Install PyTorch/model dependencies only when their phase starts.

Acceptance:
- app launches;
- diagnostics work;
- tests pass.

STOP after Phase 0.

## PHASE 1 — KITTI DATA ENGINE

Implement loader/index/playback.

Acceptance:
- real KITTI `.bin` loads;
- frame stepping works;
- no whole-sequence preload;
- malformed input handled correctly.

## PHASE 2 — 3D VIEWER

Implement:
- raw point cloud;
- playback controls;
- point count;
- frame ID;
- processing/render FPS.

Acceptance:
- responsive UI;
- repeated replay without obvious memory leak.

## PHASE 3 — PREPROCESSING + GEOMETRY

Implement filters and geometry features.

Acceptance:
- visually correct filtering;
- vectorized hot path;
- unit tests pass.

## PHASE 4 — UNIFORM 2.5D BASELINE

Implement Mode A, uniform 5 cm.

Acceptance:
- correct elevation/occupancy;
- timing measured;
- map visualization works.

## PHASE 5 — REQUIRED ADAPTIVE MAP

Implement Mode B:
- 5 cm <=10 m;
- 50 cm >10–100 m;
- integer lattice;
- hierarchy;
- boundary tests.

Acceptance:
- all mapping invariants pass;
- A/B comparison works.

## PHASE 6 — FOVEAL CONTROLLER WITHOUT AI

Implement:
- distance score;
- geometry score;
- temporal/change score;
- importance;
- max refinement budget.

Acceptance:
- high-complexity regions refine;
- budget limits refinement;
- diagnostics explain why a region was refined.

## PHASE 7 — SEMANTIC AI

Implement:
- semantic interface;
- heuristic backend;
- SalsaNext-compatible backend;
- uncertainty path.

Acceptance:
- AI can be enabled/disabled independently;
- fallback works;
- inference time measured;
- label mapping documented.

## PHASE 8 — OBJECT DETECTION + TRACKING

Implement optional CenterPoint/PointPillars backend and tracker.

Acceptance:
- object regions can trigger refinement;
- tracking does not dominate runtime.

## PHASE 9 — COMPUTE-AWARE SCHEDULING

Implement:
- MAX QUALITY;
- BALANCED;
- LOW POWER;
- CPU ONLY;
- adaptive AI cadence;
- adaptive refinement budget;
- rendering rate adaptation.

Acceptance:
- all modes work;
- no CUDA path is required for basic operation.

## PHASE 10 — BENCHMARK + DEMO MODE

Implement:
- A/B/C benchmark;
- reproducible report;
- charts;
- export;
- one-click demo flow.

Acceptance:
- results are reproducible;
- no fabricated numbers;
- UI is polished.

---

# 52. Definition of Done

The project is only considered complete when:

## Functional
- KITTI playback works;
- raw 3D view works;
- 2.5D map works;
- required 5 cm/50 cm adaptive policy works;
- foveal refinement works;
- optional AI works or cleanly falls back;
- benchmark works.

## Reliable
- unit tests pass;
- integration tests pass;
- no known normal-workflow crash;
- errors are visible/logged;
- UI remains responsive.

## Performance
- no obvious unbounded memory growth;
- asynchronous pipeline;
- actual performance metrics are displayed;
- profiling identifies major bottlenecks.

## Scientific
- assumptions documented;
- PS baseline separated from experimental/foveal mode;
- actual measurements only;
- model/license/source attribution recorded.

## Demonstration
The judge can see:

RAW LiDAR
-> ADAPTIVE MAP
-> FOVEAL IMPORTANCE
-> SELECTIVE REFINEMENT
-> RESOURCE METRICS

---

# 53. Target Performance — RTX 3050 Prototype

Initial engineering targets, not claims:

- input around ~100k points/frame;
- target 20–30+ processing FPS where practical;
- desired sub-50 ms end-to-end processing where practical;
- aim to keep VRAM below ~4 GB where practical;
- keep RAM comfortably below the machine's 16 GB capacity.

If actual values differ, report actual values and optimize.

Never put target numbers into the UI as if they were measured.

---

# 54. Git Checkpoints

Use git.

Commit after each completed phase.

Suggested messages:

- `phase0: foundation and diagnostics`
- `phase1: KITTI loader`
- `phase2: async lidar viewer`
- `phase3: preprocessing and geometry`
- `phase4: uniform 2.5d map`
- `phase5: distance adaptive map`
- `phase6: foveal controller`
- `phase7: semantic backend`
- `phase8: detection and tracking`
- `phase9: compute aware scheduling`
- `phase10: benchmark and demo`

If a change breaks a previous phase, fix/revert before continuing.

Never rewrite working modules merely to make a new module convenient.

---

# 55. Claude Code Workflow

At the beginning of every task:

1. Read `CLAUDE.md`.
2. Inspect repository structure.
3. Inspect git status.
4. Identify current phase.
5. Run current tests.
6. Implement the smallest coherent change.
7. Run targeted tests.
8. Run full test suite.
9. Update documentation.
10. Report what changed, tests run, actual results, known issues, and next step.

Do not skip phases or silently implement future phases.

When a dependency causes trouble:
- isolate it;
- keep core pipeline working;
- document the issue;
- use a fallback where configured.

---

# 56. Third-Party and Model Tracking

Create `THIRD_PARTY.md` with:
- component;
- URL;
- exact version/commit;
- purpose;
- license;
- dependency vs copied code;
- required attribution.

Create `MODEL_SOURCES.md` with:
- model;
- repository;
- weights source;
- training dataset;
- license;
- label map;
- input format;
- output format;
- hardware requirements.

Do not commit huge datasets or model weights into git.

Verify actual current licenses before redistribution. Source-code and model-weight licenses may differ.

---

# 57. Scope Control

DO NOT initially build:
- full SLAM;
- full autonomous-driving planner;
- ROS stack;
- complete Apollo;
- huge dense 3D world;
- multi-camera fusion;
- training a large network from scratch;
- complex simulator integration.

The initial objective is a reliable, measurable LiDAR perception/mapping prototype.

---

# 58. Final Product Description

Use this as the canonical product description:

> FoveaLiDAR is a modular desktop LiDAR perception system that converts sequential KITTI Velodyne point clouds into a distance-adaptive 2.5D representation. It preserves high spatial detail near the vehicle while using a coarser far-field representation, then selectively refines semantically important, uncertain, geometrically complex or dynamically changing regions. A compute-budget controller adapts AI inference and refinement effort so the architecture can operate on both GPU-equipped and low-compute systems.

---

# 59. Immediate Instruction to Claude Code

**START WITH PHASE 0 ONLY.**

Do not integrate SalsaNext yet.
Do not integrate CenterPoint yet.
Do not integrate PointPillars yet.
Do not implement the full foveal controller yet.
Do not install Apollo.
Do not install ROS.
Do not download giant model repositories.

Phase 0 must create:
- modular structure;
- configuration;
- logging;
- typed contracts;
- basic PySide6 desktop shell;
- environment diagnostic;
- test framework;
- minimal dependency set.

The program must launch successfully before Phase 0 is considered complete.

Then STOP and report Phase 0 completion. Wait for the next explicit phase instruction.

---

# 60. Reference Links

KITTI Raw Data:
https://www.cvlibs.net/datasets/kitti/raw_data.php

SemanticKITTI API:
https://github.com/PRBonn/semantic-kitti-api

Apollo:
https://github.com/ApolloAuto/apollo

SalsaNext:
https://github.com/TiagoCortinhal/SalsaNext

RangeNet++ reference:
https://github.com/PRBonn/rangenet_lib

OpenPCDet:
https://github.com/open-mmlab/OpenPCDet

Open3D:
https://github.com/isl-org/Open3D

PySide6 / Qt for Python:
https://doc.qt.io/qtforpython-6/

ANYbotics Elevation Mapping:
https://github.com/ANYbotics/elevation_mapping

OctoMap:
https://github.com/OctoMap/octomap

RoadRunner M&M:
https://leggedrobotics.github.io/roadrunner_mm/

---

# 61. Final Engineering Principle

Build a small, reliable core first.

The core remains:

    KITTI
      -> preprocessing
      -> geometry
      -> adaptive 2.5D map

Everything else is replaceable:

    semantic model
    object detector
    tracker
    importance model
    compute scheduler
    renderer

The architecture is more important than any individual model.

Do not trade correctness for visual polish, and do not trade scientific honesty for benchmark numbers.
