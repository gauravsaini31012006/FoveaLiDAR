"""Phase 3 Foveation Visualization Tab - Real-time 3D Autonomous Drive + Camera View."""

from pathlib import Path
from typing import List, Optional, Tuple, Dict, Any
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import cv2
import vispy
vispy.use('PySide6')
import vispy.scene
from vispy.scene import visuals
import vispy.color

from fovealidar.data.loaders import DatasetFactory
from fovealidar.preprocessing import PreprocessingConfig, PreprocessingPipeline
from fovealidar.foveation.gaze_simulator import GazeSimulator
from fovealidar.foveation.roi_definition import ROIDefinition
from fovealidar.foveation.adaptive_compression import AdaptiveCompression
from fovealidar.visualization.renderer import PointCloudRenderer
from fovealidar.core.types import ComputeMode
from fovealidar.foveation.compute_scheduler import ComputeScheduler
from fovealidar.perception.tracking import SimpleTracker


class FoveationVisualizerTab(QtWidgets.QWidget):
    """Phase 3: Real-time 3D Autonomous Driving View matching img2 + Synchronized Camera."""
    
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.camera_images = None
        self.gaze_sim = GazeSimulator()
        self.roi_def = ROIDefinition(fovea_radius_m=10.0, periphery_radius_m=100.0)
        self.compression = AdaptiveCompression(self.roi_def)
        self.pipeline = PreprocessingPipeline(PreprocessingConfig())
        self.renderer = PointCloudRenderer()
        self.current_frame = 0
        self.is_playing = False
        # Premium overhaul (Phases 3-8): semantic AI + tracking + compute-aware HUD
        self.scheduler = ComputeScheduler(ComputeMode.BALANCED)
        self.tracker_3d = SimpleTracker()
        self.semantic_enabled = False
        self.tracking_enabled = False
        self._scan_phase = 0
        self._last_semantic_counts = {"ground": 0, "vegetation": 0, "obstacle": 0}
        self._setup_ui()

    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header / Title
        header_layout = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Phases 3-8: Premium Foveated 3D Perception + Semantic AI & Tracking")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        header_layout.addWidget(title)
        
        sih_badge = QtWidgets.QLabel("SIH 2026 PS 26053 | 5cm Near / 50cm Far | Semantic AI + Tracking")
        sih_badge.setStyleSheet("color: #a3e635; font-weight: bold; font-size: 11px;")
        sih_badge.setStyleSheet("color: #a3e635; font-weight: bold; font-size: 11px;")
        header_layout.addStretch()
        header_layout.addWidget(sih_badge)
        layout.addLayout(header_layout)
        
        # Controls / Dataset Loading Bar
        controls = QtWidgets.QHBoxLayout()
        controls.addWidget(QtWidgets.QLabel("KITTI Root:"))
        self.path_input = QtWidgets.QLineEdit()
        self.path_input.setText("D:/CLG/SIH_2026/kitti_raw_data of lidar")
        controls.addWidget(self.path_input, 1)
        
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_dataset)
        controls.addWidget(self.browse_btn)
        
        self.load_btn = QtWidgets.QPushButton("Load Dataset")
        self.load_btn.setStyleSheet("background-color: #0284c7; color: white; font-weight: bold; padding: 4px 12px;")
        self.load_btn.clicked.connect(self._load_dataset)
        controls.addWidget(self.load_btn)
        layout.addLayout(controls)
        
        # Toolbar: Camera Presets + Color Modes + Toggles
        toolbar = QtWidgets.QHBoxLayout()
        toolbar.addWidget(QtWidgets.QLabel("Camera View:"))
        
        self.btn_chase = QtWidgets.QPushButton("🚘 Chase View")
        self.btn_chase.clicked.connect(lambda: self._set_camera_preset('chase'))
        toolbar.addWidget(self.btn_chase)
        
        self.btn_bev = QtWidgets.QPushButton("🦅 Top-Down BEV")
        self.btn_bev.clicked.connect(lambda: self._set_camera_preset('bev'))
        toolbar.addWidget(self.btn_bev)
        
        self.btn_driver = QtWidgets.QPushButton("👀 Driver POV")
        self.btn_driver.clicked.connect(lambda: self._set_camera_preset('driver'))
        toolbar.addWidget(self.btn_driver)
        
        self.btn_orbit = QtWidgets.QPushButton("🔄 Free Orbit")
        self.btn_orbit.clicked.connect(lambda: self._set_camera_preset('orbit'))
        toolbar.addWidget(self.btn_orbit)
        
        toolbar.addSpacing(15)
        toolbar.addWidget(QtWidgets.QLabel("Coloring:"))
        self.color_combo = QtWidgets.QComboBox()
        self.color_combo.addItem("🌟 Autonomous (Like img2)", "autonomous")
        self.color_combo.addItem("🌈 Elevation / Height", "height")
        self.color_combo.addItem("💡 Reflectance / Intensity", "intensity")
        self.color_combo.addItem("🎯 Foveal Resolution Map", "foveal")
        self.color_combo.addItem("🧠 Semantic AI", "semantic")
        self.color_combo.currentIndexChanged.connect(lambda: self._update_display())
        toolbar.addWidget(self.color_combo)
        
        toolbar.addSpacing(15)
        self.raw_checkbox = QtWidgets.QCheckBox("Raw 100k+ Points")
        self.raw_checkbox.setToolTip("Toggle between raw full 115k points and multi-tier foveated point cloud")
        self.raw_checkbox.setChecked(False)
        self.raw_checkbox.stateChanged.connect(lambda: self._update_display())
        toolbar.addWidget(self.raw_checkbox)
        
        self.boxes_checkbox = QtWidgets.QCheckBox("3D Bounding Boxes")
        self.boxes_checkbox.setChecked(True)
        self.boxes_checkbox.stateChanged.connect(lambda: self._update_display())
        toolbar.addWidget(self.boxes_checkbox)
        
        toolbar.addStretch()
        layout.addLayout(toolbar)

        # Premium AI & Compute bar (Phases 7-9 surfaced in the 3D view)
        ai_bar = QtWidgets.QHBoxLayout()
        self.semantic_checkbox = QtWidgets.QCheckBox("🧠 Semantic AI")
        self.semantic_checkbox.setChecked(False)
        self.semantic_checkbox.setToolTip("Phase 7: point-wise Ground / Vegetation / Obstacle heuristic")
        self.semantic_checkbox.stateChanged.connect(self._on_semantic_toggled)
        ai_bar.addWidget(self.semantic_checkbox)

        self.tracking_checkbox = QtWidgets.QCheckBox("🎯 Tracking")
        self.tracking_checkbox.setChecked(False)
        self.tracking_checkbox.setToolTip("Phase 8: tracked 3D boxes with ID badges")
        self.tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)
        ai_bar.addWidget(self.tracking_checkbox)

        ai_bar.addSpacing(15)
        ai_bar.addWidget(QtWidgets.QLabel("Compute:"))
        self.compute_combo = QtWidgets.QComboBox()
        self.compute_combo.addItem("⚡ MAX_QUALITY", ComputeMode.MAX_QUALITY.value)
        self.compute_combo.addItem("⚖️ BALANCED", ComputeMode.BALANCED.value)
        self.compute_combo.addItem("🔋 LOW_POWER", ComputeMode.LOW_POWER.value)
        self.compute_combo.addItem("🖥️ CPU_ONLY", ComputeMode.CPU_ONLY.value)
        self.compute_combo.setCurrentIndex(1)
        self.compute_combo.currentIndexChanged.connect(self._on_compute_changed)
        ai_bar.addWidget(self.compute_combo)

        self.compute_label = QtWidgets.QLabel("")
        self.compute_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        ai_bar.addWidget(self.compute_label)
        ai_bar.addStretch()
        layout.addLayout(ai_bar)
        
        # Dual Display Area (3D LiDAR Chase View + Synchronized Camera)
        dual = QtWidgets.QHBoxLayout()
        
        # Left: VisPy 3D Canvas
        left = QtWidgets.QVBoxLayout()
        left.addWidget(QtWidgets.QLabel("3D LiDAR Environment (Interactive GPU View):"))
        
        self.canvas = vispy.scene.SceneCanvas(keys='interactive', show=False, bgcolor='#0b0d12')
        self.view = self.canvas.central_widget.add_view()
        
        # Autonomous chase camera: looking forward (+X) along road
        self.camera_chase = vispy.scene.cameras.TurntableCamera(
            elevation=16.0, azimuth=180.0, distance=38.0, center=(18.0, 0.0, 0.0), fov=48.0
        )
        self.view.camera = self.camera_chase
        
        # Visual Layers:
        # 1. Ground distance rings (10m, 25m, 50m)
        self.rings_visual = visuals.Line(connect='segments', color=(0.2, 0.5, 0.7, 0.6), width=1.5, parent=self.view.scene)
        # 1b. Premium gradient ground-plane grid (subtle glow, fog-ready)
        self.grid_visual = visuals.Line(connect='segments', width=1.0, parent=self.view.scene)
        # 2. 3D Ego-car model
        self.car_visual = visuals.Line(connect='segments', color=(0.9, 0.9, 0.95, 1.0), width=2.0, parent=self.view.scene)
        self.taillights_visual = visuals.Markers(parent=self.view.scene)
        # 3. 3D Obstacle bounding boxes (geometry + Phase 8 tracked overlays)
        self.boxes_visual = visuals.Line(connect='segments', color=(0.1, 0.9, 0.85, 0.9), width=1.8, parent=self.view.scene)
        self.tracked_boxes_visual = visuals.Line(connect='segments', color=(1.0, 0.75, 0.15, 1.0), width=2.2, parent=self.view.scene)
        self.track_labels_visual = visuals.Text("", parent=self.canvas.scene, color=(1.0, 0.85, 0.4, 1.0), pos=(130, 116), font_size=9)
        # 4. Forward Attention Beam / Laser Ray + animated scan-line sweep
        self.laser_visual = visuals.Line(connect='segments', color=(1.0, 0.4, 0.1, 0.95), width=2.5, parent=self.view.scene)
        self.reticle_visual = visuals.Markers(parent=self.view.scene)
        self.scan_visual = visuals.Line(connect='segments', color=(0.2, 0.9, 1.0, 0.55), width=1.2, parent=self.view.scene)
        
        # 5. Point Cloud Scatters:
        self.scatter_ground = visuals.Markers(parent=self.view.scene)
        self.scatter_vegetation = visuals.Markers(parent=self.view.scene)
        self.scatter_objects = visuals.Markers(parent=self.view.scene)
        
        # 2D HUD Text Overlay (premium: scan-line animated title + semantic/compute metrics)
        self.hud_title = visuals.Text("FOVEALIDAR // ACTIVE", parent=self.canvas.scene, color=(0.0, 0.9, 1.0, 1.0), pos=(130, 25), font_size=11)
        self.hud_stats = visuals.Text("PTS: 0", parent=self.canvas.scene, color=(0.3, 1.0, 0.5, 1.0), pos=(130, 48), font_size=10)
        self.hud_gaze = visuals.Text("GAZE: [0, 0]", parent=self.canvas.scene, color=(1.0, 0.6, 0.2, 1.0), pos=(130, 70), font_size=10)
        self.hud_semantic = visuals.Text("", parent=self.canvas.scene, color=(0.65, 0.95, 0.55, 1.0), pos=(130, 92), font_size=10)
        
        self.lidar_display = self.canvas.native
        self.lidar_display.setMinimumSize(640, 460)
        self.lidar_display.setStyleSheet("border: 2px solid #334155; border-radius: 4px;")
        left.addWidget(self.lidar_display, 1)
        dual.addLayout(left, 1)
        
        # Right: Camera Image Feed
        right = QtWidgets.QVBoxLayout()
        right.addWidget(QtWidgets.QLabel("Synchronized Front RGB Camera (image_02):"))
        self.camera_display = QtWidgets.QLabel()
        self.camera_display.setMinimumSize(640, 460)
        self.camera_display.setAlignment(QtCore.Qt.AlignCenter)
        self.camera_display.setStyleSheet("border: 2px solid #334155; border-radius: 4px; background-color: #0b0d12;")
        right.addWidget(self.camera_display, 1)
        dual.addLayout(right, 1)
        
        layout.addLayout(dual, 1)
        
        # Playback Controls
        play = QtWidgets.QHBoxLayout()
        play.addWidget(QtWidgets.QLabel("Frame:"))
        self.slider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.slider.valueChanged.connect(self._on_frame_changed)
        play.addWidget(self.slider, 1)
        
        self.frame_label = QtWidgets.QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        play.addWidget(self.frame_label)
        
        self.prev_btn = QtWidgets.QPushButton("⏮")
        self.prev_btn.clicked.connect(self._prev_frame)
        self.prev_btn.setMaximumWidth(36)
        play.addWidget(self.prev_btn)
        
        self.play_btn = QtWidgets.QPushButton("▶")
        self.play_btn.clicked.connect(self._toggle_play)
        self.play_btn.setMaximumWidth(40)
        self.play_btn.setStyleSheet("font-weight: bold;")
        play.addWidget(self.play_btn)
        
        self.next_btn = QtWidgets.QPushButton("⏭")
        self.next_btn.clicked.connect(self._next_frame)
        self.next_btn.setMaximumWidth(36)
        play.addWidget(self.next_btn)
        
        play.addSpacing(15)
        play.addWidget(QtWidgets.QLabel("FPS:"))
        self.speed = QtWidgets.QSpinBox()
        self.speed.setMinimum(1)
        self.speed.setMaximum(60)
        self.speed.setValue(10)
        self.speed.valueChanged.connect(self._on_speed_changed)
        play.addWidget(self.speed)
        
        layout.addLayout(play)
        
        # Engineering Telemetry Box
        self.stats = QtWidgets.QTextEdit()
        self.stats.setReadOnly(True)
        self.stats.setMaximumHeight(70)
        self.stats.setStyleSheet("background-color: #0f172a; color: #94a3b8; font-family: monospace; font-size: 11px;")
        layout.addWidget(self.stats)
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._playback_tick)
        
        # Build initial static geometry (rings + car)
        self._build_static_environment()
        try:
            self._refresh_compute_label()
        except Exception:
            pass

    def _set_camera_preset(self, preset: str):
        """Set camera viewpoint."""
        if preset == 'chase':
            self.view.camera.elevation = 16.0
            self.view.camera.azimuth = 180.0
            self.view.camera.distance = 38.0
            self.view.camera.center = (18.0, 0.0, 0.0)
        elif preset == 'bev':
            self.view.camera.elevation = 90.0
            self.view.camera.azimuth = 180.0
            self.view.camera.distance = 60.0
            self.view.camera.center = (20.0, 0.0, 0.0)
        elif preset == 'driver':
            self.view.camera.elevation = 5.0
            self.view.camera.azimuth = 180.0
            self.view.camera.distance = 16.0
            self.view.camera.center = (14.0, 0.0, -0.5)
        elif preset == 'orbit':
            self.view.camera.elevation = 26.0
            self.view.camera.azimuth = 145.0
            self.view.camera.distance = 45.0
            self.view.camera.center = (16.0, 0.0, 0.0)

    def _build_static_environment(self):
        """Construct 3D ground distance rings and ego vehicle at the origin."""
        ground_z = -1.65
        
        # 1. Ground Range Rings (10m Fovea, 25m, 50m)
        ring_segments = []
        for r in [10.0, 25.0, 50.0]:
            angles = np.linspace(-np.pi * 0.9, np.pi * 0.9, 120)
            for i in range(len(angles) - 1):
                p1 = [r * np.cos(angles[i]), r * np.sin(angles[i]), ground_z]
                p2 = [r * np.cos(angles[i+1]), r * np.sin(angles[i+1]), ground_z]
                ring_segments.extend([p1, p2])
        self.rings_visual.set_data(pos=np.array(ring_segments, dtype=np.float32))

        # 1b. Premium gradient ground-plane grid (every 5 m, ±35 m) with subtle glow.
        # Per-vertex colors fade with radial distance so the grid melts into fog.
        try:
            grid_segs, grid_cols = [], []
            near = np.array([0.15, 0.75, 0.9, 0.55], dtype=np.float32)  # cyan glow
            far = np.array([0.10, 0.22, 0.32, 0.16], dtype=np.float32)  # deep slate
            for g in range(-35, 36, 5):
                for a, b in [((float(g), -35.0), (float(g), 35.0)),
                             ((-35.0, float(g)), (35.0, float(g)))]:
                    grid_segs.extend([[a[0], a[1], ground_z], [b[0], b[1], ground_z]])
                    for pt in (a, b):
                        r = float(np.hypot(pt[0], pt[1]))
                        t = float(np.clip(r / 50.0, 0.0, 1.0))
                        grid_cols.append((1.0 - t) * near + t * far)
            self.grid_visual.set_data(
                pos=np.array(grid_segs, dtype=np.float32),
                color=np.array(grid_cols, dtype=np.float32),
            )
        except Exception:
            pass
        
        # 2. Ego Passenger Car Model at (0, 0, ground_z)
        car_segments = []
        x_min, x_max = -2.1, 2.3
        y_min, y_max = -0.92, 0.92
        z_min, z_max = ground_z, ground_z + 1.45
        
        corners = [
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        for i, j in edges:
            car_segments.extend([corners[i], corners[j]])
        self.car_visual.set_data(pos=np.array(car_segments, dtype=np.float32))
        
        # Dual red taillights
        taillights = np.array([
            [x_min, y_min + 0.2, z_min + 0.65],
            [x_min, y_max - 0.2, z_min + 0.65],
        ], dtype=np.float32)
        self.taillights_visual.set_data(taillights, face_color=(1.0, 0.1, 0.1, 1.0), edge_color=(1, 0, 0, 1), size=8.0)

    # ------------------------------------------------------------------
    # Premium 3D controls (Phases 7-9 surfaced in the 3D view)
    # ------------------------------------------------------------------
    def _on_semantic_toggled(self, state):
        self.semantic_enabled = bool(state)
        if bool(state):
            # Auto-select the Semantic AI coloring for immediate feedback
            try:
                idx = self.color_combo.findData("semantic")
                if idx >= 0:
                    self.color_combo.setCurrentIndex(idx)
            except Exception:
                pass
        self._update_display()

    def _on_tracking_toggled(self, state):
        self.tracking_enabled = bool(state)
        if bool(state) and not self.semantic_checkbox.isChecked():
            self.semantic_checkbox.setChecked(True)
            self.semantic_enabled = True
        self._update_display()

    def _on_compute_changed(self):
        raw = self.compute_combo.currentData()
        try:
            mode = ComputeMode(raw) if not isinstance(raw, ComputeMode) else raw
        except Exception:
            return
        self.scheduler.set_mode(mode)
        try:
            cap = self.scheduler.target_fps
            if self.speed.value() > cap:
                self.speed.setValue(cap)
            if self.is_playing:
                self.timer.setInterval(int(1000 / max(1, self.speed.value())))
        except Exception:
            pass
        self._refresh_compute_label()
        self._update_display()

    def _refresh_compute_label(self):
        try:
            self.compute_label.setText(f"⚙️ {self.scheduler.describe()}")
        except Exception:
            pass

    @staticmethod
    def classify_semantic_points(pts: np.ndarray) -> np.ndarray:
        """Phase 7 point-wise heuristic: 1=Ground, 2=Vegetation, 3=Obstacle.

        Mirrors HeuristicSemanticBackend rules at point level:
        - Ground: low surface (drivable plane proxy, z <= -1.0)
        - Vegetation: elevated / roadside foliage proxy
          (z > 1.2, or tall/roadside points with z > -0.6)
        - Obstacle: everything else above ground (vehicles, trunks, barriers)
        """
        labels = np.full(len(pts), 3, dtype=np.int32)
        if len(pts) == 0:
            return labels
        z = pts[:, 2]
        y = pts[:, 1] if pts.shape[1] > 1 else np.zeros(len(pts))
        x = pts[:, 0] if pts.shape[1] > 0 else np.zeros(len(pts))
        is_ground = z <= -1.0
        is_veg = (~is_ground) & ((z > 1.2) | ((z > -0.6) & ((np.abs(y) > 2.8) | (z > 2.0))))
        labels[is_ground] = 1
        labels[is_veg] = 2
        # Tall forward boxes already veg; keep near-road low boxes as obstacle
        _ = x  # reserved for corridor-aware extensions
        return labels

    def _update_tracked_overlay(self, boxes):
        """Phase 8: overlay tracked-object boxes (gold) with ID badges."""
        if not boxes:
            self.tracked_boxes_visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            try:
                self.track_labels_visual.text = ""
            except Exception:
                pass
            return
        try:
            tracked = self.tracker_3d.update(
                [{"center": b["center"], "extent": b["extent"],
                  "label": b.get("label", "OBSTACLE")} for b in boxes]
            )
        except Exception:
            tracked = []
            for i, b in enumerate(boxes, start=1):
                d = dict(b)
                d["id"] = i
                tracked.append(d)
        box_lines = []
        badges = []
        for t in tracked[:6]:
            c = np.asarray(t.get("center", [0, 0, 0]), dtype=np.float32)
            e = np.asarray(t.get("extent", [1.4, 1.2, 1.3]), dtype=np.float32) / 2.0
            corners = [
                [c[0]-e[0], c[1]-e[1], c[2]-e[2]], [c[0]+e[0], c[1]-e[1], c[2]-e[2]],
                [c[0]+e[0], c[1]+e[1], c[2]-e[2]], [c[0]-e[0], c[1]+e[1], c[2]-e[2]],
                [c[0]-e[0], c[1]-e[1], c[2]+e[2]], [c[0]+e[0], c[1]-e[1], c[2]+e[2]],
                [c[0]+e[0], c[1]+e[1], c[2]+e[2]], [c[0]-e[0], c[1]+e[1], c[2]+e[2]],
            ]
            edges = [(0,1),(1,2),(2,3),(3,0),(4,5),(5,6),(6,7),(7,4),(0,4),(1,5),(2,6),(3,7)]
            for i, j in edges:
                box_lines.extend([corners[i], corners[j]])
            dist = float(np.linalg.norm(c[:2]))
            badges.append(f"ID:{t.get('id','?')} [{dist:.1f}m]")
        if box_lines:
            self.tracked_boxes_visual.set_data(pos=np.array(box_lines, dtype=np.float32))
        else:
            self.tracked_boxes_visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
        try:
            self.track_labels_visual.text = "  ".join(badges[:4]) if badges else ""
        except Exception:
            pass

    def _update_scan_line(self, gaze: np.ndarray):
        """Animated scan-line sweep across the corridor (premium HUD effect)."""
        try:
            self._scan_phase = (self._scan_phase + 1) % 120
            t = self._scan_phase / 120.0
            sweep_x = -10.0 + t * 60.0  # sweep from behind ego to far field
            z0, z1 = -1.6, 4.0
            scan_pts = np.array([
                [sweep_x, -14.0, z0], [sweep_x, 14.0, z0],
                [sweep_x, -14.0, z1], [sweep_x, 14.0, z1],
            ], dtype=np.float32)
            self.scan_visual.set_data(pos=np.array([
                scan_pts[0], scan_pts[1], scan_pts[2], scan_pts[3],
            ], dtype=np.float32))
        except Exception:
            pass

    def _browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select KITTI Root")
        if path:
            self.path_input.setText(path)

    def _load_dataset(self):
        path = self.path_input.text().strip()
        if not path:
            return
        try:
            self.dataset = DatasetFactory.auto_detect(Path(path))
            if not self.dataset:
                self.dataset = DatasetFactory.create("kitti", Path(path))
            self.camera_images = self._load_camera_images(self.dataset)
            self.slider.setMaximum(max(0, len(self.dataset) - 1))
            self.slider.setValue(0)
            self._update_display()
            QtWidgets.QMessageBox.information(
                self, "Dataset Loaded",
                f"✅ Successfully loaded {len(self.dataset)} frames from KITTI sequence.\n"
                f"Camera images: {len(self.camera_images) if self.camera_images else 'None found'}"
            )
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load dataset: {e}")

    def _load_camera_images(self, dataset):
        try:
            if hasattr(dataset, 'drive_path') and dataset.drive_path:
                img_path = dataset.drive_path / "image_02" / "data"
                if img_path.exists():
                    images = sorted(img_path.glob("*.png"))
                    if images:
                        return images
            for folder in Path(dataset.root).rglob("image_02"):
                data_folder = folder / "data"
                search_folder = data_folder if data_folder.exists() else folder
                images = sorted(search_folder.glob("*.png"))
                if images:
                    return images
            return None
        except Exception:
            return None

    def _on_frame_changed(self, value: int):
        self.current_frame = value
        self._update_display()

    def _prev_frame(self):
        if self.dataset and self.current_frame > 0:
            self.slider.setValue(self.current_frame - 1)

    def _next_frame(self):
        if self.dataset and self.current_frame < len(self.dataset) - 1:
            self.slider.setValue(self.current_frame + 1)

    def _toggle_play(self):
        self.is_playing = not self.is_playing
        if self.is_playing:
            # Phase 9: respect compute FPS cap
            try:
                cap = self.scheduler.target_fps
                if self.speed.value() > cap:
                    self.speed.setValue(cap)
            except Exception:
                pass
            self.play_btn.setText("⏸")
            self.timer.start(int(1000 / self.speed.value()))
        else:
            self.play_btn.setText("▶")
            self.timer.stop()

    def _on_speed_changed(self, value: int):
        try:
            cap = self.scheduler.target_fps
            if value > cap:
                # Clamp without recursion storms: only set once
                if self.speed.value() != cap:
                    self.speed.setValue(cap)
                value = cap
        except Exception:
            pass
        if self.is_playing:
            self.timer.setInterval(int(1000 / max(1, value)))

    def _playback_tick(self):
        if not self.dataset:
            return
        if self.current_frame < len(self.dataset) - 1:
            self.slider.setValue(self.current_frame + 1)
        else:
            self.slider.setValue(0)

    def _update_display(self):
        if not self.dataset:
            return
        try:
            idx = self.current_frame
            frame = self.dataset[idx]
            raw_pc = frame.point_cloud
            pp = self.pipeline.process(raw_pc)
            
            # Gaze simulation (driving or saliency)
            if idx % 2 == 0:
                gaze = self.gaze_sim.simulate_driving_gaze(pp)
            else:
                gaze = self.gaze_sim.simulate_saliency_gaze(pp)
            
            primary_gaze = gaze[0] if len(gaze) > 0 else np.array([20.0, 0.0, 0.0], dtype=np.float32)
            
            self.roi_def.rois.clear()
            self.roi_def.add_roi(tuple(primary_gaze))
            
            # Choose points to render: raw full (115k+) or adaptive foveated (~48k)
            is_raw = self.raw_checkbox.isChecked()
            if is_raw:
                active_pc = pp
            else:
                active_pc = self.compression.compress_adaptive(pp, strategy='voxel')

            # Phase 9: compute-aware point downsampling (LOW_POWER x2, CPU_ONLY x4)
            try:
                ds_factor = self.scheduler.downsample_factor
                if ds_factor > 1 and len(active_pc.points) > ds_factor:
                    keep = np.arange(0, len(active_pc.points), ds_factor)
                    from fovealidar.core.types import PointCloud as _PC
                    _kw = {}
                    if getattr(active_pc, "timestamp", None) is not None:
                        _kw["timestamp"] = active_pc.timestamp
                    if getattr(active_pc, "frame_id", None) is not None:
                        _kw["frame_id"] = active_pc.frame_id
                    active_pc = _PC(points=active_pc.points[keep],
                                    intensity=active_pc.intensity[keep], **_kw)
            except Exception:
                pass
            
            pts = active_pc.points
            intens = active_pc.intensity if active_pc.intensity is not None else np.ones(len(pts), dtype=np.float32)
            n_pts = len(pts)
            
            # Color mode selection
            mode = self.color_combo.currentData()
            z = pts[:, 2]
            y = pts[:, 1]
            x = pts[:, 0]
            
            # 1. Ground and lane segmentation
            is_ground = z <= -1.25
            is_lane = is_ground & (intens > 0.45) & (np.abs(y) < 7.0)
            is_veg = (z > -0.6) & (z < 7.5) & ((np.abs(y) > 2.8) | (z > 2.0))
            is_obj = (z > -1.25) & (z < 1.4) & (np.abs(y) <= 3.2) & (x > 3.0)
            
            # Setup colors
            colors = np.zeros((n_pts, 4), dtype=np.float32)
            
            if mode == 'autonomous':
                # Road surface: radial slate-blue gradient
                r_ground = np.sqrt(x**2 + y**2)
                near_g = is_ground & (r_ground <= 22.0)
                far_g = is_ground & (r_ground > 22.0)
                
                blue_fade = np.clip((22.0 - r_ground[near_g]) / 22.0, 0.0, 1.0)
                colors[near_g, 0] = (25 * blue_fade + 28 * (1 - blue_fade)) / 255.0  # R
                colors[near_g, 1] = (85 * blue_fade + 38 * (1 - blue_fade)) / 255.0  # G
                colors[near_g, 2] = (150 * blue_fade + 50 * (1 - blue_fade)) / 255.0 # B
                colors[near_g, 3] = 0.95
                
                colors[far_g, 0] = 0.12
                colors[far_g, 1] = 0.15
                colors[far_g, 2] = 0.18
                colors[far_g, 3] = 0.85
                
                # Glowing lane markings
                colors[is_lane, 0] = 1.0
                colors[is_lane, 1] = 1.0
                colors[is_lane, 2] = 1.0
                colors[is_lane, 3] = 1.0
                
                # Golden-amber tree canopies (like img2!)
                colors[is_veg, 0] = 0.96
                colors[is_veg, 1] = 0.76
                colors[is_veg, 2] = 0.12
                colors[is_veg, 3] = 1.0
                
                # Dynamic vehicles / obstacles (cyan)
                colors[is_obj, 0] = 0.12
                colors[is_obj, 1] = 0.88
                colors[is_obj, 2] = 0.94
                colors[is_obj, 3] = 1.0
                
                # Remaining buildings / structures
                other = ~(is_ground | is_veg | is_obj)
                colors[other, 0] = 0.75
                colors[other, 1] = 0.78
                colors[other, 2] = 0.82
                colors[other, 3] = 0.90
                
            elif mode == 'height':
                norm_z = np.clip((z - (-2.0)) / 5.5, 0.0, 1.0)
                colors[:, 0] = np.cos(norm_z * np.pi * 1.5) * 0.5 + 0.5
                colors[:, 1] = np.sin(norm_z * np.pi) * 0.8 + 0.2
                colors[:, 2] = np.sin(norm_z * np.pi * 1.5) * 0.5 + 0.5
                colors[:, 3] = 1.0
                
            elif mode == 'intensity':
                norm_i = np.clip(intens, 0.0, 1.0)
                colors[:, 0] = norm_i * 0.7 + 0.2
                colors[:, 1] = norm_i * 0.9 + 0.1
                colors[:, 2] = norm_i
                colors[:, 3] = 1.0
                
            elif mode == 'foveal':
                roi_ids, _ = self.roi_def.classify_points(active_pc)
                colors[roi_ids == 0] = [0.2, 1.0, 0.3, 1.0]  # Green fovea
                colors[roi_ids == 1] = [1.0, 0.6, 0.1, 0.8]  # Orange periphery
                colors[roi_ids == 2] = [0.5, 0.5, 0.5, 0.6]  # Gray far-field

            elif mode == 'semantic':
                # 🧠 Semantic AI (Phase 7 heuristic, point-level):
                # Ground = dark slate, Vegetation = emerald green, Obstacle = hot red.
                sem_labels = self.classify_semantic_points(pts)
                g = sem_labels == 1
                v = sem_labels == 2
                o = sem_labels == 3
                colors[g] = [0.16, 0.20, 0.26, 0.95]  # dark slate ground
                colors[v] = [0.06, 0.75, 0.35, 1.0]   # emerald vegetation
                colors[o] = [1.0, 0.16, 0.22, 1.0]    # hot-red obstacles
                # Preserve bright lane markings inside ground for readability
                colors[is_lane] = [1.0, 1.0, 1.0, 1.0]
                try:
                    self._last_semantic_counts = {
                        "ground": int(np.sum(g)),
                        "vegetation": int(np.sum(v)),
                        "obstacle": int(np.sum(o)),
                    }
                except Exception:
                    pass

            # Premium fog / depth fade: distant points dissolve into the background.
            try:
                r_dist = np.sqrt(x ** 2 + y ** 2)
                fog = np.clip(1.0 - (r_dist - 10.0) / 90.0, 0.22, 1.0).astype(np.float32)
                colors[:, 3] = np.clip(colors[:, 3] * fog, 0.0, 1.0)
                # Slight distance desaturation toward slate
                far_m = r_dist > 55.0
                if np.any(far_m):
                    colors[far_m, 0] = colors[far_m, 0] * 0.55 + 0.10 * 0.45
                    colors[far_m, 1] = colors[far_m, 1] * 0.55 + 0.13 * 0.45
                    colors[far_m, 2] = colors[far_m, 2] * 0.55 + 0.17 * 0.45
            except Exception:
                pass
            
            # Render points via VisPy
            # Split into ground and non-ground for crisp marker sizing
            self.scatter_ground.set_data(pts[is_ground], edge_color=None, face_color=colors[is_ground], size=2.0)
            self.scatter_vegetation.set_data(pts[~is_ground], edge_color=None, face_color=colors[~is_ground], size=2.8)
            
            # 3D Bounding Boxes for detected obstacles (geometry, cyan)
            boxes = []
            if self.boxes_checkbox.isChecked():
                boxes = self.renderer.detect_obstacles(pts, intens)
                box_lines = []
                for b in boxes:
                    c = b['center']
                    e = b['extent'] / 2.0
                    corners = [
                        [c[0]-e[0], c[1]-e[1], c[2]-e[2]], [c[0]+e[0], c[1]-e[1], c[2]-e[2]],
                        [c[0]+e[0], c[1]+e[1], c[2]-e[2]], [c[0]-e[0], c[1]+e[1], c[2]-e[2]],
                        [c[0]-e[0], c[1]-e[1], c[2]+e[2]], [c[0]+e[0], c[1]-e[1], c[2]+e[2]],
                        [c[0]+e[0], c[1]+e[1], c[2]+e[2]], [c[0]-e[0], c[1]+e[1], c[2]+e[2]],
                    ]
                    edges = [
                        (0, 1), (1, 2), (2, 3), (3, 0),
                        (4, 5), (5, 6), (6, 7), (7, 4),
                        (0, 4), (1, 5), (2, 6), (3, 7),
                    ]
                    for i, j in edges:
                        box_lines.extend([corners[i], corners[j]])
                if box_lines:
                    self.boxes_visual.set_data(pos=np.array(box_lines, dtype=np.float32))
                else:
                    self.boxes_visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
            else:
                self.boxes_visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))

            # Phase 8: tracked-object overlay (gold boxes + ID badges) when enabled
            if self.tracking_enabled:
                self._update_tracked_overlay(boxes)
            else:
                self.tracked_boxes_visual.set_data(pos=np.zeros((0, 3), dtype=np.float32))
                try:
                    self.track_labels_visual.text = ""
                except Exception:
                    pass
            
            # Forward Attention Laser Ray & Reticle + animated scan-line sweep
            laser_pts = np.array([[0.0, 0.0, 0.0], primary_gaze], dtype=np.float32)
            self.laser_visual.set_data(pos=laser_pts)
            self.reticle_visual.set_data(primary_gaze.reshape((1, 3)), face_color=(1, 0.4, 0.1, 1), size=12.0)
            self._update_scan_line(primary_gaze)

            # Premium HUD (animated scan-line prefix + semantic stats + compute mode)
            orig_cnt = len(raw_pc.points)
            comp_cnt = len(pts)
            ratio = (1.0 - comp_cnt / max(1, orig_cnt)) * 100.0
            scan_bar = "━" * (1 + (self._scan_phase % 10)) + "◆"
            sem = getattr(self, "_last_semantic_counts",
                          {"ground": 0, "vegetation": 0, "obstacle": 0})
            n_trk = 0
            try:
                if getattr(self.tracker_3d, "tracks", None) is not None:
                    n_trk = len(self.tracker_3d.tracks)
            except Exception:
                n_trk = 0
            compute_txt = self.scheduler.mode.name if hasattr(self, "scheduler") else "BALANCED"
            self.hud_title.text = f"{scan_bar} FOVEALIDAR // FRAME:{idx:04d} // {'RAW 100K+' if is_raw else 'ADAPTIVE 2.5D'}"
            self.hud_stats.text = f"RAW: {orig_cnt:,} | ACTIVE: {comp_cnt:,} ({ratio:.1f}% reduction)"
            self.hud_gaze.text = f"FOCUS LOCK: [{primary_gaze[0]:.1f}m, {primary_gaze[1]:.1f}m]"
            try:
                self.hud_semantic.text = (
                    f"SEM G:{sem.get('ground',0):,} V:{sem.get('vegetation',0):,} "
                    f"O:{sem.get('obstacle',0):,} | TRK:{n_trk} | {compute_txt}"
                )
            except Exception:
                pass
            
            # Synchronized camera image
            if self.camera_images and idx < len(self.camera_images):
                cam_bgr = cv2.imread(str(self.camera_images[idx]))
                if cam_bgr is not None:
                    self._show_camera(cv2.cvtColor(cam_bgr, cv2.COLOR_BGR2RGB))
            
            # Engineering stats box (Phases 3-8 premium telemetry)
            ai_state = f"Semantic={'ON' if self.semantic_enabled else 'OFF'} Tracking={'ON' if self.tracking_enabled else 'OFF'}"
            self.stats.setText(
                f"Frame: {idx:04d} / {len(self.dataset)-1:04d} | Mode: {mode.upper()} | "
                f"Points: {orig_cnt:,} (Raw) -> {comp_cnt:,} ({'Raw Full' if is_raw else 'Foveated Variable Grid'}) | "
                f"Reduction: {ratio:.1f}% | {ai_state} | Compute: {compute_txt}\n"
                f"Semantic 3D: Ground {sem.get('ground',0):,} / Veg {sem.get('vegetation',0):,} / Obstacle {sem.get('obstacle',0):,} | "
                f"Tracked IDs: {n_trk} | "
                f"Fovea (<=10m): 5cm grid nominal | Periphery (10-50m): 25cm grid | Far-field: 50cm grid | "
                f"Gaze Target: [{primary_gaze[0]:.2f}, {primary_gaze[1]:.2f}, {primary_gaze[2]:.2f}] m"
            )
            self.frame_label.setText(f"{idx} / {len(self.dataset)-1}")
            
        except Exception as e:
            self.stats.setText(f"Frame update error: {e}")

    def _show_camera(self, rgb_image: np.ndarray):
        """Display synchronized RGB camera feed in QLabel."""
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        qt_img = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        scaled = qt_img.scaled(self.camera_display.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
        self.camera_display.setPixmap(QtGui.QPixmap.fromImage(scaled))
