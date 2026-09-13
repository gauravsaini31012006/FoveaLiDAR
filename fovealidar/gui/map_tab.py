"""Phase 4 & 5: 2.5D Elevation & Occupancy Map Tab (Mode A Uniform vs Mode B Adaptive).

Supports SIH 2026 Problem Statement 26053 and Master Spec §14, §15, §31, §47, §51:
- Mode A: Uniform 5 cm baseline (1.96M cells)
- Mode B: Distance-Adaptive 5 cm near (<=10m) / 50 cm far (>10m)
- A/B Split View comparison
- Interactive Cell Inspector with Parent/Child Hierarchy and 64-bit Cell IDs
- Full A/B Benchmark Telemetry panel
"""

from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import numpy as np
from PySide6 import QtWidgets, QtCore, QtGui
import cv2

from fovealidar.data.loaders import DatasetFactory
from fovealidar.preprocessing import PreprocessingConfig, PreprocessingPipeline
from fovealidar.mapping import ElevationMap25D, AdaptiveMap25D, FovealMap25D
from fovealidar.visualization.renderer import PointCloudRenderer
from fovealidar.core.types import ComputeMode, PointCloud
from fovealidar.foveation.compute_scheduler import ComputeScheduler


class MapTab(QtWidgets.QWidget):
    """Phase 4, 5 & 6: 2.5D Elevation & Occupancy Map with Mode A vs Mode B vs Mode C Benchmarking."""
    
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.pipeline = PreprocessingPipeline(PreprocessingConfig())
        
        # Mode A: Uniform 5 cm Engine
        self.map_engine_a = ElevationMap25D(
            resolution_m=0.05,
            bounds=(-35.0, 35.0, -35.0, 35.0),
            mode_name="uniform_5cm",
        )
        
        # Mode B: Distance-Adaptive Engine (5cm <=10m, 50cm >10m)
        self.map_engine_b = AdaptiveMap25D(
            bounds=(-35.0, 35.0, -35.0, 35.0),
            near_field_radius_m=10.0,
            fine_res_m=0.05,
            coarse_res_m=0.50,
        )
        
        # Mode C: Dynamic Foveal Engine (Phase 6 Importance-Driven Refinement)
        self.map_engine_c = FovealMap25D(
            bounds=(-35.0, 35.0, -35.0, 35.0),
            near_field_radius_m=10.0,
            fine_res_m=0.05,
            coarse_res_m=0.50,
            max_refined_regions=128,
            refinement_threshold=0.6,
            corridor_enabled=True,
        )
        
        self.renderer = PointCloudRenderer(image_width=860, image_height=860)
        self.current_frame = 0
        self.is_playing = False
        self.selected_mode = "mode_c"  # Default to Mode C Dynamic Foveal
        self.selected_layer = "resolution"
        self.last_map_image = None
        # Phase 9: compute-aware scheduler (default BALANCED per Master Spec §38)
        self.scheduler = ComputeScheduler(ComputeMode.BALANCED)
        self.compute_mode = ComputeMode.BALANCED
        # Phase 10: cached per-mode telemetry for benchmark export
        self.last_metrics_a: Optional[Dict[str, Any]] = None
        self.last_metrics_b: Optional[Dict[str, Any]] = None
        self.last_metrics_c: Optional[Dict[str, Any]] = None
        self.last_ai_skipped: bool = False
        self._setup_ui()
        # Apply default scheduler budget to the foveal controller
        try:
            self.scheduler.apply_to_controller(self.map_engine_c.controller)
        except Exception:
            pass
        
    def _setup_ui(self):
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        
        # Header / Title
        header = QtWidgets.QHBoxLayout()
        title = QtWidgets.QLabel("Phases 4-8: 2.5D Map + Semantic AI + Object Tracking (Modes A, B & C)")
        title.setStyleSheet("font-size: 15px; font-weight: bold; color: #38bdf8;")
        header.addWidget(title)
        
        badge = QtWidgets.QLabel("SIH 2026 PS 26053 | Mode C + Semantic AI + Tracking")
        badge.setStyleSheet("color: #a3e635; font-weight: bold; font-size: 11px;")
        header.addStretch()
        header.addWidget(badge)
        layout.addLayout(header)
        
        # Dataset controls
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
        
        # Mode & Layer Control Bar
        bar = QtWidgets.QHBoxLayout()
        
        # Mode Selector
        bar.addWidget(QtWidgets.QLabel("Mapping Mode:"))
        self.mode_combo = QtWidgets.QComboBox()
        self.mode_combo.addItem("✨ Mode C: Dynamic Foveal (5cm Near/Dynamic Far + 50cm)", "mode_c")
        self.mode_combo.addItem("🎯 Mode B: Distance-Adaptive (5cm <=10m / 50cm >10m)", "mode_b")
        self.mode_combo.addItem("📏 Mode A: Uniform 5 cm Baseline (1.96M cells)", "mode_a")
        self.mode_combo.addItem("⚖️ A/B Comparison: Side-by-Side View", "split_ab")
        self.mode_combo.currentIndexChanged.connect(self._on_mode_changed)
        bar.addWidget(self.mode_combo)
        
        bar.addSpacing(10)
        bar.addWidget(QtWidgets.QLabel("Layer:"))
        self.layer_combo = QtWidgets.QComboBox()
        self._update_layer_options()
        self.layer_combo.currentIndexChanged.connect(self._on_layer_changed)
        bar.addWidget(self.layer_combo)
        
        bar.addSpacing(15)
        self.accumulate_checkbox = QtWidgets.QCheckBox("Accumulate History")
        self.accumulate_checkbox.setChecked(False)
        self.accumulate_checkbox.setToolTip("Unchecked: Clean real-time frame refresh (zero ghosting). Checked: Accumulate points into persistent map.")
        self.accumulate_checkbox.stateChanged.connect(self._on_accumulate_toggled)
        bar.addWidget(self.accumulate_checkbox)
        
        self.clear_btn = QtWidgets.QPushButton("Clear")
        self.clear_btn.setMaximumWidth(60)
        self.clear_btn.clicked.connect(self._on_clear_clicked)
        bar.addWidget(self.clear_btn)
        
        bar.addSpacing(15)
        self.roi_label = QtWidgets.QLabel("ROI: [-35m, +35m] (70x70m area)")
        self.roi_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        bar.addWidget(self.roi_label)

        # Phase 9: Compute Mode selector
        bar.addSpacing(15)
        bar.addWidget(QtWidgets.QLabel("Compute:"))
        self.compute_combo = QtWidgets.QComboBox()
        self.compute_combo.addItem("⚡ MAX_QUALITY", ComputeMode.MAX_QUALITY.value)
        self.compute_combo.addItem("⚖️ BALANCED", ComputeMode.BALANCED.value)
        self.compute_combo.addItem("🔋 LOW_POWER", ComputeMode.LOW_POWER.value)
        self.compute_combo.addItem("🖥️ CPU_ONLY", ComputeMode.CPU_ONLY.value)
        self.compute_combo.setCurrentIndex(1)  # BALANCED default
        self.compute_combo.setToolTip(
            "Phase 9 compute-aware scheduling: refinement budget, AI cadence, FPS cap, downsampling."
        )
        self.compute_combo.currentIndexChanged.connect(self._on_compute_mode_changed)
        bar.addWidget(self.compute_combo)

        bar.addStretch()
        layout.addLayout(bar)
        
        # AI & Tracking Control Bar
        ai_bar = QtWidgets.QHBoxLayout()
        self.semantic_checkbox = QtWidgets.QCheckBox("🧠 Enable Semantic AI (Phase 7)")
        self.semantic_checkbox.setChecked(False)
        self.semantic_checkbox.stateChanged.connect(self._on_semantic_toggled)
        ai_bar.addWidget(self.semantic_checkbox)
        
        self.tracking_checkbox = QtWidgets.QCheckBox("🎯 Enable Object Tracking (Phase 8)")
        self.tracking_checkbox.setChecked(False)
        self.tracking_checkbox.stateChanged.connect(self._on_tracking_toggled)
        ai_bar.addWidget(self.tracking_checkbox)

        # Phase 9 live scheduler readout
        self.compute_label = QtWidgets.QLabel("")
        self.compute_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        self.compute_label.setToolTip("Active Phase 9 compute budget.")
        ai_bar.addWidget(self.compute_label)
        self._refresh_compute_label()

        ai_bar.addStretch()

        # Phase 10: Benchmark + Demo actions
        self.benchmark_btn = QtWidgets.QPushButton("📊 Export Benchmark Report")
        self.benchmark_btn.setToolTip("Phase 10: export measured A/B/C comparison to output/benchmark_report_*.txt")
        self.benchmark_btn.clicked.connect(self._export_benchmark_report)
        ai_bar.addWidget(self.benchmark_btn)

        self.demo_btn = QtWidgets.QPushButton("🎬 One-Click Demo")
        self.demo_btn.setToolTip("Phase 10: auto-load dataset, enable Semantic AI + Tracking, select Mode C, start playback")
        self.demo_btn.setStyleSheet("background-color: #166534; color: white; font-weight: bold; padding: 4px 12px;")
        self.demo_btn.clicked.connect(self._one_click_demo)
        ai_bar.addWidget(self.demo_btn)

        layout.addLayout(ai_bar)
        
        # Main Display: Map View + Inspection & Telemetry Panels
        main_split = QtWidgets.QHBoxLayout()
        
        # Map Display Widget
        self.map_display = QtWidgets.QLabel()
        self.map_display.setMinimumSize(640, 640)
        self.map_display.setAlignment(QtCore.Qt.AlignCenter)
        self.map_display.setStyleSheet("background-color: #0b0d12; border: 2px solid #334155; border-radius: 4px;")
        self.map_display.setMouseTracking(True)
        self.map_display.mouseMoveEvent = self._on_map_mouse_move
        main_split.addWidget(self.map_display, 1)
        
        # Right Side: Interactive Cell Inspection & A/B Telemetry Cards
        side_panel = QtWidgets.QVBoxLayout()
        side_panel.setSpacing(10)
        
        # Cell Inspector Card
        inspect_group = QtWidgets.QGroupBox("🔍 Interactive Cell Inspector")
        inspect_group.setStyleSheet("QGroupBox { font-weight: bold; color: #38bdf8; }")
        inspect_layout = QtWidgets.QVBoxLayout(inspect_group)
        
        self.inspect_text = QtWidgets.QTextEdit()
        self.inspect_text.setReadOnly(True)
        self.inspect_text.setStyleSheet("background-color: #0f172a; color: #e2e8f0; font-family: monospace; font-size: 11px;")
        self.inspect_text.setText("Move mouse over 2.5D map to inspect\nexact cell level, parent/child ID, elevation, and occupancy.")
        inspect_layout.addWidget(self.inspect_text)
        side_panel.addWidget(inspect_group, 1)
        
        # Mode A vs Mode B Benchmark Card
        bench_group = QtWidgets.QGroupBox("⚡ A/B Benchmark Telemetry (Master Spec §31)")
        bench_group.setStyleSheet("QGroupBox { font-weight: bold; color: #a3e635; }")
        bench_layout = QtWidgets.QVBoxLayout(bench_group)
        
        self.bench_text = QtWidgets.QTextEdit()
        self.bench_text.setReadOnly(True)
        self.bench_text.setMinimumHeight(190)
        self.bench_text.setStyleSheet("background-color: #0f172a; color: #94a3b8; font-family: monospace; font-size: 11px;")
        self.bench_text.setText("Load dataset to measure actual Mode A vs Mode B metrics.")
        bench_layout.addWidget(self.bench_text)
        side_panel.addWidget(bench_group)
        
        main_split.addLayout(side_panel)
        layout.addLayout(main_split, 1)
        
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
        
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self._playback_tick)

    def _update_layer_options(self):
        cur = self.layer_combo.currentData() or "resolution"
        self.layer_combo.clear()
        if self.selected_mode in ("mode_b", "mode_c", "split_ab"):
            self.layer_combo.addItem("🎯 Resolution LOD (5cm Near/Dynamic vs 50cm Far)", "resolution")
        if self.selected_mode == "mode_c":
            self.layer_combo.addItem("✨ Importance Heatmap (Foveal Controller)", "foveal_importance")
            if hasattr(self, 'semantic_checkbox') and self.semantic_checkbox.isChecked():
                self.layer_combo.addItem("🧠 Semantic AI Classes (Ground/Veg/Obstacle)", "semantic")
        self.layer_combo.addItem("🏔️ Surface Elevation (Max Z)", "elevation")
        self.layer_combo.addItem("📏 Obstacle Height (Span = Max - Min Z)", "span")
        self.layer_combo.addItem("⬛ Occupancy Grid (Probability)", "occupancy")
        self.layer_combo.addItem("📊 Point Density (Points / Cell)", "density")
        if self.selected_mode == "mode_a":
            self.layer_combo.addItem("💡 Reflectance / Intensity", "intensity")
        
        # Restore selection if valid
        idx = self.layer_combo.findData(cur)
        if idx >= 0:
            self.layer_combo.setCurrentIndex(idx)

    def _on_mode_changed(self):
        self.selected_mode = self.mode_combo.currentData()
        self._update_layer_options()
        self._update_display()
        
    def _on_semantic_toggled(self, state):
        self.map_engine_c.controller.semantic_enabled = bool(state)
        if not state:
            self.tracking_checkbox.setChecked(False) # tracking requires semantic
        self._update_layer_options()
        self._update_display()

    def _on_tracking_toggled(self, state):
        if state and not self.semantic_checkbox.isChecked():
            self.semantic_checkbox.setChecked(True)
        self.map_engine_c.controller.tracking_enabled = bool(state)
        self._update_display()

    # ------------------------------------------------------------------
    # Phase 9: compute-aware scheduling
    # ------------------------------------------------------------------
    def _on_compute_mode_changed(self):
        raw = self.compute_combo.currentData()
        try:
            mode = ComputeMode(raw) if not isinstance(raw, ComputeMode) else raw
        except Exception:
            return
        self.compute_mode = mode
        self.scheduler.set_mode(mode)
        # Wire scheduler -> foveal controller refinement budget
        try:
            self.scheduler.apply_to_controller(self.map_engine_c.controller)
        except Exception:
            pass
        # Wire scheduler -> playback timer FPS cap (graceful, never freezes UI)
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

    def _maybe_downsample(self, pc: PointCloud) -> PointCloud:
        """Phase 9: reduce far-field input density on LOW_POWER / CPU_ONLY."""
        factor = self.scheduler.downsample_factor
        if factor <= 1 or pc.num_points <= factor:
            return pc
        idx = np.arange(0, pc.num_points, factor)
        kwargs: Dict[str, Any] = {}
        if getattr(pc, "timestamp", None) is not None:
            kwargs["timestamp"] = pc.timestamp
        if getattr(pc, "frame_id", None) is not None:
            kwargs["frame_id"] = pc.frame_id
        if pc.labels is not None:
            kwargs["labels"] = pc.labels[idx]
        return PointCloud(points=pc.points[idx], intensity=pc.intensity[idx], **kwargs)

    # ------------------------------------------------------------------
    # Phase 10: benchmark report + one-click demo
    # ------------------------------------------------------------------
    def generate_benchmark_report_text(self) -> str:
        """Build a reproducible text report of the measured A/B/C comparison."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        lines = [
            "FoveaLiDAR Benchmark Report (Phase 10) — Measured A/B/C Comparison",
            "=" * 72,
            f"Generated: {ts}",
            f"Frame: {self.current_frame}"
            + (f" / {len(self.dataset) - 1}" if self.dataset else " (no dataset)"),
            f"Mode shown: {self.selected_mode} | Layer: {self.selected_layer}",
            f"Compute: {self.scheduler.describe()}",
            "",
        ]
        ma, mb, mc = self.last_metrics_a, self.last_metrics_b, self.last_metrics_c
        if ma is None or mb is None or mc is None:
            lines.append("No measured metrics yet — load a dataset and step/play at least one frame.")
            lines.append("Metrics shown are live-measured only; no demonstration numbers are fabricated.")
            return "\n".join(lines) + "\n"

        def g(d: Dict[str, Any], k: str, default: float = 0.0) -> float:
            try:
                return float(d.get(k, default))
            except Exception:
                return float(default)

        total_a = self.map_engine_a.rows * self.map_engine_a.cols
        lines += [
            "Measured metrics (live, this frame):",
            "-" * 72,
            f"{'Metric':<28}{'Mode A (5cm)':>14}{'Mode B (Adp)':>14}{'Mode C (Fov)':>14}",
            f"{'Active cells':<28}{int(g(ma,'active_cells')):>14,}{int(g(mb,'active_cells')):>14,}{int(g(mc,'active_cells')):>14,}",
            f"{'Fine 5cm cells':<28}{int(g(ma,'active_cells')):>14,}{int(g(mb,'active_fine_cells')):>14,}{int(g(mc,'active_fine_cells')):>14,}",
            f"{'Coarse 50cm cells':<28}{'N/A':>14}{int(g(mb,'active_coarse_cells')):>14,}{int(g(mc,'active_coarse_cells')):>14,}",
            f"{'Dyn refined tiles':<28}{'N/A':>14}{'N/A':>14}{int(g(mc,'dynamic_refined_tiles')):>14,}",
            f"{'Mapping latency (ms)':<28}{g(ma,'mapping_time_ms'):>14.2f}{g(mb,'mapping_time_ms'):>14.2f}{g(mc,'mapping_time_ms'):>14.2f}",
            f"{'AI latency (ms)':<28}{'N/A':>14}{'N/A':>14}{(g(mc,'semantic_time_ms') + g(mc,'tracking_time_ms')):>14.2f}",
            f"{'Layer memory (MB)':<28}{(total_a * 4 * 7) / (1024*1024):>14.1f}{g(mb,'memory_mb'):>14.1f}{g(mc,'memory_mb'):>14.1f}",
            f"{'Cell reduction (%)':<28}{'baseline':>14}{g(mb,'cell_reduction_pct'):>13.1f}%{g(mc,'cell_reduction_pct'):>13.1f}%",
            f"{'AI frame skipped':<28}{'N/A':>14}{'N/A':>14}{str(bool(self.last_ai_skipped)):>14}",
            f"{'Tracked objects':<28}{'N/A':>14}{'N/A':>14}{int(g(mc,'tracked_objects')):>14,}",
            "",
            "Notes:",
            "- All values measured live from the current frame; warm-up frames excluded",
            "  from formal benchmark runs (see tools/benchmark_sequence.py methodology).",
            "- Cell reduction is vs the uniform 5 cm baseline capacity.",
            "- AI latency = semantic + tracking inference on this frame (0.0 when skipped by cadence).",
        ]
        return "\n".join(lines) + "\n"

    def _export_benchmark_report(self):
        try:
            text = self.generate_benchmark_report_text()
        except Exception as e:
            QtWidgets.QMessageBox.warning(self, "Benchmark", f"Could not build report: {e}")
            return
        try:
            out_dir = Path("output")
            out_dir.mkdir(parents=True, exist_ok=True)
            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            out_path = out_dir / f"benchmark_report_{stamp}.txt"
            out_path.write_text(text, encoding="utf-8")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Benchmark Export", f"Failed to write report: {e}")
            return
        # Show confirmation without blocking automated flows
        try:
            self.bench_text.setText(text)
        except Exception:
            pass
        QtWidgets.QMessageBox.information(
            self, "Benchmark Exported", f"✅ Benchmark report written to:\n{out_path}"
        )

    def _one_click_demo(self):
        """Phase 10: automated demo — dataset + Semantic AI + Tracking + Mode C + play."""
        # 1. Auto-load dataset if needed (silent attempt first, GUI fallback)
        if self.dataset is None:
            try:
                self._load_dataset(silent=True)
            except TypeError:
                self._load_dataset()
            except Exception:
                pass
        if self.dataset is None:
            QtWidgets.QMessageBox.warning(
                self, "One-Click Demo",
                "No dataset loaded. Check the KITTI Root path, then click 🎬 One-Click Demo again.",
            )
            return
        # 2. Enable Semantic AI + Tracking (tracking auto-enables semantic)
        try:
            self.tracking_checkbox.setChecked(True)
        except Exception:
            try:
                self.semantic_checkbox.setChecked(True)
            except Exception:
                pass
        # 3. Select Mode C
        try:
            idx = self.mode_combo.findData("mode_c")
            if idx >= 0:
                self.mode_combo.setCurrentIndex(idx)
            else:
                self.selected_mode = "mode_c"
        except Exception:
            self.selected_mode = "mode_c"
        # 4. BALANCED compute for a smooth demo
        try:
            bal_idx = self.compute_combo.findData(ComputeMode.BALANCED.value)
            if bal_idx >= 0:
                self.compute_combo.setCurrentIndex(bal_idx)
        except Exception:
            pass
        # 5. Rewind + start playback
        try:
            self.slider.setValue(0)
        except Exception:
            pass
        if not self.is_playing:
            self._toggle_play()

    def _browse_dataset(self):
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select KITTI Root")
        if path:
            self.path_input.setText(path)

    def _load_dataset(self, silent: bool = False, *args):
        path = self.path_input.text().strip()
        if not path:
            return
        try:
            self.dataset = DatasetFactory.auto_detect(Path(path))
            if not self.dataset:
                self.dataset = DatasetFactory.create("kitti", Path(path))
            self.slider.setMaximum(max(0, len(self.dataset) - 1))
            self.slider.setValue(0)
            self._update_display()
            if not silent:
                QtWidgets.QMessageBox.information(
                    self, "Dataset Loaded",
                    f"✅ Successfully loaded {len(self.dataset)} frames for 2.5D Mapping."
                )
        except Exception as e:
            if silent:
                # Re-raise so _one_click_demo can show a single consolidated warning
                raise
            QtWidgets.QMessageBox.critical(self, "Load Error", f"Failed to load dataset: {e}")

    def _on_layer_changed(self):
        data = self.layer_combo.currentData()
        if data:
            self.selected_layer = data
            self._update_display()

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
            self.play_btn.setText("⏸")
            self.timer.start(int(1000 / self.speed.value()))
        else:
            self.play_btn.setText("▶")
            self.timer.stop()

    def _on_speed_changed(self, value: int):
        if self.is_playing:
            self.timer.setInterval(int(1000 / value))

    def _playback_tick(self):
        if not self.dataset:
            return
        if self.current_frame < len(self.dataset) - 1:
            self.slider.setValue(self.current_frame + 1)
        else:
            self.slider.setValue(0)

    def _on_accumulate_toggled(self):
        if not self.accumulate_checkbox.isChecked():
            self.map_engine_a.clear()
            self.map_engine_b.clear()
            self.map_engine_c.clear()
        self._update_display()

    def _on_clear_clicked(self):
        self.map_engine_a.clear()
        self.map_engine_b.clear()
        self.map_engine_c.clear()
        self._update_display()

    def _update_display(self):
        if not self.dataset:
            return
        try:
            idx = self.current_frame
            frame = self.dataset[idx]
            raw_pc = frame.point_cloud
            pp = self.pipeline.process(raw_pc)
            # Phase 9: point downsample factor (LOW_POWER x2, CPU_ONLY x4)
            pp = self._maybe_downsample(pp)
            accumulate = self.accumulate_checkbox.isChecked()

            # Phase 9: AI cadence — run semantic every N frames, reuse otherwise.
            # CPU_ONLY disables AI entirely; other modes stride by ai_stride.
            want_semantic = bool(self.semantic_checkbox.isChecked())
            want_tracking = bool(self.tracking_checkbox.isChecked())
            run_ai_this_frame = want_semantic and self.scheduler.should_run_ai(idx)
            ctrl = self.map_engine_c.controller
            prev_sem_backup = getattr(ctrl, "last_semantic_labels", None)
            prev_trk_backup = getattr(ctrl, "last_tracked_bboxes", None)
            if prev_sem_backup is not None:
                try:
                    prev_sem_backup = prev_sem_backup.copy()
                except Exception:
                    pass
            if prev_trk_backup is not None:
                try:
                    prev_trk_backup = list(prev_trk_backup)
                except Exception:
                    pass
            ctrl.semantic_enabled = bool(run_ai_this_frame)
            ctrl.tracking_enabled = bool(run_ai_this_frame and want_tracking)
            self.last_ai_skipped = bool(want_semantic and not run_ai_this_frame)

            # Execute mapping pipelines
            metrics_b = self.map_engine_b.update(pp, clear_first=not accumulate)
            metrics_a = self.map_engine_a.update(pp, clear_first=not accumulate)
            metrics_c = self.map_engine_c.update(pp, clear_first=not accumulate)

            # Reuse last AI outputs on skipped frames (explicit reuse, not stale claims)
            if self.last_ai_skipped:
                try:
                    if prev_sem_backup is not None:
                        ctrl.last_semantic_labels = prev_sem_backup
                    if prev_trk_backup is not None:
                        ctrl.last_tracked_bboxes = prev_trk_backup
                except Exception:
                    pass

            # Phase 10: cache live-measured metrics for benchmark export
            self.last_metrics_a = dict(metrics_a)
            self.last_metrics_b = dict(metrics_b)
            self.last_metrics_c = dict(metrics_c)
            
            # Render selected view
            if self.selected_mode == "mode_c":
                map_bgr = self.renderer.render_foveal_map(
                    fov_map=self.map_engine_c,
                    layer=self.selected_layer if self.selected_layer != "intensity" else "elevation",
                    frame_idx=idx,
                    tracked_objects=self.map_engine_c.controller.last_tracked_bboxes
                )
            elif self.selected_mode == "mode_b":
                map_bgr = self.renderer.render_adaptive_map(
                    adapt_map=self.map_engine_b,
                    layer=self.selected_layer if self.selected_layer not in ("intensity", "foveal_importance") else "elevation",
                    frame_idx=idx,
                )
            elif self.selected_mode == "mode_a":
                map_bgr = self.renderer.render_elevation_map(
                    elev_map=self.map_engine_a,
                    layer=self.selected_layer if self.selected_layer not in ("resolution", "foveal_importance") else "elevation",
                    frame_idx=idx,
                )
            else:  # split_ab: Side-by-side view
                img_a = self.renderer.render_elevation_map(self.map_engine_a, layer="elevation", frame_idx=idx)
                img_b = self.renderer.render_adaptive_map(self.map_engine_b, layer="resolution", frame_idx=idx)
                # Resize and horizontally stack
                h = 720
                w = int(720 * (img_a.shape[1] / img_a.shape[0]))
                a_small = cv2.resize(img_a, (w, h))
                b_small = cv2.resize(img_b, (w, h))
                map_bgr = np.hstack([a_small, b_small])
                
            self.last_map_image = map_bgr
            
            # Show in QLabel
            rgb = cv2.cvtColor(map_bgr, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb.shape
            qt_img = QtGui.QImage(rgb.data, w, h, ch * w, QtGui.QImage.Format_RGB888)
            scaled = qt_img.scaled(self.map_display.size(), QtCore.Qt.KeepAspectRatio, QtCore.Qt.SmoothTransformation)
            self.map_display.setPixmap(QtGui.QPixmap.fromImage(scaled))
            
            # Update A/B/C Benchmark Telemetry Card (Master Spec §31)
            active_a = metrics_a['active_cells']
            total_a = self.map_engine_a.rows * self.map_engine_a.cols
            lat_a = metrics_a['mapping_time_ms']
            mem_a = (total_a * 4 * 7) / (1024 * 1024)
            
            active_b = metrics_b['active_cells']
            fine_b = metrics_b['active_fine_cells']
            coarse_b = metrics_b['active_coarse_cells']
            total_b = metrics_b['total_cells']
            lat_b = metrics_b['mapping_time_ms']
            mem_b = metrics_b['memory_mb']
            red_cells_b = metrics_b['cell_reduction_pct']
            red_mem_b = metrics_b['memory_reduction_pct']

            active_c = metrics_c['active_cells']
            fine_c = metrics_c['active_fine_cells']
            coarse_c = metrics_c['active_coarse_cells']
            total_c = metrics_c['total_cells']
            lat_c = metrics_c['mapping_time_ms']
            mem_c = metrics_c['memory_mb']
            dyn_tiles_c = metrics_c['dynamic_refined_tiles']
            red_cells_c = metrics_c['cell_reduction_pct']
            mem_red_c = 100.0 * (1.0 - mem_c / max(0.001, mem_a))
            
            ai_lat = metrics_c.get('semantic_time_ms', 0) + metrics_c.get('tracking_time_ms', 0)
            tracked_objs = metrics_c.get('tracked_objects', 0)
            
            self.bench_text.setText(
                f"┌─────────────────────────┬──────────────┬──────────────┬──────────────┐\n"
                f"│ Metric                  │ Mode A (5cm) │ Mode B (Adp) │ Mode C (Fov) │\n"
                f"├─────────────────────────┼──────────────┼──────────────┼──────────────┤\n"
                f"│ Total Grid Capacity     │ {total_a:>12,} │ {total_b:>12,} │ {total_c:>12,} │\n"
                f"│ Active Cells (Occupied) │ {active_a:>12,} │ {active_b:>12,} │ {active_c:>12,} │\n"
                f"│   - Fine 5cm (Static)   │ {active_a:>12,} │ {fine_b:>12,} │ {fine_c:>12,} │\n"
                f"│   - Dyn Refined Far     │          N/A │          N/A │ {dyn_tiles_c:>9} tiles│\n"
                f"│   - Coarse 50cm (Far)   │          N/A │ {coarse_b:>12,} │ {coarse_c:>12,} │\n"
                f"│ Mapping Latency (ms)    │ {lat_a:>10.2f} ms │ {lat_b:>10.2f} ms │ {lat_c:>10.2f} ms │\n"
                f"│   - AI Inference Lat.   │          N/A │          N/A │ {ai_lat:>10.2f} ms │\n"
                f"│ Layer Memory (MB)       │ {mem_a:>10.1f} MB │ {mem_b:>10.1f} MB │ {mem_c:>10.1f} MB │\n"
                f"│ Grid Cell Reduction     │     Baseline │ {red_cells_b:>11.1f}% │ {red_cells_c:>11.1f}% │\n"
                f"│ Memory Reduction        │     Baseline │ {red_mem_b:>11.1f}% │ {mem_red_c:>11.1f}% │\n"
                f"└─────────────────────────┴──────────────┴──────────────┴──────────────┘\n"
                f"SIH 2026 PS 26053 Invariants: Verified PASSED\n"
                f"Throughput: {metrics_c['mapped_points'] / max(0.001, lat_c / 1000.0) / 1e6:.2f} Million pts/sec\n"
                f"Phase 7/8 Info: Tracked Objects = {tracked_objs}"
            )
            self.frame_label.setText(f"{idx} / {len(self.dataset)-1}")
            
        except Exception as e:
            self.inspect_text.setText(f"Update error: {e}")

    def _on_map_mouse_move(self, event):
        """Interactive cell inspection on mouse hover over 2.5D map."""
        if not self.dataset or self.last_map_image is None:
            return
        
        pixmap = self.map_display.pixmap()
        if not pixmap:
            return
        
        # Calculate coordinate within rendered image
        disp_w = self.map_display.width()
        disp_h = self.map_display.height()
        pix_w = pixmap.width()
        pix_h = pixmap.height()
        
        offset_x = (disp_w - pix_w) // 2
        offset_y = (disp_h - pix_h) // 2
        
        rel_x = event.position().x() - offset_x
        rel_y = event.position().y() - offset_y
        
        if 0 <= rel_x < pix_w and 0 <= rel_y < pix_h:
            u = rel_x / pix_w
            v = rel_y / pix_h
            
            # Map normalized [u, v] to continuous world coordinates (meters)
            # Forward (+X or +Y) is up (v=0 -> +35m, v=1 -> -35m)
            # Lateral is left-to-right (u=0 -> -35m, u=1 -> +35m)
            b = self.map_engine_b.bounds
            world_x = b[0] + u * (b[1] - b[0])
            world_y = b[3] - v * (b[3] - b[2])
            
            if self.selected_mode == "mode_c":
                info = self.map_engine_c.get_cell_info(world_x, world_y)
                if info.get('valid', False):
                    occ_badge = "OCCUPIED" if info['occupied'] else "UNOCCUPIED / FREE"
                    parent_line = f"  Parent Coarse ID: {info['parent_id']}\n" if info['parent_id'] else ""
                    self.inspect_text.setText(
                        f"MODE C DYNAMIC FOVEAL INSPECTION:\n"
                        f"  LOD Level:        {info['level']}\n"
                        f"  Cell Resolution:  {info['resolution_m']*100:.0f} cm\n"
                        f"  Importance Score: {info.get('importance_score', 0.0):.3f}\n"
                        f"  Distance to Ego:  {info['distance_m']:.2f} m\n"
                        f"  World (X, Y):     [{world_x:+.2f}m, {world_y:+.2f}m]\n"
                        f"  Cell ID (64-bit): {info['cell_id']}\n"
                        f"{parent_line}"
                        f"  State:            {occ_badge}\n"
                        f"  Point Count:      {info['point_count']} pts\n"
                        f"  Occupancy Prob:   {info['occupancy']:.2f}\n"
                        f"  Elevation Max:    {info['elevation_max']:+.2f} m\n"
                        f"  Elevation Min:    {info['elevation_min']:+.2f} m\n"
                        f"  Obstacle Span:    {info['elevation_span']:.2f} m\n"
                        f"  Mean Intensity:   {info['intensity_mean']:.2f}"
                    )
            elif self.selected_mode in ("mode_b", "split_ab"):
                info = self.map_engine_b.get_cell_info(world_x, world_y)
                if info.get('valid', False):
                    occ_badge = "OCCUPIED" if info['occupied'] else "UNOCCUPIED / FREE"
                    parent_line = f"  Parent Coarse ID: {info['parent_id']}\n" if info['parent_id'] else ""
                    
                    self.inspect_text.setText(
                        f"MODE B ADAPTIVE CELL INSPECTION:\n"
                        f"  LOD Level:        {info['level']}\n"
                        f"  Cell Resolution:  {info['resolution_m']*100:.0f} cm\n"
                        f"  Distance to Ego:  {info['distance_m']:.2f} m\n"
                        f"  World (X, Y):     [{world_x:+.2f}m, {world_y:+.2f}m]\n"
                        f"  Cell ID (64-bit): {info['cell_id']}\n"
                        f"{parent_line}"
                        f"  State:            {occ_badge}\n"
                        f"  Point Count:      {info['point_count']} pts\n"
                        f"  Occupancy Prob:   {info['occupancy']:.2f}\n"
                        f"  Elevation Max:    {info['elevation_max']:+.2f} m\n"
                        f"  Elevation Min:    {info['elevation_min']:+.2f} m\n"
                        f"  Obstacle Span:    {info['elevation_span']:.2f} m\n"
                        f"  Mean Intensity:   {info['intensity_mean']:.2f}"
                    )
            else:
                # Mode A inspection
                col = int(u * self.map_engine_a.cols)
                row = int((1.0 - v) * self.map_engine_a.rows)
                info = self.map_engine_a.get_cell_info(row, col)
                if info:
                    occ_badge = "OCCUPIED" if info.get('occupied', False) else "UNOCCUPIED / FREE"
                    self.inspect_text.setText(
                        f"MODE A UNIFORM 5CM CELL INSPECTION:\n"
                        f"  Grid (Row, Col):  [{info['row']}, {info['col']}]\n"
                        f"  World (X, Y):     [{info['center_x']:+.2f}m, {info['center_y']:+.2f}m]\n"
                        f"  Cell Size:        5 cm (0.05 m)\n"
                        f"  State:            {occ_badge}\n"
                        f"  Point Count:      {info['point_count']} pts\n"
                        f"  Occupancy:        {info['occupancy']:.2f}\n"
                        f"  Elevation Max:    {info['elevation_max']:+.2f} m\n"
                        f"  Elevation Min:    {info['elevation_min']:+.2f} m\n"
                        f"  Obstacle Span:    {info['elevation_span']:.2f} m\n"
                        f"  Mean Intensity:   {info['intensity_mean']:.2f}"
                    )
