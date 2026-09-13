"""Phase 1 Dataset Tab for KITTI visualization."""

from pathlib import Path
from PySide6 import QtWidgets, QtCore
from fovealidar.data.loaders import DatasetFactory


class DatasetTab(QtWidgets.QWidget):
    """Phase 1: KITTI Dataset Engine tab."""
    
    def __init__(self):
        super().__init__()
        layout = QtWidgets.QVBoxLayout(self)
        
        # Title
        title = QtWidgets.QLabel("Phase 1: KITTI Dataset Engine")
        title.setStyleSheet("font-size: 14px; font-weight: bold; color: #2E7D32;")
        layout.addWidget(title)
        
        # Path input
        path_layout = QtWidgets.QHBoxLayout()
        path_layout.addWidget(QtWidgets.QLabel("KITTI Root:"))
        self.path_input = QtWidgets.QLineEdit()
        self.path_input.setPlaceholderText("Enter KITTI dataset root path...")
        path_layout.addWidget(self.path_input)
        self.browse_btn = QtWidgets.QPushButton("Browse")
        self.browse_btn.clicked.connect(self._browse_dataset)
        path_layout.addWidget(self.browse_btn)
        layout.addLayout(path_layout)
        
        # Load button
        self.load_btn = QtWidgets.QPushButton("Load Dataset")
        self.load_btn.clicked.connect(self._load_dataset)
        layout.addWidget(self.load_btn)
        
        # Dataset info
        self.dataset_info = QtWidgets.QTextEdit()
        self.dataset_info.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Dataset Information:"))
        layout.addWidget(self.dataset_info)
        
        # Frame slider
        controls_layout = QtWidgets.QHBoxLayout()
        controls_layout.addWidget(QtWidgets.QLabel("Frame:"))
        self.frame_slider = QtWidgets.QSlider()
        self.frame_slider.setOrientation(QtCore.Qt.Horizontal)
        self.frame_slider.sliderMoved.connect(self._on_frame_changed)
        controls_layout.addWidget(self.frame_slider)
        self.frame_label = QtWidgets.QLabel("0/0")
        controls_layout.addWidget(self.frame_label)
        layout.addLayout(controls_layout)
        
        # Frame details
        self.frame_details = QtWidgets.QTextEdit()
        self.frame_details.setReadOnly(True)
        layout.addWidget(QtWidgets.QLabel("Frame Details:"))
        layout.addWidget(self.frame_details)
        
        self.dataset = None
        self._update_info()
    
    def _browse_dataset(self):
        """Browse for KITTI dataset directory."""
        path = QtWidgets.QFileDialog.getExistingDirectory(self, "Select KITTI Dataset Root")
        if path:
            self.path_input.setText(path)
    
    def _load_dataset(self):
        """Load KITTI dataset."""
        path = self.path_input.text().strip()
        if not path:
            QtWidgets.QMessageBox.warning(self, "Error", "Please enter a path")
            return
        
        try:
            self.dataset = DatasetFactory.auto_detect(Path(path))
            if not self.dataset:
                self.dataset = DatasetFactory.create("kitti", Path(path))
            self.frame_slider.setMaximum(len(self.dataset) - 1)
            self.frame_slider.setValue(0)
            self._update_info()
            QtWidgets.QMessageBox.information(self, "Success", f"Loaded {len(self.dataset)} frames")
        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to load: {e}")
            self.dataset = None
            self._update_info()
    
    def _on_frame_changed(self, value):
        """Update frame details."""
        if self.dataset and 0 <= value < len(self.dataset):
            frame = self.dataset[value]
            details = f"Frame {frame.frame_id}\nPoints: {frame.num_points:,}\nTimestamp: {frame.timestamp:.4f}s"
            if frame.has_pose:
                pos = frame.get_position()
                details += f"\nPosition: ({pos[0]:.2f}, {pos[1]:.2f}, {pos[2]:.2f})"
            self.frame_details.setText(details)
            self.frame_label.setText(f"{value}/{len(self.dataset)-1}")
    
    def _update_info(self):
        """Update dataset information display."""
        if self.dataset:
            try:
                stats = self.dataset.get_stats()
                info = f"Frames: {stats.num_frames}\nAvg Points: {stats.avg_points_per_frame:.0f}"
                self.dataset_info.setText(info)
            except Exception as e:
                self.dataset_info.setText(f"Error: {e}")
        else:
            self.dataset_info.setText("No dataset loaded. Enter path and click Load.")
