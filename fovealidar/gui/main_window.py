"""Main entry point for FoveaLiDAR desktop application."""

import sys
import logging
from pathlib import Path

from PySide6 import QtWidgets, QtCore

from fovealidar.core.config import load_config, get_config
from fovealidar.core.logging import setup_logging, get_logger
from fovealidar.core.types import validate_config
from fovealidar.core.diagnostics import run_diagnostics, get_compute_recommendation
from fovealidar.gui.dataset_tab import DatasetTab
from fovealidar.gui.foveation_tab import FoveationVisualizerTab
from fovealidar.gui.map_tab import MapTab


logger = get_logger(__name__)


class FoveaLiDARMainWindow(QtWidgets.QMainWindow):
    """Main application window for FoveaLiDAR."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("FoveaLiDAR - Adaptive Variable Resolution 2.5D Perception (Phases 0-10)")
        self.setGeometry(100, 100, 1440, 920)
        
        central_widget = QtWidgets.QWidget()
        self.setCentralWidget(central_widget)
        layout = QtWidgets.QVBoxLayout(central_widget)
        
        # Title
        title = QtWidgets.QLabel("FoveaLiDAR: Adaptive Variable Resolution 2.5D LiDAR Perception Engine (Phases 0-10)")
        title.setStyleSheet("font-size: 16px; font-weight: bold; color: #38bdf8;")
        layout.addWidget(title)
        
        # Tabs
        self.tabs = QtWidgets.QTabWidget()
        layout.addWidget(self.tabs)
        
        # System info tab
        self.sys_info = QtWidgets.QTextEdit()
        self.sys_info.setReadOnly(True)
        self.tabs.addTab(self.sys_info, "System Diagnostics")
        
        # Config tab
        self.config_display = QtWidgets.QTextEdit()
        self.config_display.setReadOnly(True)
        self.tabs.addTab(self.config_display, "Configuration")
        
        # Dataset tab (Phase 1)
        self.dataset_tab = DatasetTab()
        self.tabs.addTab(self.dataset_tab, "KITTI Dataset (Phase 1)")
        
        # Foveation tab (Phase 3)
        self.foveation_tab = FoveationVisualizerTab()
        self.tabs.addTab(self.foveation_tab, "3D Foveation + Semantic AI (Phases 3-8)")
        
        # 2.5D Map tab (Phase 4, 5, 6, 7 & 8)
        self.map_tab = MapTab()
        self.tabs.addTab(self.map_tab, "2.5D Map: Mode A, B & C + Semantic AI + Tracking (Phases 4-10)")
        
        # Status
        self.statusBar().showMessage("FoveaLiDAR Phases 0-10 (Compute-Aware + Benchmark + Demo) - Ready")
        
        self._populate_tabs()
    
    def _populate_tabs(self):
        """Populate diagnostic and config tabs."""
        diag = run_diagnostics()
        
        sys_text = f"""
SYSTEM DIAGNOSTICS
==================

OS: {diag.os} {diag.os_version}
Python: {diag.python_version}

CPU: {diag.cpu.model}
  Cores: {diag.cpu.cores_physical}P / {diag.cpu.cores_logical}L
  Freq: {diag.cpu.frequency_mhz:.0f} MHz

Memory: {diag.memory.total_gb:.1f} GB total, {diag.memory.available_gb:.1f} GB available

GPUs: {len(diag.gpus)} detected
"""
        for gpu in diag.gpus:
            sys_text += f"  - {gpu.name}: {gpu.vram_total_gb:.1f} GB\n"
        
        sys_text += f"\nCUDA Available: {diag.cuda_available}\n"
        sys_text += f"Recommended Mode: {get_compute_recommendation(diag)}\n"
        
        if diag.warnings:
            sys_text += "\nWarnings:\n"
            for w in diag.warnings:
                sys_text += f"  ⚠ {w}\n"
        
        self.sys_info.setText(sys_text)
        
        config = get_config()
        cfg_text = f"""
CONFIGURATION (Phase 0 & Phase 1)
==================================

Perception: {config.perception.semantic_backend}
Mapping Mode: {config.mapping.mode}
  Near: {config.mapping.near_resolution_m} m ({config.mapping.near_range_m} m)
  Far: {config.mapping.far_resolution_m} m ({config.mapping.far_range_m} m)

Foveation: {'Enabled' if config.foveation.enabled else 'Disabled'}
Compute: {config.compute.mode}

Data (Phase 1):
  KITTI Root: {config.data.kitti_root or '(not configured)'}
  Sequence: {config.data.sequence}
  Playback FPS: {config.data.playback_target_fps}
"""
        self.config_display.setText(cfg_text)


def main():
    """Main entry point."""
    setup_logging()
    logger = get_logger(__name__)
    
    logger.info("=" * 60)
    logger.info("FoveaLiDAR Phases 0-10 Starting")
    logger.info("=" * 60)
    
    try:
        config = load_config()
        errors = validate_config(config)
        if errors:
            for e in errors:
                logger.error(f"Config: {e}")
            return 1
        
        logger.info("Configuration validated")
        
        diag = run_diagnostics()
        logger.info(f"Diagnostics complete. Mode: {get_compute_recommendation(diag)}")
        
    except Exception as e:
        logger.error(f"Error: {e}")
        return 1
    
    app = QtWidgets.QApplication(sys.argv)
    window = FoveaLiDARMainWindow()
    window.show()
    
    logger.info("UI opened - Phases 0-10 ready")
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())