"""Point cloud rendering to 2D images - BEV, Sci-Fi, and Autonomous 3D views."""

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple, Dict, Any
import cv2

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud


logger = get_logger(__name__)


class PointCloudRenderer:
    """Renders point clouds to 2D images with BEV and realistic 3D autonomous views."""
    
    def __init__(self, image_width: int = 1280, image_height: int = 720):
        self.width = image_width
        self.height = image_height
    
    def render_bev(
        self, pc: PointCloud,
        x_range: Tuple[float, float] = (-50, 50),
        y_range: Tuple[float, float] = (-50, 50),
        color_mode: str = 'intensity',
    ) -> NDArray[np.uint8]:
        """Render Bird's Eye View (top-down)."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if len(pc.points) == 0:
            return image
        
        x = pc.points[:, 0]
        y = pc.points[:, 1]
        
        x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
        y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])
        
        px = (x_norm * (self.width - 1)).astype(int)
        py = (y_norm * (self.height - 1)).astype(int)
        
        mask = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        px = px[mask]
        py = py[mask]
        
        if color_mode == 'intensity' and pc.intensity is not None:
            intensity = pc.intensity[mask]
            colors = self._intensity_to_rgb(intensity)
        elif color_mode == 'height':
            z = pc.points[mask, 2]
            colors = self._height_to_rgb(z)
        else:
            colors = np.full((len(px), 3), [100, 100, 100], dtype=np.uint8)
        
        image[py, px] = colors
        logger.debug(f"Rendered BEV: {len(px)} points visible")
        return image

    def render_elevation_map(
        self,
        elev_map: Any,
        layer: str = 'elevation',
        frame_idx: int = 0,
    ) -> NDArray[np.uint8]:
        """Render 2.5D Elevation Map top-down with layer colormaps and HUD.
        
        Args:
            elev_map: ElevationMap25D instance.
            layer: 'elevation', 'span', 'occupancy', 'density', or 'intensity'.
            frame_idx: Frame number for telemetry.
        """
        active = elev_map.active_mask
        rows, cols = elev_map.rows, elev_map.cols
        
        # Dark charcoal canvas
        rgb = np.full((rows, cols, 3), (15, 18, 22), dtype=np.uint8)
        
        if np.any(active):
            if layer == 'elevation':
                val = elev_map.elevation_max[active]
                # Normalize height between -2.0m and 4.0m
                norm = np.clip((val - (-2.0)) / 6.0, 0.0, 1.0)
                # Turbo-like colormap
                c_r = (np.cos(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                c_g = (np.sin(norm * np.pi) * 200 + 55).astype(np.uint8)
                c_b = (np.sin(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                rgb[active] = np.column_stack([c_b, c_g, c_r])
                
            elif layer == 'span':
                span = elev_map.elevation_span[active]
                norm = np.clip(span / 2.5, 0.0, 1.0)
                c_r = (norm * 255).astype(np.uint8)
                c_g = (np.sin(norm * np.pi) * 220).astype(np.uint8)
                c_b = ((1.0 - norm) * 220).astype(np.uint8)
                rgb[active] = np.column_stack([c_b, c_g, c_r])
                
            elif layer == 'occupancy':
                occ = elev_map.occupancy[active]
                c_val = (occ * 255).astype(np.uint8)
                rgb[active] = np.column_stack([c_val, c_val, c_val])
                
            elif layer == 'density':
                cnt = elev_map.point_count[active].astype(np.float32)
                norm = np.clip(cnt / 15.0, 0.0, 1.0)
                c_r = (norm * 255).astype(np.uint8)
                c_g = (norm * 200).astype(np.uint8)
                c_b = np.full(len(norm), 30, dtype=np.uint8)
                rgb[active] = np.column_stack([c_b, c_g, c_r])
                
            elif layer == 'intensity':
                i_mean = elev_map.intensity_mean[active]
                norm = np.clip(i_mean, 0.0, 1.0)
                c_r = (norm * 240).astype(np.uint8)
                c_g = (norm * 250).astype(np.uint8)
                c_b = (norm * 255).astype(np.uint8)
                rgb[active] = np.column_stack([c_b, c_g, c_r])
        
        # Invert rows so forward (+X or +Y) points upward
        rgb = np.flipud(rgb)
        
        # Resize to target display dimensions
        resized = cv2.resize(rgb, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # Draw vehicle position marker in center
        cx = self.width // 2
        cy = self.height // 2
        cv2.circle(resized, (cx, cy), 6, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(resized, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        
        # HUD overlay
        n_active = int(np.sum(active))
        total_cells = rows * cols
        ratio = (n_active / max(1, total_cells)) * 100.0
        
        cv2.putText(resized, f"FOVEALIDAR // MODE A: UNIFORM 5CM 2.5D MAP", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, f"LAYER: {layer.upper()} | FRAME: {frame_idx:04d}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1, cv2.LINE_AA)
        
        cv2.putText(resized, f"ACTIVE CELLS: {n_active:,} / {total_cells:,} ({ratio:.2f}%)", (self.width - 380, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (50, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(resized, f"MAPPING LATENCY: {elev_map.last_update_time_ms:.1f} ms", (self.width - 380, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (200, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(resized, f"GRID: {rows}x{cols} cells @ 0.05m", (self.width - 380, 76),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)
        
        return resized

    def render_adaptive_map(
        self,
        adapt_map: Any,
        layer: str = 'elevation',
        frame_idx: int = 0,
    ) -> NDArray[np.uint8]:
        """Render Phase 5 Mode B Distance-Adaptive Map top-down.
        
        Seamlessly renders fine 5 cm cells (near-field <= 10m) and coarse 50 cm
        cells (far-field > 10m) on the strictly aligned integer lattice.
        
        Args:
            adapt_map: AdaptiveMap25D instance.
            layer: 'elevation', 'resolution', 'span', 'occupancy', or 'density'.
            frame_idx: Frame number for telemetry.
        """
        f_rows, f_cols = adapt_map.fine_rows, adapt_map.fine_cols
        c_rows, c_cols = adapt_map.coarse_rows, adapt_map.coarse_cols
        
        # 1. Prepare 140x140 Coarse Canvas
        coarse_rgb = np.full((c_rows, c_cols, 3), (15, 18, 22), dtype=np.uint8)
        coarse_active = (adapt_map.coarse_point_count > 0) & (~adapt_map.refined_mask)
        
        if np.any(coarse_active):
            if layer == 'resolution':
                # Far-field coarse cells: Purple/Indigo (#a855f7)
                coarse_rgb[coarse_active] = np.array([247, 85, 168], dtype=np.uint8)
            elif layer == 'elevation':
                val = adapt_map.coarse_elevation_max[coarse_active]
                norm = np.clip((val - (-2.0)) / 6.0, 0.0, 1.0)
                c_r = (np.cos(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                c_g = (np.sin(norm * np.pi) * 200 + 55).astype(np.uint8)
                c_b = (np.sin(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                coarse_rgb[coarse_active] = np.column_stack([c_b, c_g, c_r])
            elif layer == 'span':
                span = adapt_map.coarse_elevation_max[coarse_active] - adapt_map.coarse_elevation_min[coarse_active]
                norm = np.clip(span / 2.5, 0.0, 1.0)
                coarse_rgb[coarse_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                             (np.sin(norm * np.pi) * 220).astype(np.uint8),
                                                             ((1.0 - norm) * 220).astype(np.uint8)])
            elif layer == 'occupancy':
                occ = adapt_map.coarse_occupancy[coarse_active]
                c_val = (occ * 255).astype(np.uint8)
                coarse_rgb[coarse_active] = np.column_stack([c_val, c_val, c_val])
            elif layer == 'density':
                cnt = adapt_map.coarse_point_count[coarse_active].astype(np.float32)
                norm = np.clip(cnt / 50.0, 0.0, 1.0)
                coarse_rgb[coarse_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                             (norm * 200).astype(np.uint8),
                                                             np.full(len(norm), 40, dtype=np.uint8)])
        
        # Expand 140x140 coarse canvas to 1400x1400 (each cell is exactly 10x10 fine pixels)
        full_rgb = cv2.resize(coarse_rgb, (f_cols, f_rows), interpolation=cv2.INTER_NEAREST)
        
        # 2. Overlay Near-Field Fine Cells (5 cm)
        fine_active = adapt_map.fine_point_count > 0
        if np.any(fine_active):
            if layer == 'resolution':
                # Near-field fine cells: Bright Cyan/Emerald (#06b6d4)
                full_rgb[fine_active] = np.array([212, 182, 6], dtype=np.uint8)
            elif layer == 'elevation':
                val = adapt_map.fine_elevation_max[fine_active]
                norm = np.clip((val - (-2.0)) / 6.0, 0.0, 1.0)
                c_r = (np.cos(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                c_g = (np.sin(norm * np.pi) * 200 + 55).astype(np.uint8)
                c_b = (np.sin(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                full_rgb[fine_active] = np.column_stack([c_b, c_g, c_r])
            elif layer == 'span':
                span = adapt_map.fine_elevation_max[fine_active] - adapt_map.fine_elevation_min[fine_active]
                norm = np.clip(span / 2.5, 0.0, 1.0)
                full_rgb[fine_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                        (np.sin(norm * np.pi) * 220).astype(np.uint8),
                                                        ((1.0 - norm) * 220).astype(np.uint8)])
            elif layer == 'occupancy':
                occ = adapt_map.fine_occupancy[fine_active]
                c_val = (occ * 255).astype(np.uint8)
                full_rgb[fine_active] = np.column_stack([c_val, c_val, c_val])
            elif layer == 'density':
                cnt = adapt_map.fine_point_count[fine_active].astype(np.float32)
                norm = np.clip(cnt / 15.0, 0.0, 1.0)
                full_rgb[fine_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                        (norm * 200).astype(np.uint8),
                                                        np.full(len(norm), 30, dtype=np.uint8)])
        
        # Invert rows so forward points upward
        full_rgb = np.flipud(full_rgb)
        
        # Resize to display dimensions
        resized = cv2.resize(full_rgb, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 3. Draw 10m Transition Boundary Ring (Radius: 10m / 70m * width)
        cx = self.width // 2
        cy = self.height // 2
        ring_radius_px = int((adapt_map.near_field_radius_m / (adapt_map.bounds[1] - adapt_map.bounds[0])) * self.width)
        cv2.circle(resized, (cx, cy), ring_radius_px, (0, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(resized, "10m ADAPTIVE BOUNDARY", (cx - 70, cy - ring_radius_px - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 230, 120), 1, cv2.LINE_AA)
        
        # Draw vehicle position marker in center
        cv2.circle(resized, (cx, cy), 6, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(resized, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 4. HUD overlay
        n_fine = int(np.sum(fine_active))
        n_coarse = int(np.sum(coarse_active))
        n_total = n_fine + n_coarse
        
        cv2.putText(resized, "FOVEALIDAR // MODE B: DISTANCE-ADAPTIVE 2.5D MAP", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, f"LAYER: {layer.upper()} | FRAME: {frame_idx:04d}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1, cv2.LINE_AA)
        
        cv2.putText(resized, f"ACTIVE CELLS: {n_total:,} (Fine: {n_fine:,} | Coarse: {n_coarse:,})",
                    (self.width - 430, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(resized, f"RESOLUTION: 5cm (d<=10m) / 50cm (d>10m)",
                    (self.width - 430, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(resized, f"CELL REDUCTION: >90% vs Uniform 5cm",
                    (self.width - 430, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (140, 160, 180), 1, cv2.LINE_AA)
        
        return resized

    def render_foveal_map(
        self,
        fov_map: Any,
        layer: str = 'resolution',
        frame_idx: int = 0,
        tracked_objects: Optional[List[Any]] = None,
    ) -> NDArray[np.uint8]:
        """Render Phase 6 Mode C Dynamic Foveal Map top-down.
        
        Highlights:
        - Near-field 5 cm base core (cyan)
        - Dynamic 5 cm selective refinement regions around moving vehicles/obstacles (amber/gold)
        - Far-field 50 cm coarse cells (purple/indigo)
        - Continuous spatial importance heatmap (when layer=='foveal_importance')
        - Tracked object bounding boxes (if provided)
        """
        f_rows, f_cols = fov_map.fine_rows, fov_map.fine_cols
        c_rows, c_cols = fov_map.coarse_rows, fov_map.coarse_cols
        
        # 1. Prepare Coarse Canvas
        coarse_rgb = np.full((c_rows, c_cols, 3), (15, 18, 22), dtype=np.uint8)
        
        if layer == 'foveal_importance' and fov_map.last_importance_map is not None:
            # Full spatial importance heatmap
            imp = fov_map.last_importance_map
            # Turbo / Inferno heatmap
            c_r = (np.clip(imp * 2.2, 0.0, 1.0) * 255).astype(np.uint8)
            c_g = (np.clip((imp - 0.25) * 2.0, 0.0, 1.0) * 230).astype(np.uint8)
            c_b = (np.clip((1.0 - imp * 1.5), 0.0, 1.0) * 180).astype(np.uint8)
            coarse_rgb = np.dstack([c_b, c_g, c_r])
        elif layer == 'semantic' and hasattr(fov_map, 'controller') and fov_map.controller.last_semantic_labels is not None:
            # Semantic labels: 1=GROUND(gray), 2=VEG(green), 3=OBSTACLE(red)
            labels = fov_map.controller.last_semantic_labels
            c_r = np.zeros_like(labels, dtype=np.uint8)
            c_g = np.zeros_like(labels, dtype=np.uint8)
            c_b = np.zeros_like(labels, dtype=np.uint8)
            c_r[labels == 1] = 80; c_g[labels == 1] = 80; c_b[labels == 1] = 80
            c_r[labels == 2] = 30; c_g[labels == 2] = 200; c_b[labels == 2] = 30
            c_r[labels == 3] = 255; c_g[labels == 3] = 50; c_b[labels == 3] = 50
            coarse_rgb = np.dstack([c_b, c_g, c_r])
        else:
            coarse_active = (fov_map.coarse_point_count > 0) & (~fov_map.active_refined_mask)
            if np.any(coarse_active):
                if layer == 'resolution':
                    # Unrefined Coarse Cells: Purple (#a855f7)
                    coarse_rgb[coarse_active] = np.array([247, 85, 168], dtype=np.uint8)
                elif layer == 'elevation':
                    val = fov_map.coarse_elevation_max[coarse_active]
                    norm = np.clip((val - (-2.0)) / 6.0, 0.0, 1.0)
                    c_r = (np.cos(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                    c_g = (np.sin(norm * np.pi) * 200 + 55).astype(np.uint8)
                    c_b = (np.sin(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                    coarse_rgb[coarse_active] = np.column_stack([c_b, c_g, c_r])
                elif layer == 'span':
                    span = fov_map.coarse_elevation_max[coarse_active] - fov_map.coarse_elevation_min[coarse_active]
                    norm = np.clip(span / 2.5, 0.0, 1.0)
                    coarse_rgb[coarse_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                                 (np.sin(norm * np.pi) * 220).astype(np.uint8),
                                                                 ((1.0 - norm) * 220).astype(np.uint8)])
                elif layer == 'occupancy':
                    occ = fov_map.coarse_occupancy[coarse_active]
                    c_val = (occ * 255).astype(np.uint8)
                    coarse_rgb[coarse_active] = np.column_stack([c_val, c_val, c_val])
                elif layer == 'density':
                    cnt = fov_map.coarse_point_count[coarse_active].astype(np.float32)
                    norm = np.clip(cnt / 40.0, 0.0, 1.0)
                    coarse_rgb[coarse_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                                 (norm * 200).astype(np.uint8),
                                                                 np.full(len(norm), 30, dtype=np.uint8)])
        
        full_rgb = cv2.resize(coarse_rgb, (f_cols, f_rows), interpolation=cv2.INTER_NEAREST)
        
        # 2. Overlay Fine Cells (5 cm)
        fine_active = fov_map.fine_point_count > 0
        if np.any(fine_active) and layer != 'foveal_importance':
            # Identify which fine cells belong to dynamic far-field refinement vs base near-field
            active_r5, active_c5 = np.where(fine_active)
            parent_r50 = active_r5 // fov_map.ratio
            parent_c50 = active_c5 // fov_map.ratio
            is_dynamic_fine = fov_map.dynamic_refined_mask[parent_r50, parent_c50]
            
            if layer == 'resolution':
                # Base near-field: Cyan/Teal (#06b6d4)
                full_rgb[active_r5[~is_dynamic_fine], active_c5[~is_dynamic_fine]] = np.array([212, 182, 6], dtype=np.uint8)
                # Dynamic far-field refinement: Brilliant Gold/Amber (#f59e0b)
                if np.any(is_dynamic_fine):
                    full_rgb[active_r5[is_dynamic_fine], active_c5[is_dynamic_fine]] = np.array([11, 158, 245], dtype=np.uint8)
            elif layer == 'elevation':
                val = fov_map.fine_elevation_max[fine_active]
                norm = np.clip((val - (-2.0)) / 6.0, 0.0, 1.0)
                c_r = (np.cos(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                c_g = (np.sin(norm * np.pi) * 200 + 55).astype(np.uint8)
                c_b = (np.sin(norm * np.pi * 1.5) * 127 + 128).astype(np.uint8)
                full_rgb[fine_active] = np.column_stack([c_b, c_g, c_r])
            elif layer == 'span':
                span = fov_map.fine_elevation_max[fine_active] - fov_map.fine_elevation_min[fine_active]
                norm = np.clip(span / 2.5, 0.0, 1.0)
                full_rgb[fine_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                        (np.sin(norm * np.pi) * 220).astype(np.uint8),
                                                        ((1.0 - norm) * 220).astype(np.uint8)])
            elif layer == 'occupancy':
                occ = fov_map.fine_occupancy[fine_active]
                c_val = (occ * 255).astype(np.uint8)
                full_rgb[fine_active] = np.column_stack([c_val, c_val, c_val])
            elif layer == 'density':
                cnt = fov_map.fine_point_count[fine_active].astype(np.float32)
                norm = np.clip(cnt / 15.0, 0.0, 1.0)
                full_rgb[fine_active] = np.column_stack([(norm * 255).astype(np.uint8),
                                                        (norm * 200).astype(np.uint8),
                                                        np.full(len(norm), 30, dtype=np.uint8)])

        # Invert rows so forward points upward
        full_rgb = np.flipud(full_rgb)
        resized = cv2.resize(full_rgb, (self.width, self.height), interpolation=cv2.INTER_NEAREST)
        
        # 3. Draw Tracked Objects 2D BEV Footprints
        if tracked_objects:
            for obj in tracked_objects:
                b = fov_map.bounds
                if isinstance(obj, dict):
                    # Handled Phase 8 Mock format: bbox in coarse grid indices
                    c_rows, c_cols = fov_map.coarse_rows, fov_map.coarse_cols
                    c_w = self.width / c_cols
                    c_h = self.height / c_rows
                    
                    x1 = int(obj['min_c'] * c_w)
                    x2 = int((obj['max_c'] + 1) * c_w)
                    y1 = int((c_rows - obj['max_r'] - 1) * c_h)
                    y2 = int((c_rows - obj['min_r']) * c_h)
                    
                    cv2.rectangle(resized, (x1, y1), (x2, y2), (0, 165, 255), 2)
                    cv2.putText(resized, f"ID:{obj.get('id', '?')}", (x1, max(10, y1 - 5)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 1, cv2.LINE_AA)
                else:
                    ox, oy = obj.center[0], obj.center[1]
                    dx, dy = obj.dimensions[0], obj.dimensions[1]
                    
                    px = int(((ox - b[0]) / (b[1] - b[0])) * self.width)
                    py = int(((b[3] - oy) / (b[3] - b[2])) * self.height)
                    p_dx = int((dx / (b[1] - b[0])) * self.width)
                    p_dy = int((dy / (b[3] - b[2])) * self.height)
                    
                    if 10 <= px < self.width - 10 and 10 <= py < self.height - 10:
                        box_color = (0, 240, 255) if obj.class_name == "VEHICLE" else (50, 230, 120)
                        cv2.rectangle(resized, (px - p_dx // 2, py - p_dy // 2), (px + p_dx // 2, py + p_dy // 2), box_color, 1)
                        badge = f"ID:{obj.track_id} {obj.class_name[:3]} [{obj.speed:.1f}m/s]"
                        cv2.putText(resized, badge, (px - p_dx // 2, py - p_dy // 2 - 4),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, box_color, 1, cv2.LINE_AA)
        
        # 4. Draw 10m Ring & Center Marker
        cx = self.width // 2
        cy = self.height // 2
        ring_r = int((fov_map.near_field_radius_m / (fov_map.bounds[1] - fov_map.bounds[0])) * self.width)
        cv2.circle(resized, (cx, cy), ring_r, (0, 230, 120), 1, cv2.LINE_AA)
        cv2.circle(resized, (cx, cy), 6, (0, 220, 255), -1, cv2.LINE_AA)
        cv2.circle(resized, (cx, cy), 9, (255, 255, 255), 1, cv2.LINE_AA)
        
        # 5. HUD Telemetry
        n_dyn = int(np.sum(fov_map.dynamic_refined_mask))
        n_fine = int(np.sum(fine_active))
        n_coarse = int(np.sum((fov_map.coarse_point_count > 0) & (~fov_map.active_refined_mask)))
        
        cv2.putText(resized, "FOVEALIDAR // MODE C: DYNAMIC FOVEAL MAP", (20, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.56, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(resized, f"LAYER: {layer.upper()} | FRAME: {frame_idx:04d}", (20, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, (160, 220, 255), 1, cv2.LINE_AA)
        
        cv2.putText(resized, f"ACTIVE CELLS: {n_fine + n_coarse:,} (Fine: {n_fine:,} | Coarse: {n_coarse:,})",
                    (self.width - 450, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (50, 230, 120), 1, cv2.LINE_AA)
        cv2.putText(resized, f"DYNAMIC REFINED FAR TILES: {n_dyn} tiles @ 5cm (Gold)",
                    (self.width - 450, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (245, 158, 11), 1, cv2.LINE_AA)
        cv2.putText(resized, "RESOLUTION: 5cm Near + 5cm Dynamic Far / 50cm Background",
                    (self.width - 450, 76), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (140, 160, 180), 1, cv2.LINE_AA)
        
        return resized
    

    def _intensity_to_rgb(self, intensity: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert intensity to RGB."""
        if intensity.max() > intensity.min():
            normalized = (intensity - intensity.min()) / (intensity.max() - intensity.min())
        else:
            normalized = np.zeros_like(intensity)
        
        rgb = np.zeros((len(intensity), 3), dtype=np.uint8)
        rgb[:, 0] = (normalized * 255).astype(np.uint8)
        rgb[:, 1] = (np.sin(normalized * np.pi) * 255).astype(np.uint8)
        rgb[:, 2] = ((1 - normalized) * 255).astype(np.uint8)
        
        return rgb
    
    def _height_to_rgb(self, z: NDArray[np.float32]) -> NDArray[np.uint8]:
        """Convert height to RGB."""
        z_min, z_max = z.min(), z.max()
        if z_max > z_min:
            normalized = (z - z_min) / (z_max - z_min)
        else:
            normalized = np.zeros_like(z)
        
        rgb = np.zeros((len(z), 3), dtype=np.uint8)
        for i, val in enumerate(normalized):
            if val < 0.5:
                rgb[i] = [0, int(val * 2 * 255), 255]
            else:
                rgb[i] = [int((val - 0.5) * 2 * 255), 255, 0]
        
        return rgb
    
    def render_with_roi_overlay(
        self, pc: PointCloud,
        region_ids: NDArray[np.int32],
        x_range: Tuple[float, float] = (-50, 50),
        y_range: Tuple[float, float] = (-50, 50),
    ) -> NDArray[np.uint8]:
        """Render with ROI regions color-coded."""
        image = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        
        if len(pc.points) == 0:
            return image
        
        x = pc.points[:, 0]
        y = pc.points[:, 1]
        
        x_norm = (x - x_range[0]) / (x_range[1] - x_range[0])
        y_norm = (y - y_range[0]) / (y_range[1] - y_range[0])
        
        px = (x_norm * (self.width - 1)).astype(int)
        py = (y_norm * (self.height - 1)).astype(int)
        
        mask = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        px = px[mask]
        py = py[mask]
        regions = region_ids[mask]
        
        colors = np.zeros((len(px), 3), dtype=np.uint8)
        colors[regions == 0] = [0, 255, 0]      # Fovea: Green
        colors[regions == 1] = [255, 165, 0]    # Periphery: Orange
        colors[regions == 2] = [128, 128, 128]  # Outside: Gray
        
        image[py, px] = colors
        logger.debug("Rendered ROI overlay")
        return image

    def render_scifi_3d(
        self, pc: PointCloud,
        region_ids: NDArray[np.int32],
        gaze_pos: NDArray[np.float32],
        frame_idx: int,
    ) -> NDArray[np.uint8]:
        """Sci-Fi 3D view (delegates to enhanced autonomous 3D view)."""
        return self.render_autonomous_3d(
            pc=pc,
            region_ids=region_ids,
            gaze_pos=gaze_pos,
            frame_idx=frame_idx,
        )

    def detect_obstacles(
        self,
        points: NDArray[np.float32],
        intensity: Optional[NDArray[np.float32]] = None,
        ground_z_threshold: float = -1.25,
    ) -> List[Dict[str, Any]]:
        """Fast vectorized cluster detection for obstacle 3D bounding boxes."""
        if len(points) == 0:
            return []
        
        x = points[:, 0]
        y = points[:, 1]
        z = points[:, 2]
        
        # Only above-ground points in forward roadway area
        obs_mask = (z > ground_z_threshold) & (z < 2.2) & (x > 3.0) & (x < 65.0) & (np.abs(y) < 18.0)
        obs_pts = points[obs_mask]
        if len(obs_pts) < 12:
            return []
        
        cell_size = 1.4
        grid_xy = np.floor(obs_pts[:, :2] / cell_size).astype(np.int32)
        unique_cells, inverse, counts = np.unique(grid_xy, axis=0, return_inverse=True, return_counts=True)
        
        valid_cells = np.where((counts >= 10) & (counts <= 1200))[0]
        if len(valid_cells) == 0:
            return []
        
        boxes = []
        for cell_idx in valid_cells[:12]:
            c_pts = obs_pts[inverse == cell_idx]
            min_c = c_pts.min(axis=0)
            max_c = c_pts.max(axis=0)
            extent = max_c - min_c
            
            l, w, h = extent[0], extent[1], extent[2]
            if 0.5 <= l <= 7.5 and 0.5 <= w <= 4.5 and 0.5 <= h <= 3.5:
                center = (min_c + max_c) / 2.0
                dist = float(np.linalg.norm(center[:2]))
                label = "CAR" if (l > 1.8 or w > 1.2) else "OBSTACLE"
                boxes.append({
                    'center': center,
                    'extent': np.maximum(extent, np.array([1.4, 1.2, 1.3], dtype=np.float32)),
                    'distance': dist,
                    'label': f"{label} [{dist:.1f}m]",
                    'type': label,
                })
        
        boxes.sort(key=lambda b: b['distance'])
        return boxes[:6]

    def render_autonomous_3d(
        self,
        pc: PointCloud,
        region_ids: Optional[NDArray[np.int32]] = None,
        gaze_pos: Optional[NDArray[np.float32]] = None,
        frame_idx: int = 0,
        bounding_boxes: Optional[List[Dict[str, Any]]] = None,
        color_mode: str = 'autonomous',
        raw_point_count: Optional[int] = None,
    ) -> NDArray[np.uint8]:
        """Render a 3D autonomous driving chase camera view matching img2."""
        image = np.full((self.height, self.width, 3), (14, 16, 20), dtype=np.uint8)
        
        if len(pc.points) == 0:
            return image
        
        pts = pc.points.copy()
        intens = pc.intensity if pc.intensity is not None else np.ones(len(pts), dtype=np.float32)
        n_pts = len(pts)
        
        # --- Camera Transform (Chase Camera matching img2) ---
        cam_x = -11.5
        cam_y = 0.0
        cam_z = 3.6
        pitch = np.radians(13.0)   # 13 deg downward pitch
        focal_length = 750.0
        
        dx = pts[:, 0] - cam_x
        dy = pts[:, 1] - cam_y
        dz = pts[:, 2] - cam_z
        
        x_cam = -dy
        y_cam = -dz * np.cos(pitch) - dx * np.sin(pitch)
        z_cam = -dz * np.sin(pitch) + dx * np.cos(pitch)
        
        valid = z_cam > 1.5
        if not np.any(valid):
            return image
        
        px = ((x_cam[valid] * focal_length / z_cam[valid]) + self.width / 2).astype(int)
        py = ((y_cam[valid] * focal_length / z_cam[valid]) + self.height / 2).astype(int)
        
        in_frame = (px >= 0) & (px < self.width) & (py >= 0) & (py < self.height)
        px = px[in_frame]
        py = py[in_frame]
        
        sub_pts = pts[valid][in_frame]
        sub_int = intens[valid][in_frame]
        sub_z_cam = z_cam[valid][in_frame]
        
        # Sort back-to-front
        sort_idx = np.argsort(sub_z_cam)[::-1]
        px = px[sort_idx]
        py = py[sort_idx]
        sub_pts = sub_pts[sort_idx]
        sub_int = sub_int[sort_idx]
        sub_z_cam = sub_z_cam[sort_idx]
        
        z = sub_pts[:, 2]
        x = sub_pts[:, 0]
        y = sub_pts[:, 1]
        r_ground = np.sqrt(x**2 + y**2)
        
        # --- Feature Classification matching img2 ---
        is_ground = z <= -1.25
        # Reflective road markings
        is_lane = is_ground & (sub_int > 0.45) & (np.abs(y) < 7.0)
        # Trees and foliage (golden amber in img2)
        is_vegetation = (z > -0.6) & (z < 7.5) & ((np.abs(y) > 2.8) | (z > 2.0))
        # Vehicles
        is_vehicle = (z > -1.25) & (z < 1.4) & (np.abs(y) <= 3.2) & (x > 3.0)
        # Buildings
        is_building = (z > 1.5) & (np.abs(y) > 6.5)
        
        colors = np.zeros((len(px), 3), dtype=np.uint8)
        depth_factor = np.clip(1.0 - (sub_z_cam - 5.0) / 80.0, 0.3, 1.0)
        
        if color_mode == 'autonomous':
            # 1. Road Surface (Radial blue gradient near car -> dark slate)
            # In img2, near ground is blue (BGR: [150, 85, 25]), fading to dark slate
            near_ground_mask = is_ground & (r_ground <= 22.0)
            far_ground_mask = is_ground & (r_ground > 22.0)
            
            blue_fade = np.clip((22.0 - r_ground[near_ground_mask]) / 22.0, 0.0, 1.0)
            colors[near_ground_mask, 0] = (150 * blue_fade + 50 * (1 - blue_fade)).astype(np.uint8)  # B
            colors[near_ground_mask, 1] = (85 * blue_fade + 38 * (1 - blue_fade)).astype(np.uint8)   # G
            colors[near_ground_mask, 2] = (25 * blue_fade + 28 * (1 - blue_fade)).astype(np.uint8)   # R
            
            colors[far_ground_mask, 0] = (45 * depth_factor[far_ground_mask]).astype(np.uint8)
            colors[far_ground_mask, 1] = (38 * depth_factor[far_ground_mask]).astype(np.uint8)
            colors[far_ground_mask, 2] = (32 * depth_factor[far_ground_mask]).astype(np.uint8)
            
            # 2. Road Lane Markings / Reflective lines (Glowing White/Cyan)
            colors[is_lane, 0] = 250
            colors[is_lane, 1] = 250
            colors[is_lane, 2] = 255
            
            # 3. Trees & Foliage (Golden-Amber glow matching img2)
            colors[is_vegetation, 0] = (30 * depth_factor[is_vegetation]).astype(np.uint8)   # B
            colors[is_vegetation, 1] = (195 * depth_factor[is_vegetation]).astype(np.uint8)  # G
            colors[is_vegetation, 2] = (248 * depth_factor[is_vegetation]).astype(np.uint8)  # R
            
            # 4. Vehicles & Dynamic Obstacles (Bright Cyan / Turquoise)
            colors[is_vehicle, 0] = (240 * depth_factor[is_vehicle]).astype(np.uint8)
            colors[is_vehicle, 1] = (215 * depth_factor[is_vehicle]).astype(np.uint8)
            colors[is_vehicle, 2] = (30 * depth_factor[is_vehicle]).astype(np.uint8)
            
            # 5. Buildings / Facades (Ivory / Cool Gray)
            colors[is_building, 0] = (195 * depth_factor[is_building]).astype(np.uint8)
            colors[is_building, 1] = (200 * depth_factor[is_building]).astype(np.uint8)
            colors[is_building, 2] = (205 * depth_factor[is_building]).astype(np.uint8)
            
            # Unassigned
            unassigned = ~(is_ground | is_vegetation | is_vehicle | is_building)
            colors[unassigned, 0] = (110 * depth_factor[unassigned]).astype(np.uint8)
            colors[unassigned, 1] = (110 * depth_factor[unassigned]).astype(np.uint8)
            colors[unassigned, 2] = (120 * depth_factor[unassigned]).astype(np.uint8)
            
        elif color_mode == 'height':
            norm_z = np.clip((z - (-2.0)) / 5.5, 0.0, 1.0)
            colors[:, 0] = (np.sin(norm_z * np.pi * 1.5) * 127 + 128).astype(np.uint8)
            colors[:, 1] = (np.sin(norm_z * np.pi) * 200 + 55).astype(np.uint8)
            colors[:, 2] = (np.cos(norm_z * np.pi * 1.5) * 127 + 128).astype(np.uint8)
            
        elif color_mode == 'intensity':
            norm_i = np.clip(sub_int, 0.0, 1.0)
            colors[:, 0] = (norm_i * 255).astype(np.uint8)
            colors[:, 1] = (norm_i * 230).astype(np.uint8)
            colors[:, 2] = (norm_i * 180).astype(np.uint8)
            
        elif color_mode == 'foveal' and region_ids is not None:
            sub_regs = region_ids[valid][in_frame][sort_idx]
            colors[sub_regs == 0] = [50, 255, 50]    # Fovea: Neon green
            colors[sub_regs == 1] = [30, 160, 255]   # Periphery: Orange
            colors[sub_regs == 2] = [120, 120, 120]  # Far-field: Gray
        
        # Plot points to image
        image[py, px] = colors
        
        # Draw Range Rings
        self._draw_range_rings(image, cam_x, cam_y, cam_z, pitch, focal_length)
        
        # Draw 3D Ego Vehicle (White car at origin)
        self._draw_ego_vehicle(image, cam_x, cam_y, cam_z, pitch, focal_length)
        
        # Draw 3D Bounding Boxes
        if bounding_boxes is None:
            bounding_boxes = self.detect_obstacles(pts, intens)
        self._draw_3d_bounding_boxes(image, bounding_boxes, cam_x, cam_y, cam_z, pitch, focal_length)
        
        # Draw Attention Laser Ray
        if gaze_pos is not None:
            self._draw_attention_ray(image, gaze_pos, cam_x, cam_y, cam_z, pitch, focal_length)
        
        # Draw HUD Telemetry
        orig_count = raw_point_count if raw_point_count else n_pts
        self._draw_hud(image, frame_idx, orig_count, n_pts, gaze_pos)
        
        return image

    def _project_point(
        self,
        pt: NDArray[np.float32],
        cam_x: float, cam_y: float, cam_z: float,
        pitch: float, focal: float,
    ) -> Optional[Tuple[int, int]]:
        """Project a single 3D point in KITTI coordinates to 2D image coordinates."""
        dx = pt[0] - cam_x
        dy = pt[1] - cam_y
        dz = pt[2] - cam_z
        
        x_cam = -dy
        y_cam = -dz * np.cos(pitch) - dx * np.sin(pitch)
        z_cam = -dz * np.sin(pitch) + dx * np.cos(pitch)
        
        if z_cam <= 1.0:
            return None
        
        px = int((x_cam * focal / z_cam) + self.width / 2)
        py = int((y_cam * focal / z_cam) + self.height / 2)
        return px, py

    def _draw_range_rings(
        self, image: NDArray[np.uint8],
        cam_x: float, cam_y: float, cam_z: float,
        pitch: float, focal: float,
    ):
        """Draw ground distance rings (10m fovea, 25m, 50m)."""
        ground_z = -1.65
        for r, color, label in [
            (10.0, (80, 200, 80), "FOVEA: 10m [5cm Grid]"),
            (25.0, (70, 120, 170), "25m"),
            (50.0, (50, 80, 110), "PERIPHERY: 50m [50cm Grid]"),
        ]:
            angles = np.linspace(-np.pi * 0.42, np.pi * 0.42, 60)
            ring_pts = []
            for a in angles:
                x = r * np.cos(a)
                y = r * np.sin(a)
                proj = self._project_point(np.array([x, y, ground_z], dtype=np.float32),
                                          cam_x, cam_y, cam_z, pitch, focal)
                if proj and 0 <= proj[0] < self.width and 0 <= proj[1] < self.height:
                    ring_pts.append(proj)
            
            if len(ring_pts) > 2:
                pts_arr = np.array(ring_pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(image, [pts_arr], isClosed=False, color=color, thickness=1, lineType=cv2.LINE_AA)
                mid_pt = ring_pts[len(ring_pts) // 2]
                cv2.putText(image, label, (mid_pt[0] - 65, mid_pt[1] + 16),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.36, color, 1, cv2.LINE_AA)

    def _draw_ego_vehicle(
        self, image: NDArray[np.uint8],
        cam_x: float, cam_y: float, cam_z: float,
        pitch: float, focal: float,
    ):
        """Draw 3D ego passenger car model at the sensor origin."""
        ground_z = -1.65
        car_w, car_h = 1.82, 1.42
        x_min, x_max = -2.1, 2.3
        y_min, y_max = -car_w / 2, car_w / 2
        z_min, z_max = ground_z, ground_z + car_h
        
        corners = np.array([
            [x_min, y_min, z_min], [x_max, y_min, z_min],
            [x_max, y_max, z_min], [x_min, y_max, z_min],
            [x_min, y_min, z_max], [x_max, y_min, z_max],
            [x_max, y_max, z_max], [x_min, y_max, z_max],
        ], dtype=np.float32)
        
        proj_corners = [self._project_point(c, cam_x, cam_y, cam_z, pitch, focal) for c in corners]
        if any(p is None for p in proj_corners):
            return
        
        # Rear bumper face (corners 0, 3, 7, 4)
        rear_poly = np.array([proj_corners[0], proj_corners[3], proj_corners[7], proj_corners[4]], dtype=np.int32)
        cv2.fillPoly(image, [rear_poly], (240, 240, 245), lineType=cv2.LINE_AA)
        cv2.polylines(image, [rear_poly], True, (160, 160, 170), 2, cv2.LINE_AA)
        
        # Rear windshield
        rw_bottom_y = (proj_corners[7][1] + proj_corners[3][1]) // 2
        rw_poly = np.array([
            (proj_corners[4][0] + 10, rw_bottom_y),
            (proj_corners[7][0] - 10, rw_bottom_y),
            (proj_corners[7][0] - 6, proj_corners[7][1] + 5),
            (proj_corners[4][0] + 6, proj_corners[4][1] + 5),
        ], dtype=np.int32)
        cv2.fillPoly(image, [rw_poly], (30, 40, 50), lineType=cv2.LINE_AA)
        
        # Roof face (corners 4, 5, 6, 7)
        roof_poly = np.array([proj_corners[4], proj_corners[5], proj_corners[6], proj_corners[7]], dtype=np.int32)
        cv2.fillPoly(image, [roof_poly], (215, 220, 225), lineType=cv2.LINE_AA)
        cv2.polylines(image, [roof_poly], True, (150, 150, 160), 1, cv2.LINE_AA)
        
        # Roof LiDAR puck
        puck_pt = self._project_point(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                     cam_x, cam_y, cam_z, pitch, focal)
        if puck_pt:
            cv2.circle(image, puck_pt, 5, (0, 215, 255), -1, cv2.LINE_AA)
            cv2.circle(image, puck_pt, 8, (0, 160, 255), 1, cv2.LINE_AA)
        
        # Taillights
        tl_left = ((proj_corners[0][0] + proj_corners[4][0]) // 2 + 8, (proj_corners[0][1] + proj_corners[4][1]) // 2)
        tl_right = ((proj_corners[3][0] + proj_corners[7][0]) // 2 - 8, (proj_corners[3][1] + proj_corners[7][1]) // 2)
        cv2.circle(image, tl_left, 6, (20, 20, 240), -1, cv2.LINE_AA)
        cv2.circle(image, tl_right, 6, (20, 20, 240), -1, cv2.LINE_AA)

    def _draw_3d_bounding_boxes(
        self, image: NDArray[np.uint8],
        boxes: List[Dict[str, Any]],
        cam_x: float, cam_y: float, cam_z: float,
        pitch: float, focal: float,
    ):
        """Draw 3D wireframe bounding boxes and class tags around detected obstacles."""
        for box in boxes:
            c = box['center']
            e = box['extent'] / 2.0
            
            corners_3d = np.array([
                [c[0] - e[0], c[1] - e[1], c[2] - e[2]],
                [c[0] + e[0], c[1] - e[1], c[2] - e[2]],
                [c[0] + e[0], c[1] + e[1], c[2] - e[2]],
                [c[0] - e[0], c[1] + e[1], c[2] - e[2]],
                [c[0] - e[0], c[1] - e[1], c[2] + e[2]],
                [c[0] + e[0], c[1] - e[1], c[2] + e[2]],
                [c[0] + e[0], c[1] + e[1], c[2] + e[2]],
                [c[0] - e[0], c[1] + e[1], c[2] + e[2]],
            ], dtype=np.float32)
            
            proj = [self._project_point(pt, cam_x, cam_y, cam_z, pitch, focal) for pt in corners_3d]
            if any(p is None for p in proj):
                continue
            
            color = (235, 215, 30) if box['type'] == 'CAR' else (50, 220, 120)
            edges = [
                (0, 1), (1, 2), (2, 3), (3, 0),
                (4, 5), (5, 6), (6, 7), (7, 4),
                (0, 4), (1, 5), (2, 6), (3, 7),
            ]
            for i, j in edges:
                cv2.line(image, proj[i], proj[j], color, 1, cv2.LINE_AA)
            
            top_center_x = int(np.mean([proj[i][0] for i in [4, 5, 6, 7]]))
            top_center_y = int(np.min([proj[i][1] for i in [4, 5, 6, 7]]))
            
            badge_text = box['label']
            badge_size = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)[0]
            bx = top_center_x - badge_size[0] // 2
            by = top_center_y - 10
            
            if 10 <= bx < self.width - badge_size[0] - 10 and 20 <= by < self.height:
                cv2.rectangle(image, (bx - 4, by - badge_size[1] - 3), (bx + badge_size[0] + 4, by + 4),
                              (18, 22, 30), -1)
                cv2.rectangle(image, (bx - 4, by - badge_size[1] - 3), (bx + badge_size[0] + 4, by + 4),
                              color, 1)
                cv2.putText(image, badge_text, (bx, by), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

    def _draw_attention_ray(
        self, image: NDArray[np.uint8],
        gaze: NDArray[np.float32],
        cam_x: float, cam_y: float, cam_z: float,
        pitch: float, focal: float,
    ):
        """Draw forward laser ray from vehicle sensor to primary foveal gaze target (like img2)."""
        origin_proj = self._project_point(np.array([0.0, 0.0, 0.0], dtype=np.float32),
                                          cam_x, cam_y, cam_z, pitch, focal)
        target_proj = self._project_point(gaze, cam_x, cam_y, cam_z, pitch, focal)
        
        if origin_proj and target_proj:
            cv2.line(image, origin_proj, target_proj, (35, 90, 255), 2, cv2.LINE_AA)
            tx, ty = target_proj
            if 0 <= tx < self.width and 0 <= ty < self.height:
                cv2.circle(image, (tx, ty), 14, (35, 90, 255), 2, cv2.LINE_AA)
                cv2.circle(image, (tx, ty), 3, (35, 90, 255), -1, cv2.LINE_AA)
                cv2.putText(image, f"FOVEA FOCUS [{gaze[0]:.1f}m]", (tx + 16, ty - 6),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (60, 140, 255), 1, cv2.LINE_AA)

    def _draw_hud(
        self, image: NDArray[np.uint8],
        frame_idx: int,
        orig_count: int,
        active_count: int,
        gaze_pos: Optional[NDArray[np.float32]],
    ):
        """Draw engineering HUD overlay."""
        cv2.putText(image, "FOVEALIDAR // ADAPTIVE VARIABLE RESOLUTION 2.5D PERCEPTION", (24, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.putText(image, f"FRAME: {frame_idx:04d} // CHASE CAMERA VIEW", (24, 58),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 220, 255), 1, cv2.LINE_AA)
        
        comp_ratio = (1.0 - active_count / max(1, orig_count)) * 100.0
        cv2.putText(image, f"RAW POINTS: {orig_count:,}", (self.width - 340, 34),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 240), 1, cv2.LINE_AA)
        cv2.putText(image, f"COMPRESSED: {active_count:,} ({comp_ratio:.1f}% reduction)", (self.width - 340, 56),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (50, 230, 120), 1, cv2.LINE_AA)
        
        cv2.putText(image, "SIH 2026 PS 26053: Variable 2.5D Resolution (5cm Near <=10m | 50cm Far >10m)",
                    (24, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (130, 160, 180), 1, cv2.LINE_AA)
        if gaze_pos is not None:
            cv2.putText(image, f"GAZE TENSOR: [{gaze_pos[0]:.1f}, {gaze_pos[1]:.1f}, {gaze_pos[2]:.1f}]",
                        (self.width - 340, self.height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 200, 255), 1, cv2.LINE_AA)
