"""Frame generation and video composition from visualization output."""

from pathlib import Path
from typing import Optional, Union
import numpy as np
from numpy.typing import NDArray
import cv2

from fovealidar.core.logging import get_logger
from fovealidar.core.types import PointCloud
from fovealidar.visualization.renderer import PointCloudRenderer


logger = get_logger(__name__)


class FrameGenerator:
    """Generates frames from point cloud sequences or pre-rendered images."""
    
    def __init__(self, output_dir: Union[str, Path], image_width: int = 1280, image_height: int = 720):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.renderer = PointCloudRenderer(image_width, image_height)
        self.frame_count = 0
    
    def generate_frame(
        self,
        pc_or_image: Union[PointCloud, NDArray[np.uint8]],
        frame_id: int,
        annotation: Optional[str] = None,
        render_mode: str = 'intensity',
        **kwargs,
    ) -> str:
        """Generate and save a single frame.
        
        Args:
            pc_or_image: Either a PointCloud object or a pre-rendered BGR image array.
            frame_id: Sequential frame index.
        """
        frame_path = self.output_dir / f"frame_{frame_id:06d}.png"
        
        if isinstance(pc_or_image, np.ndarray):
            # Pre-rendered image passed directly
            image = pc_or_image
        else:
            # PointCloud object passed
            if render_mode == 'autonomous_3d':
                image = self.renderer.render_autonomous_3d(pc_or_image, frame_idx=frame_id, **kwargs)
            else:
                image = self.renderer.render_bev(pc_or_image, color_mode=render_mode)
        
        self._save_image(image, str(frame_path))
        self.frame_count += 1
        logger.debug(f"Generated frame {frame_id}")
        return str(frame_path)
    
    def generate_frame_with_roi(
        self,
        pc: PointCloud,
        region_ids: NDArray[np.int32],
        frame_id: int,
        stats: Optional[dict] = None,
    ) -> str:
        """Generate frame with ROI overlay."""
        image = self.renderer.render_with_roi_overlay(pc, region_ids)
        frame_path = self.output_dir / f"frame_roi_{frame_id:06d}.png"
        self._save_image(image, str(frame_path))
        logger.debug(f"Generated ROI frame {frame_id}")
        return str(frame_path)
    
    def _save_image(self, image: NDArray[np.uint8], path: str):
        """Save image via OpenCV or PIL."""
        try:
            cv2.imwrite(path, image)
        except Exception:
            try:
                from PIL import Image
                # Convert BGR to RGB if needed
                if len(image.shape) == 3 and image.shape[2] == 3:
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                else:
                    rgb = image
                Image.fromarray(rgb).save(path)
            except Exception as e:
                logger.error(f"Failed to save image {path}: {e}")
    
    def get_frame_count(self) -> int:
        return self.frame_count
    
    def clear_frames(self):
        for f in self.output_dir.glob("frame_*.png"):
            f.unlink()
        self.frame_count = 0


class VideoComposer:
    """Composes video from frame sequences."""
    
    def __init__(self, fps: int = 30):
        self.fps = fps
    
    def compose_video(
        self,
        frame_dir: Union[str, Path],
        output_path: Union[str, Path],
        fps: Optional[int] = None,
        frame_pattern: str = "frame_*.png",
    ) -> bool:
        """Compose video (alias for compose_from_frames)."""
        return self.compose_from_frames(
            frame_dir=frame_dir,
            output_path=output_path,
            fps=fps,
            frame_pattern=frame_pattern,
        )
    
    def compose_from_frames(
        self,
        frame_dir: Union[str, Path],
        output_path: Union[str, Path],
        fps: Optional[int] = None,
        frame_pattern: str = "frame_*.png",
    ) -> bool:
        """Compose video from frame sequence."""
        frame_dir = Path(frame_dir)
        output_path = Path(output_path)
        target_fps = fps if fps is not None else self.fps
        
        frames = sorted(frame_dir.glob(frame_pattern))
        if not frames:
            logger.error(f"No frames found in {frame_dir}")
            return False
        
        logger.info(f"Found {len(frames)} frames, composing video...")
        
        first_frame = cv2.imread(str(frames[0]))
        if first_frame is None:
            logger.error("Failed to read first frame")
            return False
        
        height, width = first_frame.shape[:2]
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Try MP4V codec first, then XVID
        for codec in ['mp4v', 'avc1', 'XVID']:
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(output_path), fourcc, target_fps, (width, height))
                if writer.isOpened():
                    for f_path in frames:
                        frame = cv2.imread(str(f_path))
                        if frame is not None:
                            writer.write(frame)
                    writer.release()
                    if output_path.exists() and output_path.stat().st_size > 1000:
                        logger.info(f"Video successfully written with {codec}: {output_path}")
                        return True
            except Exception as e:
                logger.warning(f"Codec {codec} failed: {e}")
        
        return output_path.exists() and output_path.stat().st_size > 0
