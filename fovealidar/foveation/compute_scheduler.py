"""Phase 9: Compute-Aware Scheduling.

Implements Master Spec §21 (Compute-Aware Foveation) + §22 (AI Inference Scheduling).

Maps the existing ComputeMode enum to concrete runtime behaviour:
- refinement budget (max_refined_regions)
- AI cadence (run semantic every N frames)
- rendering rate (target FPS cap)
- point downsample factor

Graceful degradation order (§35): rendering -> AI cadence -> downsample ->
refinement budget -> heuristic/CPU-only.  The core near-field map is never
disabled by the scheduler.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Union

from fovealidar.core.logging import get_logger
from fovealidar.core.types import ComputeMode

logger = get_logger(__name__)


@dataclass(frozen=True)
class ComputeProfile:
    """Immutable per-mode compute budget."""

    max_refined_regions: int
    ai_stride: int  # run semantic AI every N frames (0 = disabled)
    ai_enabled: bool
    target_fps: int
    downsample_factor: int


_PROFILES: Dict[ComputeMode, ComputeProfile] = {
    ComputeMode.MAX_QUALITY: ComputeProfile(
        max_refined_regions=256,
        ai_stride=1,
        ai_enabled=True,
        target_fps=30,
        downsample_factor=1,
    ),
    ComputeMode.BALANCED: ComputeProfile(
        max_refined_regions=128,
        ai_stride=2,
        ai_enabled=True,
        target_fps=20,
        downsample_factor=1,
    ),
    ComputeMode.LOW_POWER: ComputeProfile(
        max_refined_regions=64,
        ai_stride=4,
        ai_enabled=True,
        target_fps=10,
        downsample_factor=2,
    ),
    ComputeMode.CPU_ONLY: ComputeProfile(
        max_refined_regions=32,
        ai_stride=0,
        ai_enabled=False,
        target_fps=10,
        downsample_factor=4,
    ),
}


def _coerce_mode(mode: Union[ComputeMode, str]) -> ComputeMode:
    if isinstance(mode, ComputeMode):
        return mode
    if isinstance(mode, str):
        normalized = mode.strip().lower()
        for m in ComputeMode:
            if m.value == normalized or m.name.lower() == normalized:
                return m
        raise ValueError(f"Invalid compute mode: {mode!r}")
    raise TypeError(f"mode must be ComputeMode or str, got {type(mode).__name__}")


class ComputeScheduler:
    """Compute-aware scheduler wiring ComputeMode to runtime behaviour.

    Usage:
        sched = ComputeScheduler(ComputeMode.BALANCED)
        sched.apply_to_controller(controller)  # sets max_refined_regions
        if sched.should_run_ai(frame_idx):
            ... run semantic inference ...
    """

    def __init__(self, mode: Union[ComputeMode, str] = ComputeMode.BALANCED):
        self._mode: ComputeMode = _coerce_mode(mode)
        logger.info(f"ComputeScheduler initialized in {self._mode.name} mode.")

    # -- mode management -------------------------------------------------
    @property
    def mode(self) -> ComputeMode:
        return self._mode

    def set_mode(self, mode: Union[ComputeMode, str]) -> None:
        self._mode = _coerce_mode(mode)
        logger.info(f"ComputeScheduler switched to {self._mode.name} mode.")

    @property
    def profile(self) -> ComputeProfile:
        return _PROFILES[self._mode]

    # -- budget accessors (spec-required names) --------------------------
    @property
    def max_refined_regions(self) -> int:
        return self.profile.max_refined_regions

    @property
    def ai_stride(self) -> int:
        """Run semantic AI every N frames (0 means disabled)."""
        return self.profile.ai_stride

    @property
    def ai_enabled(self) -> bool:
        return self.profile.ai_enabled

    @property
    def target_fps(self) -> int:
        """Target rendering/playback FPS cap."""
        return self.profile.target_fps

    @property
    def downsample_factor(self) -> int:
        """Point-cloud stride for far-field input density reduction."""
        return self.profile.downsample_factor

    # Backwards-compatible aliases used by GUI / configs
    @property
    def target_fps_cap(self) -> int:
        return self.target_fps

    @property
    def point_downsample_factor(self) -> int:
        return self.downsample_factor

    @property
    def ai_frequency_divider(self) -> int:
        return self.ai_stride

    # -- scheduling decisions --------------------------------------------
    def should_run_ai(self, frame_idx: int) -> bool:
        """Return True if semantic AI should run on this frame.

        CPU_ONLY disables neural/heuristic AI entirely (heuristic fallback
        remains available in the mapper itself).  Other modes run every N
        frames where N = ai_stride.
        """
        if not self.profile.ai_enabled or self.profile.ai_stride <= 0:
            return False
        try:
            idx = int(frame_idx)
        except (TypeError, ValueError):
            return True
        if idx < 0:
            return True
        return (idx % self.profile.ai_stride) == 0

    def downsample_points(self, points):
        """Stride-downsample a point array (N, ...) by the mode factor."""
        factor = self.profile.downsample_factor
        if factor <= 1:
            return points
        return points[::factor]

    # -- apply helpers ----------------------------------------------------
    def apply_to_controller(self, controller: Any) -> Any:
        """Apply refinement budget to a FovealController in-place.

        Returns the controller for chaining.  Never touches near-field
        geometry — only the dynamic far-field budget.
        """
        if hasattr(controller, "max_refined_regions"):
            controller.max_refined_regions = self.profile.max_refined_regions
            logger.debug(
                f"Applied {self._mode.name} budget: "
                f"max_refined_regions={self.profile.max_refined_regions}"
            )
        return controller

    def apply_to_foveal_map(self, foveal_map: Any) -> Any:
        """Apply budget to a FovealMap25D (forwards to its controller)."""
        controller = getattr(foveal_map, "controller", None)
        if controller is not None:
            self.apply_to_controller(controller)
        return foveal_map

    def get_config_dict(self) -> Dict[str, Any]:
        p = self.profile
        return {
            "mode": self._mode.value,
            "max_refined_regions": p.max_refined_regions,
            "ai_stride": p.ai_stride,
            "ai_enabled": p.ai_enabled,
            "target_fps": p.target_fps,
            "downsample_factor": p.downsample_factor,
        }

    def describe(self) -> str:
        p = self.profile
        ai_txt = f"every {p.ai_stride} frames" if p.ai_enabled else "disabled"
        return (
            f"{self._mode.name}: budget={p.max_refined_regions} tiles | "
            f"AI {ai_txt} | {p.target_fps} FPS cap | x{p.downsample_factor} downsample"
        )
