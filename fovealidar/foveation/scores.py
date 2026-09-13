"""Vectorized Importance Scoring Functions for Foveal Controller (Phase 6).

Implements SIH 2026 Master Spec §16–§20:
- Distance Score: Linear decay from sensor origin.
- Geometry Score: Elevation span + point density heuristic.
- Uncertainty Score: Sparsity / low-count indicator.
- Motion Score: Stub (returns 0.0, filled by Phase 8 tracker).
- Semantic Score: Stub (returns 0.0, filled by Phase 7 AI backend).

All functions operate on the 50 cm coarse grid (vectorized over all cells).
"""

import numpy as np
from numpy.typing import NDArray
from typing import Optional

from fovealidar.core.logging import get_logger

logger = get_logger(__name__)


def distance_score(
    cell_distances: NDArray[np.float32],
    near_range_m: float = 10.0,
    far_range_m: float = 70.0,
) -> NDArray[np.float32]:
    """Compute per-cell distance importance score ∈ [0, 1].

    Cells at the sensor origin score 1.0; cells at `far_range_m` score 0.0.
    Linear decay between 0 and `far_range_m`.

    Args:
        cell_distances: (M,) distance from origin to each coarse cell center.
        near_range_m: Distance at which score starts decaying from 1.0.
        far_range_m: Distance beyond which score is 0.0.

    Returns:
        (M,) float32 scores in [0, 1].
    """
    score = np.clip(1.0 - (cell_distances / far_range_m), 0.0, 1.0).astype(np.float32)
    return score


def geometry_score(
    elevation_span: NDArray[np.float32],
    point_count: NDArray[np.int32],
    span_max: float = 2.5,
    density_max: float = 50.0,
) -> NDArray[np.float32]:
    """Compute per-cell geometry importance score ∈ [0, 1].

    Cells with high elevation span (obstacles, buildings, vegetation) and
    moderate-to-high point density get higher scores. Flat ground with
    minimal span scores near 0.

    Args:
        elevation_span: (M,) max_z - min_z per cell.
        point_count: (M,) number of points per cell.
        span_max: Normalisation ceiling for elevation span.
        density_max: Normalisation ceiling for point count.

    Returns:
        (M,) float32 scores in [0, 1].
    """
    norm_span = np.clip(elevation_span / span_max, 0.0, 1.0)
    norm_density = np.clip(point_count.astype(np.float32) / density_max, 0.0, 1.0)
    # Weighted combination: span matters more than density
    score = (0.65 * norm_span + 0.35 * norm_density).astype(np.float32)
    return score


def uncertainty_score(
    point_count: NDArray[np.int32],
    elevation_variance: NDArray[np.float32],
    sparse_threshold: int = 3,
    var_max: float = 1.5,
) -> NDArray[np.float32]:
    """Compute per-cell uncertainty importance score ∈ [0, 1].

    Cells with very few points (sparse / boundary) or high height variance
    (noisy / ambiguous) need higher resolution to resolve correctly.

    Args:
        point_count: (M,) points per cell.
        elevation_variance: (M,) variance of z values per cell.
        sparse_threshold: Cells with count <= this are considered highly uncertain.
        var_max: Normalisation ceiling for variance component.

    Returns:
        (M,) float32 scores in [0, 1].
    """
    # Sparsity component: high score when count is very low
    counts_f = point_count.astype(np.float32)
    sparsity = np.clip(1.0 - (counts_f / (sparse_threshold * 3.0)), 0.0, 1.0)

    # Variance component: high score when heights are ambiguous
    norm_var = np.clip(elevation_variance / var_max, 0.0, 1.0)

    score = (0.5 * sparsity + 0.5 * norm_var).astype(np.float32)
    return score


def motion_score(
    n_cells: int,
    previous_occupancy: Optional[NDArray[np.float32]] = None,
    current_occupancy: Optional[NDArray[np.float32]] = None,
    tracked_bboxes: Optional[list] = None,
    coarse_cols: int = 140,
) -> NDArray[np.float32]:
    """Compute per-cell motion/temporal importance score ∈ [0, 1].

    Uses occupancy change (Phase 6) and optionally boosts scores inside
    tracked bounding boxes (Phase 8).

    Returns:
        (n_cells,) float32 array of scores.
    """
    score = np.zeros(n_cells, dtype=np.float32)
    
    if previous_occupancy is not None and current_occupancy is not None:
        # Simple occupancy change detection
        delta = np.abs(current_occupancy - previous_occupancy)
        score = np.clip(delta * 2.0, 0.0, 1.0).astype(np.float32)

    if tracked_bboxes:
        # Boost score inside tracked boxes
        score_2d = score.reshape(-1, coarse_cols)
        for bbox in tracked_bboxes:
            r1, r2 = bbox["min_r"], bbox["max_r"]
            c1, c2 = bbox["min_c"], bbox["max_c"]
            score_2d[r1:r2+1, c1:c2+1] = np.clip(score_2d[r1:r2+1, c1:c2+1] + 0.5, 0.0, 1.0)
        score = score_2d.flatten()

    return score


def semantic_score(
    n_cells: int,
    semantic_labels: Optional[NDArray[np.int32]] = None,
) -> NDArray[np.float32]:
    """Compute per-cell semantic importance score ∈ [0, 1].

    Args:
        n_cells: Default size if no labels provided.
        semantic_labels: (M,) 1D array of class IDs from SemanticBackend.
            Classes: 1=GROUND, 2=VEGETATION, 3=OBSTACLE.

    Returns:
        (M,) float32 array of scores.
    """
    if semantic_labels is None:
        return np.zeros(n_cells, dtype=np.float32)

    score = np.zeros_like(semantic_labels, dtype=np.float32)
    score[semantic_labels == 1] = 0.1  # Ground
    score[semantic_labels == 2] = 0.4  # Vegetation
    score[semantic_labels == 3] = 1.0  # Obstacle / Vehicle
    return score
