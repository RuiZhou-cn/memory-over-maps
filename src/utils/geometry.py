"""3D geometry utilities (point cloud queries, depth statistics)."""

from typing import Optional, Tuple

import numpy as np


def xz_dist(a: np.ndarray, b: np.ndarray) -> float:
    """2D Euclidean distance on the navigation plane (XZ in Y-up frame)."""
    d = a - b
    return float(np.sqrt(d[0] ** 2 + d[2] ** 2))


def closest_point_to_position(
    cloud: np.ndarray,
    position: np.ndarray,
    use_2d: bool = True,
) -> np.ndarray:
    """Return the point in cloud nearest to position (XZ-plane by default).

    Args:
        cloud: (N, 3) point cloud in world coordinates.
        position: (3,) robot position [x, y, z].
        use_2d: If True, compute distance in XZ-plane only (Y is up in Habitat).

    Returns:
        (3,) closest point from cloud.
    """
    if use_2d:
        diff = cloud[:, [0, 2]] - position[[0, 2]]
    else:
        diff = cloud - position
    dists = np.einsum('ij,ij->i', diff, diff)
    return cloud[np.argmin(dists)]


def masked_median_depth(
    mask: np.ndarray,
    depth_map: np.ndarray,
    depth_range: Tuple[float, float] = (0.1, 20.0),
) -> Optional[float]:
    """Compute median depth of valid pixels under a binary mask.

    Args:
        mask: Binary mask (H, W), nonzero where the object is.
        depth_map: Float depth map (H, W) in meters.
        depth_range: (min, max) valid depth range in meters.

    Returns:
        Median depth in meters, or None if no valid pixels under mask.
    """
    valid_mask = (mask > 0) & (depth_map > depth_range[0]) & (depth_map < depth_range[1])
    valid = depth_map[valid_mask]
    if len(valid) == 0:
        return None
    return float(np.median(valid))
