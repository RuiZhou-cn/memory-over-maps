"""Mask-to-3D projection and camera geometry utilities. Paper Step 3 (Sec III-D)."""

from __future__ import annotations

import numpy as np

from src.utils.geometry import closest_point_to_position


def project_mask_to_3d_cloud(
    mask: np.ndarray,
    depth_map: np.ndarray,
    intrinsics: dict[str, float],
    pose: tuple[np.ndarray, np.ndarray],
    depth_range: tuple[float, float] = (0.1, 20.0),
) -> np.ndarray | None:
    """Backproject 2D mask pixels to 3D world coordinates and return the full cloud.

    Uses the OpenCV -> Habitat/ARKit coordinate flip (Y-flip + Z-flip) so that
    the resulting 3D points are in the Y-up world frame used across all datasets.

    Args:
        mask: Binary mask (H, W), nonzero where the object is.
        depth_map: Float depth map (H, W) in meters.
        intrinsics: Dict with keys 'fx', 'fy', 'cx', 'cy'.
        pose: (R_3x3, t_3) camera-to-world rotation and translation.
        depth_range: (min, max) valid depth range in meters.

    Returns:
        (N, 3) world point cloud, or None if no valid points.
    """
    if depth_map is None or intrinsics is None or pose is None:
        return None

    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None

    z_vals = depth_map[ys, xs]
    valid = (z_vals > depth_range[0]) & (z_vals < depth_range[1])

    if not np.any(valid):
        return None

    xs = xs[valid]
    ys = ys[valid]
    zs = z_vals[valid]

    fx, fy = intrinsics['fx'], intrinsics['fy']
    cx, cy = intrinsics['cx'], intrinsics['cy']

    x_cam = (xs - cx) * zs / fx
    y_cam = (ys - cy) * zs / fy
    z_cam = zs

    # OpenCV (Y-down, Z-forward) -> Habitat/ARKit (Y-up, -Z-forward)
    P_cam = np.stack([x_cam, -y_cam, -z_cam], axis=1)  # (N, 3)

    R, t = pose
    P_world = (R @ P_cam.T).T + t

    return P_world


def quat_to_xyzw(q) -> tuple[float, float, float, float]:
    """Extract (x, y, z, w) from magnum.Quaternion, numpy-quaternion, or array-like."""
    if hasattr(q, "scalar") and hasattr(q, "vector"):
        return q.vector[0], q.vector[1], q.vector[2], q.scalar
    elif hasattr(q, "w") and hasattr(q, "x"):
        return q.x, q.y, q.z, q.w
    return float(q[0]), float(q[1]), float(q[2]), float(q[3])


def _quat_to_rotation_matrix(q) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix.

    Accepts magnum.Quaternion, numpy-quaternion, or (x, y, z, w) array-like.
    """
    x, y, z, w = quat_to_xyzw(q)

    R = np.array([
        [1 - 2*(y*y + z*z),   2*(x*y - z*w),       2*(x*z + y*w)],
        [2*(x*y + z*w),       1 - 2*(x*x + z*z),   2*(y*z - x*w)],
        [2*(x*z - y*w),       2*(y*z + x*w),       1 - 2*(x*x + y*y)],
    ], dtype=np.float64)
    return R


def _project_cloud_to_camera(
    point_cloud: np.ndarray,
    agent_position: np.ndarray,
    agent_rotation,
    depth_obs: np.ndarray,
    hfov: float,
    sensor_height: float,
    min_depth: float,
    max_depth: float,
    depth_tolerance: float,
    max_points: int = 2000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Project 3D cloud into camera and compute per-point visibility.

    Shared helper for check_cloud_visibility() and get_visible_closest_point().

    Args:
        point_cloud: (N, 3) world coordinates (must be non-empty).
        agent_position: (3,) agent base position (Habitat Y-up).
        agent_rotation: Agent rotation quaternion.
        depth_obs: (H, W) or (H, W, 1) normalized depth in [0, 1].
        hfov: Horizontal FOV in degrees.
        sensor_height: Camera height offset from agent base (meters).
        min_depth: Minimum depth (meters) matching policy normalization.
        max_depth: Maximum depth (meters) matching policy normalization.
        depth_tolerance: Tolerance for depth match (meters).
        max_points: Subsample large clouds for speed.

    Returns:
        (cloud_sub, in_bounds_indices, visible_mask) or None if no in-bounds points.
        - cloud_sub: (M, 3) subsampled cloud in world coordinates.
        - in_bounds_indices: (K,) int indices into cloud_sub for in-FOV points.
        - visible_mask: (K,) bool mask — True where depth test passes.
    """
    depth = depth_obs.squeeze()  # (H, W, 1) -> (H, W)
    H, W = depth.shape

    if len(point_cloud) > max_points:
        indices = np.random.default_rng(42).choice(len(point_cloud), max_points, replace=False)
        cloud_sub = point_cloud[indices]
    else:
        cloud_sub = point_cloud

    cam_pos = np.array(agent_position, dtype=np.float64)
    cam_pos[1] += sensor_height

    R = _quat_to_rotation_matrix(agent_rotation)

    P_centered = cloud_sub.astype(np.float64) - cam_pos[np.newaxis, :]
    P_cam_hab = (R.T @ P_centered.T).T
    x_cv = P_cam_hab[:, 0]
    y_cv = -P_cam_hab[:, 1]
    z_cv = -P_cam_hab[:, 2]

    in_front = z_cv > 1e-6
    if not np.any(in_front):
        return None

    front_idx = np.where(in_front)[0]
    x_cv = x_cv[in_front]
    y_cv = y_cv[in_front]
    z_cv = z_cv[in_front]

    hfov_rad = np.deg2rad(hfov)
    fx = W / (2.0 * np.tan(hfov_rad / 2.0))
    fy = fx  # square pixels
    cx = W / 2.0
    cy = H / 2.0

    u_f = fx * x_cv / z_cv + cx
    v_f = fy * y_cv / z_cv + cy
    _I32_MAX = np.float64(np.iinfo(np.int32).max)
    np.clip(u_f, -_I32_MAX, _I32_MAX, out=u_f)
    np.clip(v_f, -_I32_MAX, _I32_MAX, out=v_f)
    u = u_f.astype(np.int32)
    v = v_f.astype(np.int32)

    in_bounds = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    if not np.any(in_bounds):
        return None

    in_bounds_indices = front_idx[in_bounds]
    u_ib = u[in_bounds]
    v_ib = v[in_bounds]
    z_ib = z_cv[in_bounds]

    rendered_z = depth[v_ib, u_ib].astype(np.float64) * (max_depth - min_depth) + min_depth
    diff = np.abs(z_ib - rendered_z)
    visible_mask = diff < depth_tolerance

    return cloud_sub, in_bounds_indices, visible_mask


def check_cloud_visibility(
    point_cloud: np.ndarray | None,
    agent_position: np.ndarray,
    agent_rotation,
    depth_obs: np.ndarray,
    hfov: float,
    sensor_height: float = 0.88,
    min_depth: float = 0.5,
    max_depth: float = 5.0,
    depth_tolerance: float = 0.3,
    min_visible_fraction: float = 0.05,
    max_points: int = 2000,
) -> tuple[bool, float, int, int]:
    """Check if predicted object cloud is visible from current agent pose.

    Projects the 3D point cloud into the current camera view and compares
    projected depths against the rendered depth buffer to detect occlusion
    (e.g., object behind a wall).

    Args:
        point_cloud: (N, 3) world coordinates, or None.
        agent_position: (3,) agent base position (Habitat Y-up).
        agent_rotation: Agent rotation quaternion (magnum, numpy-quat, or xyzw).
        depth_obs: (H, W) or (H, W, 1) normalized depth in [0, 1].
        hfov: Horizontal FOV in degrees.
        sensor_height: Camera height offset from agent base (meters).
        min_depth: Minimum depth (meters) matching policy normalization.
        max_depth: Maximum depth (meters) matching policy normalization.
        depth_tolerance: Tolerance for depth match (meters).
        min_visible_fraction: Min fraction of in-bounds points that must be
            visible to confirm the object is not occluded.
        max_points: Subsample large clouds for speed.

    Returns:
        (is_visible, visible_fraction, n_visible, n_in_bounds).
        Graceful degradation: returns (True, 1.0, 0, 0) if cloud is None/empty.
    """
    # Graceful degradation — if no cloud, allow stop (L2-only fallback)
    if point_cloud is None or len(point_cloud) == 0:
        return True, 1.0, 0, 0

    result = _project_cloud_to_camera(
        point_cloud, agent_position, agent_rotation, depth_obs,
        hfov, sensor_height, min_depth, max_depth, depth_tolerance, max_points,
    )
    if result is None:
        return False, 0.0, 0, 0

    _, in_bounds_indices, visible_mask = result
    n_in_bounds = len(in_bounds_indices)
    n_visible = int(np.sum(visible_mask))
    fraction = n_visible / n_in_bounds

    return fraction >= min_visible_fraction, fraction, n_visible, n_in_bounds


def get_visible_closest_point(
    point_cloud: np.ndarray | None,
    agent_position: np.ndarray,
    agent_rotation,
    depth_obs: np.ndarray,
    hfov: float,
    sensor_height: float = 0.88,
    min_depth: float = 0.5,
    max_depth: float = 5.0,
    depth_tolerance: float = 0.3,
    max_points: int = 2000,
    use_2d: bool = True,
) -> np.ndarray | None:
    """Return the closest *visible* surface point from the cloud.

    Projects the cloud into the current depth observation, filters by depth
    match (same logic as check_cloud_visibility), then picks the nearest
    visible point by XZ distance (or 3D if use_2d=False).

    Args:
        point_cloud: (N, 3) world coordinates, or None.
        agent_position: (3,) agent base position (Habitat Y-up).
        agent_rotation: Agent rotation quaternion.
        depth_obs: (H, W) or (H, W, 1) normalized depth in [0, 1].
        hfov: Horizontal FOV in degrees.
        sensor_height: Camera height offset from agent base (meters).
        min_depth: Minimum depth (meters) matching policy normalization.
        max_depth: Maximum depth (meters) matching policy normalization.
        depth_tolerance: Tolerance for depth match (meters).
        max_points: Subsample large clouds for speed.
        use_2d: If True, pick closest by XZ-plane distance (Y is up).

    Returns:
        (3,) ndarray of closest visible point, or None if nothing visible.
    """
    if point_cloud is None or len(point_cloud) == 0:
        return None

    result = _project_cloud_to_camera(
        point_cloud, agent_position, agent_rotation, depth_obs,
        hfov, sensor_height, min_depth, max_depth, depth_tolerance, max_points,
    )
    if result is None:
        return None

    cloud_sub, in_bounds_indices, visible_mask = result
    if not np.any(visible_mask):
        return None

    visible_indices = in_bounds_indices[visible_mask]
    visible_points = cloud_sub[visible_indices]  # (V, 3)

    return closest_point_to_position(visible_points, np.array(agent_position, dtype=np.float64), use_2d=use_2d)


def get_visible_point_indices(
    point_cloud: np.ndarray | None,
    agent_position: np.ndarray,
    agent_rotation,
    depth_obs: np.ndarray,
    hfov: float,
    sensor_height: float = 0.88,
    min_depth: float = 0.5,
    max_depth: float = 5.0,
    depth_tolerance: float = 0.3,
) -> np.ndarray | None:
    """Return indices into the original cloud that pass the depth visibility test.

    Unlike get_visible_closest_point() which subsamples, this uses
    max_points=len(cloud) so returned indices map directly into the
    original cloud array.

    Args:
        point_cloud: (N, 3) world coordinates, or None.
        agent_position: (3,) agent base position (Habitat Y-up).
        agent_rotation: Agent rotation quaternion.
        depth_obs: (H, W) or (H, W, 1) normalized depth in [0, 1].
        hfov: Horizontal FOV in degrees.
        sensor_height: Camera height offset from agent base (meters).
        min_depth: Minimum depth (meters) matching policy normalization.
        max_depth: Maximum depth (meters) matching policy normalization.
        depth_tolerance: Tolerance for depth match (meters).

    Returns:
        1-D int array of original-cloud indices that pass visibility,
        or None if cloud is None/empty or nothing visible.
    """
    if point_cloud is None or len(point_cloud) == 0:
        return None

    result = _project_cloud_to_camera(
        point_cloud, agent_position, agent_rotation, depth_obs,
        hfov, sensor_height, min_depth, max_depth, depth_tolerance,
        max_points=len(point_cloud),  # no subsampling
    )
    if result is None:
        return None

    _, in_bounds_indices, visible_mask = result
    if not np.any(visible_mask):
        return None

    return in_bounds_indices[visible_mask]
