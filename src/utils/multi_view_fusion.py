"""Multi-view 3D localization fusion. Paper Step 3 (Sec III-D).

Filters per-view predictions to the same physical object as top-1
(via 3D point cloud overlap), then fuses into a single prediction.

Spatial fusion: finds cameras near a 3D seed point whose frustum
contains the target (frustum projection check), ensuring only cameras
actually looking at the object area contribute to the fused PCD.
"""

from __future__ import annotations

from copy import copy
from dataclasses import dataclass, field

import numpy as np
from scipy.spatial import cKDTree

from src.utils.geometry import closest_point_to_position, masked_median_depth
from src.utils.image import ensure_depth_shape, ensure_mask_shape, resize_image
from src.utils.projection import project_mask_to_3d_cloud

# Maximum accumulated cloud size during overlap-based grouping.
# Clouds exceeding this are subsampled to cap memory during clustering.
_MAX_CLOUD_POINTS = 50_000


def _merge_clouds(existing: np.ndarray, new: np.ndarray) -> np.ndarray:
    """Concatenate two point clouds, subsampling to _MAX_CLOUD_POINTS if needed."""
    merged = np.concatenate([existing, new], axis=0)
    if len(merged) > _MAX_CLOUD_POINTS:
        rng = np.random.default_rng(42)
        merged = merged[rng.choice(len(merged), _MAX_CLOUD_POINTS, replace=False)]
    return merged


def hdbscan_filter_cloud(
    cloud: np.ndarray,
    min_cluster_size: int = 20,
    min_samples: int = 5,
    min_keep_fraction: float = 0.05,
) -> np.ndarray:
    """Remove outliers from a 3D point cloud using HDBSCAN clustering.

    Keeps only the largest cluster, discarding noise (label=-1) and
    smaller clusters.

    Args:
        cloud: (N, 3) point cloud.
        min_cluster_size: Minimum points for HDBSCAN to form a cluster.
        min_samples: Core-point density threshold.
        min_keep_fraction: Safety guard — if the largest cluster has fewer
            than this fraction of the original points, return original.

    Returns:
        (M, 3) cleaned cloud. Returns original if clustering fails or
        would discard too many points.
    """
    if len(cloud) < min_cluster_size:
        return cloud

    from sklearn.cluster import HDBSCAN

    n = len(cloud)
    subsample_cap = 5000
    if n > subsample_cap:
        rng = np.random.default_rng(42)
        sub_idx = rng.choice(n, subsample_cap, replace=False)
        sub_cloud = cloud[sub_idx]
    else:
        sub_idx = None
        sub_cloud = cloud

    clusterer = HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        allow_single_cluster=True,
        copy=True,
    )
    labels = clusterer.fit_predict(sub_cloud)

    valid_labels = labels[labels >= 0]
    if len(valid_labels) == 0:
        return cloud

    unique, counts = np.unique(valid_labels, return_counts=True)
    largest_label = unique[np.argmax(counts)]
    largest_mask = labels == largest_label

    if sub_idx is None:
        kept = cloud[largest_mask]
    else:
        cluster_pts = sub_cloud[largest_mask]
        tree = cKDTree(cluster_pts)
        dists, _ = tree.query(cloud, k=1)
        inner_dists, _ = tree.query(cluster_pts, k=2)
        median_nn = np.median(inner_dists[:, 1])
        radius = median_nn * 2.0
        kept = cloud[dists <= radius]

    if len(kept) < min_keep_fraction * n:
        return cloud

    return kept


@dataclass
class GoalCandidate:
    """A navigation goal derived from one or more retrieval predictions.

    Used by MultiGoalNavigator to attempt multiple goals when some
    may be unreachable on the navmesh.
    """

    centroid: np.ndarray  # (3,) world coords — navigation target
    point_cloud: np.ndarray | None = None  # for closest-point recomputation
    source_ranks: list[int] = field(default_factory=list)  # retrieval ranks or GT goal indices
    confidence: float = 0.0  # best score in cluster
    instance_id: int = 0  # cluster ID (0-based)


def l2_sort_candidates(candidates, start_pos):
    """Sort GoalCandidates by L2 distance from start_pos (nearest first).

    Does NOT snap centroids to navmesh — raw 3D predictions are preserved.
    This avoids the through-wall snapping problem where snap_point() places
    the goal on the wrong side of a wall.

    Args:
        candidates: List of GoalCandidate.
        start_pos: Agent position (3D ndarray).

    Returns:
        Sorted list (same objects, reordered).
    """
    start = np.asarray(start_pos, dtype=np.float64)
    candidates.sort(key=lambda c: float(np.linalg.norm(c.centroid - start)))
    return candidates



def group_predictions_by_overlap(
    predictions: list[np.ndarray],
    metadata: list[dict] | None = None,
    overlap_threshold: float = 0.15,
    point_threshold: float = 0.5,
    centroid_fallback_distance: float = 3.0,
    proximity_threshold: float = 0.5,
) -> list["GoalCandidate"]:
    """Group 3D predictions by point cloud proximity into distinct object instances.

    Two predictions are merged when their point clouds are **close** — either
    by overlap fraction or by minimum pairwise distance.  This handles partial
    views of the same object that look at different parts (low IoU but
    nearby boundary points).

    Algorithm:
        Iterate in rank order.  For each prediction, compare its cloud
        against each existing cluster's **accumulated cloud**.  If overlap
        >= overlap_threshold **or** min cloud distance < proximity_threshold
        → merge (and grow the cluster cloud for transitivity).
        Else if centroid distance < fallback and both have SAM3 mask →
        merge (large-object fallback).  Else → new cluster.

    Args:
        predictions: List of (3,) centroid arrays, ordered by rank.
        metadata: Per-prediction dicts with keys:
            point_cloud (np.ndarray), score (float), sam3_score (float).
        overlap_threshold: Min fraction of candidate cloud overlapping the
            cluster cloud to merge.
        point_threshold: Distance in meters for a point to count as
            overlapping (passed to :func:`compute_cloud_overlap`).
        centroid_fallback_distance: Max centroid L2 distance for fallback
            merge when both predictions have sam3_score > 0 (meters).
            Set to 0 to disable the centroid fallback.
        proximity_threshold: Max min-distance (meters) between candidate
            and cluster cloud to merge.  Handles partial views of the same
            object that have low IoU but nearby boundary points.  Set to 0
            to disable proximity-based merging.

    Returns:
        List of GoalCandidate, one per unique instance, ordered by best rank.
    """
    if not predictions:
        return []

    clusters: list[dict] = []

    for rank, pred in enumerate(predictions):
        pred = np.asarray(pred, dtype=np.float64)
        meta = metadata[rank] if metadata else {}
        score = meta.get("score", 0.0)
        sam3_score = meta.get("sam3_score", 0.0)
        cloud = meta.get("point_cloud", None)

        assigned = False
        for cluster in clusters:
            if (cloud is not None and len(cloud) > 0
                    and cluster["acc_cloud"] is not None
                    and len(cluster["acc_cloud"]) > 0):
                overlap, min_dist = compute_cloud_overlap(
                    cluster["acc_cloud"], cloud, point_threshold,
                    _cache=cluster["_tree_cache"])
                if (overlap >= overlap_threshold
                        or (proximity_threshold > 0
                            and min_dist < proximity_threshold)):
                    cluster["ranks"].append(rank)
                    if score > cluster["best_score"]:
                        cluster["best_score"] = score
                    cluster["acc_cloud"] = _merge_clouds(cluster["acc_cloud"], cloud)
                    cluster["_tree_cache"] = {}
                    assigned = True
                    break

            if (centroid_fallback_distance > 0
                    and sam3_score > 0 and cluster["has_sam3"]
                    and np.linalg.norm(pred - cluster["centroid"]) < centroid_fallback_distance):
                cluster["ranks"].append(rank)
                if score > cluster["best_score"]:
                    cluster["best_score"] = score
                if cloud is not None and len(cloud) > 0:
                    if cluster["acc_cloud"] is not None:
                        cluster["acc_cloud"] = _merge_clouds(cluster["acc_cloud"], cloud)
                    else:
                        cluster["acc_cloud"] = cloud.copy()
                    cluster["_tree_cache"] = {}
                assigned = True
                break

        if not assigned:
            clusters.append({
                "centroid": pred.copy(),
                "best_rank": rank,
                "best_score": score,
                "ranks": [rank],
                "seed_cloud": cloud,
                "acc_cloud": cloud.copy() if cloud is not None else None,
                "_tree_cache": {},
                "has_sam3": sam3_score > 0,
            })

    clusters.sort(key=lambda c: c["best_rank"])
    candidates = []
    for idx, cluster in enumerate(clusters):
        candidates.append(GoalCandidate(
            centroid=cluster["centroid"].astype(np.float32),
            point_cloud=cluster["seed_cloud"],
            source_ranks=cluster["ranks"],
            confidence=cluster["best_score"],
            instance_id=idx,
        ))

    return candidates


@dataclass
class ViewPrediction:
    """Per-view 3D prediction data."""

    rank: int
    centroid: np.ndarray  # (3,) world coordinates
    point_cloud: np.ndarray | None = None  # (N, 3) or None
    retrieval_score: float = 0.0
    sam3_score: float = 0.0

    @property
    def confidence(self) -> float:
        """Combined confidence (average of nonzero scores)."""
        scores = [s for s in [self.retrieval_score, self.sam3_score] if s > 0]
        return float(np.mean(scores)) if scores else 0.0


@dataclass
class FusionResult:
    """Output of multi-view fusion."""

    fused_centroid: np.ndarray  # (3,) bbox center of filtered members
    fused_point_cloud: np.ndarray | None = None  # merged (M, 3) cloud
    cluster_size: int = 1
    cluster_view_indices: list[int] = field(default_factory=list)
    confidence: float = 0.0


def _find_nearby_cameras(
    point_world: np.ndarray,
    all_poses: np.ndarray,
    all_frame_ids: list[int],
    max_distance: float = 3.0,
    max_views: int = 5,
    exclude_frame_ids: set[int] | None = None,
    intrinsics: dict | None = None,
    image_hw: tuple[int, int] | None = None,
    frustum_margin: float = 0.1,
) -> list[tuple[int, float, float]]:
    """Find cameras near a 3D point whose frustum contains it.

    Uses distance filtering + frustum projection: the 3D point is
    projected into each camera's image plane, and only cameras where
    the projection falls within the image (with margin) are kept.

    Coordinate convention:
        Habitat camera: Y-up, -Z forward.
        OpenCV camera: Y-down, Z-forward.
        Conversion: x_cv, y_cv, z_cv = x_hab, -y_hab, -z_hab.

    Args:
        point_world: (3,) target 3D point in world coordinates.
        all_poses: (N, 4, 4) camera-to-world matrices.
        all_frame_ids: (N,) frame IDs corresponding to all_poses.
        max_distance: Max Euclidean distance from camera to point (meters).
        max_views: Cap on returned views (closest first).
        exclude_frame_ids: Frame IDs to skip (e.g., already processed).
        intrinsics: Camera intrinsics dict with keys 'fx', 'fy', 'cx', 'cy'.
            Required for frustum check. If None, falls back to distance-only.
        image_hw: (H, W) image dimensions. Required for frustum check.
        frustum_margin: Fraction of image to exclude at borders (0-0.5).
            0.1 means the projected point must land in the central 80% of
            the image.

    Returns:
        List of (frame_id, distance, frustum_score) sorted by distance.
        frustum_score is 1.0 for all passing cameras (placeholder for
        future scoring).
    """
    N = len(all_poses)
    if N == 0:
        return []

    cam_positions = all_poses[:, :3, 3]  # (N, 3)

    delta = point_world[None, :] - cam_positions  # (N, 3)  cam → point
    dists = np.linalg.norm(delta, axis=1)  # (N,)
    dist_mask = dists < max_distance

    if exclude_frame_ids:
        excl_set = set(exclude_frame_ids)
        excl_mask = np.array([fid not in excl_set for fid in all_frame_ids])
        dist_mask &= excl_mask

    survivors = np.where(dist_mask)[0]
    if len(survivors) == 0:
        return []

    use_frustum = intrinsics is not None and image_hw is not None

    if use_frustum:
        # Transform world point into each camera's local frame
        R_all = all_poses[survivors, :3, :3]  # (M, 3, 3)
        t_all = all_poses[survivors, :3, 3]   # (M, 3)

        # P_cam = R^T @ (P_world - t)  — world to Habitat camera coords
        p_cam = np.einsum(
            'mji,mj->mi',
            R_all,
            point_world[None, :] - t_all,
        )  # (M, 3)

        # Habitat → OpenCV: x_cv=x, y_cv=-y, z_cv=-z
        x_cv = p_cam[:, 0]
        y_cv = -p_cam[:, 1]
        z_cv = -p_cam[:, 2]

        # Must be in front of camera
        in_front = z_cv > 0.01

        fx, fy = intrinsics['fx'], intrinsics['fy']
        cx, cy = intrinsics['cx'], intrinsics['cy']
        H, W = image_hw

        u = fx * x_cv / (z_cv + 1e-8) + cx
        v = fy * y_cv / (z_cv + 1e-8) + cy

        u_min = W * frustum_margin
        u_max = W * (1.0 - frustum_margin)
        v_min = H * frustum_margin
        v_max = H * (1.0 - frustum_margin)

        in_image = (u >= u_min) & (u <= u_max) & (v >= v_min) & (v <= v_max)
        frustum_mask = in_front & in_image

        final_idx = survivors[frustum_mask]
        final_dists = dists[final_idx]
    else:
        # Fallback: distance-only (no intrinsics provided)
        final_idx = survivors
        final_dists = dists[final_idx]

    if len(final_idx) == 0:
        return []

    order = np.argsort(final_dists)
    if len(order) > max_views:
        order = order[:max_views]

    return [
        (all_frame_ids[final_idx[o]], float(final_dists[o]), 1.0)
        for o in order
    ]


def compute_cloud_overlap(
    cloud_a: np.ndarray,
    cloud_b: np.ndarray,
    point_threshold: float = 0.5,
    max_sample: int = 1000,
    _cache: dict | None = None,
) -> tuple[float, float]:
    """Fraction of cloud_b points near cloud_a, plus min distance.

    Uses cKDTree for efficient nearest-neighbor queries (O(N log N)
    instead of O(N*M) pairwise).

    Args:
        cloud_a: (N, 3) reference point cloud.
        cloud_b: (M, 3) candidate point cloud.
        point_threshold: Max distance (meters) for a point to count as overlapping.
        max_sample: Subsample clouds to this many points for speed.
        _cache: If provided, a mutable dict used to cache the cKDTree built
            from cloud_a. Pass ``{}`` on first call; the tree is reused on
            subsequent calls with the same cache dict. Set to ``{}`` again
            to invalidate after cloud_a changes.

    Returns:
        (overlap_fraction, min_distance): overlap in [0, 1], min pairwise
        distance in meters (inf if either cloud is empty).
    """
    if len(cloud_a) == 0 or len(cloud_b) == 0:
        return 0.0, float("inf")

    rng = np.random.default_rng(42)  # seeded for reproducibility and thread safety
    if _cache is not None and "tree" in _cache:
        tree = _cache["tree"]
    else:
        if len(cloud_a) > max_sample:
            idx = rng.choice(len(cloud_a), max_sample, replace=False)
            cloud_a = cloud_a[idx]
        tree = cKDTree(cloud_a)
        if _cache is not None:
            _cache["tree"] = tree
    if len(cloud_b) > max_sample:
        idx = rng.choice(len(cloud_b), max_sample, replace=False)
        cloud_b = cloud_b[idx]

    dists, _ = tree.query(cloud_b, k=1)

    n_overlap = int(np.sum(dists < point_threshold))
    min_dist = float(dists.min())

    return n_overlap / len(cloud_b), min_dist


def _filter_same_object(
    views: list[ViewPrediction],
    overlap_threshold: float = 0.3,
    point_threshold: float = 0.5,
) -> list[ViewPrediction]:
    """Keep only views whose point cloud overlaps with top-1's object.

    Uses 3D point cloud overlap: for each candidate view, compute the
    fraction of its points within point_threshold of the reference (top-1)
    cloud. Views below overlap_threshold are discarded.

    Args:
        views: Per-view predictions sorted by rank (index 0 = top-1 reference).
        overlap_threshold: Min fraction of candidate points overlapping reference.
        point_threshold: Distance in meters for a point to count as overlapping.

    Returns:
        Filtered views (always includes rank 0). Original order preserved.
    """
    if len(views) <= 1:
        return views

    ref = views[0]
    if ref.point_cloud is None or len(ref.point_cloud) == 0:
        return views[:1]

    filtered = [ref]
    ref_cloud = ref.point_cloud
    ref_cache = {}  # reuse cKDTree across comparisons

    for v in views[1:]:
        if v.point_cloud is None or len(v.point_cloud) == 0:
            continue

        overlap, _ = compute_cloud_overlap(ref_cloud, v.point_cloud, point_threshold, _cache=ref_cache)
        if overlap >= overlap_threshold:
            filtered.append(v)

    return filtered


def fuse_views(
    views: list[ViewPrediction],
    overlap_threshold: float = 0.3,
    point_threshold: float = 0.5,
    hdbscan_clean: bool = False,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int = 5,
) -> FusionResult | None:
    """Filter to same object as top-1 via point cloud overlap, then fuse.

    Pipeline:
    1. _filter_same_object() — keep views overlapping with top-1
    2. Merge point clouds from filtered views
    3. (optional) HDBSCAN-clean the merged cloud to remove outliers
    4. Compute fused centroid (single-view: seed centroid; multi-view: bbox center of merged cloud)

    The bbox center is more robust than a weighted centroid average because
    it represents the geometric center of the object's full observed extent,
    regardless of per-view point density or which surface portion each view sees.

    Args:
        views: Per-view predictions (must have centroid set).
        overlap_threshold: Min point cloud overlap to include a view (0-1).
        point_threshold: Distance threshold for overlap computation (meters).
        hdbscan_clean: If True, run HDBSCAN on the merged cloud to remove
            outlier points (flying pixels, mask bleed from multiple views).
        hdbscan_min_cluster_size: Min points to form a cluster. Higher =
            more aggressive at removing small disconnected blobs.
        hdbscan_min_samples: Core-point density threshold. Higher = stricter.

    Returns:
        FusionResult, or None if views is empty.
    """
    if not views:
        return None

    valid = [v for v in views if v.point_cloud is not None]
    if not valid:
        return None

    hdbscan_kwargs = dict(
        min_cluster_size=hdbscan_min_cluster_size,
        min_samples=hdbscan_min_samples,
    )

    if len(valid) == 1:
        v = valid[0]
        cloud = v.point_cloud
        if hdbscan_clean and cloud is not None and len(cloud) >= 10:
            cloud = hdbscan_filter_cloud(cloud, **hdbscan_kwargs)
        return FusionResult(
            fused_centroid=v.centroid.copy(),
            fused_point_cloud=cloud.copy() if cloud is not None else None,
            cluster_size=1,
            cluster_view_indices=[v.rank],
            confidence=v.confidence,
        )

    filtered = _filter_same_object(valid, overlap_threshold, point_threshold)

    clouds = [v.point_cloud for v in filtered]
    fused_cloud = np.concatenate(clouds, axis=0) if clouds else None

    if hdbscan_clean and fused_cloud is not None and len(fused_cloud) >= 10:
        cleaned = hdbscan_filter_cloud(fused_cloud, **hdbscan_kwargs)
        if len(cleaned) > 0:
            fused_cloud = cleaned

    if fused_cloud is not None and len(fused_cloud) > 0:
        bbox_min = fused_cloud.min(axis=0)
        bbox_max = fused_cloud.max(axis=0)
        fused_centroid = (bbox_min + bbox_max) / 2.0
    else:
        # No point clouds available — fall back to centroid average
        fused_centroid = np.mean([v.centroid for v in filtered], axis=0)

    mean_conf = float(np.mean([v.confidence for v in filtered]))

    return FusionResult(
        fused_centroid=fused_centroid,
        fused_point_cloud=fused_cloud,
        cluster_size=len(filtered),
        cluster_view_indices=[v.rank for v in filtered],
        confidence=mean_conf,
    )


def _fuse_with_nav_goal(
    fusion_result: FusionResult,
    robot_position: np.ndarray,
) -> np.ndarray:
    """Compute closest-surface-point from fused cloud to robot position.

    Falls back to fused centroid if no cloud is available.

    Args:
        fusion_result: Output from fuse_views().
        robot_position: (3,) robot position [x, y, z].

    Returns:
        (3,) navigation goal point.
    """
    if fusion_result.fused_point_cloud is not None and len(fusion_result.fused_point_cloud) > 0:
        return closest_point_to_position(fusion_result.fused_point_cloud, robot_position)
    return fusion_result.fused_centroid


def _build_view_from_mask(
    mask: np.ndarray,
    sam3_score: float,
    depth: np.ndarray,
    frame_id: int,
    scene_loader,
    max_mask_depth: float,
    masked_median_depth_fn,
    project_fn,
) -> ViewPrediction | None:
    """Project a segmentation mask to 3D and build a ViewPrediction.

    Returns None if depth filtering or projection fails.
    """
    mask = ensure_mask_shape(mask, depth.shape[0], depth.shape[1])

    if max_mask_depth > 0:
        med_z = masked_median_depth_fn(mask, depth)
        if med_z is not None and med_z > max_mask_depth:
            return None

    intrinsics = scene_loader.get_intrinsics_for_frame(frame_id)
    pose_rt = scene_loader.get_pose_rt(frame_id)
    cloud = project_fn(mask, depth, intrinsics, pose_rt)
    if cloud is None or len(cloud) == 0:
        return None

    centroid = np.mean(cloud, axis=0)
    return ViewPrediction(
        rank=0,
        centroid=centroid,
        point_cloud=cloud,
        retrieval_score=0.0,
        sam3_score=sam3_score,
    )


def _segment_neighbor(
    frame_id: int,
    scene_loader,
    sam3_segmenter,
    query: str,
    max_mask_depth: float = 0.0,
    query_res: int = 0,
) -> ViewPrediction | None:
    """Run SAM3 on one neighbor frame and project to 3D.

    Args:
        frame_id: Frame to segment.
        scene_loader: Scene loader (``BaseSceneDatasetLoader``).
        sam3_segmenter: SAM3Segmenter instance.
        query: Text query for segmentation.
        max_mask_depth: Skip if masked median depth exceeds this (0 = no limit).
        query_res: Resize image to this resolution before SAM3 (0 = no resize).

    Returns:
        ViewPrediction with point cloud, or None if segmentation/projection failed.
    """
    depth = scene_loader.load_depth(frame_id)
    if depth is None:
        return None

    img_h, img_w = scene_loader.intrinsics.height, scene_loader.intrinsics.width
    depth = ensure_depth_shape(depth, img_h, img_w)

    image = scene_loader.load_rgb(frame_id)
    if image is None:
        return None

    seg = sam3_segmenter.segment(
        resize_image(image, query_res), query, cache_key=(query, frame_id),
    )
    if seg["best_mask"] is None:
        return None

    return _build_view_from_mask(
        seg["best_mask"], float(seg["best_score"]), depth, frame_id,
        scene_loader, max_mask_depth, masked_median_depth, project_mask_to_3d_cloud,
    )


def batch_segment_neighbors(
    frame_ids: list[int],
    scene_loader,
    sam3_segmenter,
    query: str,
    max_mask_depth: float = 0.0,
    io_workers: int = 4,
    query_res: int = 0,
) -> dict:
    """Batch-segment multiple frames with parallel I/O and batched SAM3.

    Phase 1: Pre-load RGB for all frames in parallel (ThreadPool).
    Phase 2: Run SAM3 on GPU while loading depth in background threads.
    Phase 3: Project masks to 3D in parallel (CPU-bound).

    Args:
        frame_ids: Unique frame IDs to segment.
        scene_loader: Scene loader instance.
        sam3_segmenter: SAM3Segmenter instance.
        query: Text query for segmentation.
        max_mask_depth: Skip if masked median depth exceeds this (0 = no limit).
        io_workers: Thread pool size for I/O pre-loading and projection.

    Returns:
        Dict mapping frame_id -> ViewPrediction (only successful frames).
    """
    if not frame_ids or sam3_segmenter is None:
        return {}

    from concurrent.futures import ThreadPoolExecutor

    img_h, img_w = scene_loader.intrinsics.height, scene_loader.intrinsics.width

    # Phase 1: load RGB in parallel (needed for SAM3)
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        rgb_map = {}
        for fid, img in zip(frame_ids, pool.map(scene_loader.load_rgb, frame_ids)):
            if img is not None:
                rgb_map[fid] = img

    batch_fids = [fid for fid in frame_ids if fid in rgb_map]
    if not batch_fids:
        return {}

    batch_images = [resize_image(rgb_map[fid], query_res) for fid in batch_fids]
    batch_cache_keys = [(query, fid) for fid in batch_fids]

    # Phase 2: SAM3 on GPU + depth loading in background threads
    def _load_depth(fid):
        depth = scene_loader.load_depth(fid)
        if depth is not None:
            depth = ensure_depth_shape(depth, img_h, img_w)
        return (fid, depth)

    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        depth_futures = [pool.submit(_load_depth, fid) for fid in batch_fids]

        batch_seg = sam3_segmenter.segment_batch(
            batch_images, query, cache_keys=batch_cache_keys,
        )

        depth_map = {}
        for future in depth_futures:
            fid, depth = future.result()
            if depth is not None:
                depth_map[fid] = depth

    seg_results = {}
    for fid, seg in zip(batch_fids, batch_seg):
        if seg["best_mask"] is not None:
            seg_results[fid] = seg

    # Phase 3: parallel 3D projection
    def _project_frame(fid):
        depth = depth_map[fid]
        seg = seg_results[fid]
        return _build_view_from_mask(
            seg["best_mask"], float(seg["best_score"]), depth, fid,
            scene_loader, max_mask_depth, masked_median_depth, project_mask_to_3d_cloud,
        )

    project_fids = [fid for fid in frame_ids if fid in seg_results and fid in depth_map]
    cache = {}
    with ThreadPoolExecutor(max_workers=io_workers) as pool:
        for fid, vp in zip(project_fids, pool.map(_project_frame, project_fids)):
            if vp is not None:
                cache[fid] = vp

    return cache


def _collect_all_neighbors(
    seed_preds: list[ViewPrediction],
    all_poses_cache,
    processed_fids: set,
    spatial_max_views: int = 5,
    spatial_max_distance: float = 3.0,
    precomputed_counts: list[int] | None = None,
    intrinsics: dict | None = None,
    image_hw: tuple[int, int] | None = None,
    frustum_margin: float = 0.1,
) -> tuple[list[int], list[list[tuple[int, float, float]]]]:
    """Find nearby cameras for multiple seed predictions at once.

    Returns deduplicated frame IDs for batch segmentation, plus
    per-seed neighbor lists for later assembly.

    Args:
        seed_preds: List of seed ViewPredictions.
        all_poses_cache: (all_poses_4x4, all_fids) arrays.
        processed_fids: Frame IDs to exclude.
        spatial_max_views: Total views per seed (including seed +
            precomputed). E.g., 5 means seed + up to 4 neighbors.
        spatial_max_distance: Max camera-to-target distance (meters).
        precomputed_counts: Number of precomputed views per seed
            (reduces remaining neighbor slots). None = 0 for all.
        intrinsics: Camera intrinsics dict for frustum check.
        image_hw: (H, W) image dimensions for frustum check.
        frustum_margin: Frustum border margin (0-0.5).

    Returns:
        (all_unique_fids, per_seed_nearby) where all_unique_fids is a
        deduplicated list, and per_seed_nearby[i] is the neighbor list
        for seed_preds[i].
    """
    all_poses, all_fids = all_poses_cache

    per_seed_nearby = []
    seen_fids: set[int] = set()

    for i, seed in enumerate(seed_preds):
        n_pre = precomputed_counts[i] if precomputed_counts else 0
        # -1 accounts for the seed itself
        remaining = max(0, spatial_max_views - n_pre - 1)

        if remaining > 0:
            nearby = _find_nearby_cameras(
                seed.centroid, all_poses, all_fids,
                max_distance=spatial_max_distance,
                max_views=remaining,
                exclude_frame_ids=processed_fids,
                intrinsics=intrinsics,
                image_hw=image_hw,
                frustum_margin=frustum_margin,
            )
        else:
            nearby = []

        per_seed_nearby.append(nearby)
        for fid, _, _ in nearby:
            seen_fids.add(fid)

    return sorted(seen_fids), per_seed_nearby


def fuse_candidates(
    goal_candidates,
    query: str,
    scene_loader,
    sam3_segmenter,
    pred_metadata: list[dict],
    preds=None,
    pred_meta=None,
    all_poses_cache=None,
    spatial_max_views: int = 5,
    spatial_max_distance: float = 3.0,
    frustum_margin: float = 0.1,
    fusion_threshold: float = 0.5,
    overlap_threshold: float = 0.3,
    hdbscan_clean: bool = False,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int = 5,
    max_mask_depth: float = 0.0,
    use_centroid: bool = True,
    robot_position: np.ndarray | None = None,
    query_res: int = 0,
) -> list[list[tuple[int, float, float]]]:
    """Spatial fusion for a list of GoalCandidates (in-place update).

    For each candidate: build seed ViewPrediction, attach precomputed views
    from grouped ranks (if preds/pred_meta provided), find spatial neighbors
    to fill remaining slots, then fuse via point cloud overlap.

    Args:
        goal_candidates: GoalCandidates to fuse (updated in-place).
        query: Text query for SAM3 segmentation.
        scene_loader: Scene loader (depth, poses, intrinsics).
        sam3_segmenter: SAM3Segmenter instance.
        pred_metadata: All per-rank metadata dicts — used to build
            ``processed_fids`` (frames to exclude from neighbor search).
        preds: Per-rank 3D predictions (parallel to pred_meta).
        pred_meta: Per-rank metadata that ``source_ranks`` index into —
            may be a filtered subset of pred_metadata (e.g. nav pipeline
            filters to valid predictions only). Defaults to pred_metadata.
        all_poses_cache: Optional (all_poses, all_fids) — computed if None.
        spatial_max_views: Total views per candidate (seed + precomputed + neighbors).
        spatial_max_distance: Max camera-to-target distance (meters).
        frustum_margin: Frustum border margin (0-0.5).
        fusion_threshold: Point distance for overlap check (meters).
        overlap_threshold: Min overlap fraction to keep a view.
        hdbscan_clean: Run HDBSCAN on merged cloud to remove outliers.
        hdbscan_min_cluster_size: Min points to form a cluster.
        hdbscan_min_samples: Core-point density threshold.
        max_mask_depth: Skip neighbors whose masked median depth exceeds this (0=no limit).
        use_centroid: If True, return centroid; if False, closest surface point.
        robot_position: Agent position (used when use_centroid=False).

    Returns:
        per_seed_nearby: List of neighbor lists per candidate (for visualization).
    """
    if pred_meta is None:
        pred_meta = pred_metadata

    if all_poses_cache is None:
        all_poses, all_fids = scene_loader.get_all_poses()
        all_fids = np.asarray(all_fids)
        all_poses_cache = (all_poses, all_fids)

    processed_fids = {m["frame_id"] for m in pred_metadata if "frame_id" in m}

    seeds = []
    all_precomputed = []
    for cand in goal_candidates:
        precomputed_views = []
        if preds is not None and pred_meta is not None:
            for rank_idx in cand.source_ranks[1:]:
                if rank_idx >= len(pred_meta):
                    continue
                meta = pred_meta[rank_idx]
                cloud = meta.get("point_cloud")
                if cloud is None or len(cloud) == 0:
                    continue
                precomputed_views.append(ViewPrediction(
                    rank=1 + len(precomputed_views),
                    centroid=np.array(preds[rank_idx], dtype=np.float32),
                    point_cloud=cloud,
                    retrieval_score=meta.get("retrieval_score", 0.0),
                    sam3_score=meta.get("sam3_score", 0.0),
                ))

        max_pre = max(0, spatial_max_views - 1)
        precomputed_views = precomputed_views[:max_pre]

        seed = ViewPrediction(
            rank=0,
            centroid=cand.centroid.copy(),
            point_cloud=cand.point_cloud,
            retrieval_score=cand.confidence,
            sam3_score=cand.confidence,
        )
        seeds.append(seed)
        all_precomputed.append(precomputed_views)

    intr = scene_loader.get_intrinsics_dict()
    img_hw = (scene_loader.intrinsics.height, scene_loader.intrinsics.width)

    precomputed_counts = [len(pv) for pv in all_precomputed]
    all_neighbor_fids, per_seed_nearby = _collect_all_neighbors(
        seeds, all_poses_cache, processed_fids,
        spatial_max_views=spatial_max_views,
        spatial_max_distance=spatial_max_distance,
        precomputed_counts=precomputed_counts,
        intrinsics=intr,
        image_hw=img_hw,
        frustum_margin=frustum_margin,
    )

    segment_cache = {}
    if all_neighbor_fids and sam3_segmenter is not None:
        segment_cache = batch_segment_neighbors(
            all_neighbor_fids, scene_loader, sam3_segmenter,
            query, max_mask_depth=max_mask_depth, query_res=query_res,
        )

    for i, cand in enumerate(goal_candidates):
        fused_pred, fusion_result = _fuse_rank(
            seed_pred=seeds[i],
            query=query,
            scene_loader=scene_loader,
            sam3_segmenter=sam3_segmenter,
            precomputed_nearby=per_seed_nearby[i],
            robot_position=robot_position,
            use_centroid=use_centroid,
            fusion_threshold=fusion_threshold,
            overlap_threshold=overlap_threshold,
            spatial_max_views=spatial_max_views,
            precomputed_views=all_precomputed[i] or None,
            hdbscan_clean=hdbscan_clean,
            hdbscan_min_cluster_size=hdbscan_min_cluster_size,
            hdbscan_min_samples=hdbscan_min_samples,
            max_mask_depth=max_mask_depth,
            segment_cache=segment_cache,
            query_res=query_res,
        )
        cand.centroid = fused_pred.astype(np.float32)
        if fusion_result is not None and fusion_result.fused_point_cloud is not None:
            cand.point_cloud = fusion_result.fused_point_cloud

    return per_seed_nearby


def _fuse_rank(
    seed_pred,
    query: str,
    scene_loader,
    sam3_segmenter,
    precomputed_nearby: list[tuple[int, float, float]],
    robot_position: np.ndarray | None = None,
    use_centroid: bool = False,
    fusion_threshold: float = 0.5,
    overlap_threshold: float = 0.3,
    spatial_max_views: int = 5,
    precomputed_views: list[ViewPrediction] | None = None,
    hdbscan_clean: bool = False,
    hdbscan_min_cluster_size: int = 20,
    hdbscan_min_samples: int = 5,
    max_mask_depth: float = 0.0,
    segment_cache: dict | None = None,
    query_res: int = 0,
):
    """Unified per-rank spatial fusion for GoatCore and HM3D.

    Uses pre-computed nearby cameras from ``_collect_all_neighbors()``,
    runs SAM3 on each neighbor, fuses overlapping point clouds into
    one prediction.

    Args:
        seed_pred: ViewPrediction from the seed (top-K) rank.
        query: Text query (for SAM3 segmentation).
        scene_loader: Scene loader (``BaseSceneDatasetLoader``) — provides
            ``load_depth``, ``load_rgb``, ``get_pose_rt``, ``get_intrinsics_for_frame``,
            and ``intrinsics.height/width``.
        sam3_segmenter: SAM3Segmenter instance (or None to skip segmentation).
        precomputed_nearby: Pre-computed neighbor list from
            ``_collect_all_neighbors()``.
        robot_position: Robot 3D position (activates nav mode if provided).
        use_centroid: If True, always return centroid (not closest surface point).
        fusion_threshold: Point distance for overlap check (meters).
        overlap_threshold: Min overlap fraction to keep a view.
        spatial_max_views: Total views (seed + precomputed + neighbors).
        precomputed_views: Pre-computed ViewPredictions from retrieved images
            in the same cluster (e.g., ranks 2-4 that clustered with rank 1).
            These count toward ``spatial_max_views``, reducing neighbor search.
        hdbscan_clean: If True, run HDBSCAN on the merged cloud after fusion.
        hdbscan_min_cluster_size: Min points to form a cluster.
        hdbscan_min_samples: Core-point density threshold.
        max_mask_depth: Skip neighbor frames whose masked median depth exceeds
            this value in meters (0 = no limit).
        segment_cache: Pre-computed dict of frame_id -> ViewPrediction from
            ``batch_segment_neighbors()``. When provided, neighbor segmentation
            is skipped and cached results are used instead.

    Returns:
        ``(fused_prediction, fusion_result_or_None)`` where
        fused_prediction is ``np.ndarray`` shape (3,), and fusion_result is a ``FusionResult``
        (or ``None`` when fusion failed / fell back to seed).
    """
    # Cap precomputed views so total (seed + precomputed + neighbors) <= spatial_max_views
    max_pre = max(0, spatial_max_views - 1)
    if precomputed_views and len(precomputed_views) > max_pre:
        precomputed_views = precomputed_views[:max_pre]

    nearby = precomputed_nearby

    neighbor_preds = []
    if nearby:
        if segment_cache is not None:
            for idx, (nb_fid, _, _) in enumerate(nearby):
                vp = segment_cache.get(nb_fid)
                if vp is not None:
                    vp = copy(vp)  # avoid mutating cached rank
                    vp.rank = 1 + idx
                    neighbor_preds.append(vp)
        elif sam3_segmenter is not None:
            for idx, (nb_fid, _, _) in enumerate(nearby):
                vp = _segment_neighbor(
                    nb_fid, scene_loader, sam3_segmenter, query, max_mask_depth,
                    query_res=query_res)
                if vp is not None:
                    vp.rank = 1 + idx
                    neighbor_preds.append(vp)

    fusion_input = [seed_pred]
    if precomputed_views:
        fusion_input.extend(precomputed_views)
    fusion_input.extend(neighbor_preds)
    fusion_result = fuse_views(
        fusion_input,
        overlap_threshold=overlap_threshold,
        point_threshold=fusion_threshold,
        hdbscan_clean=hdbscan_clean,
        hdbscan_min_cluster_size=hdbscan_min_cluster_size,
        hdbscan_min_samples=hdbscan_min_samples,
    )

    fused_pred = None
    if fusion_result is not None:
        if robot_position is not None and not use_centroid:
            fused_pred = _fuse_with_nav_goal(fusion_result, robot_position)
        else:
            fused_pred = fusion_result.fused_centroid

    if fused_pred is None:
        fusion_result = None
        if robot_position is not None and not use_centroid and seed_pred.point_cloud is not None:
            fused_pred = closest_point_to_position(seed_pred.point_cloud, robot_position)
        else:
            fused_pred = seed_pred.centroid

    return fused_pred, fusion_result
