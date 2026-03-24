"""Step 3: 3D localization (SAM3 segmentation + depth backprojection).

Paper: Sec III-D (3D Object Localization).

Takes retrieval results from Step 1+2 and produces 3D point predictions:
1. SAM3 text-prompted segmentation on retrieved images
2. Depth backprojection (mask → 3D point cloud)

Spatial fusion (multi-view refinement) is handled by fuse_candidates()
in multi_view_fusion.py, called by each pipeline after localize() returns.
"""

from __future__ import annotations

import logging

import numpy as np

from src.utils.geometry import closest_point_to_position, masked_median_depth
from src.utils.image import ensure_depth_shape, ensure_mask_shape
from src.utils.projection import project_mask_to_3d_cloud

logger = logging.getLogger(__name__)



def localize(
    query: str,
    images: list[np.ndarray],
    frame_ids: list[int],
    search_result,
    scene_loader,
    sam3_segmenter,
    top_k: int,
    robot_position: np.ndarray | None = None,
    use_centroid: bool = False,
    max_mask_depth: float = 0.0,
):
    """Step 3: SAM3 segmentation + depth backprojection → 3D goals.

    Unified localization for all eval pipelines (GoatCore, HM3D, MP3D, OVON).
    Spatial fusion is handled separately by ``fuse_candidates()`` or
    ``fuse_goal_candidates()`` after this function returns.

    Args:
        query: Natural language query (used for SAM3 segmentation prompts).
        images: Scene images aligned with frame_ids.
        frame_ids: Frame IDs corresponding to images.
        search_result: SearchResult from search_scene() (Steps 1+2).
        scene_loader: Scene loader providing depth, poses, and intrinsics.
        sam3_segmenter: SAM3Segmenter instance, or None to skip segmentation
            (frames without masks are ignored).
        top_k: Number of ranked results to process.
        robot_position: Optional agent position (3,). Activates nav mode when
            provided (closest surface point returned instead of centroid).
        use_centroid: When True, loc mode — returns cloud centroid for each
            rank. When False, nav mode — returns closest cloud point to
            robot_position.
        max_mask_depth: Skip frames whose masked median depth exceeds this
            value in meters (0 = no limit).

    Returns:
        Tuple of (predictions, top1_img_idx, pred_metadata):
            predictions: List of np.ndarray (3,) goal positions, one per rank.
            top1_img_idx: Index into images for the top-1 retrieval result.
            pred_metadata: List of dicts with point_cloud, score, retrieval_score,
                sam3_score, detected, and frame_id for each prediction.
    """
    from concurrent.futures import ThreadPoolExecutor

    results = search_result.results

    img_h, img_w = scene_loader.intrinsics.height, scene_loader.intrinsics.width

    predictions = []
    pred_metadata = []
    top1_img_idx = None

    rank_info = []  # (rank, img_idx, frame_id, retrieval_score, vlm_detected)
    batch_images = []
    batch_cache_keys = []

    for rank, result in enumerate(results[:top_k]):
        img_idx = result["image_index"]
        if rank == 0:
            top1_img_idx = img_idx
        frame_id = frame_ids[img_idx]
        retrieval_score = float(result.get("confidence", result.get("stage1_score", 0.0)))
        vlm_detected = result.get("detected", True)
        rank_info.append((rank, img_idx, frame_id, retrieval_score, vlm_detected))

        if sam3_segmenter is not None:
            batch_images.append(images[img_idx])
            batch_cache_keys.append((query, frame_id))

    def _load_frame_data(frame_id):
        depth = scene_loader.load_depth(frame_id)
        if depth is not None:
            depth = ensure_depth_shape(depth, img_h, img_w)
        pose_rt = scene_loader.get_pose_rt(frame_id)
        frame_intrinsics = scene_loader.get_intrinsics_for_frame(frame_id)
        return depth, pose_rt, frame_intrinsics

    if not rank_info:
        return predictions, top1_img_idx, pred_metadata

    with ThreadPoolExecutor(max_workers=min(len(rank_info), 8)) as pool:
        depth_futures = [pool.submit(_load_frame_data, ri[2]) for ri in rank_info]

        # SAM3 batch runs on GPU while depth loads in background threads
        seg_results = []
        if sam3_segmenter is not None and batch_images:
            seg_results = sam3_segmenter.segment_batch(
                batch_images, query, cache_keys=batch_cache_keys,
            )

        frame_data = [f.result() for f in depth_futures]

    seg_idx = 0
    for ri, (rank, img_idx, frame_id, retrieval_score, vlm_detected) in enumerate(rank_info):
        depth, pose_rt, frame_intrinsics = frame_data[ri]

        if depth is None:
            logger.warning("No depth for frame %d, skipping", frame_id)
            if sam3_segmenter is not None:
                seg_idx += 1
            continue
        cloud = None
        sam3_score = 0.0

        if sam3_segmenter is not None:
            seg_result = seg_results[seg_idx]
            seg_idx += 1
            mask = seg_result["best_mask"]

            if mask is not None:
                sam3_score = float(seg_result["best_score"])
                mask = ensure_mask_shape(mask, depth.shape[0], depth.shape[1])
                if max_mask_depth > 0:
                    med_z = masked_median_depth(mask, depth)
                    if med_z is not None and med_z > max_mask_depth:
                        logger.debug("rank %d skipped: mask median depth %.1fm > %.1fm",
                                     rank, med_z, max_mask_depth)
                        continue
                cloud = project_mask_to_3d_cloud(mask, depth, frame_intrinsics, pose_rt)

        if cloud is None:
            continue

        if robot_position is not None and not use_centroid:
            predictions.append(closest_point_to_position(cloud, robot_position))
        else:
            predictions.append(np.mean(cloud, axis=0))
        pred_metadata.append({
            "point_cloud": cloud,
            "score": max(retrieval_score, sam3_score),
            "retrieval_score": retrieval_score,
            "sam3_score": sam3_score,
            "detected": vlm_detected,
            "frame_id": frame_id,
        })

    return predictions, top1_img_idx, pred_metadata
