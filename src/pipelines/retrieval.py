"""Steps 1+2: Zero-shot retrieval and VLM re-ranking.

Paper: Sec III-C (Two-Stage Hybrid Retrieval).

Step 1 (Sec III-C): Pre-computed SigLIP2/CLIP features indexed with FAISS.
    Cosine similarity search returns top-K candidates per query.
Step 2 (Sec III-C): Parallel VLM calls on Stage 1 candidates only.
    Granular confidence scoring (0.0-1.0), re-ranked by visibility/size/occlusion.

Key functions:
- ``build_retriever``: FAISS index construction with feature caching.
- ``search_scene``: Scene-level search with optional floor filtering and VLM re-ranking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

CAMERA_FLOOR_THRESHOLD = 1.0


@dataclass
class SearchResult:
    """Output of search_scene(): retrieval results + metadata for localization."""

    results: list[dict]
    """Per-candidate dicts from HybridRetriever.search() (image_index, confidence, etc.)."""
    processed_fids: set[int] = field(default_factory=set)
    """Set of frame IDs in top-K results (for spatial fusion neighbor exclusion)."""


def search_scene(
    query: str,
    images: list[np.ndarray],
    frame_ids: list[int],
    retriever,
    top_k: int,
    scene_loader=None,
    robot_position: np.ndarray | None = None,
    sensor_height: float = 0.88,
    min_retrieval_score: float = 0.0,
    query_image: np.ndarray | None = None,
    use_vlm: bool = None,
) -> SearchResult:
    """Steps 1+2: Zero-shot retrieval + VLM re-ranking.

    Handles floor filtering, FAISS cosine search, optional VLM re-ranking,
    and min_score filtering.

    Args:
        query: Natural language query.
        images: Scene images (aligned with frame_ids).
        frame_ids: Frame IDs corresponding to images.
        retriever: HybridRetriever with built FAISS index.
        top_k: Number of candidates to retrieve.
        scene_loader: Optional loader for pose-based floor filtering.
        robot_position: If provided, restricts retrieval to same-floor frames.
        sensor_height: Camera height above agent base (for floor filtering).
        min_retrieval_score: Minimum cosine similarity to keep (0 = no filter).
        query_image: Optional query image for image-mode retrieval (GoatCore).

    Returns:
        SearchResult with ranked candidates and metadata.
    """
    allowed_indices = None
    if robot_position is not None and scene_loader is not None:
        reference_cam_y = robot_position[1] + sensor_height

        all_poses_arr, all_fids = scene_loader.get_all_poses()
        all_cam_ys = all_poses_arr[:, 1, 3]
        on_floor = np.abs(all_cam_ys - reference_cam_y) <= CAMERA_FLOOR_THRESHOLD

        fid_to_idx = {fid: i for i, fid in enumerate(all_fids)}
        allowed_indices = set()
        for kf_idx, fid in enumerate(frame_ids):
            if fid in fid_to_idx and on_floor[fid_to_idx[fid]]:
                allowed_indices.add(kf_idx)

    search_kwargs = {"top_k": top_k}
    if allowed_indices is not None:
        search_kwargs["allowed_indices"] = allowed_indices
    if query_image is not None:
        search_kwargs["query_image"] = query_image
    if use_vlm is not None:
        search_kwargs["use_vlm"] = use_vlm

    results = retriever.search(query, images, **search_kwargs)

    vlm_active = retriever.has_vlm if use_vlm is None else (use_vlm and retriever.has_vlm)
    if min_retrieval_score > 0 and not vlm_active:
        results = [r for i, r in enumerate(results)
                   if i == 0 or r["confidence"] >= min_retrieval_score]

    # Pre-compute top-K frame IDs for spatial fusion neighbor exclusion
    processed_fids = {frame_ids[r["image_index"]] for r in results[:top_k]}

    return SearchResult(
        results=results,
        processed_fids=processed_fids,
    )


def _save_feature_cache(cache_dir, retriever, retrieval_model, model_type,
                        scene_name, num_images, frame_id_array, keyframing=False):
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "features.npy", retriever.image_features)
    np.save(cache_dir / "frame_ids.npy", frame_id_array)
    metadata = {
        "model_type": model_type,
        "feature_extractor_model": retrieval_model,
        "scene_id": scene_name,
        "num_frames": num_images,
        "feature_dim": retriever.image_features.shape[1],
    }
    if keyframing:
        metadata["keyframing"] = True
    with open(cache_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


def build_retriever(
    images,
    scene_name,
    retrieval_model: str,
    vlm_model: str,
    device: str,
    stage1_top_k: int,
    stage2_top_k: int,
    vlm_batch_size: int = 1,
    use_vlm: bool = True,
    existing_retriever=None,
    keyframe_ids=None,
    cache_prefix: str = "hm3d",
    extractor_kwargs: dict | None = None,
):
    """Build feature index and hybrid retriever for a scene.

    Returns a single HybridRetriever with FAISS index and VLM.

    Args:
        images: List of RGB images (np.ndarray) for the scene.
        scene_name: Scene identifier string used in cache directory names.
        retrieval_model: Feature extractor model name (e.g. SigLIP2, CLIP).
        vlm_model: VLM model name for Stage 2 re-ranking.
        device: Torch device string (e.g. "cuda:0").
        stage1_top_k: Number of candidates from FAISS retrieval.
        stage2_top_k: Number of candidates after VLM re-ranking.
        vlm_batch_size: Batch size for parallel VLM calls.
        use_vlm: Whether to enable VLM re-ranking (Stage 2).
        existing_retriever: Previous HybridRetriever to reuse feature extractor
            and VLM from. Its FAISS index is freed before building a new one.
        keyframe_ids: Optional list of frame IDs to use (keyframe subset).
            If provided, tries to load from all-frames cache with subsetting
            first, then falls back to keyframe-specific cache, then builds
            from scratch.
        cache_prefix: Prefix for cache directory names (default "hm3d").
            MP3D uses "mp3d" so caches are separate.

    Returns:
        HybridRetriever with FAISS index built and VLM loaded.
    """
    from src.models.retrieval import HybridRetriever, create_feature_extractor, get_model_type

    model_type = get_model_type(retrieval_model)

    if existing_retriever is not None:
        existing_retriever._reset_index()
        feature_extractor = existing_retriever.feature_extractor
        vlm = existing_retriever.vlm
    else:
        feature_extractor = create_feature_extractor(
            model_type=model_type, model_name=retrieval_model,
            device=device, normalize=True,
            **(extractor_kwargs or {}),
        )
        if use_vlm:
            from src.models.vlm import Qwen2_5VL
            vlm = Qwen2_5VL(model_name=vlm_model, device=device)
        else:
            print("  VLM disabled (--no-vlm): Stage 1 feature-only retrieval")
            vlm = None

    retriever = HybridRetriever(
        feature_extractor=feature_extractor,
        vlm=vlm,
        stage1_top_k=stage1_top_k,
        stage2_top_k=stage2_top_k,
        batch_size=vlm_batch_size,
    )

    project_root = Path(__file__).parent.parent.parent
    allframes_cache = project_root / "results" / "features" / f"{model_type}_{scene_name}_{cache_prefix}_allframes"

    if keyframe_ids is not None:
        # Keyframe mode: try all-frames cache + subset first, then kf cache
        kf_cache = project_root / "results" / "features" / f"{model_type}_{scene_name}_{cache_prefix}_kf{len(keyframe_ids)}"
        if allframes_cache.exists():
            retriever.load_cached_features(str(allframes_cache), keyframe_ids=keyframe_ids)
        elif kf_cache.exists():
            retriever.load_cached_features(str(kf_cache), keyframe_ids=None)
        else:
            retriever.build_index(images)
            _save_feature_cache(
                kf_cache, retriever, retrieval_model, model_type, scene_name,
                len(images), np.array(keyframe_ids), keyframing=True,
            )
    else:
        if allframes_cache.exists():
            retriever.load_cached_features(str(allframes_cache), keyframe_ids=None)
        else:
            retriever.build_index(images)
            _save_feature_cache(
                allframes_cache, retriever, retrieval_model, model_type,
                scene_name, len(images), np.arange(len(images)),
            )

    return retriever
