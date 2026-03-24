"""Hybrid retrieval system: FAISS feature search + optional VLM verification.

When vlm=None, operates as a pure feature retriever.
When vlm is set, runs two-stage: FAISS top-K -> VLM re-ranking.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from typing import Any

import faiss
import numpy as np

from .base_feature_extractor import BaseFeatureExtractor

logger = logging.getLogger(__name__)


class HybridRetriever:
    """FAISS feature retriever with optional VLM re-ranking.

    Implements Paper Steps 1+2 (Sec III-C).

    Args:
        feature_extractor: Feature extractor instance (CLIP, SigLIP, etc.)
        vlm: Optional VLM instance for stage-2 re-ranking (None = feature-only)
        use_gpu: Whether to use GPU for FAISS indexing (default: False)
        stage1_top_k: Number of candidates from stage 1 (default: 20)
        stage2_top_k: Final number of results after VLM re-ranking (default: 5)
        batch_size: Batch size for VLM processing (default: 5)
    """

    def __init__(
        self,
        feature_extractor: BaseFeatureExtractor,
        vlm: Any = None,
        use_gpu: bool = False,
        stage1_top_k: int = 20,
        stage2_top_k: int = 5,
        batch_size: int = 5,
    ):
        self.feature_extractor = feature_extractor
        self.vlm = vlm
        self.use_gpu = use_gpu

        self.index = None
        self.image_features = None
        self.frame_ids = None
        self.num_images = 0

        self.stage1_top_k = stage1_top_k
        self.stage2_top_k = stage2_top_k
        self.batch_size = batch_size

    @property
    def has_vlm(self) -> bool:
        """Whether this retriever has a VLM for stage-2 re-ranking."""
        return self.vlm is not None

    def build_index_from_features(
        self,
        features: np.ndarray,
        frame_ids: np.ndarray | None = None,
    ):
        """Build FAISS index from pre-computed features.

        Args:
            features: Pre-computed features [N, feature_dim]
            frame_ids: Optional frame IDs corresponding to features
        """
        feature_dim = self.feature_extractor.feature_dim

        self._reset_index()

        self.image_features = features
        self.num_images = len(features)
        self.frame_ids = frame_ids

        if self.feature_extractor.normalize:
            self.index = faiss.IndexFlatIP(feature_dim)
        else:
            self.index = faiss.IndexFlatL2(feature_dim)

        if self.use_gpu and faiss.get_num_gpus() > 0:
            res = faiss.StandardGpuResources()
            self.index = faiss.index_cpu_to_gpu(res, 0, self.index)

        self.index.add(features.astype(np.float32))

    def _reset_index(self):
        if self.index is not None:
            self.index.reset()
            self.index = None
        self.image_features = None
        self.frame_ids = None

    def load_cached_features(
        self,
        features_dir: str,
        keyframe_ids: list[int] | None = None,
    ):
        """Load pre-computed features from disk.

        Args:
            features_dir: Directory containing features.npy, frame_ids.npy, metadata.json
            keyframe_ids: Optional list of keyframe IDs to select (if None, use all)

        Returns:
            features: Loaded features array [N, feature_dim]
        """
        features_dir = Path(features_dir)

        metadata_path = features_dir / "metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        cached_model = metadata.get('feature_extractor_model')
        extractor = self.feature_extractor

        if cached_model and cached_model != extractor.model_name:
            logger.warning("Cached features from %s, current model: %s", cached_model, extractor.model_name)

        features_path = features_dir / "features.npy"
        frame_ids_path = features_dir / "frame_ids.npy"

        if not features_path.exists():
            raise FileNotFoundError(f"Features not found: {features_path}")

        all_features = np.load(features_path)
        all_frame_ids = np.load(frame_ids_path) if frame_ids_path.exists() else None

        if keyframe_ids is not None and all_frame_ids is not None:
            frame_id_to_idx = {fid: idx for idx, fid in enumerate(all_frame_ids)}

            selected_indices = []
            missing_ids = []

            for kf_id in keyframe_ids:
                if kf_id in frame_id_to_idx:
                    selected_indices.append(frame_id_to_idx[kf_id])
                else:
                    missing_ids.append(kf_id)

            if missing_ids:
                raise ValueError(f"Keyframe IDs not found in cached features: {missing_ids}")

            features = all_features[selected_indices]
            frame_ids = np.array(keyframe_ids)
        else:
            features = all_features
            frame_ids = all_frame_ids

        self.build_index_from_features(features, frame_ids)
        return features

    def build_index(self, images: list[np.ndarray]):
        """Build FAISS index from images.

        Args:
            images: List of RGB images (np.ndarray)
        """
        features = self.feature_extractor.extract_image_features(images)
        self.build_index_from_features(features)
        self.num_images = len(images)

    def search_features(
        self,
        query: str,
        top_k: int = 20,
        return_scores: bool = True,
        similarity_threshold: float = 0.9,
        calibrated_scores: bool = True,
        allowed_indices: set = None,
        query_image: "np.ndarray | None" = None,
    ):
        """Search for images matching the query using FAISS only.

        Args:
            query: Text query. Used when query_image is None.
            top_k: Number of top results to return
            return_scores: Whether to return similarity scores
            similarity_threshold: Skip images with cosine similarity above this (dedup)
            calibrated_scores: If True and model supports it, return calibrated probabilities
            allowed_indices: If provided, only return results whose index is in this set
            query_image: If provided, use image-to-image retrieval instead of text-to-image

        Returns:
            indices: List of image indices (sorted by relevance)
            scores: List of similarity scores (if return_scores=True)
        """
        if self.index is None:
            raise ValueError("Index not built. Call build_index() or load_cached_features() first.")

        if query_image is not None:
            query_features = self.feature_extractor.extract_image_features([query_image])
        else:
            query_features = self.feature_extractor.extract_text_features(query)

        needs_full_scan = similarity_threshold < 1.0 or allowed_indices is not None
        search_k = len(self.image_features) if needs_full_scan else top_k
        scores, indices = self.index.search(query_features.astype(np.float32), search_k)

        indices = indices[0].tolist()
        scores = scores[0].tolist()

        if calibrated_scores and hasattr(self.feature_extractor, 'similarity_to_probability'):
            scores_array = np.array(scores)
            calibrated = self.feature_extractor.similarity_to_probability(scores_array)
            scores = calibrated.tolist()

        if similarity_threshold >= 1.0 and allowed_indices is None:
            indices = indices[:top_k]
            scores = scores[:top_k]
            return list(indices), scores if return_scores else None

        filtered_indices = []
        filtered_scores = []

        for idx, score in zip(indices, scores):
            if len(filtered_indices) >= top_k:
                break

            if allowed_indices is not None and idx not in allowed_indices:
                continue

            is_unique = True
            current_features = self.image_features[idx]

            for prev_idx in filtered_indices:
                similarity = np.dot(current_features, self.image_features[prev_idx])

                if similarity > similarity_threshold:
                    is_unique = False
                    break

            if is_unique:
                filtered_indices.append(idx)
                filtered_scores.append(score)

        return filtered_indices, filtered_scores if return_scores else None

    def search(
        self,
        query: str,
        images: list[np.ndarray],
        allowed_indices: set = None,
        top_k: int = None,
        query_image: "np.ndarray | None" = None,
        use_vlm: bool = None,
    ) -> list[dict[str, Any]]:
        """Run retrieval pipeline.

        When VLM is set: two-stage hybrid retrieval (FAISS -> VLM re-ranking).
        When VLM is None: feature-only retrieval, results wrapped in dict format.

        Args:
            query: Text query
            images: List of all RGB images
            allowed_indices: If provided, only return results in this set
            top_k: Override for stage1_top_k and stage2_top_k. If None, uses self defaults.
            query_image: If provided, use image-to-image FAISS retrieval
            use_vlm: Override VLM usage. If None, uses VLM when available.

        Returns:
            List of result dicts sorted by confidence/score.
        """
        vlm_enabled = self.vlm is not None if use_vlm is None else (use_vlm and self.vlm is not None)
        if vlm_enabled:
            return self._search_with_vlm(query, images, allowed_indices, top_k=top_k, query_image=query_image)
        else:
            return self._search_features_only(query, images, allowed_indices, top_k, query_image=query_image)

    def _search_features_only(
        self,
        query: str,
        images: list[np.ndarray],
        allowed_indices: set = None,
        top_k: int = None,
        query_image: "np.ndarray | None" = None,
    ) -> list[dict[str, Any]]:
        """Feature-only search, returning results in dict format."""
        effective_k = top_k if top_k is not None else self.stage1_top_k

        indices, scores = self.search_features(
            query, top_k=effective_k, allowed_indices=allowed_indices,
            query_image=query_image,
        )
        results = [
            {
                "image_index": idx,
                "stage1_score": float(score),
                "stage1_rank": rank,
                "confidence": float(score),
                "detected": True,
                "response": "",
            }
            for rank, (idx, score) in enumerate(zip(indices, scores))
        ]

        return results

    def _search_with_vlm(
        self,
        query: str,
        images: list[np.ndarray],
        allowed_indices: set = None,
        top_k: int = None,
        query_image: "np.ndarray | None" = None,
    ) -> list[dict[str, Any]]:
        """Two-stage hybrid retrieval: FAISS -> VLM re-ranking."""
        n_pool = len(allowed_indices) if allowed_indices is not None else len(images)
        effective_stage1 = top_k if top_k is not None else self.stage1_top_k
        effective_stage2 = top_k if top_k is not None else self.stage2_top_k

        candidate_indices, stage1_scores = self.search_features(
            query,
            top_k=min(effective_stage1, n_pool),
            allowed_indices=allowed_indices,
            similarity_threshold=1.0,
            query_image=query_image,
        )

        vlm_results = self._process_with_batch(
            candidate_indices, stage1_scores, query, images,
        )

        detected_results = [r for r in vlm_results if r['detected']]

        if detected_results:
            detected_results.sort(key=lambda x: x['confidence'], reverse=True)
            final_results = detected_results[:effective_stage2]
        else:
            vlm_results.sort(key=lambda x: x['confidence'], reverse=True)
            final_results = vlm_results[:1]

        return final_results

    def _process_with_batch(
        self,
        candidate_indices: list[int],
        stage1_scores: list[float],
        query: str,
        images: list[np.ndarray],
    ) -> list[dict[str, Any]]:
        """Process candidates using local model batch processing."""
        vlm_results = []

        for batch_start in range(0, len(candidate_indices), self.batch_size):
            batch_end = min(batch_start + self.batch_size, len(candidate_indices))
            batch_indices = candidate_indices[batch_start:batch_end]
            batch_scores = stage1_scores[batch_start:batch_end]

            batch_images = [images[idx] for idx in batch_indices]

            try:
                batch_results = self.vlm.batch_query(batch_images, query, batch_size=self.batch_size)

                for i, result in enumerate(batch_results):
                    result['image_index'] = batch_indices[i]
                    result['stage1_score'] = batch_scores[i]
                    result['stage1_rank'] = batch_start + i + 1
                    vlm_results.append(result)

            except Exception as e:
                logger.error("Error processing batch %d-%d: %s", batch_start, batch_end, e)
                for i in range(len(batch_indices)):
                    vlm_results.append({
                        'image_index': batch_indices[i],
                        'stage1_score': batch_scores[i],
                        'stage1_rank': batch_start + i + 1,
                        'response': f"Error: {str(e)}",
                        'confidence': 0.0,
                        'detected': False,
                    })

        return vlm_results

