"""SAM3 (Segment Anything Model 3) wrapper for text-prompted segmentation.

Requires: pip install -e . from facebookresearch/sam3 clone.
See scripts/install.sh for installation.

Note: SAM3 requires CUDA — the upstream model only supports GPU execution.
"""

from __future__ import annotations

import logging

import numpy as np
import torch
from PIL import Image

logger = logging.getLogger(__name__)


class SAM3Segmenter:
    """Text-prompted segmentation using SAM3. Paper Step 3 (Sec III-D)."""

    def __init__(self, device: str = "cuda", confidence_threshold: float = 0.0,
                 batch_size: int = 5):
        """Load SAM3 model.

        Args:
            device: Device to run on. Must be "cuda" — SAM3 only supports
                GPU execution.
            confidence_threshold: Minimum confidence for detections.
                Default 0.0 — we select best by argmax, so even
                low-confidence masks beat the full-image fallback.
            batch_size: Max images per GPU batch for segment_batch().
        """
        try:
            import warnings
            warnings.filterwarnings("ignore", message="pkg_resources is deprecated")
            warnings.filterwarnings(
                "ignore", message="Please use the new API settings to control TF32"
            )
            from sam3.model.sam3_image_processor import Sam3Processor
            from sam3.model_builder import build_sam3_image_model
        except ImportError:
            raise ImportError(
                "SAM3 not installed. Run: bash scripts/install.sh"
            )

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.batch_size = batch_size

        self.model = build_sam3_image_model(device=device, load_from_HF=True)
        self.processor = Sam3Processor(
            self.model,
            device=device,
            confidence_threshold=confidence_threshold,
        )

        self._batch_transform = None
        self._batch_postprocessor = None
        self._batch_collate = None
        self._batch_copy_to_device = None
        self._batch_datapoint_classes = None

        self._cache: dict[tuple[str, int], dict] = {}

    def _ensure_batch_components(self):
        """Lazy-init transform, postprocessor, and imports for batch inference."""
        if self._batch_transform is not None:
            return
        from sam3.eval.postprocessors import PostProcessImage
        from sam3.model.utils.misc import copy_data_to_device
        from sam3.train.data.collator import collate_fn_api as collate
        from sam3.train.data.sam3_image_dataset import (
            Datapoint,
            FindQueryLoaded,
            InferenceMetadata,
        )
        from sam3.train.data.sam3_image_dataset import (
            Image as SAMImage,
        )
        from sam3.train.transforms.basic_for_api import (
            ComposeAPI,
            NormalizeAPI,
            RandomResizeAPI,
            ToTensorAPI,
        )

        res = self.processor.resolution
        self._batch_transform = ComposeAPI(transforms=[
            RandomResizeAPI(sizes=res, max_size=res, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])
        self._batch_postprocessor = PostProcessImage(
            max_dets_per_img=-1,
            iou_type="segm",
            use_original_sizes_box=True,
            use_original_sizes_mask=True,
            convert_mask_to_rle=False,
            detection_threshold=self.confidence_threshold,
            to_cpu=True,
        )
        self._batch_collate = collate
        self._batch_copy_to_device = copy_data_to_device
        self._batch_datapoint_classes = (Datapoint, FindQueryLoaded, SAMImage, InferenceMetadata)

    def clear_cache(self):
        """Clear the segmentation cache. Call between scenes."""
        self._cache.clear()

    def segment(
        self,
        image,
        text_query: str,
        cache_key: tuple[str, int] | None = None,
    ) -> dict:
        """Segment objects matching text query in an image.

        Args:
            image: PIL Image or numpy array (H, W, 3) RGB
            text_query: Natural language description of target object
            cache_key: Optional (query, frame_id) tuple for caching.
                When provided, results are cached and reused for the
                same key within a scene (call clear_cache() between scenes).

        Returns:
            dict with keys:
                best_mask: highest-scoring mask (H, W) or None
                best_score: highest score or 0.0
        """
        if cache_key is not None and cache_key in self._cache:
            return self._cache[cache_key]

        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)

        state = self.processor.set_image(image)
        state = self.processor.set_text_prompt(text_query, state)

        result = _extract_best(state.get('masks', None), state.get('scores', None))

        if cache_key is not None:
            self._cache[cache_key] = result

        return result

    def segment_batch(
        self,
        images: list,
        text_query: str,
        cache_keys: list[tuple[str, int]] | None = None,
        max_batch_size: int | None = None,
    ) -> list[dict]:
        """Batch-segment multiple images with the same text query.

        Uses SAM3's native batched inference (Datapoint/collate/model forward)
        for significantly faster throughput vs sequential segment() calls.

        Args:
            images: List of PIL Images or numpy arrays (H, W, 3) RGB.
            text_query: Natural language description of target object.
            cache_keys: Optional list of (query, frame_id) tuples for caching.
                Must be same length as images if provided.
            max_batch_size: Max images per GPU batch to limit VRAM usage.

        Returns:
            List of dicts (same format as segment()), one per input image.
        """
        if max_batch_size is None:
            max_batch_size = self.batch_size

        n = len(images)
        if n == 0:
            return []

        results = [None] * n
        to_process = []  # (index_in_results, pil_image)

        for i in range(n):
            if cache_keys is not None and cache_keys[i] in self._cache:
                results[i] = self._cache[cache_keys[i]]
            else:
                img = images[i]
                if isinstance(img, np.ndarray):
                    img = Image.fromarray(img)
                to_process.append((i, img))

        if not to_process:
            return results

        # Fallback to sequential segment() if only 1 image or max_batch_size=1
        if len(to_process) == 1 or max_batch_size <= 1:
            for idx, img in to_process:
                ck = cache_keys[idx] if cache_keys is not None else None
                results[idx] = self.segment(img, text_query, cache_key=ck)
            return results

        self._ensure_batch_components()
        Datapoint, FindQueryLoaded, SAMImage, InferenceMetadata = self._batch_datapoint_classes

        for chunk_start in range(0, len(to_process), max_batch_size):
            chunk = to_process[chunk_start:chunk_start + max_batch_size]
            datapoints = []
            query_id_to_idx = {}  # query_id -> index in results

            for j, (idx, pil_img) in enumerate(chunk):
                query_id = chunk_start + j
                query_id_to_idx[query_id] = idx

                w, h = pil_img.size
                dp = Datapoint(find_queries=[], images=[])
                dp.images = [SAMImage(data=pil_img, objects=[], size=[h, w])]
                dp.find_queries.append(FindQueryLoaded(
                    query_text=text_query,
                    image_id=0,
                    object_ids_output=[],
                    is_exhaustive=True,
                    query_processing_order=0,
                    inference_metadata=InferenceMetadata(
                        coco_image_id=query_id,
                        original_image_id=query_id,
                        original_category_id=1,
                        original_size=[w, h],
                        object_id=0,
                        frame_index=0,
                    ),
                ))
                datapoints.append(self._batch_transform(dp))

            batch = self._batch_collate(datapoints, dict_key="batch")["batch"]
            batch = self._batch_copy_to_device(
                batch, torch.device(self.device), non_blocking=True,
            )

            with torch.inference_mode():
                output = self.model(batch)
                processed = self._batch_postprocessor.process_results(
                    output, batch.find_metadatas,
                )

            for query_id, idx in query_id_to_idx.items():
                proc = processed.get(query_id)
                if proc is not None:
                    result = _extract_best(proc.get("masks"), proc.get("scores"))
                else:
                    result = _EMPTY_RESULT
                results[idx] = result
                if cache_keys is not None:
                    self._cache[cache_keys[idx]] = result

        for i in range(n):
            if results[i] is None:
                results[i] = _EMPTY_RESULT

        return results


_EMPTY_RESULT = {'best_mask': None, 'best_score': 0.0}


def _extract_best(mask_data, score_data) -> dict:
    """Extract the highest-scoring mask from raw mask/score arrays.

    Handles tensors, numpy arrays, and various dimensionalities from both
    the Sam3Processor (segment) and PostProcessImage (segment_batch) paths.
    """
    if mask_data is None or len(mask_data) == 0:
        return _EMPTY_RESULT

    if isinstance(mask_data, torch.Tensor):
        mask_data = mask_data.cpu().numpy()

    if score_data is not None:
        if hasattr(score_data, 'cpu'):
            score_data = score_data.cpu().numpy()
        if isinstance(score_data, np.ndarray):
            if score_data.ndim == 0:
                score_data = score_data[np.newaxis]
            scores = score_data.flatten()
        else:
            scores = np.array([float(s) for s in score_data])
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])
    else:
        best_idx = 0
        best_score = 1.0

    if isinstance(mask_data, np.ndarray):
        if mask_data.ndim == 4:
            best_mask = mask_data[best_idx, 0].astype(bool)
        elif mask_data.ndim == 3:
            best_mask = mask_data[best_idx].astype(bool)
        elif mask_data.ndim == 2:
            best_mask = mask_data.astype(bool)
        else:
            return _EMPTY_RESULT
    else:
        m = mask_data[best_idx]
        if hasattr(m, 'cpu'):
            m = m.cpu().numpy()
        if m.ndim == 3:
            m = m.squeeze(0)
        best_mask = m.astype(bool)

    return {'best_mask': best_mask, 'best_score': best_score}
