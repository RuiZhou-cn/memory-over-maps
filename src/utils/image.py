"""Image preprocessing utilities (resize, shape alignment)."""

from typing import Optional

import cv2
import numpy as np


def resize_image(rgb: np.ndarray, max_size: Optional[int]) -> np.ndarray:
    """Resize image so longest side <= *max_size* (INTER_AREA).

    Returns the original array unchanged when no resize is needed.
    """
    if not max_size:
        return rgb
    h, w = rgb.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    return rgb


def ensure_depth_shape(depth: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize depth map to (target_h, target_w) if shapes differ."""
    if depth.shape[0] != target_h or depth.shape[1] != target_w:
        depth = cv2.resize(depth, (target_w, target_h), interpolation=cv2.INTER_LINEAR)
    return depth


def ensure_mask_shape(mask: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize binary mask to (target_h, target_w) if shapes differ."""
    if mask.shape[0] != target_h or mask.shape[1] != target_w:
        mask = cv2.resize(
            mask.astype(np.uint8),
            (target_w, target_h),
            interpolation=cv2.INTER_NEAREST,
        ).astype(bool)
    return mask


def resize_images_batch(images: list, max_size: int) -> list:
    """Resize a list of images so longest side <= max_size (INTER_AREA)."""
    if max_size <= 0:
        return images
    return [resize_image(img, max_size) for img in images]
