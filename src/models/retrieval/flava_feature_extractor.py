"""FLAVA feature extractor implementation.

Uses Meta's FLAVA model with unimodal image/text encoders + projection heads.
"""

import logging
from typing import List

import torch
from PIL import Image
from transformers import FlavaModel, FlavaProcessor

from .base_feature_extractor import BaseFeatureExtractor
from .registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor(
    model_type="flava",
    default_model_name="facebook/flava-full",
    name_patterns=("flava",),
    friendly_names=("flava",),
)
class FLAVAFeatureExtractor(BaseFeatureExtractor):
    """FLAVA-based feature extraction.

    Args:
        model_name: FLAVA model name (default: "facebook/flava-full")
        device: Device to run on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features (default: True)
        batch_size: Batch size for feature extraction (default: 32)
    """

    def __init__(
        self,
        model_name: str = "facebook/flava-full",
        device: str = "cuda",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        super().__init__(model_name, device, normalize, batch_size)

        self.model = FlavaModel.from_pretrained(model_name).to(device)
        self.processor = FlavaProcessor.from_pretrained(model_name)
        self.model.eval()

        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224))
            inputs = self.processor(images=dummy_img, return_tensors="pt").to(device)
            img_out = self.model.get_image_features(**inputs)
            pooled = img_out[:, 0, :] if img_out.dim() == 3 else img_out
            projected = self.model.image_projection(pooled)
            self.feature_dim = projected.shape[-1]

        logger.info("FLAVA model loaded (dim=%d)", self.feature_dim)

    def _forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True,
        ).to(self.device)
        img_out = self.model.get_image_features(**inputs)
        pooled = img_out[:, 0, :] if img_out.dim() == 3 else img_out
        return self.model.image_projection(pooled)

    def _forward_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text],
            return_tensors="pt",
            padding="max_length",
            max_length=77,
            truncation=True,
        ).to(self.device)
        text_out = self.model.get_text_features(**inputs)
        pooled = text_out[:, 0, :] if text_out.dim() == 3 else text_out
        return self.model.text_projection(pooled)
