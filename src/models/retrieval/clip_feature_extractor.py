"""CLIP feature extractor implementation."""

import logging
from typing import List

import torch
from PIL import Image
from transformers import CLIPModel, CLIPProcessor

from .base_feature_extractor import BaseFeatureExtractor
from .registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor(
    model_type="clip",
    default_model_name="openai/clip-vit-large-patch14",
    name_patterns=("clip",),
    friendly_names=("clip-large",),
    friendly_variants={"clip-base": "openai/clip-vit-base-patch32"},
)
class CLIPFeatureExtractor(BaseFeatureExtractor):
    """CLIP-based feature extraction.

    Args:
        model_name: CLIP model name (default: "openai/clip-vit-large-patch14")
        device: Device to run on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features (default: True)
        batch_size: Batch size for feature extraction (default: 128)
    """

    def __init__(
        self,
        model_name: str = "openai/clip-vit-large-patch14",
        device: str = "cuda",
        normalize: bool = True,
        batch_size: int = 128,
    ):
        super().__init__(model_name, device, normalize, batch_size)

        self.model = CLIPModel.from_pretrained(model_name).to(device)
        self.processor = CLIPProcessor.from_pretrained(model_name, use_fast=True)
        self.model.eval()

        with torch.no_grad():
            dummy_img = Image.new('RGB', (224, 224))
            inputs = self.processor(images=dummy_img, return_tensors="pt").to(device)
            dummy_feat = self.model.get_image_features(**inputs)
            self.feature_dim = dummy_feat.shape[-1]

        logger.info("CLIP model loaded (dim=%d)", self.feature_dim)

    def _forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True,
        ).to(self.device)
        return self.model.get_image_features(**inputs)

    def _forward_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True,
        ).to(self.device)
        return self.model.get_text_features(**inputs)
