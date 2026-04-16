"""ALIGN feature extractor implementation.

Uses kakaobrain's ALIGN model (EfficientNet + BERT) for image-text embeddings.
"""

import logging
from typing import List

import torch
from PIL import Image
from transformers import AlignModel, AlignProcessor

from .base_feature_extractor import BaseFeatureExtractor
from .registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor(
    model_type="align",
    default_model_name="kakaobrain/align-base",
    name_patterns=("align",),
    friendly_names=("align",),
)
class ALIGNFeatureExtractor(BaseFeatureExtractor):
    """ALIGN-based feature extraction.

    Args:
        model_name: ALIGN model name (default: "kakaobrain/align-base")
        device: Device to run on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features (default: True)
        batch_size: Batch size for feature extraction (default: 32)
    """

    def __init__(
        self,
        model_name: str = "kakaobrain/align-base",
        device: str = "cuda",
        normalize: bool = True,
        batch_size: int = 32,
    ):
        super().__init__(model_name, device, normalize, batch_size)

        self.model = AlignModel.from_pretrained(model_name).to(device)
        self.processor = AlignProcessor.from_pretrained(model_name)
        self.model.eval()

        with torch.no_grad():
            dummy_img = Image.new("RGB", (224, 224))
            inputs = self.processor(images=dummy_img, return_tensors="pt").to(device)
            dummy_feat = self.model.get_image_features(**inputs)
            self.feature_dim = dummy_feat.shape[-1]

        logger.info("ALIGN model loaded (dim=%d)", self.feature_dim)

    def _forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True,
        ).to(self.device)
        return self.model.get_image_features(**inputs)

    def _forward_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text], return_tensors="pt", padding=True, truncation=True,
        ).to(self.device)
        return self.model.get_text_features(**inputs)
