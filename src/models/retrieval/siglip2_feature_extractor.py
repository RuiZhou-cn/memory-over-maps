"""SigLIP2 feature extractor implementation.

SigLIP2 uses learned temperature and bias parameters for calibrated
similarity scores: prob = sigmoid(cosine_sim * temp + bias).
"""

import logging
from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import AutoModel, AutoProcessor

from .base_feature_extractor import BaseFeatureExtractor
from .registry import register_extractor

logger = logging.getLogger(__name__)


@register_extractor(
    model_type="siglip2",
    default_model_name="google/siglip2-so400m-patch14-384",
    name_patterns=("siglip",),
    friendly_names=("siglip2",),
    is_default=True,
)
class SigLIP2FeatureExtractor(BaseFeatureExtractor):
    """SigLIP2-based feature extraction with calibrated similarity scores.

    Text inputs use padding="max_length" and max_length=64 as required by
    SigLIP2's training configuration.

    Args:
        model_name: SigLIP2 model name (default: "google/siglip2-so400m-patch14-384")
        device: Device to run on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features (default: True)
        batch_size: Batch size for feature extraction (default: 128)
    """

    def __init__(
        self,
        model_name: str = "google/siglip2-so400m-patch14-384",
        device: str = "cuda",
        normalize: bool = True,
        batch_size: int = 128,
    ):
        super().__init__(model_name, device, normalize, batch_size)

        logger.debug("Loading SigLIP2 model: %s", model_name)
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True).to(device)
        self.processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True, use_fast=True)
        self.model.eval()

        if hasattr(self.model, 'logit_scale') and hasattr(self.model, 'logit_bias'):
            self.logit_scale = self.model.logit_scale.exp().item()
            self.logit_bias = self.model.logit_bias.item()
        else:
            self.logit_scale = 1.0
            self.logit_bias = 0.0

        with torch.no_grad():
            dummy_img = Image.new('RGB', (224, 224))
            inputs = self.processor(images=dummy_img, return_tensors="pt").to(device)
            dummy_feat = self.model.get_image_features(**inputs)
            self.feature_dim = dummy_feat.shape[-1]

    def _forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        inputs = self.processor(
            images=pil_images, return_tensors="pt", padding=True,
        ).to(self.device)
        return self.model.get_image_features(**inputs)

    def _forward_text(self, text: str) -> torch.Tensor:
        inputs = self.processor(
            text=[text.lower()],
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        ).to(self.device)
        return self.model.get_text_features(**inputs)

    def similarity_to_probability(self, similarity_scores: np.ndarray) -> np.ndarray:
        """Convert cosine similarity to calibrated probabilities via sigmoid."""
        logits = similarity_scores * self.logit_scale + self.logit_bias
        return 1.0 / (1.0 + np.exp(-logits))
