from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm


class BaseFeatureExtractor(ABC):
    """Abstract base class for feature extraction models.

    All feature extractors must implement this interface to work with HybridRetriever.

    Subclasses must set `self.feature_dim` during __init__ and implement
    `_forward_images` and `_forward_text`. The batching, normalization, and
    numpy conversion are handled here.

    Attributes:
        device: Device to run model on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features
        model_name: Name/path of the model
        feature_dim: Dimension of extracted features
        batch_size: Batch size for feature extraction
    """

    def __init__(self, model_name: str, device: str = "cuda", normalize: bool = True,
                 batch_size: int = 128):
        self.model_name = model_name
        self.device = device
        self.normalize = normalize
        self.batch_size = batch_size
        self.feature_dim = None

    @abstractmethod
    def _forward_images(self, pil_images: list[Image.Image]) -> torch.Tensor:
        pass

    @abstractmethod
    def _forward_text(self, text: str) -> torch.Tensor:
        pass

    @property
    def _desc(self) -> str:
        return f"Extracting {self.__class__.__name__.replace('FeatureExtractor', '')} features"

    def _normalize(self, features: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            return features / features.norm(dim=-1, keepdim=True)
        return features

    def extract_image_features(self, images: list[np.ndarray]) -> np.ndarray:
        """Extract features from a list of images.

        Args:
            images: List of RGB images as numpy arrays [H, W, 3]

        Returns:
            features: Feature array of shape [N, feature_dim],
                     L2-normalized if self.normalize=True
        """
        pil_images = [Image.fromarray(img) for img in images]
        all_features = []

        with torch.no_grad():
            for i in tqdm(
                range(0, len(pil_images), self.batch_size),
                desc=self._desc,
                unit="batch",
            ):
                batch = pil_images[i:i + self.batch_size]
                features = self._forward_images(batch)
                all_features.append(self._normalize(features).cpu().numpy())

        return np.vstack(all_features)

    def extract_text_features(self, text: str) -> np.ndarray:
        """Extract features from text query.

        Args:
            text: Query text (e.g., "find the chair")

        Returns:
            features: Feature array of shape [1, feature_dim],
                     L2-normalized if self.normalize=True
        """
        with torch.no_grad():
            features = self._forward_text(text)
            return self._normalize(features).cpu().numpy()
