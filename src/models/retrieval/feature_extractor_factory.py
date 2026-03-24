"""Factory for creating feature extractors."""

from __future__ import annotations

import importlib

from .base_feature_extractor import BaseFeatureExtractor
from .model_registry import MODEL_REGISTRY

DEFAULT_MODELS = {
    entry["model_type"]: entry["model_name"]
    for entry in MODEL_REGISTRY.values()
    if entry.get("default", False)
}

_EXTRACTORS = {
    "clip": (".clip_feature_extractor", "CLIPFeatureExtractor"),
    "siglip2": (".siglip2_feature_extractor", "SigLIP2FeatureExtractor"),
    "align": (".align_feature_extractor", "ALIGNFeatureExtractor"),
    "flava": (".flava_feature_extractor", "FLAVAFeatureExtractor"),
}


def create_feature_extractor(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    normalize: bool = True,
    batch_size: int = 32,
    **kwargs,
) -> BaseFeatureExtractor:
    """Factory function to create feature extractors.

    Args:
        model_type: Type of feature extractor ("clip", "siglip2", "align", "flava")
        model_name: Model name/path (optional, uses default if None)
        device: Device to run on (e.g. "cuda" or "cpu")
        normalize: Whether to L2-normalize features (default: True)
        batch_size: Batch size for feature extraction (default: 32)

    Returns:
        Instance of feature extractor
    """
    model_type = model_type.lower()

    if model_name is None:
        if model_type not in DEFAULT_MODELS:
            raise ValueError(
                f"Unsupported model type: '{model_type}'. "
                f"Supported types: {list(DEFAULT_MODELS.keys())}"
            )
        model_name = DEFAULT_MODELS[model_type]

    if model_type in _EXTRACTORS:
        module_path, class_name = _EXTRACTORS[model_type]
        module = importlib.import_module(module_path, package=__package__)
        cls = getattr(module, class_name)
        return cls(
            model_name=model_name, device=device,
            normalize=normalize, batch_size=batch_size, **kwargs,
        )
    else:
        raise ValueError(
            f"Unsupported model type: '{model_type}'. "
            f"Supported types: {list(DEFAULT_MODELS.keys())}"
        )
