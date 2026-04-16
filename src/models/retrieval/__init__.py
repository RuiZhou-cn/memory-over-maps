"""Retrieval models for fast image search.

Importing this package registers every built-in feature extractor so
``MODEL_REGISTRY`` and ``resolve_model_type`` see all backbones. Third-party
backbones can self-register via :func:`registry.register_extractor` before
calling :func:`create_feature_extractor`.
"""

# Import extractor modules for their @register_extractor side effects.
from . import (
    align_feature_extractor,  # noqa: F401
    clip_feature_extractor,  # noqa: F401
    flava_feature_extractor,  # noqa: F401
    qwen3_vl_feature_extractor,  # noqa: F401
    siglip2_feature_extractor,  # noqa: F401
)
from .align_feature_extractor import ALIGNFeatureExtractor
from .base_feature_extractor import BaseFeatureExtractor
from .clip_feature_extractor import CLIPFeatureExtractor
from .feature_extractor_factory import create_feature_extractor
from .flava_feature_extractor import FLAVAFeatureExtractor
from .hybrid_retriever import HybridRetriever
from .model_registry import MODEL_REGISTRY, get_model_type
from .qwen3_vl_feature_extractor import Qwen3VLFeatureExtractor
from .registry import register_extractor, resolve_model_type
from .siglip2_feature_extractor import SigLIP2FeatureExtractor

__all__ = [
    "ALIGNFeatureExtractor",
    "BaseFeatureExtractor",
    "CLIPFeatureExtractor",
    "FLAVAFeatureExtractor",
    "HybridRetriever",
    "MODEL_REGISTRY",
    "Qwen3VLFeatureExtractor",
    "SigLIP2FeatureExtractor",
    "create_feature_extractor",
    "get_model_type",
    "register_extractor",
    "resolve_model_type",
]
