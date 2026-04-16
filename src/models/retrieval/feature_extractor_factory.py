"""Factory for creating feature extractors.

All extractor classes are self-registered via ``@register_extractor`` in
:mod:`registry`. This factory just looks up the class and builds it.
"""

from __future__ import annotations

from .base_feature_extractor import BaseFeatureExtractor
from .registry import get_default_model_name, get_extractor_class


def create_feature_extractor(
    model_type: str,
    model_name: str | None = None,
    device: str = "cuda",
    normalize: bool = True,
    batch_size: int = 32,
    **kwargs,
) -> BaseFeatureExtractor:
    """Instantiate a registered feature extractor.

    Args:
        model_type: Registered type key (e.g. ``"siglip2"``, ``"qwen3_vl"``).
        model_name: HF id / path. Defaults to the type's ``default_model_name``.
        device: Torch device string.
        normalize: L2-normalize features (pass-through to the extractor).
        batch_size: Per-forward batch size for image extraction.
        **kwargs: Forwarded to the extractor's constructor — backbone-specific
            knobs (e.g. Qwen's ``instruction``) go here. Unknown kwargs are
            swallowed by extractors that declare ``**_``, so mixed backbones in
            a single sweep are safe.
    """
    model_type = model_type.lower()
    cls = get_extractor_class(model_type)

    if model_name is None:
        model_name = get_default_model_name(model_type)

    return cls(
        model_name=model_name,
        device=device,
        normalize=normalize,
        batch_size=batch_size,
        **kwargs,
    )
