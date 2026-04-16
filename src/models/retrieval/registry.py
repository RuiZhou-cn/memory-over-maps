"""Self-registering retrieval backbone registry.

Adding a new backbone is one file with one decorator:

    @register_extractor(
        model_type="my_backbone",
        default_model_name="org/my-backbone-base",
        name_patterns=("my-backbone",),
        friendly_names=("my-backbone-base",),
        friendly_variants={"my-backbone-large": "org/my-backbone-large"},
    )
    class MyBackboneFeatureExtractor(BaseFeatureExtractor):
        ...

The module must be imported somewhere (see ``retrieval/__init__.py``) so the
decorator fires at import time.

To use a non-default backbone, override ``retrieval.model`` (HF id or friendly
name) in any config inheriting ``configs/base.yaml``. Backbone-specific knobs
go in ``retrieval.extractor_kwargs``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence

from .base_feature_extractor import BaseFeatureExtractor


@dataclass(frozen=True)
class ExtractorSpec:
    model_type: str
    cls: type[BaseFeatureExtractor]
    default_model_name: str
    name_patterns: tuple[str, ...] = field(default_factory=tuple)
    friendly_names: tuple[str, ...] = field(default_factory=tuple)


# model_type -> ExtractorSpec
_EXTRACTORS: dict[str, ExtractorSpec] = {}

# friendly name -> model_type (e.g. "clip-base" -> "clip")
_FRIENDLY_NAMES: dict[str, str] = {}

# friendly name -> {"model_type": ..., "model_name": ..., "default"?: True}
# Exposed to external callers (eval_sunrgbd) as MODEL_REGISTRY.
_HF_DEFAULTS: dict[str, dict] = {}


def register_extractor(
    model_type: str,
    default_model_name: str,
    name_patterns: Sequence[str] = (),
    friendly_names: Sequence[str] = (),
    friendly_variants: Mapping[str, str] | None = None,
    is_default: bool = False,
):
    """Decorator: register a ``BaseFeatureExtractor`` subclass.

    Args:
        model_type: Internal key (e.g. ``"siglip2"``). Used by ``get_model_type``
            and the factory.
        default_model_name: HF id used when only ``model_type`` is given.
        name_patterns: Substrings matched case-insensitively against unknown
            HF ids to auto-resolve ``model_type``.
        friendly_names: Short aliases (e.g. ``"siglip2"``) that map to
            ``default_model_name``. Each becomes a key in ``MODEL_REGISTRY``.
        friendly_variants: Extra ``{alias: hf_id}`` pairs sharing the same
            ``model_type`` (e.g. ``{"clip-base": "openai/clip-vit-base-patch32"}``).
        is_default: Mark this backbone's default entry with ``"default": True``
            in ``MODEL_REGISTRY``. At most one backbone should set this.
    """

    def wrap(cls: type[BaseFeatureExtractor]) -> type[BaseFeatureExtractor]:
        if not issubclass(cls, BaseFeatureExtractor):
            raise TypeError(
                f"{cls.__name__} must subclass BaseFeatureExtractor"
            )
        spec = ExtractorSpec(
            model_type=model_type,
            cls=cls,
            default_model_name=default_model_name,
            name_patterns=tuple(p.lower() for p in name_patterns),
            friendly_names=tuple(friendly_names),
        )
        _EXTRACTORS[model_type] = spec

        for fname in friendly_names:
            _FRIENDLY_NAMES[fname] = model_type
            entry = {"model_type": model_type, "model_name": default_model_name}
            if is_default:
                entry["default"] = True
            _HF_DEFAULTS[fname] = entry

        for fname, hf_id in (friendly_variants or {}).items():
            _FRIENDLY_NAMES[fname] = model_type
            _HF_DEFAULTS[fname] = {"model_type": model_type, "model_name": hf_id}

        return cls

    return wrap


def resolve_model_type(retrieval_model: str) -> str:
    """Derive ``model_type`` from a friendly name or HF id.

    Resolution order:
    1. Friendly-name exact match (``"siglip2"``, ``"clip-base"``).
    2. Exact HF-id match against any registered entry.
    3. Case-insensitive substring match on ``name_patterns``.
    Raises ``ValueError`` if nothing matches.
    """
    if retrieval_model in _FRIENDLY_NAMES:
        return _FRIENDLY_NAMES[retrieval_model]
    for entry in _HF_DEFAULTS.values():
        if entry["model_name"] == retrieval_model:
            return entry["model_type"]
    name_lower = retrieval_model.lower()
    for spec in _EXTRACTORS.values():
        if any(p in name_lower for p in spec.name_patterns):
            return spec.model_type
    raise ValueError(f"Cannot determine model_type for: {retrieval_model}")


def get_extractor_class(model_type: str) -> type[BaseFeatureExtractor]:
    try:
        return _EXTRACTORS[model_type.lower()].cls
    except KeyError as e:
        supported = list(_EXTRACTORS.keys())
        raise ValueError(
            f"Unsupported model type: '{model_type}'. Supported: {supported}"
        ) from e


def get_default_model_name(model_type: str) -> str:
    return _EXTRACTORS[model_type.lower()].default_model_name
