"""Back-compat shim over :mod:`registry`.

Kept so existing imports (``MODEL_REGISTRY``, ``get_model_type``) keep working.
New code should use :mod:`registry` directly.
"""

from .registry import _HF_DEFAULTS, resolve_model_type

MODEL_REGISTRY = _HF_DEFAULTS


def get_model_type(retrieval_model: str) -> str:
    return resolve_model_type(retrieval_model)


__all__ = ["MODEL_REGISTRY", "get_model_type"]
