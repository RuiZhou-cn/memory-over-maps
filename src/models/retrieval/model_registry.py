"""Model registry mapping friendly names to model configurations."""


MODEL_REGISTRY = {
    "clip-base": {
        "model_type": "clip",
        "model_name": "openai/clip-vit-base-patch32",
    },
    "clip-large": {
        "model_type": "clip",
        "model_name": "openai/clip-vit-large-patch14",
        "default": True,
    },
    "align": {
        "model_type": "align",
        "model_name": "kakaobrain/align-base",
        "default": True,
    },
    "flava": {
        "model_type": "flava",
        "model_name": "facebook/flava-full",
        "default": True,
    },
    "siglip2": {
        "model_type": "siglip2",
        "model_name": "google/siglip2-so400m-patch14-384",
        "default": True,
    },
}


def get_model_type(retrieval_model: str) -> str:
    """Derive model_type ('siglip2', 'clip', ...) from a retrieval model name.

    Resolution order:
    1. Registry key exact match (e.g. "clip-base").
    2. Registry model_name exact match (e.g. "openai/clip-vit-base-patch32").
    3. Substring matching on the lowercased name.
    """
    if retrieval_model in MODEL_REGISTRY:
        return MODEL_REGISTRY[retrieval_model]["model_type"]
    for entry in MODEL_REGISTRY.values():
        if entry["model_name"] == retrieval_model:
            return entry["model_type"]
    name_lower = retrieval_model.lower()
    if "siglip" in name_lower:
        return "siglip2"
    if "clip" in name_lower:
        return "clip"
    if "align" in name_lower:
        return "align"
    if "flava" in name_lower:
        return "flava"
    raise ValueError(f"Cannot determine model_type for: {retrieval_model}")


