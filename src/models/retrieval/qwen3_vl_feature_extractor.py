"""Qwen3-VL-Embedding feature extractor.

Opt-in backbone. Requires the Qwen3-VL-Embedding source cloned to
``third_party/qwen3-vl-embedding`` via ``bash scripts/install.sh`` (step ``[6/6]``).
The upstream package pins ``requires-python >=3.11`` which conflicts with this
env's Python 3.9 (pinned by habitat-sim), so we load ``Qwen3VLEmbedder`` by
file path instead of installing it as a package. All runtime deps
(transformers, qwen_vl_utils, torch) are already installed here.

Each cache file in ``results/features/`` is keyed on ``model_type`` and the HF
id, so switching between SigLIP2 and Qwen3-VL will build a separate index
(and leave the other one on disk — clean up manually if disk is tight).
"""

from __future__ import annotations

import importlib.util
import logging
from pathlib import Path
from typing import List

import numpy as np
import torch
from PIL import Image

from .base_feature_extractor import BaseFeatureExtractor
from .registry import register_extractor

logger = logging.getLogger(__name__)

_QWEN3VL_SRC = (
    Path(__file__).resolve().parents[3]
    / "third_party"
    / "qwen3-vl-embedding"
    / "src"
    / "models"
    / "qwen3_vl_embedding.py"
)


def _pick_attn_implementation() -> str:
    try:
        import flash_attn  # noqa: F401
        return "flash_attention_2"
    except ImportError:
        return "sdpa"


def _load_qwen3vl_embedder_cls():
    """Load Qwen3VLEmbedder from the third_party source file directly.

    Bypasses the upstream pyproject's Python >=3.11 pin by avoiding pip install.
    The loaded module is cached under ``sys.modules`` on a unique key so repeated
    loads reuse it.
    """
    import sys

    cache_key = "_qwen3vl_embedding_module"
    if cache_key in sys.modules:
        return sys.modules[cache_key].Qwen3VLEmbedder

    if not _QWEN3VL_SRC.exists():
        raise ImportError(
            f"Qwen3-VL-Embedding source not found at {_QWEN3VL_SRC}. "
            "Run the optional step [6/6] in scripts/install.sh, which clones "
            "https://github.com/QwenLM/Qwen3-VL-Embedding into third_party/."
        )

    spec = importlib.util.spec_from_file_location(cache_key, _QWEN3VL_SRC)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[cache_key] = mod
    spec.loader.exec_module(mod)
    return mod.Qwen3VLEmbedder


@register_extractor(
    model_type="qwen3_vl",
    default_model_name="Qwen/Qwen3-VL-Embedding-2B",
    name_patterns=("qwen3-vl", "qwen3_vl"),
    friendly_names=("qwen3-vl-2b",),
    friendly_variants={"qwen3-vl-8b": "Qwen/Qwen3-VL-Embedding-8B"},
)
class Qwen3VLFeatureExtractor(BaseFeatureExtractor):
    """Qwen3-VL-Embedding in a shared image/text embedding space.

    Variants:
        - ``Qwen/Qwen3-VL-Embedding-2B`` — 2048-dim, ~5GB bf16.
        - ``Qwen/Qwen3-VL-Embedding-8B`` — 4096-dim, ~16GB bf16.

    Args:
        model_name: HF id (default: 2B variant).
        device: Torch device string.
        normalize: Kept for interface symmetry — Qwen3VLEmbedder already
            L2-normalizes outputs, so the base class's re-normalization is a
            no-op on unit vectors.
        batch_size: Per-forward image batch size. Default 16 (safe on 24GB +
            Qwen2.5-VL-7B re-ranker; raise if you have headroom).
        instruction: Asymmetric-retrieval instruction prepended to text queries
            only (not image queries). Override via ``retrieval.extractor_kwargs``
            in YAML.
        torch_dtype: Defaults to ``torch.bfloat16``.
        attn_implementation: ``"flash_attention_2"`` if ``flash_attn`` imports,
            else ``"sdpa"``.
    """

    DEFAULT_INSTRUCTION = "Retrieve images relevant to query."

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-VL-Embedding-2B",
        device: str = "cuda",
        normalize: bool = True,
        batch_size: int = 16,
        instruction: str = DEFAULT_INSTRUCTION,
        torch_dtype: torch.dtype = torch.bfloat16,
        attn_implementation: str | None = None,
        min_pixels: int | None = None,
        max_pixels: int | None = None,
        total_pixels: int | None = None,
        max_length: int | None = None,
        **_unused,
    ):
        super().__init__(model_name, device, normalize, batch_size)

        Qwen3VLEmbedder = _load_qwen3vl_embedder_cls()

        self.instruction = instruction
        embedder_kwargs = {
            k: v for k, v in {
                "min_pixels": min_pixels,
                "max_pixels": max_pixels,
                "total_pixels": total_pixels,
                "max_length": max_length,
            }.items() if v is not None
        }
        logger.debug("Loading Qwen3-VL-Embedding: %s (%s)", model_name, embedder_kwargs)
        self.model = Qwen3VLEmbedder(
            model_name_or_path=model_name,
            torch_dtype=torch_dtype,
            attn_implementation=attn_implementation or _pick_attn_implementation(),
            **embedder_kwargs,
        )

        with torch.no_grad():
            probe = self._run([{"text": "probe", "instruction": self.instruction}])
            self.feature_dim = int(probe.shape[-1])
        logger.info("Qwen3-VL-Embedding loaded (dim=%d)", self.feature_dim)

    def _run(self, inputs: list[dict]) -> torch.Tensor:
        out = self.model.process(inputs)
        if isinstance(out, np.ndarray):
            out = torch.from_numpy(out)
        return out.to(device=self.device, dtype=torch.float32)

    def _forward_images(self, pil_images: List[Image.Image]) -> torch.Tensor:
        return self._run([{"image": img} for img in pil_images])

    def _forward_text(self, text: str) -> torch.Tensor:
        return self._run([{"text": text, "instruction": self.instruction}])
