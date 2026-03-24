from __future__ import annotations

import logging
import re
import warnings
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)


class Qwen2_5VL:
    """Qwen2.5-VL Vision Language Model. Paper Step 2 (Sec III-C).

    Supports Qwen2.5-VL-3B/7B/72B-Instruct variants.
    Uses Qwen2_5_VLForConditionalGeneration from transformers.
    Requires: pip install transformers>=4.57.0 torch pillow
    """

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        device: str = "cuda",
    ):
        """Initialize Qwen2.5-VL model.

        Args:
            model_name: HuggingFace model name (e.g., "Qwen/Qwen2.5-VL-7B-Instruct")
            device: Device to run on (cuda/cpu)
        """
        self.model_name = model_name
        self.device = device

        try:
            import torch
            from transformers import AutoProcessor
        except ImportError:
            raise ImportError(
                "Please install: pip install transformers>=4.57.0 torch pillow"
            )

        self.torch = torch

        torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

        self.model = self._load_model(model_name, device, torch_dtype, torch)

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.processor = AutoProcessor.from_pretrained(model_name, use_fast=False)

        self.processor.tokenizer.padding_side = 'left'

        logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)
        logging.getLogger("transformers.generation.configuration_utils").setLevel(logging.ERROR)

    @staticmethod
    def _load_model(model_name: str, device: str, torch_dtype, torch):
        from transformers import Qwen2_5_VLForConditionalGeneration

        attn_impl = None
        if device == "cuda":
            try:
                from flash_attn import flash_attn_func  # noqa: F401
                attn_impl = "flash_attention_2"
                logger.debug("Using Flash Attention 2")
            except ImportError:
                logger.debug("Flash Attention 2 unavailable, using default attention")

        kwargs = {"dtype": torch_dtype}
        if attn_impl:
            kwargs["attn_implementation"] = attn_impl
        if device == "cuda":
            kwargs["device_map"] = "auto"

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                model_name, **kwargs
            )

        return model

    @staticmethod
    def _to_pil(image: np.ndarray) -> Image.Image:
        """Convert numpy image to PIL, normalizing dtype if needed."""
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        return Image.fromarray(image)

    @staticmethod
    def _construct_prompt(query: str) -> str:
        """Construct a concise VLM prompt for detection + visibility scoring.

        Expected output is a short phrase: "yes 8" or "no 0".
        Visibility score reflects how large and unoccluded the object appears,
        which correlates with better downstream localization quality.
        """
        return (
            f'Is a "{query}" visible in this image? '
            f'Reply ONLY: yes/no <visibility 0-10> '
            f'0=not visible, 5=small/partially occluded, 10=large and clearly visible'
        )

    @staticmethod
    def _parse_response(response_text: str) -> dict:
        """Parse 'yes/no <score>' response into detected + confidence."""
        text = response_text.strip().lower()
        detected = text.startswith("yes")

        if match := re.search(r'\b(\d{1,2})\b', text):
            confidence = min(int(match.group(1)), 10) / 10.0
        else:
            confidence = 1.0 if detected else 0.0

        return {
            "response": response_text.strip(),
            "confidence": confidence,
            "detected": detected,
        }

    def _generate_and_decode(self, inputs, max_new_tokens: int = 48) -> list[str]:
        """Generate and decode model output."""
        try:
            with self.torch.no_grad():
                generated_ids = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=False,
                )

            generated_ids = generated_ids.cpu()
            input_ids = inputs['input_ids'] if isinstance(inputs, dict) else inputs.input_ids
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(input_ids, generated_ids)
            ]

            return self.processor.batch_decode(
                generated_ids_trimmed,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )

        except Exception as e:
            raise RuntimeError(f"Error during batch generation: {e}")

    def _prepare_inputs(self, messages):
        """Prepare inputs for Qwen2.5 models via apply_chat_template."""
        is_batch = isinstance(messages, list) and isinstance(messages[0], list)
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", message=".*max_length.*padding.*")
            return self.processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
                padding=is_batch,
            )

    def batch_query(self, images: list, query: str, batch_size: int = 10) -> list:
        """Query multiple images efficiently using native batch processing.

        Args:
            images: List of RGB images (H x W x 3)
            query: Natural language query
            batch_size: Maximum batch size for processing. Use smaller values
                (e.g., 4-8) if running out of GPU memory.

        Returns:
            List of result dictionaries
        """
        if not images:
            return []

        if len(images) > batch_size:
            all_results = []
            for i in range(0, len(images), batch_size):
                chunk = images[i:i + batch_size]
                chunk_results = self._batch_query_chunk(chunk, query)
                all_results.extend(chunk_results)
            return all_results
        else:
            return self._batch_query_chunk(images, query)

    def _batch_query_chunk(self, images: list, query: str) -> list:
        """Process a chunk of images in one batch."""
        pil_images = [self._to_pil(img) for img in images]
        prompt = self._construct_prompt(query)

        messages_list = []
        for pil_image in pil_images:
            messages_list.append([{
                "role": "user",
                "content": [
                    {"type": "image", "image": pil_image},
                    {"type": "text", "text": prompt}
                ]
            }])

        inputs = self._prepare_inputs(messages_list)

        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) if isinstance(v, self.torch.Tensor) else v
                     for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)

        response_texts = self._generate_and_decode(inputs)
        return [self._parse_response(text) for text in response_texts]
