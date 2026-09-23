from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import BinaryBackendOutput
from .image_inputs import has_images, load_transformers_images, prepare_image_prompts
from ..tokenization import _apply_chat_template, _single_token_id


class TransformersBackend:
    """Prefill-only binary scorer backed by a Transformers causal language model."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        device: str | None = None,
        dtype: str = "auto",
        yes_label: str = "yes",
        no_label: str = "no",
        batch_size: int = 8,
        enable_thinking: bool = False,
        multimodal: bool = False,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")

        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "TransformersBackend requires the 'transformers' and 'torch' packages"
            ) from error

        self._torch = torch
        self.model_path = str(model_path)
        self.batch_size = batch_size
        self.enable_thinking = enable_thinking
        self.processor = None
        model_class = AutoModelForCausalLM
        if multimodal:
            from transformers import AutoModelForImageTextToText, AutoProcessor

            self.processor = AutoProcessor.from_pretrained(
                self.model_path, local_files_only=True,
            )
            self.tokenizer = self.processor.tokenizer
            model_class = AutoModelForImageTextToText
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path,
                local_files_only=True,
            )
        if self.tokenizer.pad_token_id is None:
            if self.tokenizer.eos_token_id is None:
                raise ValueError("tokenizer must define a pad token or an EOS token")
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.yes_token_id = _single_token_id(
            self.tokenizer, yes_label, "yes_label",
        )
        self.no_token_id = _single_token_id(
            self.tokenizer, no_label, "no_label",
        )
        if self.yes_token_id == self.no_token_id:
            raise ValueError("yes_label and no_label must encode to different tokens")

        selected_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model_kwargs: dict[str, Any] = {"local_files_only": True}
        if dtype != "auto":
            try:
                model_kwargs["dtype"] = getattr(torch, dtype)
            except AttributeError as error:
                raise ValueError(f"unsupported dtype: {dtype}") from error

        self.model = model_class.from_pretrained(
            self.model_path,
            **model_kwargs,
        )
        self.model.to(selected_device)
        self.model.eval()
        self.device = selected_device

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        del model  # The loaded model path is fixed when the backend is constructed.
        prompt_values = tuple(prompts)
        if not prompt_values:
            raise ValueError("prompts must not be empty")
        contains_images = has_images(prompt_values)
        if contains_images and self.processor is None:
            raise ValueError("image prompts require TransformersBackend(multimodal=True)")
        if self.model is None:
            raise RuntimeError("TransformersBackend is closed")

        if contains_images:
            rendered_images, image_batches = prepare_image_prompts(
                self.processor, prompt_values, enable_thinking=self.enable_thinking,
            )
            image_batches = load_transformers_images(image_batches)

        yes_probabilities: list[float] = []
        input_tokens = 0
        for start in range(0, len(prompt_values), self.batch_size):
            batch = prompt_values[start : start + self.batch_size]
            if contains_images:
                images = [
                    image for row in image_batches[start : start + self.batch_size]
                    for image in row
                ]
                encoded = self.processor(
                    text=rendered_images[start : start + self.batch_size],
                    images=images or None,
                    return_tensors="pt", padding=True, add_special_tokens=False,
                )
            else:
                rendered = [
                    _apply_chat_template(
                        self.processor or self.tokenizer,
                        prompt,
                        enable_thinking=self.enable_thinking,
                    )
                    for prompt in batch
                ]
                encoded = self.tokenizer(
                    rendered,
                    return_tensors="pt",
                    padding=True,
                    add_special_tokens=False,
                )
            input_tokens += int(encoded["attention_mask"].sum().item())
            model_inputs = {
                name: tensor.to(self.device)
                for name, tensor in encoded.items()
                if contains_images or name in {"input_ids", "attention_mask", "token_type_ids"}
            }

            with self._torch.inference_mode():
                logits = self.model(**model_inputs, use_cache=False).logits

            last_positions = self._last_token_positions(model_inputs["attention_mask"])
            batch_indices = self._torch.arange(logits.shape[0], device=logits.device)
            next_token_logits = logits[batch_indices, last_positions]
            binary_logits = next_token_logits[:, [self.no_token_id, self.yes_token_id]]
            probabilities = self._torch.softmax(binary_logits.float(), dim=-1)[:, 1]
            yes_probabilities.extend(probabilities.cpu().tolist())

        return BinaryBackendOutput(
            yes_probabilities=yes_probabilities,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )

    def _last_token_positions(self, attention_mask: Any) -> Any:
        positions = self._torch.arange(
            attention_mask.shape[1],
            device=attention_mask.device,
        ).expand_as(attention_mask)
        return positions.masked_fill(attention_mask == 0, -1).max(dim=1).values

    def close(self) -> None:
        """Release references to the model and tokenizer; safe to call repeatedly."""
        self.model = None
        self.processor = None
        self.tokenizer = None

    def __enter__(self) -> TransformersBackend:
        if self.model is None:
            raise RuntimeError("TransformersBackend is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
