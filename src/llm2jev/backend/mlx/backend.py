from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import BinaryBackendOutput
from ..tokenization import _apply_chat_template, _single_token_id
from .scoring import TextScorer


class MLXBackend:
    """Prefill-only scorer with batching, prefix reuse and optional MLX-VLM images."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        yes_label: str = "yes",
        no_label: str = "no",
        enable_thinking: bool = False,
        prefill_step_size: int = 512,
        batch_size: int = 8,
        submission: Literal["all", "staged"] = "staged",
        max_cache_entries: int = 32,
        max_cache_bytes: int = 512 * 1024 * 1024,
        multimodal: bool = False,
    ) -> None:
        for name, value, minimum in (
            ("prefill_step_size", prefill_step_size, 1),
            ("batch_size", batch_size, 1),
            ("max_cache_entries", max_cache_entries, 0),
            ("max_cache_bytes", max_cache_bytes, 0),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")
        if submission not in ("staged", "all"):
            raise ValueError("submission must be 'staged' or 'all'")
        path = Path(model_path).expanduser()
        if not path.is_dir():
            raise ValueError("model_path must be an existing local MLX-LM model directory")

        try:
            import mlx.core as mx
            from mlx_lm import load
            from mlx_lm.models.cache import make_prompt_cache
        except ImportError as error:
            raise ImportError(
                "MLXBackend requires the 'mlx' extra on Apple Silicon macOS; "
                "install llm2jev[mlx]"
            ) from error

        self._mx = mx
        self._lock = RLock()
        self.model_path = str(path)
        self.enable_thinking = enable_thinking
        self.prefill_step_size = prefill_step_size
        self.batch_size = batch_size
        self.submission = submission
        self.max_cache_entries = max_cache_entries
        self.max_cache_bytes = max_cache_bytes
        self.processor = None
        self._vision_scorer = None
        self._text_scorer = None
        if multimodal:
            try:
                from mlx_vlm import load as load_vision
            except ImportError as error:
                raise ImportError(
                    "MLX image models require the 'mlx-vlm' extra; install llm2jev[mlx-vlm]"
                ) from error
            self.model, self.processor = load_vision(self.model_path, local_files_only=True)
            self.tokenizer = self.processor.tokenizer
        else:
            self.model, self.tokenizer = load(
                self.model_path, tokenizer_config={"local_files_only": True},
            )
        self.model.eval()
        self.yes_token_id = _single_token_id(
            self.tokenizer, yes_label, "yes_label",
        )
        self.no_token_id = _single_token_id(
            self.tokenizer, no_label, "no_label",
        )
        if self.yes_token_id == self.no_token_id:
            raise ValueError("yes_label and no_label must encode to different tokens")
        options = dict(
            batch_size=batch_size, prefill_step_size=prefill_step_size,
            submission=submission, max_cache_entries=max_cache_entries,
            max_cache_bytes=max_cache_bytes,
        )
        if multimodal:
            from .vision import MLXVisionScorer

            self._vision_scorer = MLXVisionScorer(
                self.model, self.processor, enable_thinking=enable_thinking, **options,
            )
        else:
            self._text_scorer = TextScorer(self.model, mx, make_prompt_cache, **options)

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        del model  # The loaded model path is fixed when the backend is constructed.
        with self._lock:
            if self.model is None:
                raise RuntimeError("MLXBackend is closed")
            return self._score(prompts)

    def _score(self, prompts: Sequence[ChatPrompt]) -> BinaryBackendOutput:
        prompt_values = tuple(prompts)
        if not prompt_values:
            raise ValueError("prompts must not be empty")
        if self._vision_scorer is not None:
            return self._vision_scorer.score(
                prompt_values, yes_token_id=self.yes_token_id, no_token_id=self.no_token_id,
            )
        # Validate the entire batch before running a text model on any evidence.
        text_prompts = tuple(_text_prompt(prompt) for prompt in prompt_values)

        token_sequences = []
        for prompt in text_prompts:
            rendered = _apply_chat_template(
                self.tokenizer, prompt, enable_thinking=self.enable_thinking,
            )
            tokens = self.tokenizer.encode(rendered, add_special_tokens=False)
            if not tokens:
                raise ValueError("each rendered prompt must encode to at least one token")
            token_sequences.append(tokens)

        return BinaryBackendOutput(
            yes_probabilities=self._text_scorer.score(
                token_sequences, no_token_id=self.no_token_id, yes_token_id=self.yes_token_id,
            ),
            usage=Usage(input_tokens=sum(map(len, token_sequences)), output_tokens=0),
        )

    def clear_cache(self) -> None:
        """Drop reusable prompt/vision caches while keeping model weights loaded."""
        with self._lock:
            for scorer in (self._text_scorer, self._vision_scorer):
                if scorer is not None:
                    scorer.clear_cache()
            self._mx.clear_cache()

    def close(self) -> None:
        """Release this backend's model and caches; repeated calls are safe."""
        with self._lock:
            if self.model is not None:
                self._mx.synchronize()
                self.clear_cache()
                self._text_scorer = self._vision_scorer = None
                self.model = self.processor = self.tokenizer = None
                self._mx.clear_cache()

    def __enter__(self) -> MLXBackend:
        if self.model is None:
            raise RuntimeError("MLXBackend is closed")
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


def _text_prompt(prompt: ChatPrompt) -> ChatPrompt:
    """Flatten text content parts and reject unsupported image evidence."""
    messages = []
    for message in prompt:
        content = message["content"]
        if not isinstance(content, str):
            text_parts = []
            for part in content:
                if part["type"] != "text":
                    raise ValueError(
                        "image prompts require MLXBackend(multimodal=True) "
                        "with an MLX-VLM model"
                    )
                text_parts.append(part["text"])
            content = "".join(text_parts)
        messages.append({**message, "content": content})
    return tuple(messages)
