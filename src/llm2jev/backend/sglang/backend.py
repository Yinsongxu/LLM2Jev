from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Literal

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import BinaryBackendOutput
from ..tokenization import _single_token_id
from .scoring import prepare_score_batches, score_output


class SGLangBackend:
    """Prefill-only binary scorer using SGLang's native scoring and prefix cache."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        yes_label: str = "yes",
        no_label: str = "no",
        enable_thinking: bool = False,
        submission: Literal["all", "staged"] = "staged",
        engine_kwargs: Mapping[str, Any] | None = None,
    ) -> None:
        options = dict(engine_kwargs or {})
        if submission not in ("all", "staged"):
            raise ValueError("submission must be 'all' or 'staged'")
        if submission == "staged" and options.get("disable_radix_cache"):
            raise ValueError("staged submission requires Radix Cache")
        if options.get("enable_mis"):
            raise ValueError("enable_mis changes the independent candidate scoring path")
        if "model_path" in options:
            raise ValueError("pass model_path directly, not in engine_kwargs")

        try:
            from sglang import Engine
            from transformers import AutoTokenizer
        except ImportError as error:
            raise ImportError(
                "SGLangBackend requires the 'sglang' extra; install llm2jev[sglang]"
            ) from error

        self.model_path = str(model_path)
        self.enable_thinking = enable_thinking
        self.submission = submission
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_path, local_files_only=True,
        )
        if not self.tokenizer.chat_template:
            import json

            template_path = Path(self.model_path) / "chat_template.json"

            if template_path.is_file():
                with template_path.open("r", encoding="utf-8") as f:
                    template_data = json.load(f)

                template = (
                    template_data.get("chat_template")
                    if isinstance(template_data, dict)
                    else template_data
                )
        self.yes_token_id = _single_token_id(
            self.tokenizer, yes_label, "yes_label",
        )
        self.no_token_id = _single_token_id(
            self.tokenizer, no_label, "no_label",
        )
        if self.yes_token_id == self.no_token_id:
            raise ValueError("yes_label and no_label must encode to different tokens")

        self.engine = Engine(model_path=self.model_path, **options)

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> BinaryBackendOutput:
        del model  # The engine's model is fixed at construction.
        if self.engine is None:
            raise RuntimeError("SGLangBackend is closed")
        prompt_values = tuple(prompts)
        if not prompt_values:
            raise ValueError("prompts must not be empty")
        batches = prepare_score_batches(
            self.tokenizer, getattr(self.engine.tokenizer_manager, "processor", None),
            prompt_values, enable_thinking=self.enable_thinking, submission=self.submission,
            no_token_id=self.no_token_id, yes_token_id=self.yes_token_id,
        )
        probabilities = [0.0] * len(prompt_values)
        input_tokens = 0
        for indices, arguments in batches:
            result = self.engine.generate(
                prompt=arguments.pop("text"), **arguments,
            )
            output = score_output(
                result, expected_count=len(indices),
                no_token_id=self.no_token_id, yes_token_id=self.yes_token_id,
            )
            for index, probability in zip(indices, output.yes_probabilities):
                probabilities[index] = probability
            input_tokens += output.usage.input_tokens
        return BinaryBackendOutput(
            yes_probabilities=probabilities,
            usage=Usage(input_tokens=input_tokens, output_tokens=0),
        )

    def close(self) -> None:
        """Shut down this backend's SGLang engine and release GPU resources."""
        if self.engine is not None:
            self.engine.shutdown()
            self.engine = None

    def __enter__(self) -> SGLangBackend:
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()
