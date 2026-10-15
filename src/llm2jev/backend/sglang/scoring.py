"""Shared text/image request preparation and zero-generation SGLang scoring."""

import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import ScoreOutput
from .prefix_plan import staged_batches
from ..tokenization import _apply_chat_template


def _prepare_inputs(
    tokenizer: Any, processor: Any, prompts: Sequence[ChatPrompt], *, enable_thinking: bool,
) -> tuple[list[str], list[list[int]] | None, list[list[Any]], bool]:
    from sglang.srt.managers.io_struct import GenerateReqInput
    from sglang.srt.parser.jinja_template_utils import process_content_for_template_format

    messages, images = [], []
    for prompt in prompts:
        row = []
        messages.append([
            process_content_for_template_format(dict(message), "openai", row, [], [], [])
            for message in prompt
        ])
        for image in row:
            if not image.url.startswith(("http://", "https://", "file://", "data:")):
                image.url = str(Path(image.url).resolve())
        images.append(row)

    contains_media = GenerateReqInput(image_data=images).contains_mm_input()
    if contains_media and getattr(processor, "image_processor", None) is None:
        raise ValueError("image prompts require a multimodal SGLang model")
    renderer = processor if contains_media else tokenizer
    texts = [
        _apply_chat_template(renderer, prompt, enable_thinking=enable_thinking)
        for prompt in messages
    ]
    input_ids = None if contains_media else tokenizer(texts, add_special_tokens=False)["input_ids"]
    return texts, input_ids, images, contains_media


def prepare_score_batches(
    tokenizer: Any, processor: Any, prompts: Sequence[ChatPrompt], *,
    enable_thinking: bool, submission: str, no_token_id: int, yes_token_id: int,
) -> list[tuple[list[int], dict[str, Any]]]:
    texts, input_ids, images, contains_media = _prepare_inputs(
        tokenizer, processor, prompts, enable_thinking=enable_thinking,
    )
    if submission == "all":
        batches = [list(range(len(prompts)))]
    elif contains_media:
        # Placeholder IDs do not identify real images for prefix planning.
        batches = [[i] for i in range(len(prompts))]
    else:
        batches = staged_batches(input_ids)
    return [(indices, {
        "text": [texts[i] for i in indices] if contains_media else None,
        "input_ids": [input_ids[i] for i in indices] if input_ids is not None else None,
        "image_data": [images[i] for i in indices] if contains_media else None,
        "sampling_params": {"max_new_tokens": 0},
        "return_logprob": True,
        "logprob_start_len": -1,
        "token_ids_logprob": [no_token_id, yes_token_id],
    }) for indices in batches]


def score_output(
    results: list[dict[str, Any]], *, expected_count: int, no_token_id: int, yes_token_id: int,
) -> ScoreOutput:
    if len(results) != expected_count:
        raise ValueError("SGLang returned the wrong number of candidate scores")
    probabilities = []
    prompt_tokens = 0
    for result in results:
        meta = result["meta_info"]
        if meta["completion_tokens"] != 0:
            raise ValueError("scoring must not generate tokens")
        positions = meta["output_token_ids_logprobs"]
        if len(positions) != 1:
            raise ValueError("SGLang must return one next-token label distribution")
        values = {entry[1]: entry[0] for entry in positions[0]}
        no, yes = values[no_token_id], values[yes_token_id]
        maximum = max(no, yes)
        no_mass, yes_mass = math.exp(no - maximum), math.exp(yes - maximum)
        probabilities.append(yes_mass / (no_mass + yes_mass))
        prompt_tokens += meta["prompt_tokens"]
    return ScoreOutput(
        yes_probabilities=probabilities,
        usage=Usage(input_tokens=prompt_tokens, output_tokens=0),
    )
