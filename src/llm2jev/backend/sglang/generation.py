"""SGLang grounding generation, shared by the library and async HTTP route."""

import math
from collections.abc import Sequence
from typing import Any

from ...core.response import Usage
from ...inference.prompt import ChatPrompt
from ..base import GenerationOutput
from ..tokenization import _generated_token_texts
from .scoring import _prepare_inputs


def prepare_generation_request(
    tokenizer: Any, processor: Any, prompts: Sequence[ChatPrompt], *, max_new_tokens: int,
) -> dict[str, Any]:
    texts, input_ids, images, contains_media = _prepare_inputs(
        tokenizer, processor, prompts, enable_thinking=False,
    )
    return {
        "text": texts if contains_media else None,
        "input_ids": input_ids,
        "image_data": images if contains_media else None,
        "sampling_params": {"max_new_tokens": max_new_tokens, "temperature": 0},
        "return_logprob": True,
        "logprob_start_len": -1,
    }


def generation_output(results: list[dict[str, Any]], *, expected_count: int, tokenizer: Any) -> GenerationOutput:
    if len(results) != expected_count:
        raise ValueError("SGLang returned the wrong number of generated results")
    token_texts, token_probabilities, input_tokens, output_tokens = [], [], 0, 0
    for result in results:
        meta = result["meta_info"]
        reason = meta["finish_reason"]
        if reason["type"] != "stop":
            raise ValueError(f"Box generation did not complete: {reason}")
        entries = meta.get("output_token_logprobs")
        if not entries or len(entries) != meta["completion_tokens"]:
            raise ValueError("SGLang must return logprobs for every generated token")
        pieces = _generated_token_texts(tokenizer, [entry[1] for entry in entries])
        if "".join(pieces) != result["text"]:
            raise ValueError("token texts must reconstruct the generated text")
        token_texts.append(pieces)
        token_probabilities.append([math.exp(entry[0]) for entry in entries])
        input_tokens += meta["prompt_tokens"]
        output_tokens += meta["completion_tokens"]
    return GenerationOutput(
        token_texts=token_texts, token_probabilities=token_probabilities,
        usage=Usage(input_tokens=input_tokens, output_tokens=output_tokens),
    )
