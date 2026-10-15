"""Single-box grounding with Qwen's normalized 0–1000 coordinates."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Sequence

from ..backend.base import GenerationOutput
from ..core.answers import BoxAnswer


BOX_MAX_NEW_TOKENS = 128
def parse_box(text: str) -> dict[str, float]:
    """Accept one JSON bbox_2d object, singleton object list, or coordinate array."""
    text = text.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*\n?(.*?)\s*```", text, re.DOTALL)
    if fenced:
        text = fenced[1]
    try:
        value = json.loads(text)
    except json.JSONDecodeError as error:
        raise ValueError("box output must be a single JSON bounding box") from error
    if isinstance(value, list) and len(value) == 1 and isinstance(value[0], dict):
        value = value[0]
    if isinstance(value, dict):
        value = value.get("bbox_2d")
    if not isinstance(value, list) or len(value) != 4 or any(
        isinstance(coordinate, bool) or not isinstance(coordinate, int)
        or not 0 <= coordinate <= 1000 for coordinate in value
    ):
        raise ValueError("bbox_2d must contain four integers in [0, 1000]")
    left, top, right, bottom = (coordinate / 1000 for coordinate in value)
    if left >= right or top >= bottom:
        raise ValueError("box must have left < right and top < bottom")
    return {"top": top, "bottom": bottom, "left": left, "right": right}


def coordinate_confidence(
    text: str, token_texts: Sequence[str], token_probabilities: Sequence[float],
) -> float:
    """Geometric mean of sampled-token probabilities overlapping coordinate digits."""
    prefix = r'"bbox_2d"\s*:\s*' if '"bbox_2d"' in text else ""
    match = re.search(prefix + r'\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]', text)
    if match is None:
        raise ValueError("cannot locate generated coordinate tokens")
    spans = [match.span(index) for index in range(1, 5)]
    selected, start = [], 0
    for piece, probability in zip(token_texts, token_probabilities):
        end = start + len(piece)
        if piece and any(start < right and end > left for left, right in spans):
            selected.append(probability)
        start = end
    if not selected:
        raise ValueError("coordinate token probabilities are missing")
    if 0 in selected:
        return 0.0
    return math.exp(math.fsum(math.log(probability) for probability in selected) / len(selected))


def assemble_box_answers(ids: Sequence[str], generated: GenerationOutput) -> dict[str, BoxAnswer]:
    if len(generated.token_texts) != len(ids):
        raise ValueError("each box question must have exactly one generated result")
    return {
        name: BoxAnswer(box=parse_box(text), confidence=coordinate_confidence(text, pieces, probabilities))
        for name, text, pieces, probabilities in zip(
            ids, generated.texts, generated.token_texts, generated.token_probabilities,
        )
    }
