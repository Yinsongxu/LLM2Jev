from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from .answers import Answer, BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer


def _format_json(value: object, level: int = 0, *, coordinates: bool = False) -> str:
    if isinstance(value, float):
        if not math.isfinite(value):
            return json.dumps(value)
        return json.dumps(value) if coordinates else f"{value:.2f}"
    if isinstance(value, Mapping):
        if not value:
            return "{}"
        indent = "  " * (level + 1)
        items = (
            f"{indent}{json.dumps(key, ensure_ascii=False)}: "
            f"{_format_json(item, level + 1, coordinates=coordinates or (key == 'box' and value.get('type') == 'box'))}"
            for key, item in value.items()
        )
        return "{\n" + ",\n".join(items) + "\n" + "  " * level + "}"
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        if not value:
            return "[]"
        indent = "  " * (level + 1)
        items = (f"{indent}{_format_json(item, level + 1)}" for item in value)
        return "[\n" + ",\n".join(items) + "\n" + "  " * level + "]"
    return json.dumps(value, ensure_ascii=False)


@dataclass(frozen=True, slots=True, kw_only=True)
class Usage:
    """Input and output token counts, when reported by the model backend."""

    input_tokens: int | None = None
    output_tokens: int | None = None

    def __post_init__(self) -> None:
        for field, value in (("input_tokens", self.input_tokens), ("output_tokens", self.output_tokens)):
            if value is not None and (isinstance(value, bool) or not isinstance(value, int) or value < 0):
                raise ValueError(f"{field} must be a non-negative integer or None")

    def to_dict(self) -> dict[str, int | None]:
        return {
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class JevResponse:
    """Answers produced by a model for a Jev request."""

    model: str
    answers: Mapping[str, Answer]
    usage: Usage

    def __post_init__(self) -> None:
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.usage, Usage):
            raise ValueError("usage must be a Usage instance")

        answers = dict(self.answers)
        if not answers:
            raise ValueError("answers must contain at least one answer")
        if any(not isinstance(answer_id, str) or not answer_id for answer_id in answers):
            raise ValueError("answer IDs must be non-empty strings")
        if any(not isinstance(answer, (BoxAnswer, ChoiceAnswer, ScoreAnswer, NoulAnswer)) for answer in answers.values()):
            raise ValueError("answers must contain BoxAnswer, ChoiceAnswer, ScoreAnswer, or NoulAnswer instances")
        object.__setattr__(self, "answers", answers)

    @property
    def choices(self) -> dict[str, ChoiceAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, ChoiceAnswer)}

    @property
    def scores(self) -> dict[str, ScoreAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, ScoreAnswer)}

    @property
    def nouls(self) -> dict[str, NoulAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, NoulAnswer)}

    @property
    def boxes(self) -> dict[str, BoxAnswer]:
        return {name: answer for name, answer in self.answers.items() if isinstance(answer, BoxAnswer)}

    @property
    def json(self) -> str:
        return _format_json(self.to_dict())

    def to_dict(self) -> dict[str, object]:
        return {
            "model": self.model,
            "answers": {
                answer_id: answer.to_dict()
                for answer_id, answer in self.answers.items()
            },
            "usage": self.usage.to_dict(),
        }
