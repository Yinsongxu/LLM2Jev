from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import ClassVar, TypeAlias

from ..utils.json import is_json_content
from ..utils.probability import copy_probability_distribution, validate_probability
from .types import JSONContent


@dataclass(frozen=True, slots=True, kw_only=True)
class ChoiceAnswer:
    """A selected choice and the probability of every available choice."""

    choice: str
    confidence: float
    probabilities: Mapping[str, float]

    type: ClassVar[str] = "choice"

    def __post_init__(self) -> None:
        if not isinstance(self.choice, str) or not self.choice:
            raise ValueError("choice must be a non-empty string")
        validate_probability(self.confidence, "confidence")
        probabilities = copy_probability_distribution(self.probabilities)
        if any(not isinstance(name, str) or not name for name in probabilities):
            raise ValueError("probability keys must be non-empty strings")
        if self.choice not in probabilities:
            raise ValueError("choice must be present in probabilities")
        if not math.isclose(probabilities[self.choice], max(probabilities.values()), abs_tol=1e-12):
            raise ValueError("choice must have the highest probability")
        object.__setattr__(self, "probabilities", probabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "choice": self.choice,
            "confidence": self.confidence,
            "probabilities": dict(self.probabilities),
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreAnswer:
    """A probability-weighted score and its ordered rubric."""

    score: float
    confidence: float
    legend: Mapping[int, JSONContent]
    probabilities: Mapping[int, float]

    type: ClassVar[str] = "score"

    def __post_init__(self) -> None:
        if isinstance(self.score, bool) or not isinstance(self.score, (int, float)) or not math.isfinite(self.score):
            raise ValueError("score must be a finite number")
        validate_probability(self.confidence, "confidence")

        legend = dict(self.legend)
        if not legend:
            raise ValueError("legend must not be empty")
        if any(isinstance(level, bool) or not isinstance(level, int) or level < 0 for level in legend):
            raise ValueError("legend keys must be non-negative integers")
        if set(legend) != set(range(len(legend))):
            raise ValueError("legend levels must be contiguous and start at 0")
        if any(not is_json_content(description) for description in legend.values()):
            raise ValueError("legend descriptions must be JSON content")

        probabilities = copy_probability_distribution(self.probabilities)
        if any(isinstance(level, bool) or not isinstance(level, int) for level in probabilities):
            raise ValueError("score probability keys must be integers")
        if set(probabilities) != set(legend):
            raise ValueError("legend and probabilities must contain the same levels")

        expected_score = math.fsum(level * probability for level, probability in probabilities.items())
        if not math.isclose(self.score, expected_score, abs_tol=1e-6):
            raise ValueError("score must equal the probability-weighted average of its levels")

        object.__setattr__(self, "legend", legend)
        object.__setattr__(self, "probabilities", probabilities)

    def to_dict(self) -> dict[str, object]:
        return {
            "type": self.type,
            "score": self.score,
            "confidence": self.confidence,
            "legend": {str(level): description for level, description in self.legend.items()},
            "probabilities": {str(level): probability for level, probability in self.probabilities.items()},
        }


@dataclass(frozen=True, slots=True, kw_only=True)
class NoulAnswer:
    """The probability that the answer is yes or the statement is true."""

    noul: float

    type: ClassVar[str] = "noul"

    def __post_init__(self) -> None:
        validate_probability(self.noul, "noul")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "noul": self.noul}


@dataclass(frozen=True, slots=True, kw_only=True)
class BoxAnswer:
    """Normalized edges and confidence from generated coordinate token probabilities."""

    box: Mapping[str, float]
    confidence: float

    type: ClassVar[str] = "box"

    def __post_init__(self) -> None:
        box = dict(self.box)
        if set(box) != {"top", "bottom", "left", "right"}:
            raise ValueError("box must contain exactly top, bottom, left, and right")
        for edge, value in box.items():
            validate_probability(value, f"box {edge}")
        if box["left"] >= box["right"] or box["top"] >= box["bottom"]:
            raise ValueError("box must have left < right and top < bottom")
        validate_probability(self.confidence, "confidence")
        object.__setattr__(self, "box", box)

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "box": dict(self.box), "confidence": self.confidence}


Answer: TypeAlias = ChoiceAnswer | ScoreAnswer | NoulAnswer | BoxAnswer
