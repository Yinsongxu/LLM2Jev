from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TypeAlias

from .questions import Box, Choice, Noul, Score
from ..utils.json import is_json_content
from .types import State
from .multimodal import contains_image, validate_multimodal


Question: TypeAlias = Box | Choice | Score | Noul


@dataclass(frozen=True, slots=True, kw_only=True)
class JevRequest:
    """A collection of questions about shared state for a specific model."""

    state: State
    model: str
    questions: Mapping[str, Question]
    include_choice_candidates: bool = True

    def __post_init__(self) -> None:
        if not is_json_content(self.state):
            raise ValueError("state must be a string, JSON object, or JSON array")
        validate_multimodal(self.state)
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("model must be a non-empty string")
        if not isinstance(self.include_choice_candidates, bool):
            raise ValueError("include_choice_candidates must be a boolean")

        questions = dict(self.questions)
        if not questions:
            raise ValueError("questions must contain at least one question")
        if any(not isinstance(question_id, str) or not question_id for question_id in questions):
            raise ValueError("question IDs must be non-empty strings")
        if any(not isinstance(question, (Box, Choice, Score, Noul)) for question in questions.values()):
            raise ValueError("questions must contain Box, Choice, Score, or Noul instances")
        for question in questions.values():
            if isinstance(question, Box) and not (
                contains_image(self.state) or contains_image(question.instructions)
            ):
                raise ValueError("box questions require an image in state or instructions")
        object.__setattr__(self, "questions", questions)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "state": self.state,
            "model": self.model,
            "questions": {
                question_id: question.to_dict()
                for question_id, question in self.questions.items()
            },
        }
        if not self.include_choice_candidates:
            result["include_choice_candidates"] = False
        return result
