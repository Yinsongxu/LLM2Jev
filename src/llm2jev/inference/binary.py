from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, TypeAlias

from ..core.questions import Choice, Noul, Score
from ..core.request import JevRequest
from ..core.types import JSONContent, State
from ..core.multimodal import validate_multimodal
from ..utils.json import copy_json_content, is_json_content


QuestionType: TypeAlias = Literal["choice", "score", "noul"]
Candidate: TypeAlias = str | int


@dataclass(frozen=True, slots=True, kw_only=True)
class BinaryQuestion:
    """A model-independent yes/no task compiled from a Jev question."""

    question_id: str
    question_type: QuestionType
    candidate: Candidate
    context: State
    objective: JSONContent | None
    condition: JSONContent | None
    choices: tuple[tuple[str, JSONContent | None], ...] | None = None
    score_levels: tuple[JSONContent, ...] | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.question_id, str) or not self.question_id:
            raise ValueError("question_id must be a non-empty string")
        if self.question_type not in {"choice", "score", "noul"}:
            raise ValueError("question_type must be choice, score, or noul")
        if not is_json_content(self.context):
            raise ValueError("context must be a string, JSON object, or JSON array")
        validate_multimodal(self.context)
        validate_multimodal(self.objective)
        for field, value in (
            ("objective", self.objective),
            ("condition", self.condition),
        ):
            if value is not None and not is_json_content(value):
                raise ValueError(f"{field} must be JSON content or None")

        if self.choices is not None:
            if self.question_type != "choice" or not self.choices:
                raise ValueError("choices are only valid and required for choice tasks")
            if any(
                not isinstance(name, str) or not name
                or (description is not None and not is_json_content(description))
                for name, description in self.choices
            ):
                raise ValueError("choices must contain non-empty names and JSON descriptions")
            if self.candidate not in {name for name, _ in self.choices}:
                raise ValueError("choice candidate must be present in choices")
        if self.score_levels is not None:
            if self.question_type != "score" or not self.score_levels:
                raise ValueError("score_levels are only valid and required for score tasks")
            if any(not is_json_content(level) for level in self.score_levels):
                raise ValueError("score_levels must contain JSON content")

        if self.question_type == "choice" and (
            not isinstance(self.candidate, str) or not self.candidate
        ):
            raise ValueError("choice candidate must be a non-empty string")
        if self.question_type == "score" and (
            isinstance(self.candidate, bool)
            or not isinstance(self.candidate, int)
            or self.candidate < 0
        ):
            raise ValueError("score candidate must be a non-negative integer")
        if self.question_type == "noul" and self.candidate not in {"true", "false"}:
            raise ValueError('noul candidate must be "true" or "false"')

        object.__setattr__(self, "context", copy_json_content(self.context))
        if self.objective is not None:
            object.__setattr__(self, "objective", copy_json_content(self.objective))
        if self.condition is not None:
            object.__setattr__(self, "condition", copy_json_content(self.condition))
        if self.choices is not None:
            object.__setattr__(
                self,
                "choices",
                tuple(
                    (name, copy_json_content(description) if description is not None else None)
                    for name, description in self.choices
                ),
            )
        if self.score_levels is not None:
            object.__setattr__(
                self,
                "score_levels",
                tuple(copy_json_content(level) for level in self.score_levels),
            )


def compile_binary_questions(request: JevRequest) -> tuple[BinaryQuestion, ...]:
    """Compile all request questions into ordered model-independent binary tasks."""

    tasks: list[BinaryQuestion] = []
    for question_id, question in request.questions.items():
        if isinstance(question, Choice):
            if not question.criteria:
                raise ValueError(f"choice question {question_id!r} must contain criteria")
            for option, description in question.criteria.items():
                tasks.append(
                    BinaryQuestion(
                        question_id=question_id,
                        question_type="choice",
                        candidate=option,
                        context=request.state,
                        objective=question.instructions,
                        condition=description,
                        choices=tuple(question.criteria.items()),
                    )
                )
        elif isinstance(question, Score):
            for level, description in enumerate(question.criteria):
                tasks.append(
                    BinaryQuestion(
                        question_id=question_id,
                        question_type="score",
                        candidate=level,
                        context=request.state,
                        objective=question.instructions,
                        condition=description,
                        score_levels=tuple(question.criteria),
                    )
                )
        elif isinstance(question, Noul):
            criteria = question.criteria or {}
            candidates = ("true", "false") if criteria else ("true",)
            for candidate in candidates:
                tasks.append(
                    BinaryQuestion(
                        question_id=question_id,
                        question_type="noul",
                        candidate=candidate,
                        context=request.state,
                        objective=question.instructions,
                        condition=criteria.get(candidate),
                    )
                )
        else:
            raise TypeError(f"unsupported question type: {type(question).__name__}")
    return tuple(tasks)
