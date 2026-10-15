from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import TypeAlias

from ..backend.base import GenerationOutput, ScoreOutput
from ..core.answers import Answer, ChoiceAnswer, NoulAnswer, ScoreAnswer
from ..core.questions import Box, Choice, Noul, Score
from ..core.request import JevRequest
from ..core.response import JevResponse, Usage
from ..utils.probability import validate_probabilities
from .box import assemble_box_answers
from .normalization import normalize_l1
from .prompt import RenderedPrompts


Normalizer: TypeAlias = Callable[[Sequence[float]], Sequence[float]]


def _normalize(
    probabilities: Sequence[float],
    normalizer: Normalizer,
) -> tuple[float, ...]:
    normalized = validate_probabilities(tuple(normalizer(probabilities)))
    if len(normalized) != len(probabilities):
        raise ValueError("normalizer must preserve the number of probabilities")
    if not math.isclose(math.fsum(normalized), 1.0, abs_tol=1e-6):
        raise ValueError("normalized probabilities must sum to 1")
    return normalized


def _confidence(probabilities: Sequence[float]) -> float:
    count = len(probabilities)
    if count < 2:
        raise ValueError("confidence requires at least two candidates")
    confidence = (count * max(probabilities) - 1) / (count - 1)
    return min(1.0, max(0.0, confidence))


def assemble_response(
    *,
    request: JevRequest,
    rendered: RenderedPrompts | None = None,
    scored: ScoreOutput | None = None,
    generated: GenerationOutput | None = None,
    normalizer: Normalizer = normalize_l1,
) -> JevResponse:
    """Assemble backend outputs into answers and combined token usage."""

    score_question_ids = (
        rendered.score_question_ids
        if rendered is not None
        else _score_question_ids(request)
    )
    raw_probabilities = scored.yes_probabilities if scored is not None else ()
    if len(raw_probabilities) != len(score_question_ids):
        raise ValueError("each binary task must have exactly one yes probability")
    box_ids = (
        rendered.box_question_ids
        if rendered is not None
        else tuple(name for name, question in request.questions.items() if isinstance(question, Box))
    )
    if box_ids and generated is None:
        raise ValueError("box questions require generated output")
    box_answers = assemble_box_answers(box_ids, generated) if generated is not None else {}

    grouped: dict[str, list[float]] = {
        question_id: [] for question_id in request.questions
    }
    for question_id, probability in zip(score_question_ids, raw_probabilities):
        if question_id not in grouped:
            raise ValueError(f"renderer returned unknown question ID {question_id!r}")
        grouped[question_id].append(probability)

    answers: dict[str, Answer] = {}
    for question_id, question in request.questions.items():
        raw = grouped[question_id]

        if isinstance(question, Box):
            answers[question_id] = box_answers[question_id]
        elif isinstance(question, Choice):
            labels = list(question.criteria)
            if len(labels) < 2:
                raise ValueError("choice answers require at least two candidates")
            probabilities = _normalize(raw, normalizer)
            by_label = dict(zip(labels, probabilities))
            choice = max(labels, key=by_label.__getitem__)
            answers[question_id] = ChoiceAnswer(
                choice=choice,
                confidence=_confidence(probabilities),
                probabilities=by_label,
            )
        elif isinstance(question, Score):
            probabilities = _normalize(raw, normalizer)
            by_level = dict(enumerate(probabilities))
            score = math.fsum(level * probability for level, probability in by_level.items())
            answers[question_id] = ScoreAnswer(
                score=score,
                confidence=_confidence(probabilities),
                legend=dict(enumerate(question.criteria)),
                probabilities=by_level,
            )
        elif isinstance(question, Noul):
            if question.criteria:
                probabilities = _normalize(raw, normalizer)
                by_candidate = dict(zip(("true", "false"), probabilities))
                answers[question_id] = NoulAnswer(noul=by_candidate["true"])
            else:
                answers[question_id] = NoulAnswer(noul=raw[0])
        else:
            raise TypeError(f"unsupported question type: {type(question).__name__}")

    return JevResponse(
        model=request.model,
        answers=answers,
        usage=sum_usage(*(output.usage for output in (scored, generated) if output is not None)),
    )


def _score_question_ids(request: JevRequest) -> tuple[str, ...]:
    ids: list[str] = []
    for question_id, question in request.questions.items():
        if isinstance(question, Box):
            continue
        if isinstance(question, Choice):
            if not question.criteria:
                raise ValueError("choice questions must contain criteria")
            count = len(question.criteria)
        elif isinstance(question, Score):
            count = len(question.criteria)
        elif isinstance(question, Noul):
            count = 2 if question.criteria else 1
        else:
            raise TypeError(f"unsupported question type: {type(question).__name__}")
        ids.extend([question_id] * count)
    return tuple(ids)


def sum_usage(*usages: Usage) -> Usage:
    def total(field: str) -> int | None:
        values = [getattr(usage, field) for usage in usages]
        return None if any(value is None for value in values) else sum(values)
    return Usage(input_tokens=total("input_tokens"), output_tokens=total("output_tokens"))
