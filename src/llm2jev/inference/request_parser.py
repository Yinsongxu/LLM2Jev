from __future__ import annotations

from collections.abc import Mapping

from ..core.questions import Choice, Noul, Score
from ..core.request import JevRequest, Question


def parse_question(value: object) -> Question:
    if not isinstance(value, Mapping):
        raise ValueError("each question must be a JSON object")

    question_type = value.get("type")
    instructions = value.get("instructions")
    if question_type == "choice":
        criteria = value.get("criteria")
        if not isinstance(criteria, Mapping):
            raise ValueError("choice criteria must be a JSON object")
        return Choice(  # type: ignore[arg-type]
            instructions=instructions,
            criteria=criteria,
        )
    if question_type == "score":
        if "criteria" not in value:
            raise ValueError("score criteria is required")
        return Score(  # type: ignore[arg-type]
            instructions=instructions,
            criteria=value["criteria"],
        )
    if question_type == "noul":
        criteria = value.get("criteria")
        if criteria is not None and not isinstance(criteria, Mapping):
            raise ValueError("noul criteria must be a JSON object")
        return Noul(  # type: ignore[arg-type]
            instructions=instructions,
            criteria=criteria,
        )
    raise ValueError("question type must be choice, score, or noul")


def parse_request(payload: object) -> JevRequest:
    if not isinstance(payload, Mapping):
        raise ValueError("request body must be a JSON object")
    for field in ("state", "model", "questions"):
        if field not in payload:
            raise ValueError(f"{field} is required")

    questions = payload["questions"]
    if not isinstance(questions, Mapping):
        raise ValueError("questions must be a JSON object")
    return JevRequest(
        state=payload["state"],  # type: ignore[arg-type]
        model=payload["model"],  # type: ignore[arg-type]
        questions={
            question_id: parse_question(question)
            for question_id, question in questions.items()
        },
    )
