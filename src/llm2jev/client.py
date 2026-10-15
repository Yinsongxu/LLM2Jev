from __future__ import annotations

from collections.abc import Mapping
from functools import cached_property
from typing import Any

import httpx
from openai import NOT_GIVEN, APIResponseValidationError, AsyncOpenAI, NotGiven, OpenAI

from .core import (
    Answer, Box, BoxAnswer, Choice, ChoiceAnswer, JevRequest, JevResponse, Noul, NoulAnswer,
    Score, ScoreAnswer, State, Usage,
)
from .core.request import Question
from .inference.request_parser import parse_question


def _request_body(
    model: str,
    state: State,
    questions: Mapping[str, Question | Mapping[str, object]],
    include_choice_candidates: bool,
) -> dict[str, object]:
    if not isinstance(questions, Mapping):
        raise ValueError("questions must be a mapping")
    return JevRequest(
        model=model,
        state=state,
        questions={
            name: question if isinstance(question, (Box, Choice, Score, Noul)) else parse_question(question)
            for name, question in questions.items()
        },
        include_choice_candidates=include_choice_candidates,
    ).to_dict()


def _mapping(value: object) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError("expected a JSON object")
    return value


def _score_levels(value: object) -> dict[int, Any]:
    levels = _mapping(value)
    if any(not isinstance(key, str) or not key.isascii() or not key.isdecimal()
           or str(int(key)) != key for key in levels):
        raise ValueError("score level keys must be non-negative integer strings")
    return {int(key): item for key, item in levels.items()}


def _parse_response(response: httpx.Response) -> JevResponse:
    payload = None
    try:
        payload = response.json()
        data = _mapping(payload)
        answers: dict[str, Answer] = {}
        for name, value in _mapping(data["answers"]).items():
            answer = _mapping(value)
            if answer["type"] == "choice":
                answers[name] = ChoiceAnswer(
                    choice=answer["choice"], confidence=answer["confidence"],
                    probabilities=_mapping(answer["probabilities"]),
                )
            elif answer["type"] == "score":
                answers[name] = ScoreAnswer(
                    score=answer["score"], confidence=answer["confidence"],
                    legend=_score_levels(answer["legend"]),
                    probabilities=_score_levels(answer["probabilities"]),
                )
            elif answer["type"] == "box":
                answers[name] = BoxAnswer(
                    box=_mapping(answer["box"]), confidence=answer["confidence"],
                )
            elif answer["type"] == "noul":
                answers[name] = NoulAnswer(noul=answer["noul"])
            else:
                raise ValueError("answer type must be box, choice, score, or noul")
        usage = _mapping(data["usage"])
        return JevResponse(
            model=data["model"], answers=answers,
            usage=Usage(input_tokens=usage.get("input_tokens"), output_tokens=usage.get("output_tokens")),
        )
    except (KeyError, TypeError, ValueError) as error:
        raise APIResponseValidationError(
            response=response, body=payload, message="Invalid systemone response",
        ) from error


def _request_options(
    extra_headers: Mapping[str, str] | None,
    extra_query: Mapping[str, object] | None,
    timeout: float | httpx.Timeout | None | NotGiven,
) -> dict[str, object]:
    options: dict[str, object] = {}
    if extra_headers is not None:
        options["headers"] = extra_headers
    if extra_query is not None:
        options["params"] = extra_query
    if not isinstance(timeout, NotGiven):
        options["timeout"] = timeout
    return options


class _Systemone:
    def __init__(self, client: OpenAI) -> None:
        self._client = client

    def create(
        self,
        *,
        model: str,
        state: State,
        questions: Mapping[str, Question | Mapping[str, object]],
        include_choice_candidates: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> JevResponse:
        """Evaluate structured questions through POST /systemone."""
        response = self._client.post(
            "/systemone", cast_to=httpx.Response,
            body=_request_body(model, state, questions, include_choice_candidates),
            options=_request_options(extra_headers, extra_query, timeout),  # type: ignore[arg-type]
        )
        return _parse_response(response)


class _AsyncSystemone:
    def __init__(self, client: AsyncOpenAI) -> None:
        self._client = client

    async def create(
        self,
        *,
        model: str,
        state: State,
        questions: Mapping[str, Question | Mapping[str, object]],
        include_choice_candidates: bool = True,
        extra_headers: Mapping[str, str] | None = None,
        extra_query: Mapping[str, object] | None = None,
        timeout: float | httpx.Timeout | None | NotGiven = NOT_GIVEN,
    ) -> JevResponse:
        """Evaluate structured questions through POST /systemone asynchronously."""
        response = await self._client.post(
            "/systemone", cast_to=httpx.Response,
            body=_request_body(model, state, questions, include_choice_candidates),
            options=_request_options(extra_headers, extra_query, timeout),  # type: ignore[arg-type]
        )
        return _parse_response(response)


class Client(OpenAI):
    """OpenAI client with an additional systemone resource for llm2jev servers."""

    @cached_property
    def systemone(self) -> _Systemone:
        return _Systemone(self)


class AsyncClient(AsyncOpenAI):
    """Asynchronous OpenAI client with an additional systemone resource."""

    @cached_property
    def systemone(self) -> _AsyncSystemone:
        return _AsyncSystemone(self)
