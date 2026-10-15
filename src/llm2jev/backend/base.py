from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol

from ..core.response import Usage
from ..utils.probability import validate_probabilities

if TYPE_CHECKING:
    from ..inference.prompt import ChatPrompt


@dataclass(frozen=True, slots=True, kw_only=True)
class ScoreOutput:
    """Ordered yes probabilities and usage returned by a binary backend."""

    yes_probabilities: Sequence[float]
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        probabilities = validate_probabilities(self.yes_probabilities)
        if not isinstance(self.usage, Usage):
            raise ValueError("usage must be a Usage instance")
        object.__setattr__(self, "yes_probabilities", probabilities)


class Backend(Protocol):
    """Inference backend providing binary scoring and coordinate generation."""

    def score(
        self,
        *,
        model: str,
        prompts: Sequence[ChatPrompt],
    ) -> ScoreOutput:
        """Score prompts in input order without generating text."""

    def generate(
        self, *, model: str, prompts: Sequence[ChatPrompt], max_new_tokens: int,
    ) -> GenerationOutput:
        """Generate complete texts and sampled-token probabilities in prompt order."""


@dataclass(frozen=True, slots=True, kw_only=True)
class GenerationOutput:
    """Token texts and full-vocabulary sampled-token probabilities, in prompt order."""

    token_texts: Sequence[Sequence[str]]
    token_probabilities: Sequence[Sequence[float]]
    usage: Usage = field(default_factory=Usage)

    def __post_init__(self) -> None:
        pieces = tuple(tuple(row) for row in self.token_texts)
        probabilities = tuple(tuple(row) for row in self.token_probabilities)
        if len(pieces) != len(probabilities):
            raise ValueError("each generated result must have token texts and probabilities")
        for row, values in zip(pieces, probabilities):
            if len(row) != len(values):
                raise ValueError("each generated token must have a probability")
            if any(not 0 <= value <= 1 for value in values):
                raise ValueError("each generated token must have a probability in [0, 1]")
        object.__setattr__(self, "token_texts", pieces)
        object.__setattr__(self, "token_probabilities", probabilities)

    @property
    def texts(self) -> tuple[str, ...]:
        return tuple("".join(row) for row in self.token_texts)
