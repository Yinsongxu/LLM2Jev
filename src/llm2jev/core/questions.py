from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import ClassVar

from ..utils.json import is_json_content
from .types import JSONContent
from .multimodal import validate_multimodal


def _validate_instructions(instructions: JSONContent | None) -> None:
    if instructions is not None and not is_json_content(instructions):
        raise ValueError("instructions must be a string, JSON object, JSON array, or None")
    validate_multimodal(instructions)


@dataclass(frozen=True, slots=True, kw_only=True)
class Box:
    """Locate one target in an image, optionally using reference images."""

    instructions: JSONContent

    type: ClassVar[str] = "box"

    def __post_init__(self) -> None:
        if self.instructions is None:
            raise ValueError("box instructions are required")
        _validate_instructions(self.instructions)
        if isinstance(self.instructions, str) and not self.instructions.strip():
            raise ValueError("box instructions must not be blank")

    def to_dict(self) -> dict[str, object]:
        return {"type": self.type, "instructions": self.instructions}


@dataclass(frozen=True, slots=True, kw_only=True)
class Choice:
    """A question that selects one option from a set of named choices."""

    criteria: Mapping[str, JSONContent | None]
    instructions: JSONContent | None = None

    type: ClassVar[str] = "choice"

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        criteria = dict(self.criteria)
        if any(not isinstance(name, str) or not name for name in criteria):
            raise ValueError("choice option names must be non-empty strings")
        if any(description is not None and not is_json_content(description) for description in criteria.values()):
            raise ValueError("choice descriptions must be JSON content or None")
        object.__setattr__(self, "criteria", criteria)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "criteria": dict(self.criteria),
        }
        if self.instructions is not None:
            result["instructions"] = self.instructions
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class Score:
    """A question that rates content against ordered descriptive levels."""

    criteria: Sequence[JSONContent]
    instructions: JSONContent | None = None

    type: ClassVar[str] = "score"

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        if isinstance(self.criteria, (str, bytes)):
            raise ValueError("score criteria must be a sequence of level descriptions")
        criteria = tuple(self.criteria)
        if not 2 <= len(criteria) <= 10:
            raise ValueError("score criteria must contain between 2 and 10 levels")
        if any(not is_json_content(description) for description in criteria):
            raise ValueError("score level descriptions must be JSON content")
        object.__setattr__(self, "criteria", criteria)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {
            "type": self.type,
            "criteria": list(self.criteria),
        }
        if self.instructions is not None:
            result["instructions"] = self.instructions
        return result


@dataclass(frozen=True, slots=True, kw_only=True)
class Noul:
    """A yes/no question that produces the probability that the answer is yes."""

    instructions: JSONContent | None = None
    criteria: Mapping[str, JSONContent | None] | None = None

    type: ClassVar[str] = "noul"

    def __post_init__(self) -> None:
        _validate_instructions(self.instructions)
        if self.criteria is None:
            return
        criteria = dict(self.criteria)
        if not set(criteria) <= {"true", "false"}:
            raise ValueError('noul criteria may only contain "true" and "false"')
        if any(description is not None and not is_json_content(description) for description in criteria.values()):
            raise ValueError("noul descriptions must be JSON content or None")
        object.__setattr__(self, "criteria", criteria)

    def to_dict(self) -> dict[str, object]:
        result: dict[str, object] = {"type": self.type}
        if self.instructions is not None:
            result["instructions"] = self.instructions
        if self.criteria is not None:
            result["criteria"] = dict(self.criteria)
        return result
