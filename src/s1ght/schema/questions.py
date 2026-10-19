"""Typed System One questions: Noul, Choice, Score and Box.

Field names match the TypeSafe `/v1/systemone` wire format one to one. `instructions` and
`criteria` may be strings, nested dicts/lists, or contain `Image` objects anywhere a value
can go; the request layer turns images into references (TypeSafe) or `images` (Ollama).
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import yaml

Content = Any
"""str, dict, list or Image, nested to any depth."""

MAX_CHOICE_OPTIONS = 255
SCORE_LEVELS = (2, 10)


def _check_instructions(instructions: Content) -> None:
    if instructions is None or (isinstance(instructions, str) and not instructions.strip()):
        raise ValueError("instructions must not be empty")


@dataclass(frozen=True)
class Noul:
    """Yes/no question; the answer is the probability of yes.

    Args:
        instructions: The yes/no question.
        criteria: Optional `{"true": ..., "false": ...}` descriptions.
    """

    instructions: Content
    criteria: dict[str, Content] | None = None

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if self.criteria is not None and (
            not isinstance(self.criteria, dict) or not set(self.criteria) <= {"true", "false"}
        ):
            raise ValueError("Noul criteria must be a dict with keys 'true' and/or 'false'")

    def to_wire(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": "noul", "instructions": self.instructions}
        if self.criteria is not None:
            out["criteria"] = self.criteria
        return out


@dataclass(frozen=True)
class Choice:
    """Pick one option from a set.

    Args:
        instructions: What to decide.
        criteria: `{option: description | None}`, or a list of option names (stored as
            `{option: None}`). 2 to 255 options.
    """

    instructions: Content
    criteria: dict[str, Content]

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        criteria = self.criteria
        if isinstance(criteria, (list, tuple)):
            if len(set(criteria)) != len(criteria):
                raise ValueError("Choice options must be unique")
            criteria = {option: None for option in criteria}
            object.__setattr__(self, "criteria", criteria)
        if not isinstance(criteria, dict) or not all(isinstance(k, str) for k in criteria):
            raise ValueError("Choice criteria must be a dict or list of option names")
        if not 2 <= len(criteria) <= MAX_CHOICE_OPTIONS:
            raise ValueError(f"Choice needs 2 to {MAX_CHOICE_OPTIONS} options, got {len(criteria)}")

    def to_wire(self) -> dict[str, Any]:
        return {"type": "choice", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Score:
    """Rate along ordered levels, lowest first; the answer is the probability-weighted level.

    Args:
        instructions: What to rate.
        criteria: 2 to 10 level descriptions, from lowest to highest.
    """

    instructions: Content
    criteria: list[Content]

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)
        if isinstance(self.criteria, tuple):
            object.__setattr__(self, "criteria", list(self.criteria))
        lo, hi = SCORE_LEVELS
        if not isinstance(self.criteria, list) or not lo <= len(self.criteria) <= hi:
            raise ValueError(f"Score needs a list of {lo} to {hi} levels")

    def to_wire(self) -> dict[str, Any]:
        return {"type": "score", "instructions": self.instructions, "criteria": self.criteria}


@dataclass(frozen=True)
class Box:
    """Find one target in an image; the answer is four fractional edges. Takes no criteria.

    Args:
        instructions: What to find and in which image, e.g. "the coffee mug in `photo`".
    """

    instructions: str

    def __post_init__(self) -> None:
        _check_instructions(self.instructions)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "box", "instructions": self.instructions}


Question: TypeAlias = Noul | Choice | Score | Box
QuestionSet = dict[str, Question]

QUESTION_TYPES: dict[str, type] = {"noul": Noul, "choice": Choice, "score": Score, "box": Box}


def question_from_wire(data: dict[str, Any]) -> Question:
    """Build a question from its wire dict (`{"type": ..., "instructions": ..., ...}`)."""
    fields = dict(data)
    kind = fields.pop("type", None)
    cls = QUESTION_TYPES.get(kind)  # type: ignore[arg-type]
    if cls is None:
        raise ValueError(f"unknown question type {kind!r}; expected one of {sorted(QUESTION_TYPES)}")
    try:
        return cls(**fields)
    except TypeError as e:
        raise ValueError(f"invalid {kind} question: {e}") from None


def questions_from_wire(data: dict[str, dict[str, Any]]) -> QuestionSet:
    """Build a `QuestionSet` from the wire `questions` map."""
    return {qid: question_from_wire(q) for qid, q in data.items()}


def load_questions(src: str | Path | dict[str, Any]) -> QuestionSet | dict[str, QuestionSet]:
    """Load questions from a YAML/JSON file or an already parsed dict.

    A flat map (`{qid: {type: ...}}`) returns a `QuestionSet`. A nested map keyed by
    detector class (`{"person": {qid: {type: ...}}}`) returns `dict[str, QuestionSet]`.
    """
    if isinstance(src, dict):
        data = src
    else:
        text = Path(src).read_text(encoding="utf-8")
        data = json.loads(text) if str(src).endswith(".json") else yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError("questions must be a mapping")
    if all(isinstance(v, dict) and "type" in v for v in data.values()):
        return questions_from_wire(data)
    return {cls: questions_from_wire(qs) for cls, qs in data.items()}
