"""Runtime contract: what a decision back end can do, and how raw scores become answers.

A runtime runs the forward pass (in-process or over HTTP). Everything around it (crops,
scheduling, calibration, fusion) lives in the S1GHT engine and only talks to this protocol.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Protocol, Self, runtime_checkable

import numpy as np

from s1ght.schema.questions import Box, Choice, Noul, Question, QuestionSet, Score
from s1ght.schema.systemone import (
    Answer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)


class CapabilityError(RuntimeError):
    """A runtime lacks a capability that a request, localizer or feature needs."""


@dataclass(frozen=True)
class Capabilities:
    """What a runtime supports.

    Attributes:
        images: Accepts images in requests.
        box: Answers Box questions.
        batch: `decide_batch` runs requests together rather than one by one.
        prefix_cache: Reuses the encoded image/state across questions.
        attention: Exposes attention maps (in-process, eager attention).
        hidden_states: Exposes per-patch hidden states (in-process).
        raw_scores: `raw()` returns pre-calibration logits.
        max_options: Most Choice options per question.
        image_px: (min, max) accepted image side in pixels, or None if unrestricted.
    """

    images: bool = False
    box: bool = False
    batch: bool = False
    prefix_cache: bool = False
    attention: bool = False
    hidden_states: bool = False
    raw_scores: bool = False
    max_options: int = 255
    image_px: tuple[int, int] | None = None


_FLAGS = frozenset(f.name for f in fields(Capabilities) if f.type in ("bool", bool))


def require(caps: Capabilities, needs: frozenset[str] | set[str], what: str) -> None:
    """Raise `CapabilityError` at setup time if any boolean capability in `needs` is missing.

    Args:
        caps: The runtime's capabilities.
        needs: Capability field names, e.g. `{"attention"}` for the attention localizer.
        what: Who needs them, for the error message.
    """
    unknown = set(needs) - _FLAGS
    if unknown:
        raise ValueError(f"unknown capabilities: {sorted(unknown)}")
    missing = sorted(n for n in needs if not getattr(caps, n))
    if missing:
        raise CapabilityError(f"{what} needs {missing}, which this runtime does not support")


def validate_request(caps: Capabilities, req: SystemOneRequest) -> None:
    """Check a request against a runtime's capabilities before sending it."""
    if req.assets():
        require(caps, {"images"}, "a request with images")
        if caps.image_px is not None:
            lo, hi = caps.image_px
            for img in req.assets().values():
                w, h = img.size
                if not (lo <= w <= hi and lo <= h <= hi):
                    raise ValueError(f"image {img.ref} is {w}x{h}; this runtime accepts sides of {lo}-{hi} px")
    for qid, q in req.questions.items():
        if isinstance(q, Box):
            require(caps, {"box"}, f"Box question {qid!r}")
        if isinstance(q, Choice) and len(q.criteria) > caps.max_options:
            raise ValueError(f"Choice {qid!r} has {len(q.criteria)} options; this runtime allows {caps.max_options}")


@dataclass
class RawScores:
    """Pre-calibration logits per question, one per option.

    Option order: Noul → (yes, no); Choice → `criteria` order; Score → levels, lowest first.
    """

    scores: dict[str, np.ndarray]


def _softmax(logits: np.ndarray, temperature: float) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64) / temperature
    z = np.exp(z - z.max())
    return z / z.sum()


def confidence(p: np.ndarray) -> float:
    """TypeSafe confidence for a distribution over n options: (p_max − 1/n) / (1 − 1/n)."""
    n = len(p)
    return float((p.max() - 1 / n) / (1 - 1 / n))


def score_to_answer(question: Question, logits: np.ndarray, temperature: float = 1.0) -> Answer:
    """Turn one question's logits into a TypeSafe answer.

    Args:
        question: Noul, Choice or Score (Box answers are not logit-based).
        logits: One logit per option, in `RawScores` order.
        temperature: Calibration temperature; logits are divided by it before softmax.
    """
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=np.float64).reshape(-1)
    if isinstance(question, Noul):
        if logits.shape != (2,):
            raise ValueError(f"Noul needs 2 logits (yes, no), got {logits.shape[0]}")
        return NoulAnswer(noul=float(_softmax(logits, temperature)[0]))
    if isinstance(question, Choice):
        options = list(question.criteria)
        if len(logits) != len(options):
            raise ValueError(f"Choice has {len(options)} options, got {len(logits)} logits")
        p = _softmax(logits, temperature)
        return ChoiceAnswer(
            choice=options[int(p.argmax())],
            probabilities={o: float(v) for o, v in zip(options, p)},
            confidence=confidence(p),
        )
    if isinstance(question, Score):
        if len(logits) != len(question.criteria):
            raise ValueError(f"Score has {len(question.criteria)} levels, got {len(logits)} logits")
        p = _softmax(logits, temperature)
        return ScoreAnswer(
            score=float(np.dot(np.arange(len(p)), p)),
            legend=dict(enumerate(question.criteria)),
            probabilities={i: float(v) for i, v in enumerate(p)},
            confidence=confidence(p),
        )
    raise TypeError(f"{type(question).__name__} answers are not computed from logits")


def answers_to_raw(questions: QuestionSet, answers: dict[str, Answer], eps: float = 1e-9) -> RawScores:
    """Log-probabilities of already computed answers, in `RawScores` order.

    For back ends that only return probabilities (servers, a model's own readout): softmax of
    these logs gives the probabilities back, so S1GHT's calibration can be applied on top.
    Box answers and missing answers are skipped.
    """
    scores: dict[str, np.ndarray] = {}
    for qid, q in questions.items():
        a = answers.get(qid)
        if isinstance(q, Noul) and isinstance(a, NoulAnswer):
            p = np.array([a.noul, 1 - a.noul])
        elif isinstance(q, Choice) and isinstance(a, ChoiceAnswer):
            p = np.array([a.probabilities.get(o, 0.0) for o in q.criteria])
        elif isinstance(q, Score) and isinstance(a, ScoreAnswer):
            p = np.array([a.probabilities.get(i, 0.0) for i in range(len(q.criteria))])
        else:
            continue
        scores[qid] = np.log(np.clip(p, eps, 1.0))
    return RawScores(scores)


def scores_to_answers(
    questions: QuestionSet, raw: RawScores, temperature: float | dict[str, float] = 1.0
) -> dict[str, Answer]:
    """Answers for every question that has scores in `raw`.

    Args:
        questions: The request's questions.
        raw: Logits per question id.
        temperature: One temperature for all, or per question id (missing ids use 1.0).
    """
    out: dict[str, Answer] = {}
    for qid, logits in raw.scores.items():
        t = temperature.get(qid, 1.0) if isinstance(temperature, dict) else temperature
        out[qid] = score_to_answer(questions[qid], logits, t)
    return out


@runtime_checkable
class Runtime(Protocol):
    """A decision back end. Implementations must be safe to call from one worker thread."""

    caps: Capabilities

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        """Answer one request."""
        ...

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        """Answer several requests; same order as `reqs`."""
        ...

    def raw(self, req: SystemOneRequest) -> RawScores:
        """Pre-calibration logits; raises `NotImplementedError` when `caps.raw_scores` is False."""
        ...

    def close(self) -> None:
        """Release models, sessions and connections."""
        ...


class RuntimeBase:
    """Optional base with default `decide_batch` (one by one), `raw` and `close`."""

    caps: Capabilities = Capabilities()

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        raise NotImplementedError

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        return [self.decide(r) for r in reqs]

    def raw(self, req: SystemOneRequest) -> RawScores:
        raise NotImplementedError(f"{type(self).__name__} does not expose raw scores")

    def close(self) -> None:
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
