"""Temporal fusion: smooth each track's answers over time and turn threshold crossings into events.

- Noul: exponential moving average in logit space, with a half-life in seconds. Two
  thresholds (hysteresis) switch it on and off, and `min_duration` must pass above the
  on-threshold before an event starts.
- Choice: moving average of the probability vector; the pick is its argmax.
- Score: moving average of the level probabilities; the score is their expected level.
- Box: the latest answer.

The weight of a new answer is `1 - 0.5 ** (dt / half_life)`, kept within [0.05, 1], where
`dt` is the time since that track's previous answer to the same question.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import numpy as np

from s1ght.results import Event
from s1ght.runtimes.base import confidence
from s1ght.schema.systemone import Answer, BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer

_EPS = 1e-6
MIN_WEIGHT = 0.05


def _logit(p: float) -> float:
    p = min(max(p, _EPS), 1 - _EPS)
    return math.log(p / (1 - p))


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


@dataclass
class _Track:
    t: float
    value: Answer
    logit: float = 0.0
    probs: np.ndarray | None = None
    keys: list | None = None
    on: bool = False
    pending: float | None = None
    event: Event | None = None


class Fusion:
    """Per-track temporal fusion of answers.

    Args:
        half_life: Seconds after which an old answer weighs half; 0 disables smoothing.
        on: Noul probability that starts an event.
        off: Noul probability that ends it (must be <= on).
        min_duration: Seconds the probability must stay above `on` before the event starts.
    """

    def __init__(self, half_life: float = 1.0, on: float = 0.6, off: float = 0.4, min_duration: float = 0.0) -> None:
        if not 0 <= off <= on <= 1:
            raise ValueError("need 0 <= off <= on <= 1")
        if half_life < 0 or min_duration < 0:
            raise ValueError("half_life and min_duration must be >= 0")
        self.half_life, self.on, self.off, self.min_duration = half_life, on, off, min_duration
        self._tracks: dict[int | None, dict[str, _Track]] = {}

    def _weight(self, dt: float) -> float:
        if self.half_life == 0:
            return 1.0
        return min(1.0, max(MIN_WEIGHT, 1 - 0.5 ** (max(dt, 0.0) / self.half_life)))

    def update(self, track_id: int | None, qid: str, answer: Answer, t: float) -> list[Event]:
        """Fold one answer in and return the events it starts or ends (snapshots)."""
        qs = self._tracks.setdefault(track_id, {})
        s = qs.get(qid)
        w = 1.0 if s is None else self._weight(t - s.t)

        if isinstance(answer, NoulAnswer):
            x = _logit(answer.noul)
            logit = x if s is None else (1 - w) * s.logit + w * x
            s = qs[qid] = replace(s, t=t, logit=logit) if s else _Track(t, answer, logit=logit)
            s.value = NoulAnswer(_sigmoid(logit))
            return self._hysteresis(track_id, qid, s, t)

        keys: list | None = None
        p: np.ndarray | None = None
        if isinstance(answer, ChoiceAnswer):
            options = list(answer.probabilities)
            p = self._mix(s, options, np.array([answer.probabilities[k] for k in options], float), w)
            keys = options
            value: Answer = ChoiceAnswer(options[int(p.argmax())], dict(zip(options, p.tolist())), confidence(p))
        elif isinstance(answer, ScoreAnswer):
            levels = sorted(answer.probabilities)
            p = self._mix(s, levels, np.array([answer.probabilities[k] for k in levels], float), w)
            keys = levels
            value = ScoreAnswer(
                float(np.dot(levels, p)), dict(answer.legend), dict(zip(levels, p.tolist())), confidence(p)
            )
        elif isinstance(answer, BoxAnswer):
            value = answer
        else:
            raise TypeError(f"cannot fuse {type(answer).__name__}")
        qs[qid] = _Track(t, value, probs=p, keys=keys)
        return []

    @staticmethod
    def _mix(s: _Track | None, keys: list, p: np.ndarray, w: float) -> np.ndarray:
        if s is not None and s.keys == keys and s.probs is not None:
            p = (1 - w) * s.probs + w * p
        return p / p.sum()

    def _hysteresis(self, track_id: int | None, qid: str, s: _Track, t: float) -> list[Event]:
        p = s.value.noul  # type: ignore[union-attr]
        if not s.on:
            if p < self.on:
                s.pending = None
                return []
            if s.pending is None:
                s.pending = t
            if t - s.pending < self.min_duration:
                return []
            s.on, s.event = True, Event(track_id, qid, start=s.pending, peak=p)
            return [replace(s.event)]
        assert s.event is not None
        s.event.peak = max(s.event.peak, p)
        if p > self.off:
            return []
        s.event.end = t
        ended = replace(s.event)
        s.on, s.event, s.pending = False, None, None
        return [ended]

    def state(self, track_id: int | None) -> dict[str, Answer]:
        """Fused answers for a track (empty when it has none)."""
        return {qid: s.value for qid, s in self._tracks.get(track_id, {}).items()}

    def close(self, track_id: int | None, t: float) -> list[Event]:
        """Forget a lost track; its open events end at `t`."""
        ended = []
        for s in self._tracks.pop(track_id, {}).values():
            if s.event is not None:
                s.event.end = t
                ended.append(replace(s.event))
        return ended

    def tracks(self) -> list[int | None]:
        return list(self._tracks)
