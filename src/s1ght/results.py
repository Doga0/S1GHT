"""Results: what `S1GHT.predict` and `S1GHT.track` yield per frame, Ultralytics-style.

for r in model.predict("bus.jpg", questions=...):
    r.answers[None]["has_people"].noul    # full-frame answers live under None
    r.plot()                              # BGR image with boxes and decisions drawn
    r.to_systemone()                      # TypeSafe responses, one per frame/track
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from s1ght.schema.systemone import Answer, SystemOneResponse


@dataclass
class Boxes:
    """Detections in one frame (all arrays have N rows).

    Attributes:
        xyxy: (N, 4) float pixel corners.
        conf: (N,) detector confidence.
        cls: (N,) integer class ids.
        id: (N,) track ids, or None when not tracking.
        names: Class id → class name.
    """

    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    id: np.ndarray | None = None
    names: dict[int, str] = field(default_factory=dict)

    @classmethod
    def empty(cls, names: dict[int, str] | None = None) -> Boxes:
        return cls(np.zeros((0, 4), np.float32), np.zeros(0, np.float32), np.zeros(0, np.int64), None, names or {})

    def __len__(self) -> int:
        return len(self.xyxy)

    def to_list(self) -> list[dict[str, Any]]:
        out = []
        for i in range(len(self)):
            c = int(self.cls[i])
            row: dict[str, Any] = {
                "xyxy": [round(float(v), 2) for v in self.xyxy[i]],
                "conf": round(float(self.conf[i]), 4),
                "cls": c,
                "name": self.names.get(c, str(c)),
            }
            if self.id is not None:
                row["id"] = int(self.id[i])
            out.append(row)
        return out


@dataclass
class Event:
    """A question crossing its threshold for a track (or the whole frame when `track_id` is None).

    Attributes:
        track_id: Track id, or None for full-frame questions.
        question: Question id.
        start: Seconds when the answer turned on.
        end: Seconds when it turned off, or None while still on.
        peak: Highest probability seen while on.
    """

    track_id: int | None
    question: str
    start: float
    end: float | None = None
    peak: float = 0.0

    @property
    def duration(self) -> float | None:
        return None if self.end is None else self.end - self.start

    def to_dict(self) -> dict[str, Any]:
        return {
            "track_id": self.track_id,
            "question": self.question,
            "start": self.start,
            "end": self.end,
            "peak": self.peak,
        }

    def __str__(self) -> str:
        who = "frame" if self.track_id is None else f"track {self.track_id}"
        span = "ongoing" if self.end is None else f"{self.duration:.1f}s"
        return f"{who}: {self.question} p={self.peak:.2f}, {span}"


@dataclass
class Results:
    """Decisions for one frame.

    Attributes:
        orig_img: The frame, BGR uint8.
        boxes: Detections (empty for full-frame `predict`).
        answers: Track id → {question id: answer}; key None holds full-frame answers.
        events: Events that started or ended on this frame.
        speed: Milliseconds per stage, e.g. {"preprocess": .., "decide": ..}.
        path: Where the frame came from (file, URL, "image0", ...).
        frame: Frame index in the source.
        model: Name of the decision model.
        usage: Token usage summed over this frame's requests.
    """

    orig_img: np.ndarray
    boxes: Boxes = field(default_factory=Boxes.empty)
    answers: dict[int | None, dict[str, Answer]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)
    speed: dict[str, float] = field(default_factory=dict)
    path: str = ""
    frame: int = 0
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)

    @property
    def orig_shape(self) -> tuple[int, int]:
        """(height, width)."""
        return self.orig_img.shape[0], self.orig_img.shape[1]

    def plot(self, **kwargs: Any) -> np.ndarray:
        """A copy of the frame with boxes, per-track decisions and the full-frame panel drawn (BGR).

        Keyword arguments go to `s1ght.io.plotting.draw` (e.g. `font_scale`, `panel=False`).
        """
        from s1ght.io.plotting import draw

        return draw(self, **kwargs)

    def save(self, filename: str | Path, **kwargs: Any) -> Path:
        """Write `plot()` to an image file and return its path."""
        import cv2

        path = Path(filename)
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(path), self.plot(**kwargs)):
            raise OSError(f"could not write {path}")
        return path

    def to_systemone(self) -> list[SystemOneResponse]:
        """One TypeSafe response per answered key: the full frame first, then tracks by id."""
        keys = sorted(self.answers, key=lambda k: (k is not None, k or 0))
        n = max(len(keys), 1)
        usage = {k: v // n for k, v in self.usage.items()}
        return [SystemOneResponse(model=self.model, answers=dict(self.answers[k]), usage=usage) for k in keys]

    def to_dict(self) -> dict[str, Any]:
        answers = {
            ("frame" if k is None else str(k)): {qid: a.to_wire() for qid, a in v.items()}
            for k, v in self.answers.items()
        }
        return {
            "path": self.path,
            "frame": self.frame,
            "shape": list(self.orig_shape),
            "model": self.model,
            "speed": {k: round(v, 2) for k, v in self.speed.items()},
            "boxes": self.boxes.to_list(),
            "answers": answers,
            "events": [e.to_dict() for e in self.events],
            "usage": self.usage,
        }

    def to_json(self, indent: int | None = None) -> str:
        """`to_dict()` as JSON; answers use the TypeSafe wire format."""
        return json.dumps(self.to_dict(), indent=indent)

    def __repr__(self) -> str:
        h, w = self.orig_shape
        n = sum(len(v) for v in self.answers.values())
        return f"Results(path={self.path!r}, shape={w}x{h}, boxes={len(self.boxes)}, answers={n}, events={len(self.events)})"
