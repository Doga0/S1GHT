"""Detector protocol and the per-frame detections it yields."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from typing import Any, Protocol, runtime_checkable

import numpy as np


@dataclass
class Frame:
    """One frame with its detections (numpy arrays, N rows).

    Attributes:
        image: BGR uint8 frame.
        index: Frame number in the source.
        t: Seconds: position in a video file, or time since a live source started.
        xyxy: (N, 4) pixel boxes.
        conf: (N,) detection confidence.
        cls: (N,) class ids.
        ids: (N,) track ids, or None when the detector does not track.
        names: Class id → name.
        path: Source name.
        speed: Detector milliseconds per stage.
    """

    image: np.ndarray
    index: int
    t: float
    xyxy: np.ndarray
    conf: np.ndarray
    cls: np.ndarray
    ids: np.ndarray | None
    names: dict[int, str]
    path: str = ""
    speed: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.xyxy)

    def class_name(self, i: int) -> str:
        c = int(self.cls[i])
        return self.names.get(c, str(c))


@runtime_checkable
class Detector(Protocol):
    """Yields detections, with track ids when tracking, frame by frame. Must never wait on decisions."""

    def stream(self, source: Any, **kwargs: Any) -> Iterator[Frame]: ...
