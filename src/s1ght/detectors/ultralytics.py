"""Ultralytics adapter: consumes `model.track(source, stream=True, persist=True)`.

The adapter reads the generator's `Results` and converts them to numpy `Frame`s. It does not
subclass the predictor or register callbacks: `track()` appends its tracker callbacks on every
call, so a user callback registered earlier would run before tracking and never see track ids.

Ultralytics is AGPL-3.0 and an optional extra (`pip install 's1ght-visual-decisions[ultralytics]'`); it is
imported only when a detector is built. YOLO handles closed classes; YOLOE (open vocabulary via
`set_classes`) and other Ultralytics models load the same way.
"""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from s1ght.detectors.base import Frame
from s1ght.io.sources import is_live


def _require_ultralytics() -> Any:
    try:
        import ultralytics
    except ImportError as e:
        raise ImportError(
            "the Ultralytics detector needs: pip install 's1ght-visual-decisions[ultralytics]' (AGPL-3.0)"
        ) from e
    return ultralytics


def _fps(source: Any) -> float | None:
    if not isinstance(source, (str, int)) or is_live(source):
        return None
    import cv2

    cap = cv2.VideoCapture(str(source))
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 0.0
    finally:
        cap.release()
    return fps or None


class UltralyticsDetector:
    """YOLO/YOLOE detection and tracking through Ultralytics.

    Args:
        model: Weights name or path ("yolo11n.pt", "yoloe-11s-seg.pt", ...), or a loaded Ultralytics model.
        classes: Class names (or ids) to keep. For YOLOE, names not in the model's vocabulary
            are set as open-vocabulary prompts with `set_classes`.
        conf: Detection confidence threshold.
        tracker: Tracker config ("bytetrack.yaml" or "botsort.yaml").
        imgsz: Inference size.
        device: Ultralytics device string ("cpu", "0", ...); Ultralytics picks when None.
    """

    def __init__(
        self,
        model: Any = "yolo11n.pt",
        *,
        classes: list[str | int] | None = None,
        conf: float = 0.25,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        device: str | None = None,
    ) -> None:
        ul = _require_ultralytics()
        self.model = ul.YOLO(model) if isinstance(model, str) else model
        self.conf, self.tracker, self.imgsz, self.device = conf, tracker, imgsz, device
        self.class_ids = self._class_ids(classes)

    def _class_ids(self, classes: list[str | int] | None) -> list[int] | None:
        if classes is None:
            return None
        names: dict[int, str] = dict(self.model.names)
        by_name = {v: k for k, v in names.items()}
        unknown = [c for c in classes if isinstance(c, str) and c not in by_name]
        if unknown:
            if not hasattr(self.model, "set_classes"):
                raise ValueError(f"{unknown} are not classes of this model: {sorted(by_name)}")
            self.model.set_classes([str(c) for c in classes])  # YOLOE open vocabulary
            return None
        return [by_name[c] if isinstance(c, str) else int(c) for c in classes]

    @property
    def names(self) -> dict[int, str]:
        return dict(self.model.names)

    def stream(self, source: Any, **kwargs: Any) -> Iterator[Frame]:
        """Track `source` and yield a `Frame` per frame; extra kwargs go to `model.track`."""
        fps = _fps(source)
        stride = int(kwargs.get("vid_stride", 1))
        t0 = time.monotonic()
        args = {"conf": self.conf, "tracker": self.tracker, "imgsz": self.imgsz, "verbose": False, **kwargs}
        if self.device is not None:
            args.setdefault("device", self.device)
        if self.class_ids is not None:
            args.setdefault("classes", self.class_ids)
        for i, r in enumerate(self.model.track(source, stream=True, persist=True, **args)):
            b = r.boxes
            n = 0 if b is None else len(b)
            ids = None if b is None or b.id is None else b.id.cpu().numpy().astype(np.int64)
            yield Frame(
                image=r.orig_img,
                index=i,
                t=i * stride / fps if fps else time.monotonic() - t0,
                xyxy=b.xyxy.cpu().numpy().astype(np.float32) if n else np.zeros((0, 4), np.float32),
                conf=b.conf.cpu().numpy().astype(np.float32) if n else np.zeros(0, np.float32),
                cls=b.cls.cpu().numpy().astype(np.int64) if n else np.zeros(0, np.int64),
                ids=ids,
                names=dict(r.names),
                path=str(r.path),
                speed={k: float(v) for k, v in (r.speed or {}).items() if v is not None},
            )
