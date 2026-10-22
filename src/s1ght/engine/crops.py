"""Image preparation for runtimes: crops around detections, appearance signatures, size limits."""

from __future__ import annotations

from collections.abc import Sequence

import cv2
import numpy as np

SIGNATURE_PX = 32


def expand_box(xyxy: Sequence[float], shape: tuple[int, ...], pad: float = 0.2) -> tuple[int, int, int, int]:
    """Grow a box by `pad` of its width/height on every side and clip it to the image.

    State questions need context around the object: a bare crop of a fur coat scored only
    +0.5 nat for "cat" in YOLOJ. Returns integer pixel corners with at least 1 px per side.
    """
    h, w = shape[:2]
    x1, y1, x2, y2 = (float(v) for v in xyxy)
    dx, dy = (x2 - x1) * pad, (y2 - y1) * pad
    nx1, ny1 = max(0, int(np.floor(x1 - dx))), max(0, int(np.floor(y1 - dy)))
    nx2, ny2 = min(w, int(np.ceil(x2 + dx))), min(h, int(np.ceil(y2 + dy)))
    return nx1, ny1, max(nx2, nx1 + 1), max(ny2, ny1 + 1)


def crop(img: np.ndarray, xyxy: Sequence[float], pad: float = 0.2) -> np.ndarray:
    """A copy of the expanded box region (safe to hand to another thread)."""
    x1, y1, x2, y2 = expand_box(xyxy, img.shape, pad)
    return img[y1:y2, x1:x2].copy()


def signature(img: np.ndarray) -> np.ndarray:
    """A 32×32 color thumbnail in [0, 1], used to notice when a track's appearance changes.

    Color, not grayscale: a red and a blue helmet can have the same gray level, and color-only
    changes (a vest put on, a helmet swapped) are exactly the state changes worth re-asking about.
    """
    bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR) if img.ndim == 2 else img
    small = cv2.resize(bgr, (SIGNATURE_PX, SIGNATURE_PX), interpolation=cv2.INTER_AREA)
    return small.astype(np.float32) / 255.0


def signature_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Mean absolute difference of two signatures after removing each one's overall brightness (0 = same)."""
    return float(np.mean(np.abs((a - a.mean()) - (b - b.mean()))))


def with_frame(crop_img: np.ndarray, frame: np.ndarray, max_side: int = 512) -> dict[str, np.ndarray]:
    """The `context="crop+frame"` state: the crop plus a downscaled full frame for relation questions."""
    h, w = frame.shape[:2]
    s = min(1.0, max_side / max(h, w))
    small = cv2.resize(frame, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)
    return {"image": crop_img, "frame": small}


def fit_to_limits(img: np.ndarray, image_px: tuple[int, int] | None) -> np.ndarray:
    """Resize (and if needed pad) so both sides are within `image_px = (min_side, max_side)`.

    TypeSafe rejects images outside 280–1568 px rather than resizing them. The long side is
    scaled down to `max_side` and a short side below `min_side` is scaled up; when one scale
    cannot satisfy both (very elongated images), the short side is padded with black, centered.
    The input is returned unchanged when it already fits or `image_px` is None.
    """
    if image_px is None:
        return img
    lo, hi = image_px
    h, w = img.shape[:2]
    if lo <= min(h, w) and max(h, w) <= hi:
        return img
    s = min(1.0, hi / max(h, w))
    if min(h, w) * s < lo and max(h, w) * (lo / min(h, w)) <= hi:
        s = lo / min(h, w)
    nh, nw = max(1, round(h * s)), max(1, round(w * s))
    out = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
    ph, pw = max(0, lo - nh), max(0, lo - nw)
    if ph or pw:
        out = cv2.copyMakeBorder(out, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_CONSTANT, value=0)
    return out
