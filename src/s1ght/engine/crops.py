"""Image preparation for runtimes: fit frames and crops to a runtime's size limits."""

from __future__ import annotations

import cv2
import numpy as np


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
