"""Drawing results: boxes, per-track decisions and a panel of full-frame decisions."""

from __future__ import annotations

from typing import TYPE_CHECKING

import cv2
import numpy as np

from s1ght.schema.systemone import Answer, BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer

if TYPE_CHECKING:
    from s1ght.results import Results

# BGR, Ultralytics-like palette
PALETTE = [
    (56, 56, 255), (151, 157, 255), (31, 112, 255), (29, 178, 255), (49, 210, 207), (10, 249, 72),
    (23, 204, 146), (134, 219, 61), (52, 147, 26), (187, 212, 0), (168, 153, 44), (255, 194, 0),
    (147, 69, 52), (255, 115, 100), (236, 24, 0), (255, 56, 132), (133, 0, 82), (255, 56, 203),
]  # fmt: skip
ON = (60, 200, 60)
OFF = (90, 90, 230)
NEUTRAL = (235, 235, 235)


def color(i: int) -> tuple[int, int, int]:
    return PALETTE[int(i) % len(PALETTE)]


def describe(qid: str, a: Answer, threshold: float = 0.5) -> tuple[str, tuple[int, int, int]]:
    """One line of text and its color for an answer."""
    if isinstance(a, NoulAnswer):
        yes = a.noul >= threshold
        return f"{qid}: {'yes' if yes else 'no'} {a.noul:.2f}", ON if yes else OFF
    if isinstance(a, ChoiceAnswer):
        return f"{qid}: {a.choice} {a.probabilities.get(a.choice, 0.0):.2f}", NEUTRAL
    if isinstance(a, ScoreAnswer):
        level = a.legend.get(round(a.score), "")
        label = level if isinstance(level, str) and len(level) <= 24 else ""
        return f"{qid}: {a.score:.2f}{f' ({label})' if label else ''}", NEUTRAL
    if isinstance(a, BoxAnswer):
        return f"{qid}: box {a.confidence:.2f}", NEUTRAL
    return f"{qid}: ?", NEUTRAL


def _text_block(
    img: np.ndarray,
    lines: list[tuple[str, tuple[int, int, int]]],
    origin: tuple[int, int],
    font_scale: float,
    thickness: int,
    below: bool = True,
) -> None:
    """Lines of text on a dark translucent box; `origin` is the box's top-left (or bottom-left)."""
    if not lines:
        return
    font = cv2.FONT_HERSHEY_SIMPLEX
    sizes = [cv2.getTextSize(t, font, font_scale, thickness)[0] for t, _ in lines]
    pad = max(2, int(4 * font_scale))
    line_h = max(h for _, h in sizes) + 2 * pad
    w = max(sw for sw, _ in sizes) + 2 * pad
    h = line_h * len(lines)
    H, W = img.shape[:2]
    x = int(min(max(origin[0], 0), max(W - w, 0)))
    y = int(origin[1] if below else origin[1] - h)
    y = min(max(y, 0), max(H - h, 0))
    roi = img[y : y + h, x : x + w]
    roi[:] = (roi * 0.35).astype(np.uint8)
    for i, (text, col) in enumerate(lines):
        baseline_y = y + i * line_h + line_h - pad - 1
        cv2.putText(img, text, (x + pad, baseline_y), font, font_scale, col, thickness, cv2.LINE_AA)


def draw(
    result: Results,
    *,
    boxes: bool = True,
    labels: bool = True,
    panel: bool = True,
    font_scale: float | None = None,
    line_width: int | None = None,
    threshold: float = 0.5,
) -> np.ndarray:
    """A copy of `result.orig_img` with detections and decisions drawn.

    Args:
        result: The frame's `Results`.
        boxes: Draw detection boxes.
        labels: Write class, track id and that track's decisions next to each box.
        panel: Write full-frame decisions (and events) in the top-left corner.
        font_scale: Text size; scales with the image when None.
        line_width: Box line width; scales with the image when None.
        threshold: Noul probability shown as "yes".
    """
    img = result.orig_img.copy()
    H, W = img.shape[:2]
    scale = max(H, W) / 1000
    fs = font_scale or max(0.4, 0.6 * scale)
    lw = line_width or max(1, round(2 * scale))
    th = max(1, lw // 2)

    b = result.boxes
    for i in range(len(b) if boxes else 0):
        x1, y1, x2, y2 = (int(v) for v in b.xyxy[i])
        tid = None if b.id is None else int(b.id[i])
        col = color(tid if tid is not None else int(b.cls[i]))
        cv2.rectangle(img, (x1, y1), (x2, y2), col, lw, cv2.LINE_AA)
        if not labels:
            continue
        name = b.names.get(int(b.cls[i]), str(int(b.cls[i])))
        head = f"{name}{f' #{tid}' if tid is not None else ''} {float(b.conf[i]):.2f}"
        lines = [(head, col)] + [describe(q, a, threshold) for q, a in result.answers.get(tid, {}).items()]
        _text_block(img, lines, (x1, y1), fs * 0.9, th, below=False)

    if panel:
        lines = [describe(q, a, threshold) for q, a in result.answers.get(None, {}).items()]
        lines += [(str(e), (0, 215, 255)) for e in result.events]
        _text_block(img, lines, (0, 0), fs, th)
    return img
