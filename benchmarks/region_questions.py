"""E1 (decision-space detection): can a frozen decision model answer questions about named regions?

    python benchmarks/region_questions.py --models qwen3-vl-4b d1-omni-600m --device cuda --out e1.json
    python benchmarks/region_questions.py --images 50 --grid 4 --methods grid crop
    python benchmarks/region_questions.py --methods --line-methods line strip  # E1b

E0 (`decision_ceiling.py`) showed what decision-space detection can reach with perfect answers:
an 8x8 grid of cell questions, then threshold questions about each object's edges, reaches
0.469 COCO mAP (YOLO26n: 0.429). E1 measures whether a model gives those answers, and which way
of naming a region in the prompt works. Synthetic images first: coloured shapes on a noisy
background, each class at most once per image, so "the {class}" names one object.

Region questions (a Noul each, asked for classes present in the image, as after a presence gate):
- **occupancy:** "Does any part of the {class} lie in {region}?" Yes when the shape's pixels cover
  any of the cell.
- **center:** "Is the center of the {class} in {region}?" Yes in the cell holding its box center.

Ways to name a region (`--methods`), on a `--grid` x `--grid` grid:
- `text`: coordinates as shares of the image ("x from 0.25 to 0.50, y from 0.50 to 0.75").
- `grid`: the grid drawn on the image with column letters and row numbers; questions name a cell
  ("cell C5"). One image for all questions, so a prefix cache or vLLM's prefix caching can share it.
- `mark`: the cell outlined in red on its own copy of the image ("inside the red rectangle").
- `crop`: the cell cropped and enlarged ("in `photo`"), as P2.0's tiles.
- `crops`: every cell's crop in one request (`cell_A1`, `cell_A2` ...), the question naming one;
  only for grids up to 4x4.

Boundary questions, for a side of the object and a line at a known offset from its true edge
("Is the whole {class} to the right of {line}?"; also left of, above, below):
- `text`: the line as a share of the image ("the vertical line at 0.40 of the width").
- `line`: the line drawn in red on its own copy of the image.
- `strip` (E1b): no line; the region beyond it is framed in red, across the object's band (the
  rows, or columns, of its occupied `--grid` cells, as a decoder would know them), and the
  question is an occupancy one: "Does any part of the {class} lie inside the red rectangle?"
  Yes means the object reaches past the line, the opposite of the other methods, so its P(yes)
  is flipped before scoring. E1 found occupancy of a framed region answered at AUC 0.99 and the
  line questions at 0.72; this asks for the same edge in the form that works.
Offsets: +-0.02, 0.05, 0.1, 0.2 of the image side.

Presence (E1b, unless `--no-presence`): "Is there a {class} in `photo`?" for every class present
and as many absent ones, on the whole image. It tells a model that cannot see the shapes apart
from one that cannot place them.

Reported per model and method: AUC of P(yes); AUC against neighbouring cells only (the hard
negatives a decoder must reject); recall at 0.5 by how much of the cell the shape covers; for
boundaries, accuracy by offset and the edge error after projecting each side's answers onto a
monotone curve (isotonic regression) and reading where it crosses 0.5; ms per question.

Negatives are sampled: per (image, class), every occupied cell plus as many empty ones (half
next to the shape), at most `--cells` in all; for centers the center cell, up to 3 other
occupied cells and one far cell.
"""

from __future__ import annotations

import argparse
import json
import random
import string
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw, ImageFont

from s1ght import Image, Noul

COLORS = {"blue": (40, 90, 220), "green": (40, 160, 60), "orange": (240, 140, 20), "brown": (120, 70, 30)}
SHAPES = ("circle", "square", "triangle")
MARK = (230, 0, 0)  # grids, frames and lines: red, a colour no shape has
OFFSETS = (0.02, 0.05, 0.1, 0.2)
SIDES = {  # side -> (question wording, box index, sign: True when edge > line)
    "left": ("to the right of", 0, True),
    "top": ("below", 1, True),
    "right": ("to the left of", 2, False),
    "bottom": ("above", 3, False),
}
REGION_METHODS = ("text", "grid", "mark", "crop", "crops")
LINE_METHODS = ("text", "line", "strip")
FLIPPED = {"strip"}  # methods whose yes means the edge is NOT past the line


# --- synthetic scenes -------------------------------------------------------------------------


@dataclass
class Shape:
    name: str  # "blue circle"
    box: np.ndarray  # xyxy pixels
    mask: np.ndarray  # (h, w) bool


@dataclass
class Scene:
    image: PILImage.Image
    shapes: list[Shape] = field(default_factory=list)


def make_scene(rng: random.Random, size: int = 448, max_shapes: int = 3, tries: int = 50) -> Scene:
    """Coloured shapes (distinct classes, little overlap) on a noisy light background."""
    nprng = np.random.default_rng(rng.randrange(2**32))
    base = np.array([rng.randint(170, 235) for _ in range(3)], np.float32)
    pixels = np.clip(base + nprng.normal(0, 12, (size, size, 3)), 0, 255).astype(np.uint8)
    image = PILImage.fromarray(pixels)
    draw = ImageDraw.Draw(image)
    for _ in range(rng.randint(2, 5)):  # grey distractor blobs
        x, y, r = rng.randrange(size), rng.randrange(size), rng.randint(8, 40)
        g = rng.randint(90, 160)
        draw.ellipse((x - r, y - r, x + r, y + r), fill=(g, g, g))
    classes = [f"{c} {s}" for c in COLORS for s in SHAPES]
    scene = Scene(image)
    for name in rng.sample(classes, rng.randint(1, max_shapes)):
        for _ in range(tries):
            w, h = rng.randint(24, size // 2), rng.randint(24, size // 2)
            if name.endswith(("circle", "square")):
                h = w
            x0, y0 = rng.randint(0, size - w), rng.randint(0, size - h)
            box = np.array([x0, y0, x0 + w, y0 + h], np.float64)
            if all(_iou(box, s.box) < 0.05 for s in scene.shapes):
                break
        else:
            continue
        layer = PILImage.new("L", (size, size), 0)
        _draw_shape(ImageDraw.Draw(layer), name.split()[1], box, 255)
        _draw_shape(draw, name.split()[1], box, COLORS[name.split()[0]])
        scene.shapes.append(Shape(name, box, np.array(layer) > 0))
    for i, s in enumerate(scene.shapes):  # later shapes are drawn over earlier ones
        for t in scene.shapes[i + 1 :]:
            s.mask &= ~t.mask
    scene.shapes = [s for s in scene.shapes if s.mask.any()]
    for s in scene.shapes:  # truth: the visible pixels
        ys, xs = np.nonzero(s.mask)
        s.box = np.array([xs.min(), ys.min(), xs.max() + 1, ys.max() + 1], np.float64)
    return scene


def _draw_shape(draw: ImageDraw.ImageDraw, shape: str, box: np.ndarray, fill: Any) -> None:
    x0, y0, x1, y1 = box.tolist()
    if shape == "circle":
        draw.ellipse((x0, y0, x1 - 1, y1 - 1), fill=fill)
    elif shape == "square":
        draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=fill)
    else:
        draw.polygon([((x0 + x1) / 2, y0), (x1 - 1, y1 - 1), (x0, y1 - 1)], fill=fill)


def _iou(a: np.ndarray, b: np.ndarray) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


# --- cells and truth --------------------------------------------------------------------------


def cell_edges(size: int, g: int) -> np.ndarray:
    return np.round(np.linspace(0, size, g + 1)).astype(np.int64)


def coverage(mask: np.ndarray, g: int) -> np.ndarray:
    """(g, g): the share of each cell the mask covers."""
    ys, xs = cell_edges(mask.shape[0], g), cell_edges(mask.shape[1], g)
    return np.array([[mask[ys[r] : ys[r + 1], xs[c] : xs[c + 1]].mean() for c in range(g)] for r in range(g)])


def center_cell(box: np.ndarray, width: int, height: int, g: int) -> tuple[int, int]:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return min(int(cy / height * g), g - 1), min(int(cx / width * g), g - 1)


@dataclass
class Probe:
    """One region question with its truth."""

    scene: int
    shape: str
    kind: str  # occupancy, center
    cell: tuple[int, int]
    truth: bool
    cover: float  # share of the cell the shape covers
    near: bool  # the cell touches an occupied cell (8-neighbours) or is occupied


def sample_cells(scene: Scene, index: int, g: int, cells: int, rng: random.Random) -> list[Probe]:
    """Occupancy and center probes for each shape: positives and sampled negatives."""
    w, h = scene.image.size
    out = []
    for s in scene.shapes:
        cov = coverage(s.mask, g)
        occ = cov > 0
        grown = occ.copy()
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                grown |= np.roll(np.roll(occ, dr, 0), dc, 1) & _valid_shift(g, dr, dc)
        pos = [tuple(map(int, rc)) for rc in np.argwhere(occ)]
        near = [tuple(map(int, rc)) for rc in np.argwhere(grown & ~occ)]
        far = [tuple(map(int, rc)) for rc in np.argwhere(~grown)]
        rng.shuffle(pos)
        rng.shuffle(near)
        rng.shuffle(far)
        n_pos = min(len(pos), cells // 2)
        n_near = min(len(near), (cells - n_pos) // 2)
        n_far = min(len(far), cells - n_pos - n_near)
        picked = pos[:n_pos] + near[:n_near] + far[:n_far]
        for rc in picked:
            out.append(Probe(index, s.name, "occupancy", rc, bool(occ[rc]), float(cov[rc]), bool(grown[rc])))
        c = center_cell(s.box, w, h, g)
        others = [rc for rc in pos if rc != c][:3] + far[:1]
        for rc in [c, *others]:
            out.append(Probe(index, s.name, "center", rc, rc == c, float(cov[rc]), bool(grown[rc])))
    return out


def _valid_shift(g: int, dr: int, dc: int) -> np.ndarray:
    """Mask that drops np.roll's wrap-around for a (dr, dc) shift."""
    m = np.ones((g, g), bool)
    if dr > 0:
        m[:dr] = False
    elif dr < 0:
        m[dr:] = False
    if dc > 0:
        m[:, :dc] = False
    elif dc < 0:
        m[:, dc:] = False
    return m


@dataclass
class LineProbe:
    """One boundary question: is the whole shape beyond a line? `at` is the line's share of the side."""

    scene: int
    shape: str
    side: str
    offset: float  # line minus true edge, as a share of the image side
    at: float
    truth: bool


def sample_lines(scene: Scene, index: int) -> list[LineProbe]:
    w, h = scene.image.size
    out = []
    for s in scene.shapes:
        for side, (_, k, beyond) in SIDES.items():
            extent = w if k % 2 == 0 else h
            edge = s.box[k] / extent
            for off in OFFSETS:
                for d in (-off, off):
                    at = edge + d
                    if 0.01 <= at <= 0.99:
                        out.append(LineProbe(index, s.name, side, d, at, (edge > at) if beyond else (edge < at)))
    return out


def sample_presence(scene: Scene, index: int, rng: random.Random) -> list[tuple[int, str, bool]]:
    """(scene, class, present) for every present class and as many absent ones."""
    present = [s.name for s in scene.shapes]
    absent = [f"{c} {s}" for c in COLORS for s in SHAPES if f"{c} {s}" not in present]
    return [(index, n, True) for n in present] + [(index, n, False) for n in rng.sample(absent, len(present))]


# --- prompts ----------------------------------------------------------------------------------


def label(cell: tuple[int, int]) -> str:
    """Column letter, row number: (row 4, col 2) -> "C5"."""
    return f"{string.ascii_uppercase[cell[1]]}{cell[0] + 1}"


def draw_grid(image: PILImage.Image, g: int) -> PILImage.Image:
    out = image.copy()
    draw = ImageDraw.Draw(out)
    xs, ys = cell_edges(out.size[0], g), cell_edges(out.size[1], g)
    for x in xs[1:-1]:
        draw.line((x, 0, x, out.size[1]), fill=MARK, width=2)
    for y in ys[1:-1]:
        draw.line((0, y, out.size[0], y), fill=MARK, width=2)
    side = int(min(np.diff(xs).min(), np.diff(ys).min()))
    try:  # labels about a quarter of a cell high (Pillow >= 10.1 scales the default font)
        font: Any = ImageFont.load_default(size=max(10, side // 4))
    except TypeError:
        font = ImageFont.load_default()
    for r in range(g):
        for c in range(g):
            draw.text((xs[c] + 3, ys[r] + 2), label((r, c)), fill=MARK, font=font)
    return out


def region_text(cell: tuple[int, int], g: int) -> str:
    r, c = cell
    return f"the region of `photo` from x = {c / g:.2f} to {(c + 1) / g:.2f} and y = {r / g:.2f} to {(r + 1) / g:.2f}"


def region_request(method: str, scene: Scene, probe: Probe, g: int, crop_side: int) -> tuple[dict[str, Any], str]:
    """(state, question text) for one region probe; the state's images are PIL images."""
    shape = probe.shape
    ask = {
        "occupancy": "Does any part of the {shape} lie in {where}?",
        "center": "Is the center of the {shape} in {where}?",
    }[probe.kind]
    if method == "text":
        where = region_text(probe.cell, g) + " (x from the left edge, y from the top, as shares of the image)"
        return {"photo": scene.image}, ask.format(shape=shape, where=where)
    if method == "grid":
        note = (
            f"`photo` has a red {g} x {g} grid; columns are lettered A-{string.ascii_uppercase[g - 1]} from "
            f"left to right and rows numbered 1-{g} from top to bottom. "
        )
        return {"photo": draw_grid(scene.image, g)}, note + ask.format(shape=shape, where=f"cell {label(probe.cell)}")
    if method == "mark":
        xs, ys = cell_edges(scene.image.size[0], g), cell_edges(scene.image.size[1], g)
        r, c = probe.cell
        marked = scene.image.copy()
        ImageDraw.Draw(marked).rectangle((xs[c], ys[r], xs[c + 1] - 1, ys[r + 1] - 1), outline=MARK, width=3)
        return {"photo": marked}, ask.format(shape=shape, where="the red rectangle in `photo`")
    if method == "crop":
        return {"photo": crop(scene.image, probe.cell, g, crop_side)}, ask.format(shape=shape, where="`photo`")
    if method == "crops":
        state = {f"cell_{label((r, c))}": crop(scene.image, (r, c), g, crop_side) for r in range(g) for c in range(g)}
        note = (
            f"The images `cell_A1` ... `cell_{label((g - 1, g - 1))}` are the cells of a {g} x {g} grid over one "
            "photo (letter: column from the left, number: row from the top). "
        )
        return state, note + ask.format(shape=shape, where=f"`cell_{label(probe.cell)}`")
    raise ValueError(f"unknown region method {method!r}")


def crop(image: PILImage.Image, cell: tuple[int, int], g: int, side: int) -> PILImage.Image:
    xs, ys = cell_edges(image.size[0], g), cell_edges(image.size[1], g)
    r, c = cell
    return image.crop((xs[c], ys[r], xs[c + 1], ys[r + 1])).resize((side, side), PILImage.Resampling.BICUBIC)


def strip_rect(scene: Scene, probe: LineProbe, g: int) -> tuple[int, int, int, int]:
    """The region beyond the line (toward the image border the side faces), across the object's
    band: the rows (vertical line) or columns (horizontal line) of its occupied cells."""
    w, h = scene.image.size
    _, k, _ = SIDES[probe.side]
    shape = next(s for s in scene.shapes if s.name == probe.shape)
    occ = np.argwhere(coverage(shape.mask, g) > 0)
    (r0, c0), (r1, c1) = occ.min(0), occ.max(0)
    xs, ys = cell_edges(w, g), cell_edges(h, g)
    if k % 2 == 0:  # left/right: a vertical line at x
        x = round(probe.at * w)
        x0, x1 = (0, x) if k == 0 else (x, w)
        return int(x0), int(ys[r0]), int(x1), int(ys[r1 + 1])
    y = round(probe.at * h)
    y0, y1 = (0, y) if k == 1 else (y, h)
    return int(xs[c0]), int(y0), int(xs[c1 + 1]), int(y1)


def line_request(method: str, scene: Scene, probe: LineProbe, g: int = 8) -> tuple[dict[str, Any], str]:
    words, k, _ = SIDES[probe.side]
    vertical = k % 2 == 0
    if method == "strip":
        x0, y0, x1, y1 = strip_rect(scene, probe, g)
        marked = scene.image.copy()
        ImageDraw.Draw(marked).rectangle((x0, y0, x1 - 1, y1 - 1), outline=MARK, width=3)
        return {"photo": marked}, f"Does any part of the {probe.shape} lie inside the red rectangle in `photo`?"
    if method == "text":
        where = (
            f"the vertical line at {probe.at:.2f} of the width of `photo` (0 is the left edge)"
            if vertical
            else f"the horizontal line at {probe.at:.2f} of the height of `photo` (0 is the top edge)"
        )
        return {"photo": scene.image}, f"Is the whole {probe.shape} {words} {where}?"
    if method == "line":
        w, h = scene.image.size
        marked = scene.image.copy()
        draw = ImageDraw.Draw(marked)
        if vertical:
            x = round(probe.at * w)
            draw.line((x, 0, x, h), fill=MARK, width=2)
        else:
            y = round(probe.at * h)
            draw.line((0, y, w, y), fill=MARK, width=2)
        kind = "vertical" if vertical else "horizontal"
        return {"photo": marked}, f"Is the whole {probe.shape} {words} the red {kind} line in `photo`?"
    raise ValueError(f"unknown line method {method!r}")


# --- asking -----------------------------------------------------------------------------------


def ask_all(model: Any, requests: list[tuple[dict[str, Any], str]], batch: int) -> tuple[np.ndarray, float]:
    """P(yes) per (state, question), and ms per question.

    Requests sharing a state object are sent together as one request with several questions.
    """
    from s1ght.schema.systemone import SystemOneRequest

    groups: dict[int, list[int]] = {}
    for i, (state, _) in enumerate(requests):
        groups.setdefault(id(state), []).append(i)
    reqs, order = [], []
    for idx in groups.values():
        state = {k: Image(v) for k, v in requests[idx[0]][0].items()}
        for j in range(0, len(idx), batch):
            part = idx[j : j + batch]
            reqs.append(SystemOneRequest(model.name, state, {f"q{n}": Noul(requests[n][1]) for n in part}))
            order.append(part)
    out = np.zeros(len(requests))
    t0 = time.perf_counter()
    for j in range(0, len(reqs), batch):
        for part, resp in zip(order[j : j + batch], model.runtime.decide_batch(reqs[j : j + batch]), strict=True):
            for n in part:
                out[n] = float(resp.answers[f"q{n}"].noul)
    return out, (time.perf_counter() - t0) * 1000 / max(len(requests), 1)


# --- metrics ----------------------------------------------------------------------------------


def auc(scores: np.ndarray, truth: np.ndarray) -> float:
    """Area under the ROC curve (ties count half); nan without both classes."""
    pos, neg = scores[truth], scores[~truth]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    ranks = np.argsort(np.argsort(np.concatenate([pos, neg]), kind="stable"), kind="stable") + 1.0
    allv = np.concatenate([pos, neg])
    for v in np.unique(allv):  # average ranks of ties
        tie = allv == v
        ranks[tie] = ranks[tie].mean()
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def isotonic(y: np.ndarray, increasing: bool = True) -> np.ndarray:
    """Least-squares monotone fit (pool adjacent violators)."""
    v = list(y if increasing else -np.asarray(y))
    blocks: list[list[float]] = []  # [mean, weight]
    for x in v:
        blocks.append([float(x), 1.0])
        while len(blocks) > 1 and blocks[-2][0] > blocks[-1][0]:
            m2, w2 = blocks.pop()
            m1, w1 = blocks.pop()
            blocks.append([(m1 * w1 + m2 * w2) / (w1 + w2), w1 + w2])
    out = np.concatenate([[m] * int(w) for m, w in blocks])
    return out if increasing else -out


def edge_estimate(at: np.ndarray, p: np.ndarray, beyond: bool) -> float:
    """Where a side's monotone P(yes) crosses 0.5 along the line position.

    `beyond` (left, top): yes means the edge is past the line, so P(yes) falls as the line moves on.
    """
    order = np.argsort(at)
    a, q = at[order], isotonic(p[order], increasing=not beyond)
    falling = q if beyond else q[::-1]
    pos = a if beyond else a[::-1]
    # first line where the (falling) answer drops below 0.5: the edge is between it and the previous
    below = np.nonzero(falling < 0.5)[0]
    if len(below) == 0:
        return float(pos[-1])
    i = int(below[0])
    return float(pos[0]) if i == 0 else float((pos[i - 1] + pos[i]) / 2)


def region_metrics(probes: list[Probe], p: np.ndarray, ms: float) -> dict[str, Any]:
    out: dict[str, Any] = {"ms": ms}
    for kind in ("occupancy", "center"):
        sel = np.array([pr.kind == kind for pr in probes])
        if not sel.any():
            continue
        t = np.array([pr.truth for pr in probes])[sel]
        near = np.array([pr.near for pr in probes])[sel]
        cov = np.array([pr.cover for pr in probes])[sel]
        q = p[sel]
        row: dict[str, Any] = {
            "auc": auc(q, t),
            "auc_near": auc(q[near], t[near]),
            "accuracy": float(((q > 0.5) == t).mean()),
            "n": int(sel.sum()),
        }
        if kind == "occupancy":
            for name, lo, hi in (
                ("recall_cover_10", 0, 0.1),
                ("recall_cover_50", 0.1, 0.5),
                ("recall_cover_100", 0.5, 1.01),
            ):
                m = t & (cov > lo) & (cov <= hi)
                row[name] = float((q[m] > 0.5).mean()) if m.any() else float("nan")
        out[kind] = row
    return out


def line_metrics(probes: list[LineProbe], scenes: list[Scene], p: np.ndarray, ms: float) -> dict[str, Any]:
    t = np.array([pr.truth for pr in probes])
    off = np.abs(np.array([pr.offset for pr in probes]))
    out: dict[str, Any] = {"ms": ms, "auc": auc(p, t), "n": len(probes)}
    for o in OFFSETS:
        m = np.isclose(off, o)
        out[f"accuracy_{o}"] = float(((p[m] > 0.5) == t[m]).mean()) if m.any() else float("nan")
    errors = []
    keys = sorted({(pr.scene, pr.shape, pr.side) for pr in probes})
    for key in keys:
        idx = [i for i, pr in enumerate(probes) if (pr.scene, pr.shape, pr.side) == key]
        at = np.array([probes[i].at for i in idx])
        _, k, beyond = SIDES[key[2]]
        shape = next(s for s in scenes[key[0]].shapes if s.name == key[1])
        extent = scenes[key[0]].image.size[k % 2]
        errors.append(abs(edge_estimate(at, p[idx], beyond) - shape.box[k] / extent))
    out["edge_error"] = float(np.mean(errors)) if errors else float("nan")  # 0 for perfect answers
    return out


# --- experiment -------------------------------------------------------------------------------


def run_model(model: Any, scenes: list[Scene], args: argparse.Namespace) -> dict[str, Any]:
    rng = random.Random(args.seed)
    probes = [pr for i, sc in enumerate(scenes) for pr in sample_cells(sc, i, args.grid, args.cells, rng)]
    lines = [pr for i, sc in enumerate(scenes) for pr in sample_lines(sc, i)]
    report: dict[str, Any] = {
        "presence": {},
        "regions": {},
        "lines": {},
        "probes": len(probes),
        "line_probes": len(lines),
        "errors": {},
    }
    if args.presence:
        asked = [q for i, sc in enumerate(scenes) for q in sample_presence(sc, i, rng)]
        states: dict[int, dict[str, Any]] = {}
        requests = [
            (states.setdefault(i, {"photo": scenes[i].image}), f"Is there a {name} in `photo`?") for i, name, _ in asked
        ]
        p, ms = ask_all(model, requests, args.batch)
        t = np.array([present for _, _, present in asked])
        report["presence"] = {"auc": auc(p, t), "accuracy": float(((p > 0.5) == t).mean()), "n": len(asked), "ms": ms}
        print(f"{model.name} presence: {json.dumps(report['presence'])}", flush=True)
    for method in args.methods:
        if method == "crops" and args.grid > 4:
            print(f"skipping crops: {args.grid}x{args.grid} is too many images per request", flush=True)
            continue
        shared: dict[tuple[int, str], dict[str, Any]] = {}  # one state per scene for shared-image methods
        requests = []
        for pr in probes:
            state, text = region_request(method, scenes[pr.scene], pr, args.grid, args.crop_side)
            if method in ("text", "grid", "crops"):
                state = shared.setdefault((pr.scene, method), state)
            requests.append((state, text))
        try:
            p, ms = ask_all(model, requests, args.batch)
        except (ValueError, RuntimeError) as e:  # e.g. too many images per request, or out of memory
            report["errors"][method] = f"{type(e).__name__}: {e}"
            print(f"{model.name} {method}: failed, {report['errors'][method]}", flush=True)
            continue
        report["regions"][method] = region_metrics(probes, p, ms)
        print(f"{model.name} {method}: {json.dumps(report['regions'][method])}", flush=True)
    for method in args.line_methods:
        shared = {}
        requests = []
        for pr in lines:
            state, text = line_request(method, scenes[pr.scene], pr, args.grid)
            if method == "text":
                state = shared.setdefault((pr.scene, method), state)
            requests.append((state, text))
        try:
            p, ms = ask_all(model, requests, args.batch)
        except (ValueError, RuntimeError) as e:
            report["errors"][f"line {method}"] = f"{type(e).__name__}: {e}"
            print(f"{model.name} lines {method}: failed, {report['errors'][f'line {method}']}", flush=True)
            continue
        if method in FLIPPED:
            p = 1.0 - p
        report["lines"][method] = line_metrics(lines, scenes, p, ms)
        print(f"{model.name} lines {method}: {json.dumps(report['lines'][method])}", flush=True)
    return report


def markdown(report: dict[str, Any]) -> str:
    g = report["grid"]
    out = [f"E1 region questions: {report['images']} synthetic images, {g}x{g} grid", ""]
    if any(r["presence"] for r in report["models"].values()):
        out += ["| model | presence AUC | accuracy | ms/question |", "| --- | ---: | ---: | ---: |"]
        for name, r in report["models"].items():
            if r["presence"]:
                pr = r["presence"]
                out.append(f"| {name} | {pr['auc']:.3f} | {pr['accuracy']:.2f} | {pr['ms']:.0f} |")
        out.append("")
    out += [
        (
            "| model | naming | occupancy AUC | vs neighbours | recall cover <=0.1 / <=0.5 / >0.5 "
            "| center AUC | vs neighbours | ms/question |"
        ),
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for name, r in report["models"].items():
        for method, m in r["regions"].items():
            o, c = m.get("occupancy", {}), m.get("center", {})
            rec = " / ".join(
                f"{o.get(k, float('nan')):.2f}" for k in ("recall_cover_10", "recall_cover_50", "recall_cover_100")
            )
            out.append(
                f"| {name} | {method} | {o.get('auc', float('nan')):.3f} | {o.get('auc_near', float('nan')):.3f} | {rec} "
                f"| {c.get('auc', float('nan')):.3f} | {c.get('auc_near', float('nan')):.3f} | {m['ms']:.0f} |"
            )
    acc = " | ".join(f"acc +-{o}" for o in OFFSETS)
    out += [
        "",
        f"| model | line | AUC | {acc} | edge error | ms/question |",
        "| --- | --- | ---: |" + " ---: |" * (len(OFFSETS) + 2),
    ]
    for name, r in report["models"].items():
        for method, m in r["lines"].items():
            accs = " | ".join(f"{m[f'accuracy_{o}']:.2f}" for o in OFFSETS)
            out.append(f"| {name} | {method} | {m['auc']:.3f} | {accs} | {m['edge_error']:.3f} | {m['ms']:.0f} |")
    errors = [f"- {name} {what}: {err}" for name, r in report["models"].items() for what, err in r["errors"].items()]
    if errors:
        out += ["", "Failed:", *errors]
    return "\n".join(out)


def main(argv: list[str] | None = None, models: dict[str, Any] | None = None) -> dict[str, Any]:
    """`models` (name -> loaded S1GHT) replaces `--models`, for tests."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--models", nargs="+", default=["qwen3-vl-4b", "d1-omni-600m"])
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--size", type=int, default=448, help="synthetic image side (px)")
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--cells", type=int, default=12, help="occupancy probes per (image, shape)")
    ap.add_argument("--methods", nargs="*", choices=REGION_METHODS, default=["text", "grid", "mark", "crop"])
    ap.add_argument("--line-methods", nargs="*", choices=LINE_METHODS, default=list(LINE_METHODS))
    ap.add_argument("--no-presence", dest="presence", action="store_false", help="skip whole-image presence")
    ap.add_argument("--crop-side", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16, help="requests per runtime call, questions per request")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    if args.grid > 26:
        raise SystemExit("--grid: at most 26 columns (letters)")

    rng = random.Random(args.seed)
    scenes = [make_scene(rng, args.size) for _ in range(args.images)]
    report: dict[str, Any] = {"images": len(scenes), "grid": args.grid, "models": {}}
    if models is None:
        import gc

        from s1ght import S1GHT

        for name in args.models:
            with S1GHT(name, device=args.device) as model:
                report["models"][name] = run_model(model, scenes, args)
            gc.collect()
    else:
        for name, model in models.items():
            report["models"][name] = run_model(model, scenes, args)
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
