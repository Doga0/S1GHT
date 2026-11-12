"""E2 (decision-space detection): the whole pipeline on a frozen decision model, scored by mAP.

    python benchmarks/dsod_pipeline.py --model d1-3b --device cuda --out e2b.json
    python benchmarks/dsod_pipeline.py --model none  # the oracle rows only, no model
    python benchmarks/dsod_pipeline.py --images 20 --cells mark --bits 2

E0 (`decision_ceiling.py`) measured what the question designs reach with perfect answers; E1 and
E1b (`region_questions.py`) measured single answers of Qwen3-VL-4B: presence AUC 1.000,
occupancy of a cropped or framed cell 0.99, center 0.79-0.86, boundaries asked as occupancy of a
framed strip 0.958 (edge error 0.048 of the image side). E2 chains them into detection on E1's
synthetic scenes and measures mAP, so the errors of each stage meet the decoder:

1. **Presence:** "Is there a {class} in `photo`?" for all 12 classes, on the whole image.
2. **Cells:** for each present class (P > 0.5), occupancy of every cell of a `--grid` grid, the
   cell cropped and enlarged (`--cells crop`) or framed in red (`mark`). One image per cell
   serves every class.
3. **Centers:** "Is the center of the {class} in {cell}?" in the occupied cells, the cell framed
   (`--centers mark`) or cropped.
4. **Decoding (deterministic):** occupied cells (P > 0.5) form connected components; center cells
   (P > 0.5 and the highest among their 8 neighbours) seed objects and share out the component's
   cells (nearest seed); a component without a seed is one object. Box: the cells' bounding box.
   Score: presence x the mean occupancy of the object's cells.
5. **Edges:** per object and side, 2^`--bits` - 1 lines across the edge cell and its outer
   neighbour (E0's `+bBw`). Each is asked as E1b's strip: the region beyond the line, across the
   object's rows (or columns), is framed in red, "Does any part of the {class} lie inside the red
   rectangle?". The answers are projected onto a monotone curve and the edge is read where it
   crosses 0.5 (as in E1); edges that would cross keep the cells' box.

Configurations (each stage answered by the model or by the truth, `oracle`) separate the stages:
- `model` and `model, no edges`: the model answers everything;
- `model, oracle edges`: the truth answers the edge questions about the model's objects;
- `oracle objects, model edges`: the truth answers presence, cells and centers, the model the
  edges;
- `oracle` and `oracle, no edges`: perfect answers, this pipeline's ceiling on these scenes.

A model's answers are cached by (image, question), so a question shared by configurations is
asked once. Reported: mAP (COCO style, 12 classes, 0.50:0.95), mAP50, mAP75, by object size,
model questions per image and ms per image (questions x the measured ms per question).

E2b (E2 with Qwen3-VL-4B: 0.116 mAP, 0.504 with true edges, so the edges were the bottleneck):
- **The model defaults to `d1-3b`,** a decision model; Qwen3-VL-4B is a general VLM, kept only as
  a reference (`--model qwen3-vl-4b`).
- **Edge regions (`--edges`):** `strip` as in E2 (the region beyond the line, to the image
  border, framed in red), and `near`: only the cell-wide part next to the line, cropped and
  enlarged, so a sliver of the object fills more of the image. Configurations whose edges the
  model answers run once per region (`model [near]` ...).
- **Edge diagnostics** for objects matched to a true shape: signed edge error (positive: the edge
  lies inside the true box, i.e. the model missed slivers) per side, and P(yes) by where the
  line lies relative to the true edge (in cells), next to the truth's yes rate.
- **Crossing level:** the edge is read where the monotone answers cross a level. It is chosen
  (0.05-0.95) on the first half of the images by edge error and tested on the second half. It is
  a constant of the decoder, fitted on ground truth like a calibration temperature; the model
  is not trained. Re-reading at another level asks nothing new.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Callable, Hashable
from dataclasses import dataclass, field
from itertools import pairwise
from pathlib import Path
from typing import Any

import numpy as np
from decision_ceiling import cell_box, components, split_by_seeds
from patch_head import AREAS, average_precision, box_iou
from PIL import Image as PILImage
from PIL import ImageDraw
from region_questions import COLORS, MARK, SHAPES, Scene, cell_edges, center_cell, coverage, crop, edge_estimate
from region_questions import make_scene as _make_scene

CLASSES = [f"{c} {s}" for c in COLORS for s in SHAPES]
STAGES = ("presence", "cells", "centers", "edges")
CONFIGS: dict[str, tuple[dict[str, str], bool]] = {  # name -> (source per stage, refine edges)
    "model": ({s: "model" for s in STAGES}, True),
    "model, no edges": ({s: "model" for s in STAGES}, False),
    "model, oracle edges": ({"presence": "model", "cells": "model", "centers": "model", "edges": "oracle"}, True),
    "oracle objects, model edges": (
        {"presence": "oracle", "cells": "oracle", "centers": "oracle", "edges": "model"},
        True,
    ),
    "oracle": ({s: "oracle" for s in STAGES}, True),
    "oracle, no edges": ({s: "oracle" for s in STAGES}, False),
}
TAU = 0.5
EDGE_METHODS = ("strip", "near")
Cell = tuple[int, int]
Rect = tuple[int, int, int, int]


def make_scenes(n: int, size: int, seed: int) -> list[Scene]:
    """E1's scenes: the same seed gives the same images."""
    rng = random.Random(seed)
    return [_make_scene(rng, size) for _ in range(n)]


def framed(image: PILImage.Image, rect: Rect) -> PILImage.Image:
    out = image.copy()
    x0, y0, x1, y1 = rect
    ImageDraw.Draw(out).rectangle((x0, y0, x1 - 1, y1 - 1), outline=MARK, width=3)
    return out


def zoom(image: PILImage.Image, rect: Rect, side: int) -> PILImage.Image:
    """The region cropped and enlarged: the shorter side to `side` px, the longer at most 2x that."""
    x0, y0, x1, y1 = rect
    w, h = max(x1 - x0, 1), max(y1 - y0, 1)
    scale = min(side / min(w, h), 2 * side / max(w, h))
    size = (max(round(w * scale), 1), max(round(h * scale), 1))
    return image.crop((x0, y0, x0 + w, y0 + h)).resize(size, PILImage.Resampling.BICUBIC)


def cell_rect(scene: Scene, cell: Cell, g: int) -> Rect:
    xs, ys = cell_edges(scene.image.size[0], g), cell_edges(scene.image.size[1], g)
    r, c = cell
    return int(xs[c]), int(ys[r]), int(xs[c + 1]), int(ys[r + 1])


# --- answers ----------------------------------------------------------------------------------


class Oracle:
    """Perfect answers about one scene."""

    def __init__(self, scene: Scene, g: int) -> None:
        self.scene, self.g = scene, g
        self.masks = {s.name: s.mask for s in scene.shapes}

    def presence(self, classes: list[str]) -> np.ndarray:
        return np.array([float(c in self.masks) for c in classes])

    def cells(self, classes: list[str]) -> dict[str, np.ndarray]:
        empty = np.zeros((self.g, self.g))
        return {c: (coverage(self.masks[c], self.g) > 0).astype(float) if c in self.masks else empty for c in classes}

    def centers(self, asks: dict[str, list[Cell]]) -> dict[str, dict[Cell, float]]:
        w, h = self.scene.image.size
        out = {}
        for c, cells in asks.items():
            shape = next((s for s in self.scene.shapes if s.name == c), None)
            at = center_cell(shape.box, w, h, self.g) if shape is not None else None
            out[c] = {cell: float(cell == at) for cell in cells}
        return out

    def strips(self, asks: list[tuple[str, Rect]], method: str = "strip") -> np.ndarray:
        out = np.zeros(len(asks))
        for i, (c, (x0, y0, x1, y1)) in enumerate(asks):
            if c in self.masks:
                out[i] = float(self.masks[c][y0:y1, x0:x1].any())
        return out


class Asker:
    """Asks a model Nouls, grouped by image, batched, and cached by (image key, question)."""

    def __init__(self, model: Any, batch: int) -> None:
        self.model, self.batch = model, batch
        self.cache: dict[tuple[Hashable, str], float] = {}
        self.asked, self.seconds = 0, 0.0

    def ask(self, items: list[tuple[Hashable, Callable[[], dict[str, PILImage.Image]], str]]) -> np.ndarray:
        """P(yes) per (state key, state builder, question)."""
        from s1ght import Image, Noul
        from s1ght.schema.systemone import SystemOneRequest

        pending: dict[Hashable, tuple[Callable[[], dict[str, PILImage.Image]], list[str]]] = {}
        for key, build, text in items:
            if (key, text) not in self.cache:
                texts = pending.setdefault(key, (build, []))[1]
                if text not in texts:
                    texts.append(text)
        reqs, order = [], []
        for key, (build, texts) in pending.items():
            state = {k: Image(v) for k, v in build().items()}
            for j in range(0, len(texts), self.batch):
                part = texts[j : j + self.batch]
                reqs.append(SystemOneRequest(self.model.name, state, {f"q{n}": Noul(t) for n, t in enumerate(part)}))
                order.append((key, part))
        t0 = time.perf_counter()
        for j in range(0, len(reqs), self.batch):
            resps = self.model.runtime.decide_batch(reqs[j : j + self.batch])
            for (key, part), resp in zip(order[j : j + self.batch], resps, strict=True):
                for n, t in enumerate(part):
                    self.cache[(key, t)] = float(resp.answers[f"q{n}"].noul)
        self.seconds += time.perf_counter() - t0
        self.asked += sum(len(p) for _, p in order)
        return np.array([self.cache[(key, text)] for key, _, text in items])


class ModelAnswers:
    """The model's answers about one scene (scene `index` keys the cache)."""

    def __init__(self, asker: Asker, scene: Scene, index: int, g: int, cells: str, centers: str, crop_side: int):
        self.asker, self.scene, self.index, self.g = asker, scene, index, g
        self.cell_method, self.center_method, self.crop_side = cells, centers, crop_side
        self.count = 0  # questions this scene's configuration used (cached or not)

    def _ask(self, items: list[tuple[Hashable, Callable[[], dict[str, PILImage.Image]], str]]) -> np.ndarray:
        self.count += len(items)
        return self.asker.ask(items)

    def _cell_view(self, method: str, cell: Cell) -> tuple[Hashable, Callable[[], dict[str, PILImage.Image]], str]:
        """(key, builder, how the question names the cell)."""
        if method == "crop":
            return (
                ("crop", self.index, cell),
                lambda: {"photo": crop(self.scene.image, cell, self.g, self.crop_side)},
                "`photo`",
            )
        rect = cell_rect(self.scene, cell, self.g)
        return (
            ("mark", self.index, cell),
            lambda: {"photo": framed(self.scene.image, rect)},
            "the red rectangle in `photo`",
        )

    def presence(self, classes: list[str]) -> np.ndarray:
        key, build = ("image", self.index), lambda: {"photo": self.scene.image}
        return self._ask([(key, build, f"Is there a {c} in `photo`?") for c in classes])

    def cells(self, classes: list[str]) -> dict[str, np.ndarray]:
        grid = [(r, c) for r in range(self.g) for c in range(self.g)]
        items, where = [], []
        for cell in grid:
            key, build, name = self._cell_view(self.cell_method, cell)
            for cls in classes:
                items.append((key, build, f"Does any part of the {cls} lie in {name}?"))
                where.append((cls, cell))
        p = self._ask(items) if items else np.zeros(0)
        out = {cls: np.zeros((self.g, self.g)) for cls in classes}
        for (cls, cell), v in zip(where, p, strict=True):
            out[cls][cell] = v
        return out

    def centers(self, asks: dict[str, list[Cell]]) -> dict[str, dict[Cell, float]]:
        items, where = [], []
        for cls, cells in asks.items():
            for cell in cells:
                key, build, name = self._cell_view(self.center_method, cell)
                items.append((key, build, f"Is the center of the {cls} in {name}?"))
                where.append((cls, cell))
        p = self._ask(items) if items else np.zeros(0)
        out: dict[str, dict[Cell, float]] = {cls: {} for cls in asks}
        for (cls, cell), v in zip(where, p, strict=True):
            out[cls][cell] = float(v)
        return out

    def strips(self, asks: list[tuple[str, Rect]], method: str = "strip") -> np.ndarray:
        """`strip`: the region framed in red on the image; `near`: the region cropped and enlarged."""
        if method == "strip":
            items = [
                (
                    ("strip", self.index, rect),
                    lambda rect=rect: {"photo": framed(self.scene.image, rect)},
                    f"Does any part of the {cls} lie inside the red rectangle in `photo`?",
                )
                for cls, rect in asks
            ]
        else:
            items = [
                (
                    ("near", self.index, rect),
                    lambda rect=rect: {"photo": zoom(self.scene.image, rect, self.crop_side)},
                    f"Does any part of the {cls} lie in `photo`?",
                )
                for cls, rect in asks
            ]
        return self._ask(items) if items else np.zeros(0)


# --- decoding ---------------------------------------------------------------------------------


def peaks(prob: np.ndarray, tau: float = TAU) -> list[Cell]:
    """Cells above `tau` that are the highest among their 8 neighbours (ties: the first)."""
    out = []
    g0, g1 = prob.shape
    for r, c in zip(*np.nonzero(prob > tau), strict=True):
        window = prob[max(r - 1, 0) : r + 2, max(c - 1, 0) : c + 2]
        if prob[r, c] >= window.max() and not any(
            (rr, cc) in out
            for rr in range(max(r - 1, 0), min(r + 2, g0))
            for cc in range(max(c - 1, 0), min(c + 2, g1))
        ):
            out.append((int(r), int(c)))
    return out


def objects(
    occ: np.ndarray, centers: dict[Cell, float], xs: np.ndarray, ys: np.ndarray, tau: float = TAU
) -> list[list[Cell]]:
    """Each object's cells: components of occupied cells, split among center peaks."""
    grid = np.zeros_like(occ)
    for cell, p in centers.items():
        grid[cell] = p
    seeds = peaks(grid, tau)
    out = []
    for comp in components(occ > tau):
        inside = [s for s in seeds if s in comp]
        out += split_by_seeds(comp, inside, xs, ys) if inside else [comp]
    return out


def edge_lines(
    box: np.ndarray, side: int, xs: np.ndarray, ys: np.ndarray, bits: int
) -> tuple[float, float, np.ndarray]:
    """(lo, hi, lines): 2^bits - 1 lines across the edge cell and its outer neighbour."""
    e = xs if side % 2 == 0 else ys
    k = int(np.searchsorted(e, box[side]))  # cell-box edges lie on cell edges
    last = len(e) - 1
    lo, hi = e[max(k - 1, 0)], e[min(k + 1, last)]  # the cells on both sides of the edge
    n = 2**bits
    return float(lo), float(hi), lo + (hi - lo) * np.arange(1, n) / n


def strip(box: np.ndarray, side: int, at: float, width: int, height: int) -> Rect:
    """The region beyond a line at `at` (toward the border the side faces), across the box's band."""
    x0, y0, x1, y1 = (int(v) for v in box)
    a = round(at)
    if side == 0:
        return 0, y0, a, y1
    if side == 2:
        return a, y0, width, y1
    if side == 1:
        return x0, 0, x1, a
    return x0, a, x1, height


def edge_region(method: str, box: np.ndarray, side: int, at: float, width: int, height: int, cell: float) -> Rect:
    """The region an edge question asks about: `strip` reaches the image border, `near` only one
    cell past the line (where a sliver of the object would be)."""
    x0, y0, x1, y1 = strip(box, side, at, width, height)
    if method == "strip":
        return x0, y0, x1, y1
    d = round(cell)
    if side == 0:
        return max(x1 - d, 0), y0, x1, y1
    if side == 2:
        return x0, y0, min(x0 + d, width), y1
    if side == 1:
        return x0, max(y1 - d, 0), x1, y1
    return x0, y0, x1, min(y0 + d, height)


@dataclass
class Found:
    """A decoded object before its edges are read: the cells' box, its edge questions and answers."""

    cls: str
    box: np.ndarray
    score: float
    spans: list[tuple[float, float]] = field(default_factory=list)
    lines: list[np.ndarray] = field(default_factory=list)
    rects: list[list[Rect]] = field(default_factory=list)
    answers: np.ndarray | None = None  # (4, lines): P(part of the object in the region)


def refine(found: Found, level: float = 0.5) -> np.ndarray:
    """Edges read from the region answers, crossing `level`.

    Left/top: a yes means the edge is before the line. Right/bottom: after it. The span's ends
    act as sure answers (the edge is inside the span), then E1's monotone readout.
    """
    box = found.box.astype(float)
    if found.answers is None:
        return box
    out = box.copy()
    for side in range(4):
        lo, hi = found.spans[side]
        beyond = side < 2  # truth "edge past the line" in the side's inward direction
        p_truth = 1.0 - found.answers[side]
        at = np.concatenate([[lo], found.lines[side], [hi]])
        p = np.concatenate([[1.0 if beyond else 0.0], p_truth, [0.0 if beyond else 1.0]])
        out[side] = edge_estimate(at, p, beyond, level)
    if out[2] <= out[0] or out[3] <= out[1]:
        return box
    return out


def decode(scene: Scene, src: dict[str, Any], g: int, bits: int, edges: str | None) -> list[Found]:
    """Objects of one scene, with their edge answers when `edges` names a region method."""
    w, h = scene.image.size
    xs, ys = cell_edges(w, g), cell_edges(h, g)
    pres = src["presence"].presence(CLASSES)
    present = [c for c, p in zip(CLASSES, pres, strict=True) if p > TAU]
    occ = src["cells"].cells(present) if present else {}
    cen = src["centers"].centers({c: [tuple(map(int, rc)) for rc in np.argwhere(occ[c] > TAU)] for c in present})
    found = []
    for c in present:
        for part in objects(occ[c], cen[c], xs, ys):
            score = pres[CLASSES.index(c)] * float(np.mean([occ[c][cell] for cell in part]))
            found.append(Found(c, cell_box(part, xs, ys), score))
    if edges and found:
        asks = []
        for f in found:
            for side in range(4):
                lo, hi, ln = edge_lines(f.box, side, xs, ys, bits)
                cell = (w if side % 2 == 0 else h) / g
                f.spans.append((lo, hi))
                f.lines.append(ln)
                f.rects.append([edge_region(edges, f.box, side, a, w, h, cell) for a in ln])
                asks += [(f.cls, r) for r in f.rects[-1]]
        p = src["edges"].strips(asks, edges).reshape(len(found), 4, -1)
        for f, a in zip(found, p, strict=True):
            f.answers = a
    return found


def finish(found: list[Found], level: float = 0.5) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Boxes, scores and class indices."""
    boxes = np.array([refine(f, level) for f in found], np.float32).reshape(-1, 4)
    scores = np.array([f.score for f in found], np.float32)
    return boxes, scores, np.array([CLASSES.index(f.cls) for f in found], np.int64)


def detect(
    scene: Scene, src: dict[str, Any], g: int, bits: int, edges: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """E2's entry point: decode, then read `strip` edges at 0.5."""
    return finish(decode(scene, src, g, bits, "strip" if edges else None))


# --- edge diagnostics -------------------------------------------------------------------------

BINS = np.arange(-2.0, 2.01, 0.25)  # line position past the true edge, inward, in cells


def matched(found: list[Found], scene: Scene) -> list[tuple[Found, np.ndarray]]:
    """Each true shape's decoded object of its class with the highest cell-box IoU (> 0.1)."""
    out = []
    for s in scene.shapes:
        cands = [f for f in found if f.cls == s.name and f.answers is not None]
        if cands:
            ious = box_iou(np.array([f.box for f in cands], float), s.box[None].astype(float))[:, 0]
            if ious.max() > 0.1:
                out.append((cands[int(ious.argmax())], s.box))
    return out


def inward(box: np.ndarray, truth: np.ndarray) -> np.ndarray:
    """Signed edge errors, positive when an edge lies inside the true box (the box is too small)."""
    return np.array([box[0] - truth[0], box[1] - truth[1], truth[2] - box[2], truth[3] - box[3]], float)


def edge_stats(found: list[list[Found]], scenes: list[Scene], g: int, level: float = 0.5) -> dict[str, Any]:
    """Signed and absolute edge errors (px) of matched objects, and P(yes) by the line's position
    relative to the true edge (inward, in cells), with the truth's yes rate."""
    errors, dist, p_yes, truth_yes = [], [], [], []
    for fs, sc in zip(found, scenes, strict=True):
        w, h = sc.image.size
        mask = {s.name: s.mask for s in sc.shapes}
        for f, tb in matched(fs, sc):
            assert f.answers is not None
            errors.append(inward(refine(f, level), tb))
            for side in range(4):
                cell = (w if side % 2 == 0 else h) / g
                sign = 1.0 if side < 2 else -1.0
                for line, rect, p in zip(f.lines[side], f.rects[side], f.answers[side], strict=True):
                    dist.append(sign * (line - tb[side]) / cell)
                    p_yes.append(p)
                    x0, y0, x1, y1 = rect
                    truth_yes.append(float(mask[f.cls][y0:y1, x0:x1].any()))
    if not errors:
        return {"objects": 0}
    e = np.array(errors)
    d, py, ty = np.array(dist), np.array(p_yes), np.array(truth_yes)
    curve = []
    for lo, hi in pairwise(BINS):
        m = (d >= lo) & (d < hi)
        if m.any():
            curve.append(
                {
                    "from": float(lo),
                    "to": float(hi),
                    "n": int(m.sum()),
                    "p_yes": float(py[m].mean()),
                    "truth": float(ty[m].mean()),
                }
            )
    return {
        "objects": len(e),
        "signed_px": float(e.mean()),
        "abs_px": float(np.abs(e).mean()),
        "signed_px_by_side": e.mean(0).tolist(),
        "curve": curve,
    }


def calibrate(found: list[list[Found]], scenes: list[Scene], g: int) -> float:
    """The crossing level with the smallest mean absolute edge error on these scenes."""
    levels = np.round(np.arange(0.05, 0.96, 0.05), 2)
    errs = [edge_stats(found, scenes, g, float(lv)).get("abs_px", np.inf) for lv in levels]
    return float(levels[int(np.argmin(errs))])


# --- experiment -------------------------------------------------------------------------------


def summary(dets: list, truth: list) -> dict[str, Any]:
    def stats(area: tuple[float, float] | None) -> dict[str, float]:
        aps = [a for c in range(len(CLASSES)) if (a := average_precision(dets, truth, c, area)) is not None]
        if not aps:
            return {"map": float("nan"), "map50": float("nan"), "map75": float("nan")}
        a = np.stack(aps)
        return {"map": float(a.mean()), "map50": float(a[:, 0].mean()), "map75": float(a[:, 5].mean())}

    return {"all": stats(None), **{name: stats(rng) for name, rng in AREAS.items()}}


def run(scenes: list[Scene], args: argparse.Namespace, model: Any | None) -> dict[str, Any]:
    truth = [
        (
            np.array([s.box for s in sc.shapes], np.float32).reshape(-1, 4),
            np.array([CLASSES.index(s.name) for s in sc.shapes], np.int64),
        )
        for sc in scenes
    ]
    half = len(scenes) // 2  # calibration: the first half; test: the second
    asker = Asker(model, args.batch) if model is not None else None
    report: dict[str, Any] = {"configs": {}}
    for name in args.configs:
        sources, edges = CONFIGS[name]
        if asker is None and "model" in sources.values():
            continue
        model_edges = edges and sources["edges"] == "model"
        methods: list[str | None] = list(args.edges) if model_edges else (["strip"] if edges else [None])
        for method in methods:
            label = f"{name} [{method}]" if model_edges else name
            found, questions = [], 0
            t0 = time.perf_counter()
            for i, sc in enumerate(scenes):
                oracle = Oracle(sc, args.grid)
                answers = (
                    ModelAnswers(asker, sc, i, args.grid, args.cells, args.centers, args.crop_side) if asker else None
                )
                src = {stage: answers if sources[stage] == "model" else oracle for stage in STAGES}
                found.append(decode(sc, src, args.grid, args.bits, method))
                questions += answers.count if answers is not None else 0
            row = summary([finish(f) for f in found], truth)
            row["questions"] = questions / len(scenes)
            row["objects"] = float(np.mean([len(f) for f in found]))
            row["seconds"] = time.perf_counter() - t0
            if edges:
                row["edges"] = edge_stats(found, scenes, args.grid)
            if model_edges and half:
                level = calibrate(found[:half], scenes[:half], args.grid)
                test = found[half:]
                row["calibrated"] = {
                    "level": level,
                    "test_map_at_0.5": summary([finish(f) for f in test], truth[half:])["all"],
                    "test_map_calibrated": summary([finish(f, level) for f in test], truth[half:])["all"],
                    "test_edges_at_0.5": edge_stats(test, scenes[half:], args.grid),
                    "test_edges_calibrated": edge_stats(test, scenes[half:], args.grid, level),
                }
            report["configs"][label] = row
            print(f"{label}: {json.dumps(row['all'])}, {row['questions']:.0f} model questions/image", flush=True)
    if asker is not None and asker.asked:
        report["ms_per_question"] = asker.seconds * 1000 / asker.asked
    return report


def markdown(report: dict[str, Any]) -> str:
    ms = report.get("ms_per_question")
    lines = [
        f"E2 decision-space detection with {report['model']}: {report['images']} synthetic images, "
        f"{report['grid']}x{report['grid']} grid, cells by {report['cells']}, centers by {report['centers']}, "
        f"{2 ** report['bits'] - 1} lines per edge" + (f", {ms:.0f} ms per model question" if ms else ""),
        "",
        (
            "| configuration | model questions/image | ms/image | objects/image | mAP | mAP50 | mAP75 "
            "| small | medium | large |"
        ),
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in report["configs"].items():
        a = r["all"]
        cost = f"{r['questions'] * ms:.0f}" if ms and r["questions"] else "-"
        lines.append(
            f"| {name} | {r['questions']:.0f} | {cost} | {r['objects']:.1f} | {a['map']:.3f} | {a['map50']:.3f} "
            f"| {a['map75']:.3f} | {r['small']['map']:.3f} | {r['medium']['map']:.3f} | {r['large']['map']:.3f} |"
        )
    edged = {n: r for n, r in report["configs"].items() if r.get("edges", {}).get("objects")}
    if edged:
        lines += [
            "",
            "Edges of matched objects (px; signed > 0: the edge lies inside the true box)",
            "",
            "| configuration | objects | signed | abs | signed left / top / right / bottom |",
            "| --- | ---: | ---: | ---: | --- |",
        ]
        for n, r in edged.items():
            e = r["edges"]
            by = " / ".join(f"{v:.1f}" for v in e["signed_px_by_side"])
            lines.append(f"| {n} | {e['objects']} | {e['signed_px']:.1f} | {e['abs_px']:.1f} | {by} |")
    cal = {n: r["calibrated"] for n, r in report["configs"].items() if "calibrated" in r}
    if cal:
        lines += [
            "",
            "Crossing level calibrated on the first half of the images, tested on the second half",
            "",
            (
                "| configuration | level | test mAP at 0.5 | calibrated | test mAP75 at 0.5 | calibrated "
                "| abs edge error px at 0.5 | calibrated |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
        for n, c in cal.items():
            a, b = c["test_map_at_0.5"], c["test_map_calibrated"]
            ea = c["test_edges_at_0.5"].get("abs_px", float("nan"))
            eb = c["test_edges_calibrated"].get("abs_px", float("nan"))
            lines.append(
                f"| {n} | {c['level']:.2f} | {a['map']:.3f} | {b['map']:.3f} | {a['map75']:.3f} | {b['map75']:.3f} "
                f"| {ea:.1f} | {eb:.1f} |"
            )
    curves = {n: r["edges"]["curve"] for n, r in edged.items() if "[" in n}
    if curves:
        names = list(curves)
        lines += [
            "",
            "P(yes) by the line's position past the true edge (inward, in cells; the truth says yes for > 0)",
            "",
            "| line position | truth yes | " + " | ".join(names) + " |",
            "| --- | ---: |" + " ---: |" * len(names),
        ]
        rows: dict[tuple[float, float], dict[str, float]] = {}
        for n, curve in curves.items():
            for b in curve:
                rows.setdefault((b["from"], b["to"]), {"truth": b["truth"]})[n] = b["p_yes"]
        for (lo, hi), v in sorted(rows.items()):
            vals = " | ".join(f"{v[n]:.2f}" if n in v else "-" for n in names)
            lines.append(f"| {lo:+.2f} to {hi:+.2f} | {v['truth']:.2f} | {vals} |")
    return "\n".join(lines)


def main(argv: list[str] | None = None, model: Any | None = None) -> dict[str, Any]:
    """`model` (a loaded S1GHT) replaces `--model`, for tests; `--model none` runs the oracle rows only."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="d1-3b", help="catalog name, or none for the oracle rows only")
    ap.add_argument("--images", type=int, default=100)
    ap.add_argument("--size", type=int, default=448)
    ap.add_argument("--grid", type=int, default=8)
    ap.add_argument("--bits", type=int, default=3, help="2^bits - 1 edge questions per side")
    ap.add_argument("--cells", choices=["crop", "mark"], default="crop")
    ap.add_argument("--centers", choices=["crop", "mark"], default="mark")
    ap.add_argument("--edges", nargs="+", choices=EDGE_METHODS, default=list(EDGE_METHODS), help="edge regions")
    ap.add_argument("--configs", nargs="+", choices=list(CONFIGS), default=list(CONFIGS))
    ap.add_argument("--crop-side", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16, help="requests per runtime call, questions per request")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0, help="E1's scenes with the default 0")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    scenes = make_scenes(args.images, args.size, args.seed)
    report: dict[str, Any] = {k: getattr(args, k) for k in ("images", "grid", "bits", "cells", "centers")}
    report["model"] = getattr(model, "name", None) or args.model
    if model is not None or args.model == "none":
        report.update(run(scenes, args, model))
    else:
        from s1ght import S1GHT

        with S1GHT(args.model, device=args.device) as loaded:
            report.update(run(scenes, args, loaded))
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
