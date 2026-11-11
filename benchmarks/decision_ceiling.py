"""E0 (decision-space detection): the COCO mAP a question design reaches with perfect answers.

    python benchmarks/decision_ceiling.py --out ceiling.json
    python benchmarks/decision_ceiling.py --designs occ8 count16+b3 --min-cover 0.1

Decision-space detection asks a frozen decision model typed questions about regions of the image
and turns the answers into boxes with a deterministic decoder. Before any model is asked, this
script answers every question from the COCO ground truth and runs the decoders. The result is
each design's ceiling: the mAP it cannot exceed however good the model is, and the number of
decisions it needs per image. No model, no images, no GPU.

Designs (`--designs`, G = grid side, a power of 2; cells split the image G x G):
- `occG`: per class and cell, a Noul "does a {class} overlap this cell?". Decoder: connected
  occupied cells (4-neighbours) are one object, its box the cells' bounding box.
- `scoreG`: a Score "how much of this cell does a {class} cover?" with 5 levels (none, up to 1/4,
  1/2, 3/4, all). Decoder: connected components as in `occ`, then a box fitted to the coverage
  levels (each edge on 8 positions inside its edge cell, least squares).
- `centerG`: `occ` plus a Noul "is the center of a {class} in this cell?". Decoder: each center
  cell seeds one object; the occupied cells of a component go to the nearest seed.
- `countG`: `center` with a Choice for how many {class} centers the cell holds (0, 1, 2, 3+), so a
  cell can seed several objects (same box: they cannot be told apart).
- `+bB` (after `center`/`count`): a second stage asks, per object and side, threshold questions
  ("is its left edge right of this line?") at 2^B - 1 lines across the edge cell. The object is
  named by its seed cell; the answer is about the largest object with its center there (the
  2nd largest for a cell's 2nd seed ...).
- `boxes`: the ground truth itself (sanity check, mAP 1).

Every class's presence is asked first (80 Nouls) and taken as perfect, so absent classes give
no detections. All detections score 1: perfect answers carry no ranking.

Decisions per image, two counts:
- **parallel:** every question of a stage asked at once for every class (presence, then G^2
  questions per class and question type; `+bB` adds 4 (2^B - 1) per decoded object in a second
  stage).
- **adaptive:** presence first, then only present classes, coarse to fine (2x2 cells, then the
  4 children of each occupied cell, ... down to G); center questions only in occupied cells;
  `+bB` as a binary search (4 B per object).

`--min-cover` sets when an occupancy answer is yes: the class covers more than this fraction of
the cell (0: any overlap). A model will likely miss slivers, so 0.1 or 0.25 is more realistic.
Higher values make the coarse-to-fine count approximate (a parent can be "no" over a "yes" child).

mAP is COCO style (IoU 0.50:0.95, by object size too), computed with `patch_head.py`'s code on
the same first `--val-images` val2017 images as P2.1-P2.3, so the rows compare with the trained
heads (best 0.261) and YOLO26n (0.429) there. The annotation zip (253 MB) is downloaded once.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from patch_head import AREAS, average_precision, fetch_annotations

LEVELS = 4  # Score: none, then coverage in quarters
STEPS = 8  # sub-cell positions per box edge when fitting coverage
MAX_COUNT = 3  # Choice: 0, 1, 2, 3+
DEFAULT_DESIGNS = [
    "occ4", "occ8", "occ16", "occ32",
    "score4", "score8", "score16",
    "center8", "center16", "count8", "count16",
    "count8+b2", "count8+b3", "count16+b2", "count16+b3",
]  # fmt: skip


# --- data -------------------------------------------------------------------------------------


@dataclass
class Truth:
    """One COCO image: boxes (xyxy pixels), contiguous labels and its size."""

    boxes: np.ndarray
    labels: np.ndarray
    width: int
    height: int


def load_truth(root: Path, n: int) -> tuple[list[Truth], list[str]]:
    """The first `n` val2017 images with a non-crowd box, as `patch_head.load_split` picks them."""
    data = json.loads((root / "annotations" / "instances_val2017.json").read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    index = {c["id"]: k for k, c in enumerate(cats)}
    sizes = {im["id"]: (im["width"], im["height"]) for im in data["images"]}
    by_image: dict[int, list[tuple[list[float], int]]] = {}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w > 1 and h > 1:
            by_image.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], index[a["category_id"]]))
    out = []
    for i in sorted(by_image)[:n]:
        boxes, labels = zip(*by_image[i], strict=True)
        out.append(Truth(np.array(boxes, np.float64), np.array(labels, np.int64), *sizes[i]))
    return out, [c["name"] for c in cats]


# --- perfect answers --------------------------------------------------------------------------


def edges(size: int, g: int) -> np.ndarray:
    """Pixel edges of g cells across `size` pixels (g + 1 values)."""
    return np.round(np.linspace(0, size, g + 1)).astype(np.int64)


class Oracle:
    """Perfect answers about one class in one image, from its ground-truth boxes."""

    def __init__(self, boxes: np.ndarray, width: int, height: int, min_cover: float = 0.0) -> None:
        self.boxes, self.width, self.height, self.min_cover = boxes, width, height, min_cover
        mask = np.zeros((height, width), np.int32)
        for x0, y0, x1, y1 in boxes:
            mask[int(np.floor(y0)) : int(np.ceil(y1)), int(np.floor(x0)) : int(np.ceil(x1))] = 1
        self.integral = np.pad(mask.cumsum(0).cumsum(1), ((1, 0), (1, 0)))  # summed-area table

    def grid(self, g: int) -> tuple[np.ndarray, np.ndarray]:
        return edges(self.width, g), edges(self.height, g)

    def coverage(self, g: int) -> np.ndarray:
        """(g, g): the fraction of each cell the class covers (union of its boxes)."""
        xs, ys = self.grid(g)
        s = self.integral
        area = s[ys[1:]][:, xs[1:]] - s[ys[:-1]][:, xs[1:]] - s[ys[1:]][:, xs[:-1]] + s[ys[:-1]][:, xs[:-1]]
        return area / np.maximum(np.outer(np.diff(ys), np.diff(xs)), 1)

    def occupied(self, g: int) -> np.ndarray:
        return self.coverage(g) > self.min_cover

    def level(self, g: int) -> np.ndarray:
        """Score answers: 0 for unoccupied cells, else coverage in quarters (1..4)."""
        cov = self.coverage(g)
        return np.where(cov > self.min_cover, np.clip(np.ceil(cov * LEVELS), 1, LEVELS), 0).astype(np.int64)

    def cell_of(self, g: int) -> np.ndarray:
        """(n, 2) row, col of each box's center cell."""
        cx = (self.boxes[:, 0] + self.boxes[:, 2]) / 2
        cy = (self.boxes[:, 1] + self.boxes[:, 3]) / 2
        col = np.minimum((cx / self.width * g).astype(np.int64), g - 1)
        row = np.minimum((cy / self.height * g).astype(np.int64), g - 1)
        return np.stack([row, col], 1)

    def centers(self, g: int) -> np.ndarray:
        """(g, g) Choice answers: centers per cell, capped at 3+."""
        counts = np.zeros((g, g), np.int64)
        for r, c in self.cell_of(g):
            counts[r, c] += 1
        return np.minimum(counts, MAX_COUNT)

    def seeded(self, g: int, cell: tuple[int, int], k: int) -> np.ndarray | None:
        """The k-th largest box with its center in `cell` (whom a boundary question is about)."""
        at = np.where((self.cell_of(g) == cell).all(1))[0]
        if k >= len(at):
            return None
        b = self.boxes[at]
        return b[np.argsort(-(b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]), kind="stable")[k]]


# --- decoders ---------------------------------------------------------------------------------


def components(mask: np.ndarray) -> list[list[tuple[int, int]]]:
    """Connected cells (4-neighbours), each as a list of (row, col)."""
    seen = np.zeros_like(mask, bool)
    out = []
    for r0, c0 in zip(*np.nonzero(mask), strict=True):
        if seen[r0, c0]:
            continue
        seen[r0, c0] = True
        stack, cells = [(int(r0), int(c0))], []
        while stack:
            r, c = stack.pop()
            cells.append((r, c))
            for rr, cc in ((r - 1, c), (r + 1, c), (r, c - 1), (r, c + 1)):
                if 0 <= rr < mask.shape[0] and 0 <= cc < mask.shape[1] and mask[rr, cc] and not seen[rr, cc]:
                    seen[rr, cc] = True
                    stack.append((rr, cc))
        out.append(sorted(cells))
    return out


def cell_box(cells: list[tuple[int, int]], xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    return np.array([xs[min(cols)], ys[min(rows)], xs[max(cols) + 1], ys[max(rows) + 1]], np.float64)


def split_by_seeds(
    cells: list[tuple[int, int]], seeds: list[tuple[int, int]], xs: np.ndarray, ys: np.ndarray
) -> list[list[tuple[int, int]]]:
    """The component's cells shared out to their nearest seed (cell centers, pixel distance)."""
    if len(seeds) <= 1:
        return [cells]
    cx, cy = (xs[:-1] + xs[1:]) / 2, (ys[:-1] + ys[1:]) / 2
    sp = np.array([[cx[c], cy[r]] for r, c in seeds])
    parts: list[list[tuple[int, int]]] = [[] for _ in seeds]
    for r, c in cells:
        parts[int(((sp - [cx[c], cy[r]]) ** 2).sum(1).argmin())].append((r, c))
    return [p or [s] for p, s in zip(parts, seeds, strict=True)]


def overlap(lo: np.ndarray, hi: np.ndarray, e: np.ndarray) -> np.ndarray:
    """(k, n): the fraction of each of n cells (edges `e`) inside each of k intervals [lo, hi]."""
    a = np.clip(np.minimum(hi[:, None], e[None, 1:]) - np.maximum(lo[:, None], e[None, :-1]), 0, None)
    return a / np.maximum(np.diff(e), 1)[None]


def fit_box(
    cells: list[tuple[int, int]], level: np.ndarray, xs: np.ndarray, ys: np.ndarray, rounds: int = 3
) -> np.ndarray:
    """A box whose cell coverage best matches the Score levels of the object's cells.

    Each edge takes one of `STEPS` positions inside its edge cell; x and y edges are fitted in
    turn (coverage of a box is separable: x share times y share), least squares over the cells of
    the object's bounding box (cells outside the object count as empty).
    """
    rows, cols = [r for r, _ in cells], [c for _, c in cells]
    r0, r1, c0, c1 = min(rows), max(rows), min(cols), max(cols)
    target = np.zeros((r1 - r0 + 1, c1 - c0 + 1))
    for r, c in cells:
        target[r - r0, c - c0] = (level[r, c] - 0.5) / LEVELS  # middle of the answered quarter
    ex, ey = xs[c0 : c1 + 2].astype(np.float64), ys[r0 : r1 + 2].astype(np.float64)

    def candidates(e: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        lo = e[0] + (e[1] - e[0]) * np.arange(STEPS) / STEPS
        hi = e[-2] + (e[-1] - e[-2]) * np.arange(1, STEPS + 1) / STEPS
        lo, hi = np.repeat(lo, STEPS), np.tile(hi, STEPS)
        ok = hi > lo
        return lo[ok], hi[ok]

    cand_x, cand_y = candidates(ex), candidates(ey)
    fx_all, fy_all = overlap(*cand_x, ex), overlap(*cand_y, ey)  # (k, cols), (k, rows)
    bx, by = len(cand_x[0]) - 1, len(cand_y[0]) - 1  # start: the full cells (the last candidate)
    for _ in range(rounds):
        bx = int(((fy_all[by][None, :, None] * fx_all[:, None, :] - target[None]) ** 2).sum((1, 2)).argmin())
        by = int(((fy_all[:, :, None] * fx_all[bx][None, None, :] - target[None]) ** 2).sum((1, 2)).argmin())
    return np.array([cand_x[0][bx], cand_y[0][by], cand_x[1][bx], cand_y[1][by]])


def refine(box: np.ndarray, target: np.ndarray | None, xs: np.ndarray, ys: np.ndarray, bits: int) -> np.ndarray:
    """Each edge moved inside its edge cell by 2^bits - 1 perfect threshold answers about `target`.

    The answers ("is the edge past line k?") give the count of lines the true edge passes; the
    edge goes to the middle of that bracket. A true edge outside the cell clamps to the cell.
    """
    if target is None:
        return box
    out = box.copy()
    n = 2**bits
    for side, (e, inward) in enumerate(((xs, 1), (ys, 1), (xs, -1), (ys, -1))):
        k = int(np.searchsorted(e, box[side]))  # box edges lie on cell edges
        lo, hi = (e[k], e[min(k + 1, len(e) - 1)]) if inward > 0 else (e[max(k - 1, 0)], e[k])
        lines = lo + (hi - lo) * np.arange(1, n) / n
        passed = int((target[side] > lines).sum())
        out[side] = lo + (hi - lo) * (passed + 0.5) / n
    return out


# --- designs ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class Design:
    name: str
    kind: str  # occ, score, center, count, boxes
    grid: int = 0
    bits: int = 0  # boundary questions per edge: 2^bits - 1 lines (0: none)


def parse_design(name: str) -> Design:
    if name == "boxes":
        return Design(name, "boxes")
    m = re.fullmatch(r"(occ|score|center|count)(\d+)(?:\+b(\d+))?", name)
    if m is None:
        raise ValueError(f"{name}: expected occG, scoreG, centerG, countG (+bB for the last two) or boxes")
    kind, g, bits = m.group(1), int(m.group(2)), int(m.group(3) or 0)
    if g < 2 or g & (g - 1):
        raise ValueError(f"{name}: the grid side must be a power of 2 (2, 4, 8 ...)")
    if bits and kind not in ("center", "count"):
        raise ValueError(f"{name}: boundary questions need objects named by a center cell (center/count)")
    return Design(name, kind, g, bits)


def coarse_to_fine(oracle: Oracle, g: int) -> int:
    """Cell questions asked from 2x2 down to g x g, children only under occupied parents."""
    asked, size = 4, 2
    while size < g:
        asked += 4 * int(oracle.occupied(size).sum())
        size *= 2
    return asked


def decode(design: Design, oracle: Oracle) -> tuple[list[np.ndarray], int]:
    """Boxes of one present class from perfect answers, and the decisions the adaptive order
    asks about it (presence aside)."""
    if design.kind == "boxes":
        return list(oracle.boxes), 0
    g = design.grid
    xs, ys = oracle.grid(g)
    adaptive = coarse_to_fine(oracle, g)
    if design.kind in ("occ", "score"):
        level = oracle.level(g)
        found = components(level > 0)
        if design.kind == "occ":
            return [cell_box(c, xs, ys) for c in found], adaptive
        return [fit_box(c, level, xs, ys) for c in found], adaptive
    occupied, counts = oracle.occupied(g), oracle.centers(g)
    occupied |= counts > 0  # a center implies its cell is occupied
    adaptive += int(occupied.sum())  # center questions only where the class is
    if design.kind == "center":
        counts = np.minimum(counts, 1)
    boxes = []
    for comp in components(occupied):
        seeds = [(r, c) for r, c in comp if counts[r, c]]
        if not seeds:  # only with --min-cover: occupancy without a center
            boxes.append(cell_box(comp, xs, ys))
            continue
        for seed, part in zip(seeds, split_by_seeds(comp, seeds, xs, ys), strict=True):
            for k in range(int(counts[seed])):  # a 2+ count: same cells, k-th object for boundaries
                box = cell_box(part, xs, ys)
                if design.bits:
                    box = refine(box, oracle.seeded(g, seed, k), xs, ys, design.bits)
                boxes.append(box)
    adaptive += len(boxes) * 4 * design.bits
    return boxes, adaptive


def parallel_decisions(design: Design, n_classes: int, objects: int) -> int:
    """Every question of each stage asked at once, for every class."""
    if design.kind == "boxes":
        return 0
    per_class = design.grid**2 * (1 if design.kind in ("occ", "score") else 2)
    return n_classes * (1 + per_class) + objects * 4 * (2**design.bits - 1)


# --- experiment -------------------------------------------------------------------------------


def summary(dets: list, truth: list, n_classes: int) -> dict[str, Any]:
    """mAP, mAP50, mAP75 and mAP by object size (COCO rules)."""

    def stats(area: tuple[float, float] | None) -> dict[str, float]:
        aps = [a for c in range(n_classes) if (a := average_precision(dets, truth, c, area)) is not None]
        if not aps:
            return {"map": float("nan"), "map50": float("nan"), "map75": float("nan")}
        a = np.stack(aps)
        return {"map": float(a.mean()), "map50": float(a[:, 0].mean()), "map75": float(a[:, 5].mean())}

    return {"all": stats(None), **{name: stats(rng) for name, rng in AREAS.items()}}


def run(design: Design, images: list[Truth], n_classes: int, min_cover: float) -> dict[str, Any]:
    dets, parallel, adaptive, objects = [], 0, 0, 0
    for im in images:
        boxes, labels = [], []
        adaptive += n_classes if design.kind != "boxes" else 0  # presence of every class
        for cls in np.unique(im.labels):
            oracle = Oracle(im.boxes[im.labels == cls], im.width, im.height, min_cover)
            found, asked = decode(design, oracle)
            boxes += found
            labels += [int(cls)] * len(found)
            adaptive += asked
        parallel += parallel_decisions(design, n_classes, len(boxes))
        objects += len(boxes)
        b = np.array(boxes, np.float32).reshape(-1, 4)
        dets.append((b, np.ones(len(b), np.float32), np.array(labels, np.int64)))
    truth = [(im.boxes.astype(np.float32), im.labels) for im in images]
    n = len(images)
    return {
        **summary(dets, truth, n_classes),
        "decisions_parallel": parallel / n,
        "decisions_adaptive": adaptive / n,
        "stages": 2 if design.bits else 1,
        "objects": objects / n,
    }


def markdown(report: dict[str, Any]) -> str:
    title = (
        f"E0 decision ceilings: {report['images']} COCO val2017 images, perfect answers, "
        f"min cover {report['min_cover']} ({report['truth_objects']:.1f} objects per image)"
    )
    header = (
        "| design | stages | decisions/image parallel | adaptive | objects/image | mAP | mAP50 | mAP75 "
        "| small | medium | large |"
    )
    lines = [
        title,
        "",
        header,
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in report["designs"].items():
        a = r["all"]
        lines.append(
            f"| {name} | {r['stages']} | {r['decisions_parallel']:.0f} | {r['decisions_adaptive']:.0f} "
            f"| {r['objects']:.1f} | {a['map']:.3f} | {a['map50']:.3f} | {a['map75']:.3f} "
            f"| {r['small']['map']:.3f} | {r['medium']['map']:.3f} | {r['large']['map']:.3f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--coco", default="datasets/coco", help="COCO root (annotations/ is downloaded once)")
    ap.add_argument("--val-images", type=int, default=500, help="the first N val2017 images, as in P2.1-P2.3")
    ap.add_argument("--designs", nargs="+", default=DEFAULT_DESIGNS)
    ap.add_argument("--min-cover", type=float, default=0.0, help="occupied: covers more than this share of a cell")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)
    designs = [parse_design(d) for d in args.designs]  # fail before downloading

    root = Path(args.coco)
    fetch_annotations(root)
    images, names = load_truth(root, args.val_images)
    report: dict[str, Any] = {
        "images": len(images),
        "min_cover": args.min_cover,
        "truth_objects": float(np.mean([len(im.labels) for im in images])),
        "designs": {},
    }
    for design in designs:
        t0 = time.perf_counter()
        report["designs"][design.name] = run(design, images, len(names), args.min_cover)
        print(f"{design.name}: {time.perf_counter() - t0:.0f}s", flush=True)
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
