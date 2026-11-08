"""P2.0: does a decision model's patch hidden state say *where* the queried object is?

    python benchmarks/patch_signal.py --model qwen3-vl-4b --device cuda --out patch_signal.json
    python benchmarks/patch_signal.py --data path/to/yolo-seg-dataset --images 256 --layers 12 16 20 24

The gate for the standalone `head` localizer (detection-research.md, P2.0). For every image and
class, the model reads the normal S1GHT Noul prompt "Is there a {class} in `image`?" once. From
that single forward pass, each method turns the image's patch tokens into a per-patch score map,
which is compared with the class's COCO masks:

- `cosine`: cosine between each patch's hidden state and the decision token's hidden state at
  the same layer (training-free). `cosine_centered` subtracts the image's mean patch first.
- `attention_mean` / `attention_top5`: the decision token's attention to the patches, over all
  heads or over the 5 heads that score best on the training images (YOLOJ's baseline).
- `objectness`: a logistic probe on the patch state, "is this patch inside any object"
  (class-agnostic; shows whether location is there at all).
- `bilinear`: a low-rank probe (A·patch)·(B·query) + w·patch, trained on half the images and
  on 3/4 of the classes; reported separately for seen and held-out ("unseen") classes.
- `tiles_N`: S1GHT's black-box baseline, N×N crops each asked the same Noul.
- `oracle`: the class's own mask coverage. Its boxes are the ceiling of the box extraction
  (threshold + connected components on 32 px patches), not of the model.

The decision token sits after the image, and attention is causal, so patch states never see
the question: any query dependence comes from the query vector. Every method and layer gets a
patch AUC (patches covered ≥ 50% by the class mask vs the rest) on the test images, and box
recall / precision from the connected components of the thresholded map (threshold and layer
chosen on the training images), against the class's instance boxes. The presence AUC is the
model's ordinary answer over present vs absent classes.

Data: a YOLO segmentation dataset (images/ + labels/ with normalized polygons). `coco128-seg`
(Ultralytics, 128 COCO train2017 images, ~7 MB) is downloaded when asked for by name.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage
from PIL import ImageDraw

from s1ght import Noul
from s1ght.runtimes import letters

COCO128_SEG = "https://github.com/ultralytics/assets/releases/download/v0.0.0/coco128-seg.zip"
COVERED = 0.5  # a patch is "inside" when the class mask covers at least this fraction of it
THRESHOLDS = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8)


def coco_names() -> list[str]:
    """The 80 COCO class names (multi-word names joined back)."""
    words = (
        "person|bicycle|car|motorcycle|airplane|bus|train|truck|boat|traffic light|fire hydrant|stop sign|"
        "parking meter|bench|bird|cat|dog|horse|sheep|cow|elephant|bear|zebra|giraffe|backpack|umbrella|"
        "handbag|tie|suitcase|frisbee|skis|snowboard|sports ball|kite|baseball bat|baseball glove|skateboard|"
        "surfboard|tennis racket|bottle|wine glass|cup|fork|knife|spoon|bowl|banana|apple|sandwich|orange|"
        "broccoli|carrot|hot dog|pizza|donut|cake|chair|couch|potted plant|bed|dining table|toilet|tv|laptop|"
        "mouse|remote|keyboard|cell phone|microwave|oven|toaster|sink|refrigerator|book|clock|vase|scissors|"
        "teddy bear|hair drier|toothbrush"
    )
    return words.split("|")


# --- data -------------------------------------------------------------------------------------


@dataclass
class Sample:
    """One image at the size the model reads it, with per-class masks and instance boxes (pixels)."""

    name: str
    image: PILImage.Image
    masks: dict[int, np.ndarray] = field(default_factory=dict)
    boxes: dict[int, list[tuple[float, float, float, float]]] = field(default_factory=dict)


def fetch_coco128_seg(root: Path) -> Path:
    """Download and unpack coco128-seg under `root` (once)."""
    target = root / "coco128-seg"
    if not (target / "images").exists():
        root.mkdir(parents=True, exist_ok=True)
        archive = root / "coco128-seg.zip"
        urllib.request.urlretrieve(COCO128_SEG, archive)
        with zipfile.ZipFile(archive) as z:
            z.extractall(root)
        archive.unlink()
    return target


def load_samples(root: Path, limit: int, max_side: int) -> list[Sample]:
    """Images and YOLO-seg labels (`labels/` mirrors `images/`), resized as S1GHT sends them."""
    paths = sorted(p for p in root.rglob("*") if p.suffix.lower() in (".jpg", ".jpeg", ".png") and "images" in p.parts)
    out = []
    for path in paths:
        parts = list(path.parts)
        parts[len(parts) - 1 - parts[::-1].index("images")] = "labels"
        label = Path(*parts).with_suffix(".txt")
        if not label.exists():
            continue
        image = letters.fit_image(PILImage.open(path), max_side)
        w, h = image.size
        sample = Sample(path.stem, image)
        for line in label.read_text().splitlines():
            values = line.split()
            if len(values) < 7:
                continue
            cls = int(values[0])
            xy = np.array(values[1:], dtype=np.float64).reshape(-1, 2) * (w, h)
            mask = PILImage.new("1", (w, h))
            ImageDraw.Draw(mask).polygon([tuple(p) for p in xy], fill=1)
            sample.masks[cls] = sample.masks.get(cls, np.zeros((h, w), bool)) | np.array(mask)
            sample.boxes.setdefault(cls, []).append((xy[:, 0].min(), xy[:, 1].min(), xy[:, 0].max(), xy[:, 1].max()))
        if sample.masks:
            out.append(sample)
        if len(out) == limit:
            break
    return out


# --- model reading ----------------------------------------------------------------------------


@dataclass
class Reading:
    """What one image's forward passes give, per selected layer and class."""

    classes: list[int]
    grid: tuple[int, int]  # patch rows, columns
    patches: dict[int, np.ndarray]  # layer -> (P, D) float16
    queries: dict[int, np.ndarray]  # layer -> (C, D) float16
    attention: dict[int, np.ndarray]  # layer -> (C, H, P) float16 (empty without --attention)
    presence: np.ndarray  # (C,) P(yes)
    tiles: dict[int, np.ndarray] = field(default_factory=dict)  # n -> (C, P) tile P(yes) per patch


class QwenReader:
    """Reads patch states, decision-token states and attention from a `qwen_vl` model.

    Other families plug in here: they need the image tokens' positions and grid and the hidden
    states of one forward pass.
    """

    def __init__(
        self, path: str, device: str, dtype: str, max_side: int, chunk: int, attention: bool, tile_batch: int = 8
    ) -> None:
        from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily

        self.family = QwenVlFamily(flip="never", batch=chunk, max_side=max_side)
        self.family.load(path, device, dtype)
        self.model, self.processor = self.family.model, self.family.processor
        self.device, self.chunk, self.attention, self.tile_batch = device, chunk, attention, tile_batch
        self.image_token = self.model.config.image_token_id
        self.merge = int(self.processor.image_processor.merge_size)

    def prompt(self, name: str) -> letters.Prompt:
        question = Noul(f"Is there a {name} in `image`?").to_wire()
        return letters.build(0, name, {"image": "[image 1]"}, question, 1, (0, 1))

    def read(self, image: PILImage.Image, classes: list[int], names: list[str], layers: list[int]) -> Reading:
        import torch

        if self.attention:
            self.model.set_attn_implementation("eager")
        texts = [
            self.processor.apply_chat_template(
                self.family.messages(self.prompt(names[c])),
                tokenize=False,
                add_generation_prompt=True,
                **self.family.chat_kwargs,
            )
            for c in classes
        ]
        patches: dict[int, np.ndarray] = {}
        queries: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
        attention: dict[int, list[np.ndarray]] = {layer: [] for layer in layers} if self.attention else {}
        presence: list[float] = []
        grid = (0, 0)
        for start in range(0, len(texts), self.chunk):
            batch = texts[start : start + self.chunk]
            inputs = self.processor(text=batch, images=[image] * len(batch), padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                out = self.model(
                    **inputs, output_hidden_states=True, output_attentions=self.attention, logits_to_keep=1
                )
            _, gh, gw = (int(v) for v in inputs["image_grid_thw"][0])
            grid = (gh // self.merge, gw // self.merge)
            for r in range(len(batch)):
                where = (inputs["input_ids"][r] == self.image_token).nonzero().flatten()
                if len(where) != grid[0] * grid[1]:
                    raise RuntimeError(f"{len(where)} image tokens for a {grid} grid")
                for layer in layers:
                    hs = out.hidden_states[layer][r]
                    if layer not in patches:  # causal: every row has the same patch states
                        patches[layer] = hs[where].float().cpu().numpy().astype(np.float16)
                    queries[layer].append(hs[-1].float().cpu().numpy().astype(np.float16))
                    if self.attention and layer > 0:  # hidden_states[0] is the embedding output
                        att = out.attentions[layer - 1][r][:, -1, where]
                        attention[layer].append(att.float().cpu().numpy().astype(np.float16))
                row = out.logits[r, -1].float()
                yes, no = (float(torch.logsumexp(row[self.family.letter_ids[c]], 0)) for c in "AB")
                presence.append(1 / (1 + math.exp(no - yes)))
        return Reading(
            classes,
            grid,
            patches,
            {k: np.stack(v) for k, v in queries.items()},
            {k: np.stack(v) for k, v in attention.items() if v},
            np.array(presence),
        )

    def tiles(
        self, image: PILImage.Image, classes: list[int], ask: set[int], names: list[str], grid: tuple[int, int], n: int
    ) -> np.ndarray:
        """Each of the n×n crops asked the same Noul; the answer spread over its patches: (C, P).

        Only classes in `ask` are asked (the others stay 0: tiles are graded on present classes)."""
        if self.attention:
            self.model.set_attn_implementation("sdpa")
        self.family.batch = self.tile_batch
        rows = np.array_split(np.arange(grid[0]), n)
        cols = np.array_split(np.arange(grid[1]), n)
        cells = [(r, c) for r in rows for c in cols if len(r) and len(c)]
        crops = [image.crop((c[0] * 32, r[0] * 32, (c[-1] + 1) * 32, (r[-1] + 1) * 32)) for r, c in cells]
        items, asked = [], [ci for ci, cls in enumerate(classes) if cls in ask]
        for ci in asked:
            cls = classes[ci]
            question = Noul(f"Is there a {names[cls]} in `image`?").to_wire()
            items += [({"image": "[image 1]"}, {"q": question}, [crop]) for crop in crops]
        answers = self.family.run(items) if items else []
        out = np.zeros((len(classes), grid[0], grid[1]))
        for k, (ans, _) in enumerate(answers):
            ai, cell = divmod(k, len(cells))
            ci = asked[ai]
            r, c = cells[cell]
            out[ci, r[0] : r[-1] + 1, c[0] : c[-1] + 1] = ans["q"]["noul"]
        return out.reshape(len(classes), -1)

    def close(self) -> None:
        self.family.close()


# --- metrics ----------------------------------------------------------------------------------


def auc(scores: np.ndarray, labels: np.ndarray) -> float:
    """ROC AUC by ranks (ties averaged); nan when one class is missing."""
    pos, neg = labels.sum(), (~labels).sum()
    if pos == 0 or neg == 0:
        return float("nan")
    _, inverse, counts = np.unique(scores, return_inverse=True, return_counts=True)
    last = np.cumsum(counts)
    ranks = ((last - counts + 1 + last) / 2)[inverse]  # 1-based ranks, ties averaged
    return float((ranks[labels].sum() - pos * (pos + 1) / 2) / (pos * neg))


def coverage(mask: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Fraction of each 32 px patch covered by `mask`, flattened row-major like the image tokens."""
    gh, gw = grid
    if mask.shape != (gh * 32, gw * 32):
        mask = np.array(PILImage.fromarray(mask).resize((gw * 32, gh * 32), PILImage.Resampling.NEAREST))
    return mask.reshape(gh, 32, gw, 32).mean(axis=(1, 3)).reshape(-1)


def components(on: np.ndarray, grid: tuple[int, int]) -> list[tuple[float, float, float, float]]:
    """Pixel boxes of the 8-connected groups of `on` patches."""
    gh, gw = grid
    on = on.reshape(gh, gw).copy()
    boxes = []
    for r0 in range(gh):
        for c0 in range(gw):
            if not on[r0, c0]:
                continue
            stack, cells = [(r0, c0)], []
            on[r0, c0] = False
            while stack:
                r, c = stack.pop()
                cells.append((r, c))
                for dr in (-1, 0, 1):
                    for dc in (-1, 0, 1):
                        rr, cc = r + dr, c + dc
                        if 0 <= rr < gh and 0 <= cc < gw and on[rr, cc]:
                            on[rr, cc] = False
                            stack.append((rr, cc))
            rs, cs = zip(*cells, strict=True)
            boxes.append((min(cs) * 32.0, min(rs) * 32.0, (max(cs) + 1) * 32.0, (max(rs) + 1) * 32.0))
    return boxes


def iou(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = w * h
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(pred: list, truth: list, at: float) -> int:
    """True positives of a greedy one-to-one IoU matching."""
    pairs = sorted(((iou(p, t), i, j) for i, p in enumerate(pred) for j, t in enumerate(truth)), reverse=True)
    used_p, used_t, tp = set(), set(), 0
    for v, i, j in pairs:
        if v < at:
            break
        if i not in used_p and j not in used_t:
            used_p.add(i)
            used_t.add(j)
            tp += 1
    return tp


def normalize(m: np.ndarray) -> np.ndarray:
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else np.zeros_like(m)


# --- probes -----------------------------------------------------------------------------------


def train_probes(
    readings: list[Reading], samples: list[Sample], layer: int, seen: set[int], device: str, rank: int, steps: int
) -> tuple[Any, Any]:
    """Objectness (logistic) and bilinear probes for one layer on the given (training) images.

    Returns two functions mapping a Reading to (P,) and (C, P) score maps.
    """
    import torch

    pmax = max(r.patches[layer].shape[0] for r in readings)
    dim = readings[0].patches[layer].shape[1]
    patches = torch.zeros(len(readings), pmax, dim)
    valid = torch.zeros(len(readings), pmax, dtype=torch.bool)
    anyobj = torch.zeros(len(readings), pmax)
    q_rows, img_idx, labels = [], [], []
    for i, (rd, s) in enumerate(zip(readings, samples, strict=True)):
        n = rd.patches[layer].shape[0]
        patches[i, :n] = torch.from_numpy(rd.patches[layer].astype(np.float32))
        valid[i, :n] = True
        union = np.zeros(s.image.size[::-1], bool)
        for m in s.masks.values():
            union |= m
        anyobj[i, :n] = torch.from_numpy((coverage(union, rd.grid) >= COVERED).astype(np.float32))
        for ci, cls in enumerate(rd.classes):
            if cls not in seen:
                continue
            y = np.zeros(pmax, np.float32)
            if cls in s.masks:
                y[:n] = coverage(s.masks[cls], rd.grid) >= COVERED
            q_rows.append(rd.queries[layer][ci].astype(np.float32))
            img_idx.append(i)
            labels.append(y)
    mu, sd = patches[valid].mean(0), patches[valid].std(0) + 1e-6
    queries = torch.from_numpy(np.stack(q_rows))
    qmu, qsd = queries.mean(0), queries.std(0) + 1e-6
    x = ((patches - mu) / sd * valid[..., None]).to(device)
    q = ((queries - qmu) / qsd).to(device)
    y = torch.from_numpy(np.stack(labels)).to(device)
    idx = torch.tensor(img_idx, device=device)
    vmask, anyobj, valid_d = valid[idx.cpu()].to(device), anyobj.to(device), valid.to(device)

    def fit(params: list, loss_fn: Any) -> None:
        opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-4)
        for _ in range(steps):
            opt.zero_grad()
            loss_fn().backward()
            opt.step()

    def bce(logits: Any, target: Any, mask: Any) -> Any:
        pos = (target * mask).sum().clamp(min=1)
        weight = torch.where(target > 0, (mask.sum() - pos) / pos, torch.ones_like(target)) * mask
        return torch.nn.functional.binary_cross_entropy_with_logits(logits, target, weight=weight)

    w_obj = torch.zeros(dim, device=device, requires_grad=True)
    b_obj = torch.zeros(1, device=device, requires_grad=True)
    fit([w_obj, b_obj], lambda: bce(x @ w_obj + b_obj, anyobj, valid_d.float()))

    gen = torch.Generator().manual_seed(0)
    a = (torch.randn(rank, dim, generator=gen) / math.sqrt(dim)).to(device).requires_grad_()
    b = (torch.randn(rank, dim, generator=gen) / math.sqrt(dim)).to(device).requires_grad_()
    w = torch.zeros(dim, device=device, requires_grad=True)
    bias = torch.zeros(1, device=device, requires_grad=True)

    def bilinear_logits(xx: Any, qq: Any, img: Any) -> Any:
        return ((xx @ a.T)[img] * (qq @ b.T)[:, None, :]).sum(-1) + (xx @ w)[img] + bias

    fit([a, b, w, bias], lambda: bce(bilinear_logits(x, q, idx), y, vmask.float()))

    def objectness(rd: Reading) -> np.ndarray:
        with torch.no_grad():
            xx = (torch.from_numpy(rd.patches[layer].astype(np.float32)) - mu) / sd
            return (xx.to(device) @ w_obj + b_obj).cpu().numpy()

    def bilinear(rd: Reading) -> np.ndarray:
        with torch.no_grad():
            xx = ((torch.from_numpy(rd.patches[layer].astype(np.float32)) - mu) / sd).to(device)[None]
            qq = ((torch.from_numpy(rd.queries[layer].astype(np.float32)) - qmu) / qsd).to(device)
            img = torch.zeros(len(qq), dtype=torch.long, device=device)
            return torch.sigmoid(bilinear_logits(xx, qq, img)).cpu().numpy()

    return objectness, bilinear


# --- experiment -------------------------------------------------------------------------------


def score_maps(rd: Reading, layer: int, probes: dict[str, Any], heads: dict[int, list[int]]) -> dict[str, np.ndarray]:
    """Per method, a (C, P) map for one image at one layer."""
    p = rd.patches[layer].astype(np.float32)
    q = rd.queries[layer].astype(np.float32)

    def cos(pp: np.ndarray) -> np.ndarray:
        pn = pp / (np.linalg.norm(pp, axis=1, keepdims=True) + 1e-6)
        qn = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-6)
        return qn @ pn.T

    maps = {"cosine": cos(p), "cosine_centered": cos(p - p.mean(0))}
    if layer in rd.attention:
        att = rd.attention[layer].astype(np.float32)
        maps["attention_mean"] = att.mean(1)
        if heads.get(layer):
            maps["attention_top5"] = att[:, heads[layer]].mean(1)
    if "objectness" in probes:
        maps["objectness"] = np.repeat(probes["objectness"](rd)[None], len(rd.classes), 0)
        maps["bilinear"] = probes["bilinear"](rd)
    return maps


def pair_labels(rd: Reading, s: Sample) -> list[np.ndarray | None]:
    """Per class of the reading: patch labels when the class is present and gradable, else None."""
    out = []
    for cls in rd.classes:
        if cls not in s.masks:
            out.append(None)
            continue
        y = coverage(s.masks[cls], rd.grid) >= COVERED
        out.append(y if 0 < y.sum() < len(y) else None)
    return out


def evaluate(
    readings: list[Reading],
    samples: list[Sample],
    split: list[bool],
    layers: list[int],
    seen: set[int],
    device: str,
    rank: int,
    steps: int,
) -> dict[str, Any]:
    train = [i for i, t in enumerate(split) if t]
    # attention heads: the 5 with the best mean AUC on the training images, per layer
    heads: dict[int, list[int]] = {}
    for layer in layers:
        per_head: list[list[float]] = []
        for i in train:
            rd = readings[i]
            if layer not in rd.attention:
                break
            for ci, y in enumerate(pair_labels(rd, samples[i])):
                if y is not None:
                    per_head.append([auc(h, y) for h in rd.attention[layer][ci].astype(np.float32)])
        if per_head:
            heads[layer] = list(np.argsort(np.nanmean(per_head, 0))[::-1][:5])

    results: dict[str, dict[int, dict[str, float]]] = {}
    boxes_by: dict[str, dict[int, dict[str, Any]]] = {}
    for layer in layers:
        t0 = time.perf_counter()
        probes: dict[str, Any] = {}
        if layer > 0:
            obj, bil = train_probes(
                [readings[i] for i in train], [samples[i] for i in train], layer, seen, device, rank, steps
            )
            probes = {"objectness": obj, "bilinear": bil}
        collected: dict[str, dict[str, list]] = {}
        for i in range(len(readings)):
            rd, s = readings[i], samples[i]
            maps = score_maps(rd, layer, probes, heads)
            for ci, y in enumerate(pair_labels(rd, s)):
                if y is None:
                    continue
                cls = rd.classes[ci]
                part = "train" if split[i] else ("test_unseen" if cls not in seen else "test")
                for name, m in maps.items():
                    if name == "bilinear" and part == "test_unseen":
                        name = "bilinear_unseen"
                    c = collected.setdefault(name, {"train": [], "test": [], "pairs": []})
                    key = "train" if split[i] else "test"
                    c[key].append(auc(m[ci], y))
                    c["pairs"].append((key, normalize(m[ci]), rd.grid, s.boxes[cls]))
        for name, c in collected.items():
            results.setdefault(name, {})[layer] = {
                "train_auc": float(np.nanmean(c["train"])) if c["train"] else float("nan"),
                "test_auc": float(np.nanmean(c["test"])) if c["test"] else float("nan"),
                "test_pairs": len(c["test"]),
            }
            boxes_by.setdefault(name, {})[layer] = {"pairs": c["pairs"]}
        print(f"layer {layer}: {time.perf_counter() - t0:.0f}s", flush=True)

    # layer-free maps: tiles, and the oracle (the class's own mask coverage), whose boxes are the
    # ceiling of thresholding + connected components on 32 px patches
    free: dict[str, Any] = {
        f"tiles_{n}": (lambda rd, s, ci, n=n: rd.tiles[n][ci]) for n in sorted({n for rd in readings for n in rd.tiles})
    }
    free["oracle"] = lambda rd, s, ci: coverage(s.masks[rd.classes[ci]], rd.grid)
    for name, get in free.items():
        c: dict[str, list] = {"train": [], "test": [], "pairs": []}
        for i, (rd, s) in enumerate(zip(readings, samples, strict=True)):
            for ci, y in enumerate(pair_labels(rd, s)):
                if y is None:
                    continue
                key = "train" if split[i] else "test"
                m = get(rd, s, ci)
                c[key].append(auc(m, y))
                c["pairs"].append((key, m, rd.grid, s.boxes[rd.classes[ci]]))
        results[name] = {
            -1: {
                "train_auc": float(np.nanmean(c["train"])),
                "test_auc": float(np.nanmean(c["test"])),
                "test_pairs": len(c["test"]),
            }
        }
        boxes_by[name] = {-1: {"pairs": c["pairs"]}}

    best: dict[str, dict[str, Any]] = {}
    # held-out classes have no training pairs: they take the layer and threshold of their probe
    for name in sorted(results, key=lambda n: n.endswith("_unseen")):
        per_layer = results[name]
        if name.endswith("_unseen") and name.removesuffix("_unseen") in best:
            base = best[name.removesuffix("_unseen")]
            layer, threshold = base["layer"], base["threshold"]
            pairs = boxes_by[name][layer]["pairs"]
        else:
            layer = max(per_layer, key=lambda k: np.nan_to_num(per_layer[k]["train_auc"], nan=-1))
            pairs = boxes_by[name][layer]["pairs"]
            threshold = max(THRESHOLDS, key=lambda t: _f1(pairs, "train", t, 0.5))
        best[name] = {
            "layer": layer,
            "test_auc": per_layer[layer]["test_auc"],
            "threshold": threshold,
            **{f"{k}@{at}": v for at in (0.3, 0.5) for k, v in _box_stats(pairs, "test", threshold, at).items()},
        }
    presence = _presence_auc(readings, samples, split)
    return {
        "per_layer": results,
        "best": best,
        "presence_test_auc": presence,
        "heads": {k: [int(h) for h in v] for k, v in heads.items()},
    }


def _box_stats(pairs: list, part: str, threshold: float, at: float) -> dict[str, float]:
    tp = n_pred = n_true = 0
    for key, m, grid, truth in pairs:
        if key != part:
            continue
        pred = components(m >= threshold, grid)
        tp += match(pred, truth, at)
        n_pred += len(pred)
        n_true += len(truth)
    return {"recall": tp / n_true if n_true else float("nan"), "precision": tp / n_pred if n_pred else float("nan")}


def _f1(pairs: list, part: str, threshold: float, at: float) -> float:
    s = _box_stats(pairs, part, threshold, at)
    r, p = np.nan_to_num(s["recall"]), np.nan_to_num(s["precision"])
    return 2 * r * p / (r + p) if r + p else 0.0


def _presence_auc(readings: list[Reading], samples: list[Sample], split: list[bool]) -> float:
    scores, labels = [], []
    for rd, s, is_train in zip(readings, samples, split, strict=True):
        if is_train:
            continue
        for ci, cls in enumerate(rd.classes):
            scores.append(rd.presence[ci])
            labels.append(cls in s.masks)
    return auc(np.array(scores), np.array(labels))


def markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    head = (
        f"**{cfg['model']}** on {cfg['data']} ({report['images']} images, half for training probes, "
        f"thresholds and layers; {cfg['max_side']} px). Presence AUC (the model's own Noul answer, "
        f"test): {report['presence_test_auc']:.3f}."
    )
    lines = [
        head,
        "",
        "| method | layer | patch AUC | box recall@0.5 | box precision@0.5 | box recall@0.3 | box precision@0.3 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, b in report["best"].items():
        layer = "-" if b["layer"] < 0 else str(b["layer"])
        lines.append(
            f"| {name} | {layer} | {b['test_auc']:.3f} | {b['recall@0.5']:.3f} | {b['precision@0.5']:.3f} "
            f"| {b['recall@0.3']:.3f} | {b['precision@0.3']:.3f} |"
        )
    layered = [n for n, per in report["per_layer"].items() if -1 not in per]
    layers = sorted({int(k) for n in layered for k in report["per_layer"][n]})
    lines += ["", "Test patch AUC by layer:", "", "| method | " + " | ".join(map(str, layers)) + " |"]
    lines.append("| --- | " + " | ".join("---:" for _ in layers) + " |")
    for n in layered:
        per = report["per_layer"][n]
        lines.append(f"| {n} | " + " | ".join(f"{per[k]['test_auc']:.3f}" if k in per else "" for k in layers) + " |")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3-vl-4b", help="catalog name or local model folder (qwen_vl family)")
    ap.add_argument("--data", default="coco128-seg", help="'coco128-seg' (downloaded) or a YOLO-seg dataset folder")
    ap.add_argument("--cache", default="datasets", help="where coco128-seg is unpacked")
    ap.add_argument("--images", type=int, default=128)
    ap.add_argument("--layers", type=int, nargs="+", help="hidden-state indices (0 = embeddings); default every 4th")
    ap.add_argument("--present", type=int, default=3, help="present classes asked per image")
    ap.add_argument("--absent", type=int, default=3, help="absent classes asked per image")
    ap.add_argument("--tiles", type=int, nargs="*", default=[2, 4])
    ap.add_argument("--attention", action=argparse.BooleanOptionalAction, default=True, help="needs eager attention")
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=3, help="prompts per forward pass when reading hidden states")
    ap.add_argument("--tile-batch", type=int, default=8, help="crops per forward pass for tiles")
    ap.add_argument("--rank", type=int, default=64)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None, help="JSON report path")
    args = ap.parse_args(argv)

    from s1ght.runtimes.native.engine import select_device, select_dtype

    device = select_device(args.device)
    if Path(args.model).is_dir():
        path, dtype = args.model, args.dtype or select_dtype(device)
    else:
        from s1ght.hub import download, resolve

        spec = resolve(args.model)
        if spec.family != "qwen_vl":
            raise SystemExit(f"{args.model}: only qwen_vl models expose patch states here (family {spec.family})")
        assert spec.source is not None
        path = str(download(spec.source))
        dtype = args.dtype or select_dtype(device, (spec.options or {}).get("dtype"))
    root = fetch_coco128_seg(Path(args.cache)) if args.data == "coco128-seg" else Path(args.data)
    samples = load_samples(root, args.images, args.max_side)
    names = coco_names()
    rng = random.Random(args.seed)
    seen = {c for c in range(len(names)) if c % 4 != 3}  # every 4th class is held out from the probes

    reader = QwenReader(path, device, dtype, args.max_side, args.chunk, args.attention, args.tile_batch)
    n_layers = int(reader.model.config.text_config.num_hidden_layers)
    layers = args.layers or list(range(0, n_layers + 1, 4))
    readings = []
    t0 = time.perf_counter()
    try:
        for k, s in enumerate(samples):
            present = sorted(s.masks)
            absent = [c for c in range(len(names)) if c not in s.masks]
            classes = rng.sample(present, min(args.present, len(present))) + rng.sample(absent, args.absent)
            rd = reader.read(s.image, classes, names, layers)
            for n in args.tiles:
                rd.tiles[n] = reader.tiles(s.image, classes, set(s.masks), names, rd.grid, n)
            readings.append(rd)
            if k % 8 == 0:
                print(f"read {k + 1}/{len(samples)} images, {time.perf_counter() - t0:.0f}s", flush=True)
    finally:
        reader.close()

    split = [k % 2 == 0 for k in range(len(samples))]
    report = evaluate(readings, samples, split, layers, seen, device, args.rank, args.steps)
    report["images"] = len(samples)
    report["config"] = {k: v for k, v in vars(args).items()} | {"device": device, "dtype": dtype, "layers": layers}
    report["seconds"] = time.perf_counter() - t0
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
