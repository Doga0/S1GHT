"""P2.1: does a small trained head turn a frozen decision model into a detector? (COCO mAP)

    python benchmarks/patch_head.py --device cuda --presence --reference yolo26n.pt --out head.json
    python benchmarks/patch_head.py --train-images 8000 --layers 4 16 20 --epochs 20

P2.0 (`patch_signal.py`) showed that a frozen Qwen3-VL-4B knows *whether* an object is there
(presence AUC 0.99) but that location does not fall out of training-free readouts or linear
probes (patch AUC ≤ 0.81). This experiment trains what detectors actually train: a small head
on the frozen model's patch states, with a score and a box per patch, and measures COCO mAP.

- **Features:** one forward pass per image (S1GHT's prompt with the image). The decision prompt
  comes after the image and attention is causal, so patch states do not depend on the question:
  they are computed once per frame and serve every query. States of `--layers` are concatenated
  and cached on disk (float16 memmap).
- **Queries:** per class, the hidden state at the same layers of the last token of a text-only
  Noul prompt "Is there a {class} in the image?", computed once ("prompt-then-detect").
- **Head:** LayerNorm + linear, 2D position, a 2-layer transformer over the patches; class score
  = scaled cosine(patch, query) + bias (OWL-ViT style); a class-agnostic box per patch as
  distances to the four sides (FCOS style). Focal loss for scores, GIoU + L1 for boxes. A patch
  is positive for a class when its center lies in a box of that class; the patch holding a box's
  center is always positive (small objects).
- **Classes:** every 4th COCO class (20) is held out from training; mAP is reported for seen
  and unseen classes.
- **Inference:** sigmoid scores, top-k per class, NMS, 100 detections per image. `--presence`
  multiplies each class's scores by the model's own Noul answer for that image (asked for the
  `--presence-top` classes with the highest head score; the others are dropped).
- **mAP:** COCO style (IoU 0.50:0.95, 101-point interpolation, 100 detections, crowd boxes
  removed), computed here so seen and unseen classes can be split. `--reference` runs Ultralytics
  models on the same pixels and scores them with the same code: they are trained on all of COCO
  train (118k images), so they are a yardstick, not a like-for-like comparison.

Data: COCO 2017 from images.cocodataset.org. The annotation zip (253 MB) is downloaded once,
then only the sampled images (~160 kB each).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image as PILImage

from s1ght import Noul
from s1ght.runtimes import letters

COCO_URL = "http://images.cocodataset.org"
STRIDE = 32  # Qwen3-VL: 16 px patches merged 2x2
IOUS = np.linspace(0.5, 0.95, 10)


# --- data -------------------------------------------------------------------------------------


@dataclass
class Item:
    """One COCO image: boxes (xyxy, pixels of the original image) and contiguous labels."""

    id: int
    file: Path
    boxes: np.ndarray
    labels: np.ndarray


def fetch_annotations(root: Path) -> None:
    """Download COCO 2017 instance annotations into `root/annotations` (once)."""
    if (root / "annotations" / "instances_val2017.json").exists():
        return
    root.mkdir(parents=True, exist_ok=True)
    archive = root / "annotations_trainval2017.zip"
    urllib.request.urlretrieve(f"{COCO_URL}/annotations/annotations_trainval2017.zip", archive)
    with zipfile.ZipFile(archive) as z:
        for name in ("annotations/instances_train2017.json", "annotations/instances_val2017.json"):
            z.extract(name, root)
    archive.unlink()


def load_split(root: Path, split: str, n: int, seed: int) -> tuple[list[Item], list[str]]:
    """`n` images with at least one non-crowd box, and the class names in contiguous order.

    Val takes the first `n` image ids; train a seeded random sample. Category ids are mapped to
    0..K-1 in sorted order (Ultralytics' COCO order).
    """
    data = json.loads((root / "annotations" / f"instances_{split}.json").read_text())
    cats = sorted(data["categories"], key=lambda c: c["id"])
    index = {c["id"]: k for k, c in enumerate(cats)}
    files = {im["id"]: im["file_name"] for im in data["images"]}
    by_image: dict[int, list[tuple[list[float], int]]] = {}
    for a in data["annotations"]:
        if a.get("iscrowd"):
            continue
        x, y, w, h = a["bbox"]
        if w > 1 and h > 1:
            by_image.setdefault(a["image_id"], []).append(([x, y, x + w, y + h], index[a["category_id"]]))
    ids = sorted(by_image)
    ids = ids[:n] if split.startswith("val") else sorted(random.Random(seed).sample(ids, min(n, len(ids))))
    items = []
    for i in ids:
        boxes, labels = zip(*by_image[i], strict=True)
        items.append(Item(i, root / split / files[i], np.array(boxes, np.float32), np.array(labels, np.int64)))
    return items, [c["name"] for c in cats]


def fetch_images(items: list[Item], split: str, workers: int = 32) -> None:
    """Download the images that are not on disk yet."""
    missing = [it for it in items if not it.file.exists()]
    if not missing:
        return
    missing[0].file.parent.mkdir(parents=True, exist_ok=True)

    def get(it: Item) -> None:
        tmp = it.file.with_suffix(".part")
        urllib.request.urlretrieve(f"{COCO_URL}/{split}/{it.file.name}", tmp)
        tmp.replace(it.file)

    with ThreadPoolExecutor(workers) as pool:
        list(pool.map(get, missing))


def fitted(item: Item, max_side: int) -> tuple[PILImage.Image, np.ndarray]:
    """The image as S1GHT sends it (sides multiples of 32) and its boxes in those pixels."""
    original = PILImage.open(item.file)
    image = letters.fit_image(original, max_side)
    sx, sy = image.size[0] / original.size[0], image.size[1] / original.size[1]
    return image, item.boxes * np.array([sx, sy, sx, sy], np.float32)


# --- features ---------------------------------------------------------------------------------


class Extractor:
    """Patch states (one forward per image), text query states and presence from a `qwen_vl` model."""

    def __init__(self, path: str, device: str, dtype: str, layers: list[int], chunk: int, max_side: int) -> None:
        from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily

        self.family = QwenVlFamily(flip="never", batch=chunk, max_side=max_side)
        self.family.load(path, device, dtype)
        self.model, self.processor = self.family.model, self.family.processor
        self.device, self.layers, self.chunk = device, layers, chunk
        self.image_token = self.model.config.image_token_id
        self.dim = int(self.model.config.text_config.hidden_size) * len(layers)

    def text(self, prompt: letters.Prompt) -> str:
        return str(
            self.processor.apply_chat_template(
                self.family.messages(prompt), tokenize=False, add_generation_prompt=True, **self.family.chat_kwargs
            )
        )

    def patches(self, images: list[PILImage.Image]) -> list[np.ndarray]:
        """Per image: (P, dim) float16, patches in row-major order."""
        import torch

        question = Noul("Is there an object in `image`?").to_wire()
        text = self.text(letters.build(0, "q", {"image": "[image 1]"}, question, 1, (0, 1)))
        out_rows = []
        for start in range(0, len(images), self.chunk):
            batch = images[start : start + self.chunk]
            inputs = self.processor(text=[text] * len(batch), images=batch, padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                hs = self.model(**inputs, output_hidden_states=True, logits_to_keep=1).hidden_states
            for r, image in enumerate(batch):
                where = (inputs["input_ids"][r] == self.image_token).nonzero().flatten()
                expected = (image.size[1] // STRIDE) * (image.size[0] // STRIDE)
                if len(where) != expected:
                    raise RuntimeError(f"{len(where)} image tokens, expected {expected} for {image.size}")
                rows = torch.cat([hs[layer][r, where] for layer in self.layers], dim=-1)
                out_rows.append(rows.to(torch.float16).cpu().numpy())
        return out_rows

    def queries(self, names: list[str]) -> np.ndarray:
        """(C, dim) float32: last-token states of a text-only Noul prompt per class."""
        import torch

        texts = [
            self.text(letters.build(0, "q", None, Noul(f"Is there a {n} in the image?").to_wire(), 0, (0, 1)))
            for n in names
        ]
        out = []
        for start in range(0, len(texts), 16):
            inputs = self.processor(text=texts[start : start + 16], padding=True, return_tensors="pt").to(self.device)
            with torch.inference_mode():
                hs = self.model(**inputs, output_hidden_states=True, logits_to_keep=1).hidden_states
            out.append(torch.cat([hs[layer][:, -1] for layer in self.layers], dim=-1).float().cpu().numpy())
        return np.concatenate(out)

    def presence(self, image: PILImage.Image, classes: list[int], names: list[str]) -> np.ndarray:
        """The model's own Noul answer, P(yes), for each class (S1GHT's ordinary readout)."""
        questions = {str(c): Noul(f"Is there a {names[c]} in `image`?").to_wire() for c in classes}
        answers, _ = self.family.run([({"image": "[image 1]"}, questions, [image])])[0]
        return np.array([answers[str(c)]["noul"] for c in classes])

    def close(self) -> None:
        self.family.close()


class Store:
    """Patch states of many images, appended to one float16 file, then read back as a memmap."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path, self.dim = path, dim
        self.grids: list[tuple[int, int]] = []
        self.offsets = np.zeros(1, np.int64)
        self._sizes = [0]
        self._file: Any = open(path, "wb")  # noqa: SIM115 - closed in finish()
        self.data: Any = None

    def append(self, rows: np.ndarray, grid: tuple[int, int]) -> None:
        self._file.write(np.ascontiguousarray(rows, np.float16).tobytes())
        self.grids.append(grid)
        self._sizes.append(self._sizes[-1] + len(rows))

    def finish(self) -> None:
        self._file.close()
        self.offsets = np.array(self._sizes, np.int64)
        self.data = np.memmap(self.path, np.float16, "r", shape=(int(self.offsets[-1]), self.dim))

    def __getitem__(self, i: int) -> np.ndarray:
        return np.asarray(self.data[self.offsets[i] : self.offsets[i + 1]])


# --- targets ----------------------------------------------------------------------------------


def targets(
    boxes: np.ndarray, labels: np.ndarray, grid: tuple[int, int], classes: list[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per patch: class labels (C, P) for `classes`, a box target (P, 4) and a box mask (P,).

    A patch is positive for a class when its center lies in a box of that class; the patch that
    holds a box's center is always positive. Its box target is the smallest such box.
    """
    gh, gw = grid
    cy, cx = np.meshgrid((np.arange(gh) + 0.5) * STRIDE, (np.arange(gw) + 0.5) * STRIDE, indexing="ij")
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    col = {c: k for k, c in enumerate(classes)}
    cls = np.zeros((len(classes), gh * gw), np.float32)
    box = np.zeros((gh * gw, 4), np.float32)
    best = np.full(gh * gw, np.inf)
    for b, label in sorted(zip(boxes, labels, strict=True), key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1])):
        inside = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
        center = min(int((b[1] + b[3]) / 2 // STRIDE), gh - 1) * gw + min(int((b[0] + b[2]) / 2 // STRIDE), gw - 1)
        inside[center] = True
        area = (b[2] - b[0]) * (b[3] - b[1])
        take = inside & (area <= best)  # boxes come largest first: smaller ones overwrite
        box[take], best[take] = b, area
        if label in col:
            cls[col[label], inside] = 1.0
    return cls, box, np.isfinite(best)


# --- head -------------------------------------------------------------------------------------


def build_head(dim: int, width: int = 256, depth: int = 2) -> Any:
    import torch
    from torch import nn

    class Head(nn.Module):
        """Patch states + queries -> per-patch class logits (B, C, P) and boxes (B, P, 4) in pixels."""

        def __init__(self) -> None:
            super().__init__()
            self.inp = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width))
            self.pos = nn.Sequential(nn.Linear(4, width), nn.GELU(), nn.Linear(width, width))
            layer = nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
            self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
            self.norm = nn.LayerNorm(width)
            self.key = nn.Linear(width, width)
            self.query = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, width), nn.GELU(), nn.Linear(width, width))
            self.box = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 4))
            self.scale = nn.Parameter(torch.tensor(math.log(10.0)))
            self.bias = nn.Parameter(torch.tensor(-4.0))  # low prior: most patches are background

        def forward(self, x: Any, pos: Any, valid: Any, q: Any) -> tuple[Any, Any]:
            h = self.inp(x) + self.pos(pos)
            h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
            k = nn.functional.normalize(self.key(h), dim=-1)
            e = nn.functional.normalize(self.query(q), dim=-1)
            logits = torch.einsum("bpd,cd->bcp", k, e) * self.scale.exp() + self.bias
            ltrb = nn.functional.softplus(self.box(h)) * STRIDE
            centers = pos[..., 2:] * STRIDE  # (x, y) of patch centers in pixels
            boxes = torch.cat([centers - ltrb[..., :2], centers + ltrb[..., 2:]], dim=-1)
            return logits, boxes

    return Head()


def positions(grid: tuple[int, int]) -> np.ndarray:
    """Per patch: (y, x) normalized to [0, 1] and (x, y) centers in patch units, (P, 4)."""
    gh, gw = grid
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r, c = r.reshape(-1), c.reshape(-1)
    return np.stack([(r + 0.5) / gh, (c + 0.5) / gw, c + 0.5, r + 0.5], 1).astype(np.float32)


def batch_tensors(store: Store, idx: list[int], device: str) -> tuple[Any, Any, Any]:
    import torch

    pmax = max(store.offsets[i + 1] - store.offsets[i] for i in idx)
    x = torch.zeros(len(idx), pmax, store.dim)
    pos = torch.zeros(len(idx), pmax, 4)
    valid = torch.zeros(len(idx), pmax, dtype=torch.bool)
    for b, i in enumerate(idx):
        rows = store[i]
        x[b, : len(rows)] = torch.from_numpy(rows.astype(np.float32))
        pos[b, : len(rows)] = torch.from_numpy(positions(store.grids[i]))
        valid[b, : len(rows)] = True
    return x.to(device), pos.to(device), valid.to(device)


def train_head(
    store: Store,
    boxes: list[np.ndarray],
    labels: list[np.ndarray],
    queries: np.ndarray,
    seen: list[int],
    device: str,
    epochs: int,
    batch: int,
    negatives: int,
    seed: int,
) -> Any:
    import torch
    from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = build_head(store.dim).to(device)
    q_all = torch.from_numpy(queries).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=3e-4, weight_decay=0.05)
    order = list(range(len(boxes)))
    steps = epochs * math.ceil(len(order) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=steps, pct_start=0.05)
    seen_set = set(seen)
    for epoch in range(epochs):
        rng.shuffle(order)
        t0, total = time.perf_counter(), 0.0
        for start in range(0, len(order), batch):
            idx = order[start : start + batch]
            present = sorted({int(c) for i in idx for c in labels[i] if c in seen_set})
            classes = present + rng.sample(
                [c for c in seen if c not in present], min(negatives, len(seen) - len(present))
            )
            x, pos, valid = batch_tensors(store, idx, device)
            cls_t = torch.zeros(len(idx), len(classes), x.shape[1], device=device)
            box_t = torch.zeros(len(idx), x.shape[1], 4, device=device)
            box_m = torch.zeros(len(idx), x.shape[1], dtype=torch.bool, device=device)
            for b, i in enumerate(idx):
                c, bx, m = targets(boxes[i], labels[i], store.grids[i], classes)
                n = c.shape[1]
                cls_t[b, :, :n], box_t[b, :n], box_m[b, :n] = (
                    torch.from_numpy(c),
                    torch.from_numpy(bx),
                    torch.from_numpy(m),
                )
            logits, pred = head(x, pos, valid, q_all[classes])
            mask = valid[:, None, :].expand_as(logits)
            n_pos = max(1.0, float(cls_t.sum()))
            loss_cls = sigmoid_focal_loss(logits[mask], cls_t[mask], reduction="sum") / n_pos
            loss_box = torch.zeros((), device=device)
            if box_m.any():
                p, t = pred[box_m], box_t[box_m]
                loss_box = generalized_box_iou_loss(p, t, reduction="mean") + (p - t).abs().mean() / 64
            loss = loss_cls + 2.0 * loss_box
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
            opt.step()
            sched.step()
            total += float(loss.detach())
        n_batches = math.ceil(len(order) / batch)
        print(f"epoch {epoch + 1}/{epochs}: loss {total / n_batches:.3f}, {time.perf_counter() - t0:.0f}s", flush=True)
    return head.eval()


def detect(
    head: Any, store: Store, i: int, queries: np.ndarray, device: str, top: int = 50, iou: float = 0.6
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(boxes, scores, labels) for one image over all classes, at most 100, before presence."""
    import torch
    from torchvision.ops import batched_nms

    x, pos, valid = batch_tensors(store, [i], device)
    with torch.inference_mode():
        logits, boxes = head(x, pos, valid, torch.from_numpy(queries).to(device))
    scores = logits[0].sigmoid()  # (C, P)
    k = min(top, scores.shape[1])
    s, p = scores.topk(k, dim=1)
    labels = torch.arange(scores.shape[0], device=device)[:, None].expand(-1, k).reshape(-1)
    b, s = boxes[0][p.reshape(-1)], s.reshape(-1)
    keep = batched_nms(b, s, labels, iou)[:100]
    return b[keep].cpu().numpy(), s[keep].cpu().numpy(), labels[keep].cpu().numpy()


# --- mAP --------------------------------------------------------------------------------------


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])
    return inter / (area(a)[:, None] + area(b)[None] - inter + 1e-9)


def average_precision(
    dets: list[tuple[np.ndarray, np.ndarray, np.ndarray]], truth: list[tuple[np.ndarray, np.ndarray]], cls: int
) -> np.ndarray | None:
    """COCO AP of one class at each IoU in `IOUS` (101-point); None when it has no ground truth."""
    n_true = sum(int((lab == cls).sum()) for _, lab in truth)
    if n_true == 0:
        return None
    records = []  # (score, image, box)
    for img, (boxes, scores, labels) in enumerate(dets):
        for b, s in zip(boxes[labels == cls], scores[labels == cls], strict=True):
            records.append((float(s), img, b))
    records.sort(key=lambda r: -r[0])
    out = np.zeros(len(IOUS))
    for t, thr in enumerate(IOUS):
        used = {img: np.zeros(int((lab == cls).sum()), bool) for img, (_, lab) in enumerate(truth)}
        tp = np.zeros(len(records))
        for k, (_, img, b) in enumerate(records):
            gt = truth[img][0][truth[img][1] == cls]
            if len(gt) == 0:
                continue
            ious = box_iou(b[None], gt)[0]
            ious[used[img]] = -1
            j = int(ious.argmax())
            if ious[j] >= thr:
                used[img][j] = True
                tp[k] = 1
        ctp = np.cumsum(tp)
        recall = ctp / n_true
        precision = ctp / np.arange(1, len(records) + 1) if records else np.zeros(0)
        precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
        points = np.linspace(0, 1, 101)
        where = np.searchsorted(recall, points, side="left")
        out[t] = np.mean([precision[w] if w < len(precision) else 0.0 for w in where])
    return out


def coco_map(dets: list, truth: list, classes: list[int]) -> dict[str, float]:
    aps = [ap for c in classes if (ap := average_precision(dets, truth, c)) is not None]
    if not aps:
        return {"map": float("nan"), "map50": float("nan"), "classes": 0}
    a = np.stack(aps)
    return {"map": float(a.mean()), "map50": float(a[:, 0].mean()), "classes": len(aps)}


def scores_by_split(dets: list, truth: list, n_classes: int, seen: set[int]) -> dict[str, dict[str, float]]:
    every = list(range(n_classes))
    return {
        "all": coco_map(dets, truth, every),
        "seen": coco_map(dets, truth, [c for c in every if c in seen]),
        "unseen": coco_map(dets, truth, [c for c in every if c not in seen]),
    }


def reference(weights: str, images: list[PILImage.Image], device: str) -> tuple[list, float]:
    """An Ultralytics model on the same pixels: detections and ms per image."""
    from ultralytics import YOLO

    model = YOLO(weights)
    model.predict(images[0], verbose=False, device=device)  # warm-up
    dets, t0 = [], time.perf_counter()
    for image in images:
        r = model.predict(image, conf=0.001, max_det=100, verbose=False, device=device)[0]
        dets.append((r.boxes.xyxy.cpu().numpy(), r.boxes.conf.cpu().numpy(), r.boxes.cls.cpu().numpy().astype(int)))
    return dets, (time.perf_counter() - t0) * 1000 / len(images)


# --- experiment -------------------------------------------------------------------------------


def markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    head = (
        f"**{cfg['model']}**, layers {cfg['layers']}: head trained on {report['train_images']} COCO train images "
        f"({report['seen_classes']} classes; {report['classes'] - report['seen_classes']} held out), evaluated on "
        f"{report['val_images']} val images at {cfg['max_side']} px."
    )
    lines = [
        head,
        "",
        "| detector | mAP50-95 | mAP50 | mAP seen | mAP unseen | ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in report["results"].items():
        lines.append(
            f"| {name} | {r['all']['map']:.3f} | {r['all']['map50']:.3f} | {r['seen']['map']:.3f} "
            f"| {r['unseen']['map']:.3f} | {r.get('ms', float('nan')):.0f} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3-vl-4b", help="catalog name or local model folder (qwen_vl family)")
    ap.add_argument("--coco", default="datasets/coco", help="COCO root: annotations/ and train2017/, val2017/")
    ap.add_argument("--work", default="head_work", help="feature cache folder")
    ap.add_argument("--train-images", type=int, default=4000)
    ap.add_argument("--val-images", type=int, default=500)
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 20])
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=4, help="images per forward pass")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--negatives", type=int, default=16, help="absent classes added per training batch")
    ap.add_argument("--presence", action="store_true", help="also score head x the model's own Noul answer")
    ap.add_argument("--presence-top", type=int, default=10, help="classes asked per image for --presence")
    ap.add_argument("--reference", nargs="*", default=[], help="Ultralytics weights to score on the same images")
    ap.add_argument("--device", default=None)
    ap.add_argument("--dtype", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    import torch

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

    root, work = Path(args.coco), Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    fetch_annotations(root)
    train, names = load_split(root, "train2017", args.train_images, args.seed)
    val, _ = load_split(root, "val2017", args.val_images, args.seed)
    t0 = time.perf_counter()
    fetch_images(train, "train2017")
    fetch_images(val, "val2017")
    print(f"{len(train)} train and {len(val)} val images ready, {time.perf_counter() - t0:.0f}s", flush=True)
    seen = [c for c in range(len(names)) if c % 4 != 3]

    extractor = Extractor(path, device, dtype, args.layers, args.chunk, args.max_side)
    stores: dict[str, Store] = {}
    fit_boxes: dict[str, list[np.ndarray]] = {}
    val_images: list[PILImage.Image] = []
    extract_ms: dict[str, float] = {}
    try:
        queries = extractor.queries(names)
        for split, items in (("train", train), ("val", val)):
            store, fit_boxes[split], spent = Store(work / f"{split}.f16", extractor.dim), [], 0.0
            for start in range(0, len(items), 64):
                pairs = [fitted(it, args.max_side) for it in items[start : start + 64]]
                t1 = time.perf_counter()
                rows = extractor.patches([im for im, _ in pairs])
                spent += time.perf_counter() - t1
                for (image, b), r in zip(pairs, rows, strict=True):
                    store.append(r, (image.size[1] // STRIDE, image.size[0] // STRIDE))
                    fit_boxes[split].append(b)
                if split == "val":
                    val_images += [im for im, _ in pairs]
                print(f"{split}: {start + len(pairs)}/{len(items)} images, {spent:.0f}s in the model", flush=True)
            store.finish()
            stores[split] = store
            extract_ms[split] = spent * 1000 / len(items)

        head = train_head(
            stores["train"],
            fit_boxes["train"],
            [it.labels for it in train],
            queries,
            seen,
            device,
            args.epochs,
            args.batch,
            args.negatives,
            args.seed,
        )
        truth = [(b, it.labels) for b, it in zip(fit_boxes["val"], val, strict=True)]
        t0 = time.perf_counter()
        dets = [detect(head, stores["val"], i, queries, device) for i in range(len(val))]
        head_ms = (time.perf_counter() - t0) * 1000 / len(val)
        results: dict[str, dict[str, Any]] = {
            "head": {**scores_by_split(dets, truth, len(names), set(seen)), "ms": extract_ms["val"] + head_ms}
        }
        if args.presence:
            t0 = time.perf_counter()
            gated = []
            for image, (b, s, lab) in zip(val_images, dets, strict=True):
                best: dict[int, float] = {}
                for c, v in zip(lab.tolist(), s.tolist(), strict=True):
                    best[c] = max(best.get(c, 0.0), v)
                ask = sorted(best, key=lambda c: -best[c])[: args.presence_top]
                p = dict(zip(ask, extractor.presence(image, ask, names).tolist(), strict=True)) if ask else {}
                factor = np.array([p.get(c, 0.0) for c in lab.tolist()])
                keep = factor > 0
                gated.append((b[keep], (s * factor)[keep], lab[keep]))
            presence_ms = (time.perf_counter() - t0) * 1000 / len(val)
            results["head x presence"] = {
                **scores_by_split(gated, truth, len(names), set(seen)),
                "ms": extract_ms["val"] + head_ms + presence_ms,
            }
    finally:
        extractor.close()
    for weights in args.reference:
        ref, ms = reference(weights, val_images, device)
        results[weights] = {**scores_by_split(ref, truth, len(names), set(seen)), "ms": ms}

    report = {
        "results": results,
        "train_images": len(train),
        "val_images": len(val),
        "seen_classes": len(seen),
        "classes": len(names),
        "config": {k: v for k, v in vars(args).items()} | {"device": device, "dtype": dtype},
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else device,
    }
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
