"""P2.1-P2.3: does a small trained head turn a frozen decision model into a detector? (COCO mAP)

    python benchmarks/patch_head.py --device cuda --reference yolo26n.pt --out head.json
    python benchmarks/patch_head.py --train-sizes 2000 4000 8000 --queries text image
    python benchmarks/patch_head.py --queries image --heads s32 s16 s16q --train-sizes 4000 8000

P2.0 (`patch_signal.py`) showed that a frozen Qwen3-VL-4B knows *whether* an object is there
(presence AUC 0.99) but that location does not fall out of training-free readouts or linear
probes (patch AUC <= 0.81). P2.1 trained a small head on the frozen patch states: 0.071 mAP,
0.145 with the model's presence answer, 0.007 on held-out classes (YOLO26n: 0.436). P2.2 asks
why, in one run:

- **Query from the decision token (`--queries image`):** the query vector is the hidden state of
  the decision token of S1GHT's ordinary Noul prompt *with the image* ("Is there a {class} in
  `image`?"), the representation that gives presence AUC 0.99, instead of a text-only prompt's
  (`text`). One forward pass per (image, class) gives the patch states, the query vector and
  P(yes) together. Each prompt re-encodes the image here (no prefix cache), which the cost
  column shows.
- **Diagnostics:** `x GT presence` multiplies scores by the ground truth's class presence, the
  upper bound of a perfect presence answer, so classification and localization errors separate;
  class-agnostic mAP (all classes merged, class-agnostic NMS); and mAP for small, medium and
  large objects (COCO area ranges), to see what the 32 px patch stride costs.
- **Data scaling (`--train-sizes`):** heads trained on 2k, 4k, 8k ... images (nested random
  subsets of one sample), to see whether mAP grows with data or plateaus.

P2.2 found classification solved by the decision-token query (unseen classes 0.223 vs seen 0.213)
and the gap to YOLO in localization, worst for small objects (0.012 vs 0.168). P2.3 targets it:

- **Stride 16 (`--heads s16`):** the vision encoder's patch states before its 2x2 merge (16 px
  patches, captured at the merger's input, reordered from merge blocks to raster order) are added
  to the decision model's 32 px states, upsampled 2x; the head scores and boxes every 16 px cell.
- **IoU quality (`s16q`):** a branch predicts each box's IoU with its target; detections are
  ranked by class score x predicted IoU, so well-placed boxes come first (mAP at high IoU).
- **Features on the GPU (`--features-on`):** with enough GPU memory the cached features are
  loaded there once, so training does not reread tens of GB from disk every epoch.

The rest is P2.1:
- **Features:** the decision prompt comes after the image and attention is causal, so patch
  states do not depend on the question. States of `--layers` are concatenated and cached on
  disk (float16 memmap).
- **Head:** LayerNorm + linear, 2D position, two 3x3 convolutions on the cell grid (added in
  P2.3: box extents come from neighbouring cells, and attention alone overfit at stride 16), a
  2-layer transformer over the patches; class score
  = scaled cosine(patch, query) + bias (OWL-ViT style); a class-agnostic box per patch as
  distances to the four sides (FCOS style). Focal loss for scores, GIoU + L1 for boxes. A patch
  is positive for a class when its center lies in a box of that class; the patch holding a box's
  center is always positive (small objects).
- **Classes:** every 4th COCO class (20) is held out from training; mAP is reported for seen
  and unseen classes. With image queries, each training image is asked about its present seen
  classes plus `--negatives` absent seen ones.
- **Inference:** sigmoid scores, top-k per class, NMS, 100 detections per image. `x presence`
  multiplies each class's scores by the model's own Noul answer, asked for all classes.
- **mAP:** COCO style (IoU 0.50:0.95, 101-point interpolation, 100 detections, crowd boxes
  removed, COCO's ignore rules for area ranges), computed here so classes can be split.
  `--reference` runs Ultralytics models on the same pixels and scores them with the same code:
  they are trained on all of COCO train (118k images), so they are a yardstick, not a
  like-for-like comparison.

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
FINE = STRIDE // 2  # the vision encoder's patches, before the merge
IOUS = np.linspace(0.5, 0.95, 10)
AREAS = {"small": (0.0, 32.0**2), "medium": (32.0**2, 96.0**2), "large": (96.0**2, float("inf"))}


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

    Val takes the first `n` image ids; train a seeded random sample in random order, so its
    prefixes are nested random subsets. Category ids are mapped to 0..K-1 in sorted order
    (Ultralytics' COCO order).
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
    ids = ids[:n] if split.startswith("val") else random.Random(seed).sample(ids, min(n, len(ids)))
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
    """Patch states, query states (text-only or decision token) and presence from a `qwen_vl` model."""

    def __init__(
        self, path: str, device: str, dtype: str, layers: list[int], chunk: int, max_side: int, fine: bool = False
    ) -> None:
        from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily

        self.family = QwenVlFamily(flip="never", batch=chunk, max_side=max_side)
        self.family.load(path, device, dtype)
        self.model, self.processor = self.family.model, self.family.processor
        self.device, self.layers, self.chunk = device, layers, chunk
        self.image_token = self.model.config.image_token_id
        self.dim = int(self.model.config.text_config.hidden_size) * len(layers)
        self.fine_dim = int(self.model.config.vision_config.hidden_size) if fine else 0
        self._merger_input: Any = None
        if fine:  # the vision encoder's 16 px patch states enter the 2x2 merger

            def keep(_: Any, args: tuple) -> None:
                self._merger_input = args[0]

            self.model.model.visual.merger.register_forward_pre_hook(keep)

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

    def decide(
        self, image: PILImage.Image, classes: list[int], names: list[str]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
        """S1GHT's Noul prompt per class, with the image: patch states (P, dim) float16, the
        decision token's states (C, dim) float16, the model's answer P(yes) (C,), and with `fine`
        the vision encoder's 16 px patch states in raster order (4P, fine_dim) float16."""
        import torch

        state = {"image": "[image 1]"}
        texts = [
            self.text(letters.build(0, "q", state, Noul(f"Is there a {names[c]} in `image`?").to_wire(), 1, (0, 1)))
            for c in classes
        ]
        yes, no = self.family.letter_ids["A"], self.family.letter_ids["B"]
        patches, fine, states, presence = None, None, [], []
        for start in range(0, len(texts), self.chunk):
            batch = texts[start : start + self.chunk]
            inputs = self.processor(text=batch, images=[image] * len(batch), padding=True, return_tensors="pt")
            inputs = inputs.to(self.device)
            with torch.inference_mode():
                out = self.model(**inputs, output_hidden_states=True, logits_to_keep=1)
            hs = out.hidden_states
            if patches is None:  # causal: every row has the same patch states
                where = (inputs["input_ids"][0] == self.image_token).nonzero().flatten()
                expected = (image.size[1] // STRIDE) * (image.size[0] // STRIDE)
                if len(where) != expected:
                    raise RuntimeError(f"{len(where)} image tokens, expected {expected} for {image.size}")
                rows = torch.cat([hs[layer][0, where] for layer in self.layers], -1)
                patches = rows.to(torch.float16).cpu().numpy()
                if self.fine_dim:
                    grid = (image.size[1] // FINE, image.size[0] // FINE)
                    first = self._merger_input[: grid[0] * grid[1]]  # the batch holds copies of one image
                    fine = merge_to_raster(first.to(torch.float16).cpu().numpy(), grid)
            last = torch.cat([hs[layer][:, -1] for layer in self.layers], -1)
            states.append(last.to(torch.float16).cpu().numpy())
            logits = out.logits[:, -1].float()
            margin = torch.logsumexp(logits[:, yes], -1) - torch.logsumexp(logits[:, no], -1)
            presence.append(torch.sigmoid(margin).cpu().numpy())
        assert patches is not None
        return patches, np.concatenate(states), np.concatenate(presence), fine

    def close(self) -> None:
        self.family.close()


def merge_to_raster(rows: np.ndarray, grid: tuple[int, int]) -> np.ndarray:
    """Vision-encoder patch states from Qwen-VL's 2x2 merge-block order to raster order.

    The image processor emits patches as (rows/2, cols/2, 2, 2) so each merge block is contiguous.
    """
    gh, gw = grid
    return rows.reshape(gh // 2, gw // 2, 2, 2, -1).transpose(0, 2, 1, 3, 4).reshape(gh * gw, -1)


class Store:
    """Patch states of many images, appended to one float16 file, then read back as a memmap."""

    def __init__(self, path: Path, dim: int) -> None:
        self.path, self.dim = path, dim
        self.grids: list[tuple[int, int]] = []
        self.offsets = np.zeros(1, np.int64)
        self._sizes = [0]
        self._file: Any = open(path, "wb")  # noqa: SIM115 - closed in finish()
        self.data: Any = None
        self.tensor: Any = None  # the whole store on a device, after load()

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

    @property
    def nbytes(self) -> int:
        return int(self.offsets[-1]) * self.dim * 2

    def load(self, device: str) -> None:
        """Copy the store to `device` in chunks (no full host copy)."""
        import torch

        n = int(self.offsets[-1])
        self.tensor = torch.empty((n, self.dim), dtype=torch.float16, device=device)
        step = max(1, (1 << 28) // (self.dim * 2))  # 256 MB at a time
        for start in range(0, n, step):
            self.tensor[start : start + step] = torch.from_numpy(np.asarray(self.data[start : start + step]))

    def rows(self, i: int, device: str) -> Any:
        import torch

        if self.tensor is not None:
            return self.tensor[self.offsets[i] : self.offsets[i + 1]].float()
        return torch.from_numpy(self[i].astype(np.float32)).to(device)


# --- targets ----------------------------------------------------------------------------------


def targets(
    boxes: np.ndarray, labels: np.ndarray, grid: tuple[int, int], classes: list[int], stride: int = STRIDE
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per patch: class labels (C, P) for `classes`, a box target (P, 4) and a box mask (P,).

    A patch is positive for a class when its center lies in a box of that class; the patch that
    holds a box's center is always positive. Its box target is the smallest such box.
    """
    gh, gw = grid
    cy, cx = np.meshgrid((np.arange(gh) + 0.5) * stride, (np.arange(gw) + 0.5) * stride, indexing="ij")
    cx, cy = cx.reshape(-1), cy.reshape(-1)
    col = {c: k for k, c in enumerate(classes)}
    cls = np.zeros((len(classes), gh * gw), np.float32)
    box = np.zeros((gh * gw, 4), np.float32)
    best = np.full(gh * gw, np.inf)
    for b, label in sorted(zip(boxes, labels, strict=True), key=lambda t: -(t[0][2] - t[0][0]) * (t[0][3] - t[0][1])):
        inside = (cx >= b[0]) & (cx <= b[2]) & (cy >= b[1]) & (cy <= b[3])
        center = min(int((b[1] + b[3]) / 2 // stride), gh - 1) * gw + min(int((b[0] + b[2]) / 2 // stride), gw - 1)
        inside[center] = True
        area = (b[2] - b[0]) * (b[3] - b[1])
        take = inside & (area <= best)  # boxes come largest first: smaller ones overwrite
        box[take], best[take] = b, area
        if label in col:
            cls[col[label], inside] = 1.0
    return cls, box, np.isfinite(best)


# --- head -------------------------------------------------------------------------------------


def build_head(dim: int, width: int = 256, depth: int = 2, fine_dim: int = 0, quality: bool = False) -> Any:
    import torch
    from torch import nn

    class Head(nn.Module):
        """Patch states + queries -> per-cell class logits (B, C, P), boxes (B, P, 4) in pixels and,
        with `quality`, IoU logits (B, P). With `fine_dim` the cells are the 16 px patches: the 32 px
        states are upsampled onto them and the vision encoder's states added."""

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
            self.fine = nn.Sequential(nn.LayerNorm(fine_dim), nn.Linear(fine_dim, width)) if fine_dim else None
            self.iou = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1)) if quality else None
            self.stride = FINE if fine_dim else STRIDE
            # 3x3 convolutions on the cell grid: box extents come from neighbouring cells, and a
            # local, translation-invariant layer learns that from far less data than attention
            self.local = nn.Sequential(
                nn.Conv2d(width, width, 3, padding=1), nn.GELU(), nn.Conv2d(width, width, 3, padding=1)
            )

        def grid_conv(self, h: Any, cells: Any) -> Any:
            index, rows, cols = cells  # index: each cell's place on a rows x cols canvas (padding: the extra slot)
            b, _, w = h.shape
            canvas = h.new_zeros(b, rows * cols + 1, w)
            canvas.scatter_(1, index[..., None].expand(-1, -1, w), h)
            y = self.local(canvas[:, :-1].reshape(b, rows, cols, w).permute(0, 3, 1, 2))
            y = torch.cat([y.permute(0, 2, 3, 1).reshape(b, rows * cols, w), canvas[:, -1:]], 1)
            return h + torch.gather(y, 1, index[..., None].expand(-1, -1, w))

        def forward(
            self, x: Any, pos: Any, valid: Any, q: Any, fine: Any = None, cells: Any = None
        ) -> tuple[Any, Any, Any]:
            h = self.inp(x)
            if self.fine is not None:
                xf, pos, valid, up, cells = fine  # cells become the 16 px patches
                h = torch.gather(h, 1, up[..., None].expand(-1, -1, h.shape[-1])) + self.fine(xf)
            h = h + self.pos(pos)
            if cells is not None:
                h = self.grid_conv(h * valid[..., None], cells)
            h = self.norm(self.encoder(h, src_key_padding_mask=~valid))
            k = nn.functional.normalize(self.key(h), dim=-1)
            e = nn.functional.normalize(self.query(q), dim=-1)
            pattern = "bpd,cd->bcp" if e.dim() == 2 else "bpd,bcd->bcp"  # shared or per-image queries
            logits = torch.einsum(pattern, k, e) * self.scale.exp() + self.bias
            ltrb = nn.functional.softplus(self.box(h)) * self.stride
            centers = pos[..., 2:] * self.stride  # (x, y) of cell centers in pixels
            boxes = torch.cat([centers - ltrb[..., :2], centers + ltrb[..., 2:]], dim=-1)
            iou = self.iou(h).squeeze(-1) if self.iou is not None else None
            return logits, boxes, iou

    return Head()


def positions(grid: tuple[int, int]) -> np.ndarray:
    """Per patch: (y, x) normalized to [0, 1] and (x, y) centers in patch units, (P, 4)."""
    gh, gw = grid
    r, c = np.meshgrid(np.arange(gh), np.arange(gw), indexing="ij")
    r, c = r.reshape(-1), c.reshape(-1)
    return np.stack([(r + 0.5) / gh, (c + 0.5) / gw, c + 0.5, r + 0.5], 1).astype(np.float32)


def batch_tensors(
    store: Store, idx: list[int], device: str, fine: Store | None = None
) -> tuple[Any, Any, Any, Any, Any]:
    """Padded (x, pos, valid, cells) for the 32 px states and, with `fine`, (x, pos, valid, up, cells)
    for the 16 px states, where `up` maps each 16 px cell to its 32 px patch and `cells` places the
    cells on a rows x cols canvas for the head's convolutions."""
    import torch

    def canvas(st: Store, n: int) -> tuple[Any, int, int]:
        rows, cols = max(st.grids[i][0] for i in idx), max(st.grids[i][1] for i in idx)
        index = torch.full((len(idx), n), rows * cols, dtype=torch.long)  # padding: the extra slot
        for b, i in enumerate(idx):
            gh, gw = st.grids[i]
            r, c = np.divmod(np.arange(gh * gw), gw)
            index[b, : gh * gw] = torch.from_numpy(r * cols + c)
        return index.to(device), rows, cols

    def pad(st: Store) -> tuple[Any, Any, Any]:
        pmax = max(st.offsets[i + 1] - st.offsets[i] for i in idx)
        x = torch.zeros(len(idx), pmax, st.dim, device=device)
        pos = torch.zeros(len(idx), pmax, 4, device=device)
        valid = torch.zeros(len(idx), pmax, dtype=torch.bool, device=device)
        for b, i in enumerate(idx):
            rows = st.rows(i, device)
            x[b, : len(rows)] = rows
            pos[b, : len(rows)] = torch.from_numpy(positions(st.grids[i])).to(device)
            valid[b, : len(rows)] = True
        return x, pos, valid

    x, pos, valid = pad(store)
    if fine is None:
        return x, pos, valid, canvas(store, x.shape[1]), None
    xf, posf, validf = pad(fine)
    up = torch.zeros(len(idx), xf.shape[1], dtype=torch.long, device=device)
    for b, i in enumerate(idx):
        (fh, fw), (_, gw) = fine.grids[i], store.grids[i]
        r, c = np.divmod(np.arange(fh * fw), fw)
        up[b, : fh * fw] = torch.from_numpy((r // 2) * gw + c // 2).to(device)
    return x, pos, valid, None, (xf, posf, validf, up, canvas(fine, xf.shape[1]))


def pair_iou(a: Any, b: Any) -> Any:
    """IoU of matching rows of two (N, 4) xyxy tensors."""
    import torch

    lt, rb = torch.maximum(a[:, :2], b[:, :2]), torch.minimum(a[:, 2:], b[:, 2:])
    inter = (rb - lt).clamp(min=0).prod(-1)
    area = lambda x: (x[:, 2] - x[:, 0]).clamp(min=0) * (x[:, 3] - x[:, 1]).clamp(min=0)
    return inter / (area(a) + area(b) - inter + 1e-9)


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
    image_queries: list[tuple[list[int], np.ndarray]] | None = None,
    fine: Store | None = None,
    quality: bool = False,
) -> Any:
    """Train a head on the first `len(boxes)` images of `store`.

    Text queries (`image_queries` None): each batch is scored against its present seen classes
    plus `negatives` absent ones, with the shared `queries`. Image queries: each image against
    the classes it was asked about, with its own decision-token states. With `fine` the head
    works on 16 px cells; with `quality` it also learns each box's IoU.
    """
    import torch
    from torchvision.ops import generalized_box_iou_loss, sigmoid_focal_loss

    torch.manual_seed(seed)
    rng = random.Random(seed)
    head = build_head(store.dim, fine_dim=fine.dim if fine else 0, quality=quality).to(device)
    cells, stride = (fine, FINE) if fine is not None else (store, STRIDE)
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
            x, pos, valid, cells_t, fine_t = batch_tensors(store, idx, device, fine)
            n_cells = fine_t[0].shape[1] if fine_t is not None else x.shape[1]
            if image_queries is None:
                present = sorted({int(c) for i in idx for c in labels[i] if c in seen_set})
                absent = [c for c in seen if c not in present]
                shared = present + rng.sample(absent, min(negatives, len(absent)))
                lists = [shared] * len(idx)
                q = q_all[shared]
            else:
                lists = [image_queries[i][0] for i in idx]
                q = torch.zeros(len(idx), max(map(len, lists)), store.dim)
                for b, i in enumerate(idx):
                    q[b, : len(lists[b])] = torch.from_numpy(image_queries[i][1].astype(np.float32))
                q = q.to(device)
            cmax = max(map(len, lists))
            cmask = torch.zeros(len(idx), cmax, dtype=torch.bool, device=device)
            cls_t = torch.zeros(len(idx), cmax, n_cells, device=device)
            box_t = torch.zeros(len(idx), n_cells, 4, device=device)
            box_m = torch.zeros(len(idx), n_cells, dtype=torch.bool, device=device)
            for b, i in enumerate(idx):
                c, bx, m = targets(boxes[i], labels[i], cells.grids[i], lists[b], stride)
                n = c.shape[1]
                cmask[b, : len(lists[b])] = True
                cls_t[b, : len(lists[b]), :n] = torch.from_numpy(c)
                box_t[b, :n], box_m[b, :n] = torch.from_numpy(bx), torch.from_numpy(m)
            logits, pred, iou = head(x, pos, valid, q, fine_t, cells_t)
            if fine_t is not None:
                valid = fine_t[2]
            mask = valid[:, None, :] & cmask[:, :, None]
            n_pos = max(1.0, float(cls_t.sum()))
            loss_cls = sigmoid_focal_loss(logits[mask], cls_t[mask], reduction="sum") / n_pos
            loss_box = torch.zeros((), device=device)
            if box_m.any():
                p, t = pred[box_m], box_t[box_m]
                loss_box = generalized_box_iou_loss(p, t, reduction="mean") + (p - t).abs().mean() / 64
                if iou is not None:
                    target_iou = pair_iou(p.detach(), t)
                    loss_box = loss_box + torch.nn.functional.binary_cross_entropy_with_logits(iou[box_m], target_iou)
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
    head: Any,
    store: Store,
    i: int,
    queries: np.ndarray,
    device: str,
    top: int = 50,
    iou: float = 0.6,
    fine: Store | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(boxes, scores, labels) for one image over all classes, at most 100, before presence.

    With an IoU branch, a detection's score is its class score times its predicted IoU."""
    import torch
    from torchvision.ops import batched_nms

    x, pos, valid, cells_t, fine_t = batch_tensors(store, [i], device, fine)
    with torch.inference_mode():
        logits, boxes, quality = head(x, pos, valid, torch.from_numpy(queries).to(device), fine_t, cells_t)
    scores = logits[0].sigmoid()  # (C, P)
    if quality is not None:
        scores = scores * quality[0].sigmoid()[None]
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


def _area(b: np.ndarray) -> np.ndarray:
    return (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])


def average_precision(
    dets: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    truth: list[tuple[np.ndarray, np.ndarray]],
    cls: int,
    area: tuple[float, float] | None = None,
) -> np.ndarray | None:
    """COCO AP of one class at each IoU in `IOUS` (101-point); None when it has no ground truth.

    With `area` (min, max), COCO's rules: ground truth outside the range is ignored (a detection
    matched to it counts for nothing), and so are unmatched detections outside the range.
    """
    gts, ignore = [], []
    for boxes, labels in truth:
        g = boxes[labels == cls]
        ign = np.zeros(len(g), bool) if area is None else ~((_area(g) >= area[0]) & (_area(g) < area[1]))
        order = np.argsort(ign, kind="stable")  # matchable ground truth first
        gts.append(g[order])
        ignore.append(ign[order])
    n_true = int(sum((~ign).sum() for ign in ignore))
    if n_true == 0:
        return None
    records = []  # (score, image, box)
    for img, (boxes, scores, labels) in enumerate(dets):
        for b, sc in zip(boxes[labels == cls], scores[labels == cls], strict=True):
            records.append((float(sc), img, b))
    records.sort(key=lambda r: -r[0])
    out = np.zeros(len(IOUS))
    for t, thr in enumerate(IOUS):
        used = [np.zeros(len(g), bool) for g in gts]
        tp, keep = np.zeros(len(records)), np.ones(len(records), bool)
        for k, (_, img, b) in enumerate(records):
            g, ign = gts[img], ignore[img]
            match = -1
            if len(g):
                ious = box_iou(b[None], g)[0]
                ious[used[img]] = -1
                for subset in (~ign, ign):  # a matchable box first, an ignored one only if none fits
                    cand = np.where(subset & (ious >= thr))[0]
                    if len(cand):
                        match = int(cand[ious[cand].argmax()])
                        break
            if match >= 0:
                used[img][match] = True
                tp[k], keep[k] = (0, False) if ign[match] else (1, True)
            elif area is not None and not (area[0] <= _area(b[None])[0] < area[1]):
                keep[k] = False
        tp = tp[keep]
        ctp = np.cumsum(tp)
        recall = ctp / n_true
        precision = ctp / np.arange(1, len(tp) + 1) if len(tp) else np.zeros(0)
        precision = np.maximum.accumulate(precision[::-1])[::-1] if len(precision) else precision
        where = np.searchsorted(recall, np.linspace(0, 1, 101), side="left")
        out[t] = np.mean([precision[w] if w < len(precision) else 0.0 for w in where])
    return out


def coco_map(dets: list, truth: list, classes: list[int], area: tuple[float, float] | None = None) -> dict[str, float]:
    aps = [ap for c in classes if (ap := average_precision(dets, truth, c, area)) is not None]
    if not aps:
        return {"map": float("nan"), "map50": float("nan"), "classes": 0}
    a = np.stack(aps)
    return {"map": float(a.mean()), "map50": float(a[:, 0].mean()), "classes": len(aps)}


def agnostic(dets: list, iou: float = 0.6) -> list:
    """Detections with every class merged into one, after class-agnostic NMS."""
    import torch
    from torchvision.ops import nms

    out = []
    for boxes, scores, _ in dets:
        keep = nms(torch.from_numpy(boxes).float(), torch.from_numpy(scores).float(), iou).numpy()
        out.append((boxes[keep], scores[keep], np.zeros(len(keep), int)))
    return out


def evaluate(dets: list, truth: list, n_classes: int, seen: set[int]) -> dict[str, Any]:
    """mAP for all, seen and unseen classes, by object size, and class-agnostic."""
    every = list(range(n_classes))
    out: dict[str, Any] = {
        "all": coco_map(dets, truth, every),
        "seen": coco_map(dets, truth, [c for c in every if c in seen]),
        "unseen": coco_map(dets, truth, [c for c in every if c not in seen]),
    }
    for name, rng in AREAS.items():
        out[name] = coco_map(dets, truth, every, rng)
    out["agnostic"] = coco_map(agnostic(dets), [(b, np.zeros(len(b), int)) for b, _ in truth], [0])
    return out


def rescore(dets: list, factors: list[np.ndarray]) -> list:
    """Scores times a per-image, per-class factor (e.g. presence); zero-factor detections dropped."""
    out = []
    for (b, s, lab), f in zip(dets, factors, strict=True):
        w = f[lab] if len(lab) else np.zeros(0)
        keep = w > 0
        out.append((b[keep], (s * w)[keep], lab[keep]))
    return out


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


HEADS = {"s32": "", "s16": ", stride 16", "s16q": ", stride 16 + IoU"}  # --heads: row label suffix


def markdown(report: dict[str, Any]) -> str:
    cfg = report["config"]
    head = (
        f"**{cfg['model']}**, layers {cfg['layers']}: heads trained on {cfg['train_sizes']} COCO train images "
        f"({report['seen_classes']} classes; {report['classes'] - report['seen_classes']} held out), evaluated on "
        f"{report['val_images']} val images at {cfg['max_side']} px. Cost: ms per image for all "
        f"{report['classes']} classes; `GT presence` is an upper bound, not a detector."
    )
    lines = [
        head,
        "",
        "| detector | mAP50-95 | mAP50 | seen | unseen | small | medium | large | agnostic mAP50 | ms/image |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for name, r in report["results"].items():
        ms = r.get("ms")
        lines.append(
            f"| {name} | {r['all']['map']:.3f} | {r['all']['map50']:.3f} | {r['seen']['map']:.3f} "
            f"| {r['unseen']['map']:.3f} | {r['small']['map']:.3f} | {r['medium']['map']:.3f} "
            f"| {r['large']['map']:.3f} | {r['agnostic']['map50']:.3f} | {'-' if ms is None else f'{ms:.0f}'} |"
        )
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="qwen3-vl-4b", help="catalog name or local model folder (qwen_vl family)")
    ap.add_argument("--coco", default="datasets/coco", help="COCO root: annotations/ and train2017/, val2017/")
    ap.add_argument("--work", default="head_work", help="feature cache folder")
    ap.add_argument("--train-sizes", type=int, nargs="+", default=[4000], help="a head is trained per size")
    ap.add_argument("--val-images", type=int, default=500)
    ap.add_argument("--queries", nargs="+", choices=["text", "image"], default=["text", "image"])
    ap.add_argument("--layers", type=int, nargs="+", default=[4, 20])
    ap.add_argument("--max-side", type=int, default=640)
    ap.add_argument("--chunk", type=int, default=8, help="prompts (or images) per forward pass")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--negatives", type=int, default=8, help="absent classes per image (image) or batch (text)")
    ap.add_argument("--heads", nargs="+", choices=list(HEADS), default=["s32"], help="s16/s16q need the decision pass")
    ap.add_argument(
        "--features-on", choices=["auto", "gpu", "disk"], default="auto", help="where training reads features"
    )
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
    sizes = sorted(set(args.train_sizes))
    train, names = load_split(root, "train2017", sizes[-1], args.seed)
    sizes = [n for n in sizes if n <= len(train)] or [len(train)]
    val, _ = load_split(root, "val2017", args.val_images, args.seed)
    t0 = time.perf_counter()
    fetch_images(train, "train2017")
    fetch_images(val, "val2017")
    print(f"{len(train)} train and {len(val)} val images ready, {time.perf_counter() - t0:.0f}s", flush=True)
    seen = [c for c in range(len(names)) if c % 4 != 3]
    seen_set = set(seen)
    everything = list(range(len(names)))
    rng = random.Random(args.seed)

    fine_heads = any(h != "s32" for h in args.heads)
    if fine_heads and "image" not in args.queries:
        raise SystemExit("--heads s16/s16q read the vision encoder during the decision pass: add --queries image")
    extractor = Extractor(path, device, dtype, args.layers, args.chunk, args.max_side, fine=fine_heads)
    train_store, val_store = Store(work / "train.f16", extractor.dim), Store(work / "val.f16", extractor.dim)
    fine_dim = extractor.fine_dim
    train_fine = Store(work / "train_fine.f16", fine_dim) if fine_heads else None
    val_fine = Store(work / "val_fine.f16", fine_dim) if fine_heads else None
    train_boxes: list[np.ndarray] = []
    train_q: list[tuple[list[int], np.ndarray]] = []
    val_boxes: list[np.ndarray] = []
    val_q: list[np.ndarray] = []
    val_presence: list[np.ndarray] = []
    val_images: list[PILImage.Image] = []
    try:
        text_q = extractor.queries(names)
        t0 = time.perf_counter()
        for start in range(0, len(train), 64):
            pairs = [fitted(it, args.max_side) for it in train[start : start + 64]]
            if "image" in args.queries:  # the decision pass gives patches and image queries together
                for (image, b), it in zip(pairs, train[start : start + 64], strict=True):
                    present = sorted({int(c) for c in it.labels if c in seen_set})
                    absent = [c for c in seen if c not in present]
                    asked = present + rng.sample(absent, min(args.negatives, len(absent)))
                    rows, states, _, fine = extractor.decide(image, asked, names)
                    train_store.append(rows, (image.size[1] // STRIDE, image.size[0] // STRIDE))
                    if train_fine is not None:
                        train_fine.append(fine, (image.size[1] // FINE, image.size[0] // FINE))
                    train_q.append((asked, states))
                    train_boxes.append(b)
            else:
                for (image, b), rows in zip(pairs, extractor.patches([im for im, _ in pairs]), strict=True):
                    train_store.append(rows, (image.size[1] // STRIDE, image.size[0] // STRIDE))
                    train_boxes.append(b)
            print(f"train: {start + len(pairs)}/{len(train)} images, {time.perf_counter() - t0:.0f}s", flush=True)
        train_store.finish()
        if train_fine is not None:
            train_fine.finish()
        # val: every class asked with the image (presence and image queries for all classes)
        t0 = time.perf_counter()
        for k, it in enumerate(val):
            image, b = fitted(it, args.max_side)
            rows, states, presence, fine = extractor.decide(image, everything, names)
            val_store.append(rows, (image.size[1] // STRIDE, image.size[0] // STRIDE))
            if val_fine is not None:
                val_fine.append(fine, (image.size[1] // FINE, image.size[0] // FINE))
            val_boxes.append(b)
            val_q.append(states.astype(np.float32))
            val_presence.append(presence)
            val_images.append(image)
            if k % 50 == 0:
                print(f"val: {k + 1}/{len(val)} images, {time.perf_counter() - t0:.0f}s", flush=True)
        decide_ms = (time.perf_counter() - t0) * 1000 / len(val)
        val_store.finish()
        if val_fine is not None:
            val_fine.finish()
        probe = val_images[: min(32, len(val_images))]
        t0 = time.perf_counter()
        extractor.patches(probe)
        patch_ms = (time.perf_counter() - t0) * 1000 / len(probe)
    finally:
        extractor.close()

    stores = [st for st in (train_store, val_store, train_fine, val_fine) if st is not None]
    total = sum(st.nbytes for st in stores)
    on_gpu = args.features_on == "gpu"
    if args.features_on == "auto" and device.startswith("cuda"):
        on_gpu = torch.cuda.mem_get_info()[0] > 1.3 * total
    print(f"features: {total / 2**30:.1f} GB, read from {'the GPU' if on_gpu else 'disk'} during training", flush=True)
    if on_gpu:
        for st in stores:
            st.load(device)

    truth = [(b, it.labels) for b, it in zip(val_boxes, val, strict=True)]
    gt_presence = [np.isin(everything, it.labels).astype(np.float32) for it in val]
    labels = [it.labels for it in train]
    results: dict[str, dict[str, Any]] = {}
    for mode in args.queries:
        for kind in args.heads:
            if kind != "s32" and mode != "image":
                continue  # stride-16 heads are measured with the decision-token query
            fine_train, fine_val = (train_fine, val_fine) if kind != "s32" else (None, None)
            for n in sizes:
                print(f"training: {mode} queries, {kind}, {n} images", flush=True)
                head = train_head(
                    train_store,
                    train_boxes[:n],
                    labels[:n],
                    text_q,
                    seen,
                    device,
                    args.epochs,
                    args.batch,
                    args.negatives,
                    args.seed,
                    image_queries=train_q[:n] if mode == "image" else None,
                    fine=fine_train,
                    quality=kind == "s16q",
                )
                t0 = time.perf_counter()
                dets = [
                    detect(head, val_store, i, text_q if mode == "text" else val_q[i], device, fine=fine_val)
                    for i in range(len(val))
                ]
                head_ms = (time.perf_counter() - t0) * 1000 / len(val)
                base = (patch_ms if mode == "text" else decide_ms) + head_ms
                name = f"{mode} query, {n} images{HEADS[kind]}"
                results[name] = {**evaluate(dets, truth, len(names), seen_set), "ms": base}
                results[f"{name} x presence"] = {
                    **evaluate(rescore(dets, val_presence), truth, len(names), seen_set),
                    "ms": base + (decide_ms if mode == "text" else 0.0),
                }
                gt = evaluate(rescore(dets, gt_presence), truth, len(names), seen_set)
                results[f"{name} x GT presence"] = gt
                del head
    for weights in args.reference:
        ref, ms = reference(weights, val_images, device)
        results[weights] = {**evaluate(ref, truth, len(names), seen_set), "ms": ms}

    report = {
        "results": results,
        "train_images": len(train),
        "val_images": len(val),
        "seen_classes": len(seen),
        "classes": len(names),
        "cost_ms": {"patch_pass": patch_ms, "decision_pass_all_classes": decide_ms},
        "features_gb": total / 2**30,
        "config": {k: v for k, v in vars(args).items()} | {"device": device, "dtype": dtype, "train_sizes": sizes},
        "gpu": torch.cuda.get_device_name() if device.startswith("cuda") else device,
    }
    print(markdown(report))
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, default=float))
    return report


if __name__ == "__main__":
    main()
