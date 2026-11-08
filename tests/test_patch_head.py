"""benchmarks/patch_head.py (P2.1): mAP, targets and COCO loading offline; a tiny-model run end to end.

The end-to-end test uses a tiny random Qwen3-VL built from the real processor files (~12 MB,
downloaded once from the pinned catalog revision, or S1GHT_QWEN_PROCESSOR=<dir>) and a
synthetic COCO-format dataset, so nothing else is downloaded.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import patch_head as ph

from test_patch_signal import tiny_qwen  # noqa: F401 - pytest fixture

BOX = np.array([[10, 10, 50, 50]], np.float32)


def test_average_precision() -> None:
    truth = [(BOX, np.array([0]))]
    perfect = [(BOX, np.array([0.9]), np.array([0]))]
    assert ph.average_precision(perfect, truth, 0).tolist() == [1.0] * 10
    assert ph.average_precision(perfect, truth, 1) is None  # no ground truth: not counted
    # IoU 0.625 with the truth: a hit at thresholds 0.50-0.60 only
    shifted = [(np.array([[10, 10, 50, 74]], np.float32), np.array([0.9]), np.array([0]))]
    ap = ph.average_precision(shifted, truth, 0)
    assert ap is not None and ap[:3].tolist() == [1.0] * 3 and ap[3:].sum() == 0
    # a higher-scored false positive halves precision at full recall
    with_fp = [(np.array([[200, 200, 240, 240], [10, 10, 50, 50]], np.float32), np.array([0.9, 0.8]), np.array([0, 0]))]
    assert ph.average_precision(with_fp, truth, 0)[0] == pytest.approx(0.5)
    nothing = [(np.zeros((0, 4), np.float32), np.zeros(0), np.zeros(0, int))]
    assert ph.average_precision(nothing, truth, 0).tolist() == [0.0] * 10
    assert ph.coco_map(perfect, truth, [0, 1]) == {"map": 1.0, "map50": 1.0, "classes": 1}


def test_targets() -> None:
    # 4 x 6 patch grid (128 x 192 px); a big class-0 box and a small class-2 box inside it
    boxes = np.array([[0, 0, 96, 96], [36, 36, 52, 52]], np.float32)
    cls, box, mask = ph.targets(boxes, np.array([0, 2]), (4, 6), [0, 1, 2])
    grid = cls.reshape(3, 4, 6)
    assert grid[0, :3, :3].all() and grid[0].sum() == 9  # patch centers inside the big box
    assert grid[1].sum() == 0  # class 1 is absent
    assert grid[2].sum() == 1 and grid[2, 1, 1] == 1  # too small for any center: its center patch
    assert box.reshape(4, 6, 4)[1, 1].tolist() == [36, 36, 52, 52]  # smallest box wins
    assert box.reshape(4, 6, 4)[0, 0].tolist() == [0, 0, 96, 96]
    assert mask.sum() == 9


def coco_dataset(root: Path, n_train: int = 6, n_val: int = 4) -> Path:
    """Colored rectangles in COCO format; category ids are sparse to test the mapping."""
    cats = [
        {"id": 7, "name": "yellow box"},
        {"id": 1, "name": "red box"},
        {"id": 3, "name": "blue box"},
        {"id": 5, "name": "green box"},
    ]
    colors = {1: (220, 20, 20), 3: (20, 20, 220), 5: (20, 200, 20), 7: (230, 220, 20)}
    rng = np.random.default_rng(0)
    (root / "annotations").mkdir(parents=True)
    ann_id = 1
    for split, n in (("train2017", n_train), ("val2017", n_val)):
        (root / split).mkdir()
        images, anns = [], []
        for k in range(n):
            image_id = 100 * k + 1
            img = PILImage.new("RGB", (320, 256), "white")
            for cat in rng.choice([1, 3, 5, 7], size=2, replace=False):
                x, y = int(rng.integers(0, 200)), int(rng.integers(0, 150))
                img.paste(colors[int(cat)], (x, y, x + 96, y + 80))
                anns.append(
                    {"id": ann_id, "image_id": image_id, "category_id": int(cat), "bbox": [x, y, 96, 80], "iscrowd": 0}
                )
                ann_id += 1
            anns.append({"id": ann_id, "image_id": image_id, "category_id": 1, "bbox": [0, 0, 320, 256], "iscrowd": 1})
            ann_id += 1
            name = f"{image_id:012d}.jpg"
            img.save(root / split / name)
            images.append({"id": image_id, "file_name": name, "width": 320, "height": 256})
        data = {"images": images, "annotations": anns, "categories": cats}
        (root / "annotations" / f"instances_{split}.json").write_text(json.dumps(data))
    return root


def test_load_split(tmp_path: Path) -> None:
    root = coco_dataset(tmp_path)
    items, names = ph.load_split(root, "val2017", 3, 0)
    assert names == ["red box", "blue box", "green box", "yellow box"]  # sorted by category id
    assert [it.id for it in items] == [1, 101, 201]  # val: the first ids
    assert all(len(it.boxes) == 2 for it in items)  # crowd boxes dropped
    assert items[0].boxes[0, 2] - items[0].boxes[0, 0] == 96
    train, _ = ph.load_split(root, "train2017", 4, 0)
    assert len(train) == 4 and train == sorted(train, key=lambda it: it.id)


@pytest.mark.network
def test_end_to_end_tiny_model(tiny_qwen: str, tmp_path: Path) -> None:  # noqa: F811
    root = coco_dataset(tmp_path / "coco")
    out = tmp_path / "head.json"
    report = ph.main(
        [
            *("--model", tiny_qwen, "--coco", str(root), "--work", str(tmp_path / "work")),
            *("--train-images", "6", "--val-images", "4", "--layers", "1", "2", "--chunk", "2"),
            *("--epochs", "2", "--batch", "3", "--negatives", "2", "--presence", "--presence-top", "2"),
            *("--device", "cpu", "--dtype", "fp32", "--out", str(out)),
        ]
    )
    assert set(report["results"]) == {"head", "head x presence"}
    for r in report["results"].values():
        assert 0.0 <= r["all"]["map"] <= 1.0
        assert r["seen"]["classes"] + r["unseen"]["classes"] == r["all"]["classes"]
    assert report["seen_classes"] == 3 and report["classes"] == 4  # class index 3 is held out
    assert out.exists() and "| head |" in ph.markdown(report)


@pytest.mark.slow
def test_head_learns_planted_location(tmp_path: Path) -> None:
    """Training, decoding and mAP together: with location planted in the features, mAP is high.

    Each patch inside a box carries its class vector; boxes sit on the patch grid, so they are
    recoverable from the features. A pipeline bug (targets, box decoding, NMS, AP) would keep mAP
    low. Box regression needs a few hundred images not to overfit (64 gave val IoU 0.62 vs 0.92
    on train). About 30 s on a CPU.
    """
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    rng = np.random.default_rng(0)
    n_cls, dim, grid = 4, 32, (8, 10)
    vectors = rng.normal(size=(n_cls, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centers = ph.positions(grid)[:, 2:] * ph.STRIDE

    def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        boxes, labels = [], []
        for c in rng.choice(n_cls, size=2, replace=False):
            w, h = rng.integers(2, 5) * 32, rng.integers(2, 4) * 32
            x, y = rng.integers(0, 10 - w // 32 + 1) * 32, rng.integers(0, 8 - h // 32 + 1) * 32
            boxes.append([x, y, x + w, y + h])
            labels.append(c)
        boxes_a = np.array(boxes, np.float32)
        feats = rng.normal(scale=0.1, size=(grid[0] * grid[1], dim)).astype(np.float32)
        for b, c in zip(boxes_a, labels, strict=True):
            inside = (centers[:, 0] > b[0]) & (centers[:, 0] < b[2]) & (centers[:, 1] > b[1]) & (centers[:, 1] < b[3])
            feats[inside] += vectors[c]
        return feats, boxes_a, np.array(labels)

    def store(name: str, n: int) -> tuple[ph.Store, list, list]:
        s, boxes, labels = ph.Store(tmp_path / name, dim), [], []
        for _ in range(n):
            f, b, lab = sample()
            s.append(f, grid)
            boxes.append(b)
            labels.append(lab)
        s.finish()
        return s, boxes, labels

    train, tb, tl = store("train.f16", 256)
    val, vb, vl = store("val.f16", 16)
    head = ph.train_head(train, tb, tl, vectors, list(range(n_cls)), "cpu", epochs=10, batch=8, negatives=2, seed=0)
    dets = [ph.detect(head, val, i, vectors, "cpu") for i in range(16)]
    result = ph.coco_map(dets, list(zip(vb, vl, strict=True)), list(range(n_cls)))
    assert result["map50"] > 0.8, result
    assert result["map"] > 0.25, result
