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
    again, _ = ph.load_split(root, "train2017", 4, 0)
    assert len(train) == 4 and [it.id for it in train] == [it.id for it in again]  # seeded, random order


def test_average_precision_by_area() -> None:
    """COCO area rules: out-of-range ground truth and unmatched out-of-range detections are ignored."""
    small, large = np.array([[0, 0, 20, 20]], np.float32), np.array([[100, 100, 300, 300]], np.float32)
    truth = [(np.concatenate([small, large]), np.array([0, 0]))]
    found_large = [(large, np.array([0.9]), np.array([0]))]
    assert ph.average_precision(found_large, truth, 0, ph.AREAS["large"]).tolist() == [1.0] * 10
    assert ph.average_precision(found_large, truth, 0, ph.AREAS["small"]).tolist() == [0.0] * 10
    assert ph.average_precision(found_large, truth, 0, ph.AREAS["medium"]) is None  # nothing medium
    assert ph.average_precision(found_large, truth, 0)[0] == pytest.approx(51 / 101)  # recall 0.5: 51 of 101 points
    # a stray large false positive does not hurt the small range
    both = [
        (np.concatenate([np.array([[400, 0, 600, 200]], np.float32), small]), np.array([0.9, 0.8]), np.array([0, 0]))
    ]
    assert ph.average_precision(both, truth, 0, ph.AREAS["small"]).tolist() == [1.0] * 10


def test_agnostic_and_rescore() -> None:
    # the same object found as class 0 and class 1: one box after class-agnostic NMS
    dets = [(np.array([[10, 10, 50, 50], [11, 10, 50, 50]], np.float32), np.array([0.9, 0.7]), np.array([0, 1]))]
    merged = ph.agnostic(dets)
    assert len(merged[0][0]) == 1 and merged[0][2].tolist() == [0]
    truth = [(BOX, np.array([1]))]
    assert ph.evaluate(dets, truth, 2, {0})["agnostic"]["map"] == 1.0
    gated = ph.rescore(dets, [np.array([0.0, 0.5])])  # class 0 judged absent
    assert gated[0][2].tolist() == [1] and gated[0][1].tolist() == pytest.approx([0.35])


@pytest.mark.network
def test_end_to_end_tiny_model(tiny_qwen: str, tmp_path: Path) -> None:  # noqa: F811
    root = coco_dataset(tmp_path / "coco")
    out = tmp_path / "head.json"
    report = ph.main(
        [
            *("--model", tiny_qwen, "--coco", str(root), "--work", str(tmp_path / "work")),
            *("--train-sizes", "3", "6", "--val-images", "4", "--layers", "1", "2", "--chunk", "3"),
            *("--epochs", "2", "--batch", "3", "--negatives", "2", "--reference"),
            *("--device", "cpu", "--dtype", "fp32", "--out", str(out)),
        ]
    )
    expected = {
        f"{mode} query, {n} images{suffix}"
        for mode in ("text", "image")
        for n in (3, 6)
        for suffix in ("", " x presence", " x GT presence")
    }
    assert set(report["results"]) == expected
    for r in report["results"].values():
        for key in ("all", "small", "medium", "large", "agnostic"):
            assert np.isnan(r[key]["map"]) or 0.0 <= r[key]["map"] <= 1.0
        assert r["seen"]["classes"] + r["unseen"]["classes"] == r["all"]["classes"]
    assert "ms" not in report["results"]["image query, 6 images x GT presence"]  # an upper bound, not a detector
    assert report["seen_classes"] == 3 and report["classes"] == 4  # class index 3 is held out
    assert set(report["cost_ms"]) == {"patch_pass", "decision_pass_all_classes"}
    assert out.exists() and "| image query, 6 images |" in ph.markdown(report)


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


def test_merge_to_raster() -> None:
    gh, gw = 4, 6
    raster = np.arange(gh * gw)
    merge_order = raster.reshape(gh // 2, 2, gw // 2, 2).transpose(0, 2, 1, 3).reshape(-1)  # as the processor emits
    assert ph.merge_to_raster(merge_order[:, None], (gh, gw))[:, 0].tolist() == raster.tolist()


@pytest.mark.network
def test_processor_patch_order(tiny_qwen: str) -> None:  # noqa: F811
    """Qwen-VL's image processor emits 16 px patches in 2x2 merge blocks; merge_to_raster undoes it."""
    from transformers import AutoProcessor

    processor = AutoProcessor.from_pretrained(tiny_qwen)
    gh, gw = 16, 24  # 256 x 384 px: above the processor's minimum area, so it is not resized
    pixels = np.zeros((gh * 16, gw * 16, 3), np.uint8)
    for r in range(gh):
        for c in range(gw):
            pixels[r * 16 : (r + 1) * 16, c * 16 : (c + 1) * 16] = (10 * r + 5, 10 * c + 5, 0)
    out = processor.image_processor(images=[PILImage.fromarray(pixels)], return_tensors="pt")
    assert out["image_grid_thw"].tolist() == [[1, gh, gw]]
    pv = out["pixel_values"].float().numpy()
    third = pv.shape[1] // 3  # channel-major rows: R, G, B
    ids = []
    for channel, n in ((0, gh), (1, gw)):
        m = pv[:, channel * third : (channel + 1) * third].mean(1)
        ids.append(np.round((m - m.min()) / (m.max() - m.min()) * (n - 1)).astype(int))
    patch_ids = ids[0] * gw + ids[1]
    assert ph.merge_to_raster(patch_ids[:, None], (gh, gw))[:, 0].tolist() == list(range(gh * gw))


@pytest.mark.slow
def test_fine_head_learns_small_objects(tmp_path: Path) -> None:
    """The stride-16 head with an IoU branch: location planted only in the 16 px features, boxes of
    16-64 px on that grid, coarse 32 px features pure noise. Without the head's 3x3 convolutions
    this stayed at mAP50 0.14 (boxes overfit: IoU 0.64 train, 0.32 val). About 2 min on a CPU."""
    torch = pytest.importorskip("torch")
    torch.set_num_threads(4)
    rng = np.random.default_rng(1)
    n_cls, dim, grid, fine_grid = 4, 32, (8, 10), (16, 20)
    vectors = rng.normal(size=(n_cls, dim)).astype(np.float32)
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    centers = ph.positions(fine_grid)[:, 2:] * ph.FINE

    def sample() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        boxes, labels = [], []
        for c in rng.choice(n_cls, size=2, replace=False):
            w, h = rng.integers(1, 5) * 16, rng.integers(1, 5) * 16
            x, y = rng.integers(0, 20 - w // 16 + 1) * 16, rng.integers(0, 16 - h // 16 + 1) * 16
            boxes.append([x, y, x + w, y + h])
            labels.append(c)
        b = np.array(boxes, np.float32)
        fine = rng.normal(scale=0.1, size=(fine_grid[0] * fine_grid[1], dim)).astype(np.float32)
        for bb, c in zip(b, labels, strict=True):
            inside = (
                (centers[:, 0] > bb[0]) & (centers[:, 0] < bb[2]) & (centers[:, 1] > bb[1]) & (centers[:, 1] < bb[3])
            )
            fine[inside] += vectors[c]
        coarse = rng.normal(scale=0.1, size=(grid[0] * grid[1], dim)).astype(np.float32)  # queries share its width
        return coarse, fine, b, np.array(labels)

    def stores(name: str, n: int) -> tuple[ph.Store, ph.Store, list, list]:
        coarse_s, fine_s = ph.Store(tmp_path / f"{name}.f16", dim), ph.Store(tmp_path / f"{name}_fine.f16", dim)
        boxes, labels = [], []
        for _ in range(n):
            c, f, b, lab = sample()
            coarse_s.append(c, grid)
            fine_s.append(f, fine_grid)
            boxes.append(b)
            labels.append(lab)
        coarse_s.finish()
        fine_s.finish()
        return coarse_s, fine_s, boxes, labels

    train, train_f, tb, tl = stores("train", 128)
    val, val_f, vb, vl = stores("val", 16)
    head = ph.train_head(
        train,
        tb,
        tl,
        vectors,
        list(range(n_cls)),
        "cpu",
        epochs=12,
        batch=8,
        negatives=2,
        seed=0,
        fine=train_f,
        quality=True,
    )
    dets = [ph.detect(head, val, i, vectors, "cpu", fine=val_f) for i in range(16)]
    result = ph.coco_map(dets, list(zip(vb, vl, strict=True)), list(range(n_cls)))
    assert result["map50"] > 0.6, result  # broken (no convolutions): 0.14
    assert result["map"] > 0.2, result


@pytest.mark.network
def test_end_to_end_fine_heads(tiny_qwen: str, tmp_path: Path) -> None:  # noqa: F811
    root = coco_dataset(tmp_path / "coco")
    report = ph.main(
        [
            *("--model", tiny_qwen, "--coco", str(root), "--work", str(tmp_path / "work")),
            *("--train-sizes", "6", "--val-images", "4", "--layers", "1", "2", "--chunk", "3"),
            *("--queries", "image", "--heads", "s32", "s16", "s16q", "--features-on", "disk"),
            *("--epochs", "2", "--batch", "3", "--negatives", "2", "--device", "cpu", "--dtype", "fp32"),
        ]
    )
    heads = ("", ", stride 16", ", stride 16 + IoU")
    suffixes = ("", " x presence", " x GT presence")
    assert set(report["results"]) == {f"image query, 6 images{h}{s}" for h in heads for s in suffixes}
    assert report["features_gb"] > 0
