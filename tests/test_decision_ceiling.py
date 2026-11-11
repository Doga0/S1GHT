"""benchmarks/decision_ceiling.py (E0): perfect answers and decoders, offline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import decision_ceiling as dc
import patch_head as ph


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(ph.box_iou(np.asarray(a, float)[None], np.asarray(b, float)[None])[0, 0])


def test_oracle_answers() -> None:
    # 100 x 100 image, 4 x 4 cells of 25 px; a box over cells (1..2, 1..2), half of column 3
    o = dc.Oracle(np.array([[25, 25, 87.5, 75]]), 100, 100)
    cov = o.coverage(4)
    assert cov[1:3, 1:3].tolist() == [[1, 1], [1, 1]]
    assert cov[1:3, 3].tolist() == [0.52, 0.52]  # 13 of 25 px: the box edge rounds outward
    assert cov[0].sum() == cov[3].sum() == cov[:, 0].sum() == 0
    assert o.level(4)[1].tolist() == [0, 4, 4, 3]
    assert o.centers(4).sum() == 1 and o.centers(4)[2, 2] == 1  # center (56, 50): row 2
    # min cover: the half-covered column is "no" above 0.6
    assert o.occupied(4).sum() == 6 and dc.Oracle(o.boxes, 100, 100, 0.6).occupied(4).sum() == 4


def test_components() -> None:
    m = np.zeros((4, 4), bool)
    m[0, 0] = m[0, 1] = m[1, 1] = True
    m[2, 2] = m[3, 3] = True  # diagonal: separate
    found = dc.components(m)
    assert sorted(map(len, found)) == [1, 1, 3]


def test_occ_decoder_is_exact_on_aligned_boxes() -> None:
    o = dc.Oracle(np.array([[25, 0, 75, 50]]), 100, 100)
    boxes, asked = dc.decode(dc.parse_design("occ4"), o)
    assert len(boxes) == 1 and boxes[0].tolist() == [25, 0, 75, 50]
    assert asked == 4 + 4 * 2  # 2x2: all 4, then the children of the 2 occupied cells


def test_score_fit_beats_cells() -> None:
    box = np.array([31, 12, 83, 61])
    o = dc.Oracle(box[None], 128, 128)
    cells = dc.decode(dc.parse_design("occ4"), o)[0][0]
    fitted = dc.decode(dc.parse_design("score4"), o)[0][0]
    assert iou(fitted, box) > iou(cells, box) + 0.15
    assert iou(fitted, box) > 0.85


def test_center_splits_touching_objects() -> None:
    # two same-class boxes side by side: one component, two centers
    boxes = np.array([[0, 0, 50, 50], [50, 0, 100, 50]])
    o = dc.Oracle(boxes, 100, 100)
    merged = dc.decode(dc.parse_design("occ8"), o)[0]
    split = dc.decode(dc.parse_design("center8"), o)[0]
    assert len(merged) == 1 and merged[0].tolist() == [0, 0, 100, 50]
    # centers in columns 2 and 6; column 4 is as near to both and goes to the first seed
    assert [b.tolist() for b in split] == [[0, 0, 62, 50], [62, 0, 100, 50]]


def test_count_seeds_several_objects_in_one_cell() -> None:
    boxes = np.array([[10, 10, 40, 40], [5, 5, 45, 45]])  # nested, same center cell
    o = dc.Oracle(boxes, 100, 100)
    assert len(dc.decode(dc.parse_design("center4"), o)[0]) == 1
    found = dc.decode(dc.parse_design("count4+b3"), o)[0]
    assert len(found) == 2  # the boundary stage tells them apart: largest first
    assert iou(found[0], boxes[1]) > 0.9 and iou(found[1], boxes[0]) > 0.85


def test_boundary_refinement() -> None:
    box = np.array([31, 12, 83, 61])
    o = dc.Oracle(box[None], 128, 128)
    coarse = dc.decode(dc.parse_design("center4"), o)[0][0]
    for bits in (1, 2, 3):
        fine = dc.decode(dc.parse_design(f"center4+b{bits}"), o)[0][0]
        step = 32 / 2**bits  # cell width over 2^bits brackets
        assert np.abs(fine - box).max() <= step / 2 + 1e-9
    assert iou(fine, box) > 0.9
    assert iou(coarse, box) < 0.7


def test_parse_design() -> None:
    assert dc.parse_design("count16+b3") == dc.Design("count16+b3", "count", 16, 3)
    for bad in ("occ6", "occ8+b2", "grid8", "center1"):
        with pytest.raises(ValueError):
            dc.parse_design(bad)


def test_parallel_decisions() -> None:
    assert dc.parallel_decisions(dc.parse_design("occ8"), 80, 5) == 80 * (1 + 64)
    assert dc.parallel_decisions(dc.parse_design("count8+b2"), 80, 5) == 80 * (1 + 128) + 5 * 4 * 3


def test_main_end_to_end(tmp_path: Path) -> None:
    ann = tmp_path / "annotations"
    ann.mkdir()
    data = {
        "categories": [{"id": 3, "name": "b"}, {"id": 1, "name": "a"}],
        "images": [{"id": 1, "width": 128, "height": 96}, {"id": 2, "width": 64, "height": 64}],
        "annotations": [
            {"image_id": 1, "category_id": 1, "bbox": [0, 0, 64, 48], "iscrowd": 0},
            {"image_id": 1, "category_id": 3, "bbox": [70, 50, 40, 30], "iscrowd": 0},
            {"image_id": 1, "category_id": 3, "bbox": [0, 0, 128, 96], "iscrowd": 1},  # dropped
            {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 20], "iscrowd": 0},
        ],
    }
    (ann / "instances_val2017.json").write_text(json.dumps(data))
    out = tmp_path / "e0.json"
    report = dc.main(["--coco", str(tmp_path), "--designs", "boxes", "occ2", "--out", str(out)])
    assert report["images"] == 2 and report["truth_objects"] == 1.5
    assert report["designs"]["boxes"]["all"]["map"] == pytest.approx(1.0)
    assert report["designs"]["occ2"]["all"]["map"] < 1.0
    assert report["designs"]["occ2"]["decisions_parallel"] == 2 * (1 + 4)
    assert json.loads(out.read_text())["designs"].keys() == {"boxes", "occ2"}
