"""benchmarks/dsod_pipeline.py (E2): decoding, edges and caching offline; a tiny-model run."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import dsod_pipeline as dp
import patch_head as ph

from test_patch_signal import tiny_qwen  # noqa: F401 - pytest fixture


def iou(a: np.ndarray, b: np.ndarray) -> float:
    return float(ph.box_iou(np.asarray(a, float)[None], np.asarray(b, float)[None])[0, 0])


def test_peaks() -> None:
    p = np.zeros((4, 4))
    p[0, 0], p[0, 1], p[3, 3], p[2, 2] = 0.9, 0.8, 0.7, 0.4
    assert dp.peaks(p) == [(0, 0), (3, 3)]  # (0, 1) is next to a higher one; (2, 2) is below 0.5
    p[0, 1] = 0.9
    assert dp.peaks(p) == [(0, 0), (3, 3)]  # a tie keeps the first


def test_objects_split_by_centers() -> None:
    xs = ys = np.arange(0, 129, 16)
    occ = np.zeros((8, 8))
    occ[2, 0:6] = 0.9  # one component, two centers far apart
    assert len(dp.objects(occ, {(2, 1): 0.9, (2, 4): 0.8}, xs, ys)) == 2
    assert len(dp.objects(occ, {}, xs, ys)) == 1  # no center: one object


def test_edges_from_perfect_strip_answers() -> None:
    sc = dp.make_scenes(1, 448, 3)[0]
    shape = sc.shapes[0]
    oracle = dp.Oracle(sc, 8)
    src = {stage: oracle for stage in dp.STAGES}
    boxes, scores, labels = dp.detect(sc, src, 8, 4, edges=True)
    mine = boxes[labels == dp.CLASSES.index(shape.name)]
    assert len(mine) == 1 and np.all(scores == 1)
    step = 2 * 56 / 16  # two cells over 2^4 brackets
    assert np.abs(mine[0] - shape.box).max() <= step / 2 + 1
    coarse = dp.detect(sc, src, 8, 4, edges=False)[0][labels == dp.CLASSES.index(shape.name)][0]
    assert iou(mine[0], shape.box) > 0.85 > iou(coarse, shape.box)  # edges land mid-bracket


def test_strip_rects() -> None:
    box = np.array([56, 112, 168, 224])
    assert dp.strip(box, 0, 60.4, 448, 448) == (0, 112, 60, 224)
    assert dp.strip(box, 2, 160, 448, 448) == (160, 112, 448, 224)
    assert dp.strip(box, 1, 120, 448, 448) == (56, 0, 168, 120)
    assert dp.strip(box, 3, 220, 448, 448) == (56, 220, 168, 448)
    lo, hi, lines = dp.edge_lines(box, 0, np.arange(0, 449, 56), np.arange(0, 449, 56), 3)
    assert (lo, hi) == (0, 112) and len(lines) == 7 and lines[0] == 14


def test_oracle_rows() -> None:
    report = dp.main(["--model", "none", "--images", "30", "--configs", "oracle", "oracle, no edges"])
    rows = report["configs"]
    assert rows["oracle"]["all"]["map50"] > 0.9
    assert rows["oracle"]["all"]["map"] > rows["oracle, no edges"]["all"]["map"] + 0.3


class Stub:
    """Random answers; counts questions per runtime call."""

    name = "stub"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.questions = 0
        self.runtime = SimpleNamespace(decide_batch=self.decide_batch)

    def decide_batch(self, reqs: list[Any]) -> list[Any]:
        self.questions += sum(len(r.questions) for r in reqs)
        return [
            SimpleNamespace(answers={q: SimpleNamespace(noul=float(self.rng.random())) for q in r.questions})
            for r in reqs
        ]


def test_asker_groups_and_caches() -> None:
    stub = Stub()
    asker = dp.Asker(stub, batch=4)
    built = []

    def build() -> dict[str, Any]:
        built.append(1)
        return {"photo": dp.make_scenes(1, 64, 0)[0].image}

    items = [("a", build, f"q{i}") for i in range(6)] + [("a", build, "q0")]
    p = asker.ask(items)
    assert stub.questions == 6 and len(built) == 1 and p[0] == p[6]
    assert np.array_equal(asker.ask(items[:3]), p[:3]) and stub.questions == 6  # all cached


def test_main_with_stub(tmp_path: Path) -> None:
    stub = Stub()
    out = tmp_path / "e2.json"
    report = dp.main(["--images", "2", "--size", "128", "--grid", "4", "--bits", "1", "--out", str(out)], model=stub)
    assert set(report["configs"]) == set(dp.CONFIGS)
    assert report["configs"]["oracle"]["questions"] == 0 and report["configs"]["model"]["questions"] > 0
    assert stub.questions <= sum(r["questions"] for r in report["configs"].values()) * 2  # shared, cached
    assert out.exists()


@pytest.mark.network
def test_tiny_model_end_to_end(tiny_qwen: str) -> None:  # noqa: F811
    from s1ght import S1GHT
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.native.engine import NativeRuntime

    spec = RuntimeSpec("native", "tiny-qwen3-vl", family="qwen_vl")
    with S1GHT(NativeRuntime(spec, path=tiny_qwen, device="cpu", dtype="fp32")) as model:
        report = dp.main(
            ["--images", "1", "--size", "128", "--grid", "4", "--bits", "1", "--crop-side", "64", "--batch", "8"],
            model=model,
        )
    assert report["configs"]["model"]["questions"] > 0 and report["ms_per_question"] > 0
