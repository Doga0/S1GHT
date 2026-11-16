"""benchmarks/region_questions.py (E1): scenes, probes, prompts and metrics offline; a tiny-model run."""

from __future__ import annotations

import random
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import region_questions as rq

from test_patch_signal import tiny_qwen  # noqa: F401 - pytest fixture


def scenes(n: int = 6, size: int = 256) -> list[rq.Scene]:
    rng = random.Random(1)
    return [rq.make_scene(rng, size) for _ in range(n)]


def test_scene_truth_matches_pixels() -> None:
    for sc in scenes():
        names = [s.name for s in sc.shapes]
        assert len(names) == len(set(names)) >= 1
        for s in sc.shapes:
            ys, xs = np.nonzero(s.mask)
            assert s.box.tolist() == [xs.min(), ys.min(), xs.max() + 1, ys.max() + 1]
            color = np.array(rq.COLORS[s.name.split()[0]])
            assert (np.abs(np.asarray(sc.image)[s.mask] - color).max(1) == 0).mean() > 0.95
        masks = [s.mask for s in sc.shapes]
        assert sum(m.sum() for m in masks) == np.logical_or.reduce(masks).sum()  # no pixel twice


def test_probes() -> None:
    sc = scenes(1)[0]
    probes = rq.sample_cells(sc, 0, 8, 12, random.Random(0))
    for s in sc.shapes:
        cov = rq.coverage(s.mask, 8)
        mine = [p for p in probes if p.shape == s.name]
        occ = [p for p in mine if p.kind == "occupancy"]
        assert 0 < len(occ) <= 12 and all(p.truth == (cov[p.cell] > 0) for p in occ)
        assert any(p.truth for p in occ) and any(not p.truth for p in occ)
        centers = [p for p in mine if p.kind == "center"]
        assert sum(p.truth for p in centers) == 1
    lines = rq.sample_lines(sc, 0)
    for p in lines:
        _, k, beyond = rq.SIDES[p.side]
        s = next(s for s in sc.shapes if s.name == p.shape)
        edge = s.box[k] / sc.image.size[k % 2]
        assert p.truth == ((edge > p.at) if beyond else (edge < p.at))
        assert p.truth == ((p.offset < 0) == beyond)


def test_requests() -> None:
    sc = scenes(1)[0]
    probe = rq.Probe(0, sc.shapes[0].name, "occupancy", (2, 1), True, 0.5, True)
    for method in rq.REGION_METHODS:
        state, text = rq.region_request(method, sc, probe, 4, 64)
        assert sc.shapes[0].name in text
        if method == "crops":
            assert len(state) == 16 and "`cell_B3`" in text and state["cell_B3"].size == (64, 64)
        else:
            assert list(state) == ["photo"] and "`photo`" in text
    assert "cell B3" in rq.region_request("grid", sc, probe, 4, 64)[1]
    assert "x = 0.25 to 0.50 and y = 0.50 to 0.75" in rq.region_request("text", sc, probe, 4, 64)[1]
    marked = np.asarray(rq.region_request("mark", sc, probe, 4, 64)[0]["photo"])
    assert (marked[128, 64] == rq.MARK).all()  # the frame's top-left corner: cell B3 starts at (64, 128)
    line = rq.LineProbe(0, sc.shapes[0].name, "left", 0.05, 0.5, False)
    assert "to the right of the vertical line at 0.50" in rq.line_request("text", sc, line)[1]
    drawn = np.asarray(rq.line_request("line", sc, line)[0]["photo"])
    assert (drawn[:, 128] == rq.MARK).all()


def test_metrics() -> None:
    t = np.array([True, True, False, False])
    assert rq.auc(np.array([0.9, 0.8, 0.1, 0.2]), t) == 1.0
    assert rq.auc(np.array([0.1, 0.2, 0.9, 0.8]), t) == 0.0
    assert rq.auc(np.full(4, 0.5), t) == 0.5
    assert np.isnan(rq.auc(np.ones(2), np.array([True, True])))
    assert rq.isotonic(np.array([1.0, 3.0, 2.0, 4.0])).tolist() == [1.0, 2.5, 2.5, 4.0]
    assert rq.isotonic(np.array([4.0, 2.0, 3.0, 1.0]), increasing=False).tolist() == [4.0, 2.5, 2.5, 1.0]
    at = np.array([0.3, 0.35, 0.38, 0.42, 0.45, 0.5])
    # left edge at 0.40: yes (edge past line) for lines before it; one wrong answer is smoothed out
    assert rq.edge_estimate(at, np.array([0.9, 0.2, 0.8, 0.1, 0.1, 0.0]), True) == pytest.approx(0.40)
    # right edge at 0.40: yes (edge before line) for lines after it
    assert rq.edge_estimate(at, np.array([0.0, 0.1, 0.1, 0.9, 0.9, 1.0]), False) == pytest.approx(0.40)


def test_perfect_answers_score_perfectly() -> None:
    sc = scenes()
    probes = [p for i, s in enumerate(sc) for p in rq.sample_cells(s, i, 8, 12, random.Random(0))]
    m = rq.region_metrics(probes, np.array([p.truth for p in probes], float), 1.0)
    assert m["occupancy"]["auc"] == m["center"]["auc"] == m["occupancy"]["accuracy"] == 1.0
    lines = [p for i, s in enumerate(sc) for p in rq.sample_lines(s, i)]
    lm = rq.line_metrics(lines, sc, np.array([p.truth for p in lines], float), 1.0)
    assert lm["auc"] == 1.0 and all(lm[f"accuracy_{o}"] == 1.0 for o in rq.OFFSETS)
    assert lm["edge_error"] < 0.005  # 0 except at the image border, where only inner lines are asked


class Stub:
    """A model that answers every question with a random P(yes); counts images per request."""

    name = "stub"

    def __init__(self) -> None:
        self.rng = np.random.default_rng(0)
        self.images: list[int] = []
        self.runtime = SimpleNamespace(decide_batch=self.decide_batch)

    def decide_batch(self, reqs: list[Any]) -> list[Any]:
        self.images += [len(r.assets()) for r in reqs]
        return [
            SimpleNamespace(answers={q: SimpleNamespace(noul=float(self.rng.random())) for q in r.questions})
            for r in reqs
        ]


def test_main_with_stub(tmp_path: Path) -> None:
    stub = Stub()
    argv = [
        "--images",
        "3",
        "--size",
        "128",
        "--grid",
        "4",
        "--methods",
        *rq.REGION_METHODS,
        "--out",
        str(tmp_path / "e1.json"),
    ]
    report = rq.main(argv, models={"stub": stub})
    r = report["models"]["stub"]
    assert set(r["regions"]) == set(rq.REGION_METHODS) and set(r["lines"]) == set(rq.LINE_METHODS)
    assert 0.0 <= r["regions"]["grid"]["occupancy"]["auc"] <= 1.0
    assert max(stub.images) == 16  # crops: one request holds every cell
    assert (tmp_path / "e1.json").exists()


@pytest.mark.network
def test_tiny_model_end_to_end(tiny_qwen: str, tmp_path: Path) -> None:  # noqa: F811
    from s1ght import S1GHT
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.native.engine import NativeRuntime

    spec = RuntimeSpec("native", "tiny-qwen3-vl", family="qwen_vl")
    with S1GHT(NativeRuntime(spec, path=tiny_qwen, device="cpu", dtype="fp32")) as model:
        report = rq.main(
            ["--images", "2", "--size", "128", "--grid", "4", "--cells", "4", "--crop-side", "64", "--batch", "4"]
            + ["--methods", *rq.REGION_METHODS],
            models={"tiny": model},
        )
    regions = report["models"]["tiny"]["regions"]
    assert set(regions) == set(rq.REGION_METHODS)  # crops: 16 images in one request
    assert all(0.0 <= m["occupancy"]["accuracy"] <= 1.0 for m in regions.values())
