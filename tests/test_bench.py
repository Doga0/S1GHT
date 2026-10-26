"""`s1ght.bench` and `s1ght bench`, on runtimes with known speed."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, cli
from s1ght.bench import bench, scene, table


class SlowRuntime(FakeRuntime):
    """Takes 2 ms per call plus 1 ms per image, and records image sizes."""

    def __init__(self) -> None:
        super().__init__()
        self.sizes: list[tuple[int, int]] = []
        self.batches: list[int] = []

    def decide_batch(self, reqs):
        self.batches.append(len(reqs))
        self.sizes += [next(iter(r.assets().values())).size for r in reqs]
        time.sleep(0.002 + 0.001 * len(reqs))
        return [self.decide(r) for r in reqs]


def test_bench_rows() -> None:
    rt = SlowRuntime()
    rows = bench(S1GHT(rt), sizes=(64, 128), batches=(1, 4), runs=3, warmup=1)
    assert [(r.size, r.batch, r.runs) for r in rows] == [(64, 1, 3), (64, 4, 3), (128, 1, 3), (128, 4, 3)]
    assert rt.batches == [1, 1, 1, 1, 4, 4, 4, 4] * 2  # warmup + 3 timed per pair
    assert (128, 96) in rt.sizes and (64, 48) in rt.sizes
    one, four = rows[2], rows[3]
    assert four.median_ms > one.median_ms and four.ms_per_image < one.median_ms  # batching amortizes the 2 ms
    assert four.images_per_s == pytest.approx(4000 / four.median_ms, rel=0.01)
    assert one.p90_ms >= one.median_ms and one.questions == 1 and one.runtime == "custom"
    text = table(rows)
    assert text.splitlines()[0].startswith("fake-1 (custom")
    assert "ms/image" in text


def test_bench_time_limit_and_errors() -> None:
    rows = bench(S1GHT(SlowRuntime()), sizes=(32,), batches=(2,), runs=50, warmup=0, max_time=0.0)
    assert rows[0].runs == 1  # at least one timed call, then the limit stops it
    with pytest.raises(ValueError):
        bench(S1GHT(SlowRuntime()), sizes=(), batches=(1,))


def test_images_differ_within_a_batch() -> None:
    seen = []

    class Recording(FakeRuntime):
        def decide_batch(self, reqs):
            seen.extend(next(iter(r.assets().values())).to_bytes() for r in reqs)
            return super().decide_batch(reqs)

    bench(S1GHT(Recording()), sizes=(64,), batches=(3,), runs=1, warmup=0)
    assert len(set(seen)) == 3  # no runtime can answer from a cache
    img = scene(200)
    assert img.shape == (150, 200, 3) and np.array_equal(img, scene(200))


def test_cli_bench(monkeypatch, tmp_path: Path, capsys) -> None:
    monkeypatch.setattr(cli, "load_model", lambda cfg: S1GHT(SlowRuntime()))
    out = tmp_path / "bench.json"
    code = cli.main(["bench", "sizes=[48,96]", "batches=2", "runs=2", "warmup=0", f"save={out}"])
    assert code == 0
    text = capsys.readouterr().out
    assert "size" in text and "ms/image" in text
    rows = json.loads(out.read_text(encoding="utf-8"))
    assert [(r["size"], r["batch"]) for r in rows] == [(48, 2), (96, 2)]
    assert cli.main(["bench", "sizes=[0]"]) == 2
    assert cli.main(["bench", "batches=[1,x]"]) == 2


def test_cli_bench_json(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "load_model", lambda cfg: S1GHT(SlowRuntime()))
    assert cli.main(["bench", "sizes=32", "batches=1", "runs=1", "warmup=0", "json=True"]) == 0
    assert json.loads(capsys.readouterr().out)[0]["size"] == 32


def test_latency_script_markdown() -> None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
    try:
        import latency
    finally:
        sys.path.pop(0)
    rows = bench(S1GHT(SlowRuntime()), sizes=(32, 64), batches=(1, 2), runs=1, warmup=0)
    md = latency.markdown(rows, "test cpu")
    assert "| size | batch 1 | batch 2 |" in md and "| 64 |" in md and "**fake-1**" in md
