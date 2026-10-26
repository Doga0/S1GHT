"""Latency benchmark: how fast a model answers, by image size and batch size.

    from s1ght import S1GHT
    from s1ght.bench import bench, table
    rows = bench(S1GHT("d1-omni-600m"), sizes=(128, 256, 384, 640), batches=(1, 4, 16))
    print(table(rows))

For every (image size, batch size) pair, `batch` requests with distinct images (longer side
`size` px, 4:3) are answered by one `runtime.decide_batch` call. After `warmup` untimed calls,
calls are timed until `runs` are done or `max_time` seconds have passed (at least one timed
call). Reported: median and p90 milliseconds per call, milliseconds per image, images per
second, and peak memory (CUDA: allocated by torch; otherwise the process's resident memory
when psutil is installed).

The same numbers come from the command line: `s1ght bench model=d1-omni-600m sizes=[256,640]`.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import cv2
import numpy as np

from s1ght.engine.crops import fit_to_limits
from s1ght.schema.assets import Image
from s1ght.schema.questions import Noul, QuestionSet
from s1ght.schema.systemone import SystemOneRequest

DEFAULT_QUESTIONS: QuestionSet = {"person": Noul("Is there a person in `image`?")}


@dataclass
class BenchRow:
    """One (image size, batch size) measurement."""

    model: str
    runtime: str
    device: str
    size: int
    batch: int
    questions: int
    runs: int
    median_ms: float
    p90_ms: float
    ms_per_image: float
    images_per_s: float
    peak_mem_mb: float | None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def scene(size: int, seed: int = 0) -> np.ndarray:
    """A deterministic synthetic street-like BGR image, longer side `size` px (4:3)."""
    w, h = size, max(1, round(size * 3 / 4))
    rng = np.random.default_rng(seed)
    img = np.zeros((h, w, 3), np.uint8)
    img[: h // 2] = (200, 170, 120)  # sky
    img[h // 2 :] = (90, 90, 90)  # road
    for _ in range(6):
        x, y = int(rng.integers(0, w)), int(rng.integers(h // 3, h))
        bw, bh = int(rng.integers(w // 20 + 1, w // 5 + 2)), int(rng.integers(h // 10 + 1, h // 3 + 2))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.rectangle(img, (x, y), (x + bw, y + bh), color, -1)
    noise = rng.integers(-12, 13, img.shape, dtype=np.int16)
    return np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = size / max(h, w)
    return cv2.resize(img, (max(1, round(w * s)), max(1, round(h * s))), interpolation=cv2.INTER_AREA)


def _requests(
    model: str,
    base: np.ndarray,
    batch: int,
    questions: QuestionSet,
    image_px: tuple[int, int] | None,
    rng: np.random.Generator,
) -> list[SystemOneRequest]:
    """`batch` requests, each with `base` plus its own faint noise."""
    reqs = []
    for _ in range(batch):
        noise = rng.integers(-3, 4, base.shape, dtype=np.int16)
        img = np.clip(base.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        reqs.append(SystemOneRequest(model, {"image": Image(fit_to_limits(img, image_px))}, questions))
    return reqs


class _Memory:
    """Peak memory in MB: torch CUDA allocations on a GPU runtime, else resident set size."""

    def __init__(self, device: str) -> None:
        self.cuda = device.startswith("cuda")
        self.peak: float | None = None
        self._proc: Any = None
        if not self.cuda:
            try:
                import psutil  # type: ignore[import-untyped]  # optional; installed with ultralytics

                self._proc = psutil.Process()
            except ImportError:
                pass

    def reset(self) -> None:
        self.peak = None
        if self.cuda:
            import torch

            torch.cuda.reset_peak_memory_stats()

    def sample(self) -> None:
        if self.cuda:
            import torch

            self.peak = torch.cuda.max_memory_allocated() / 2**20
        elif self._proc is not None:
            rss = self._proc.memory_info().rss / 2**20
            self.peak = rss if self.peak is None else max(self.peak, rss)


def bench(
    model: Any,
    *,
    sizes: Sequence[int] = (128, 256, 384, 640),
    batches: Sequence[int] = (1, 4, 16),
    questions: QuestionSet | None = None,
    image: np.ndarray | None = None,
    runs: int = 5,
    warmup: int = 1,
    max_time: float = 60.0,
    progress: Callable[[BenchRow], None] | None = None,
) -> list[BenchRow]:
    """Measure `model` (an `S1GHT`) at every (size, batch) pair.

    Args:
        model: The model to measure; its runtime's `decide_batch` is timed.
        sizes: Longer image side in pixels.
        batches: Images per `decide_batch` call.
        questions: Questions per image; one Noul by default.
        image: A BGR image to resize instead of the synthetic scene. Each image in a batch gets a
            little different noise, so runtimes cannot answer from a cache.
        runs: Timed calls per pair (fewer if `max_time` runs out).
        warmup: Untimed calls per pair first (model load, kernel selection, server cache).
        max_time: Seconds of timed calls per pair.
        progress: Called with each finished row.
    """
    if runs < 1 or warmup < 0 or not sizes or not batches:
        raise ValueError("need runs >= 1, warmup >= 0 and at least one size and batch")
    qs = questions or DEFAULT_QUESTIONS
    runtime = model.runtime
    device = str(getattr(getattr(model, "base_runtime", runtime), "device", model.spec.runtime))
    memory = _Memory(device)
    image_px = runtime.caps.image_px
    rows = []
    for size in sizes:
        base = scene(size) if image is None else _resize(image, size)
        for batch in batches:
            rng = np.random.default_rng(size * 1000 + batch)
            for _ in range(warmup):
                runtime.decide_batch(_requests(model.name, base, batch, qs, image_px, rng))
            memory.reset()
            times: list[float] = []
            start = time.perf_counter()
            while len(times) < runs and (not times or time.perf_counter() - start < max_time):
                reqs = _requests(model.name, base, batch, qs, image_px, rng)
                t0 = time.perf_counter()
                runtime.decide_batch(reqs)
                times.append((time.perf_counter() - t0) * 1000)
                memory.sample()
            median = statistics.median(times)
            p90 = float(np.percentile(times, 90))
            row = BenchRow(
                model=model.name,
                runtime=model.spec.runtime,
                device=device,
                size=size,
                batch=batch,
                questions=len(qs),
                runs=len(times),
                median_ms=round(median, 1),
                p90_ms=round(p90, 1),
                ms_per_image=round(median / batch, 1),
                images_per_s=round(batch * 1000 / median, 2),
                peak_mem_mb=None if memory.peak is None else round(memory.peak),
            )
            rows.append(row)
            if progress is not None:
                progress(row)
    return rows


def table(rows: Sequence[BenchRow]) -> str:
    """The rows as a fixed-width text table."""
    head = ("size", "batch", "runs", "median ms", "p90 ms", "ms/image", "images/s", "peak MB")
    body = [
        (r.size, r.batch, r.runs, r.median_ms, r.p90_ms, r.ms_per_image, r.images_per_s,
         "-" if r.peak_mem_mb is None else r.peak_mem_mb)
        for r in rows
    ]  # fmt: skip
    cells = [tuple(str(c) for c in row) for row in [head, *body]]
    widths = [max(len(row[i]) for row in cells) for i in range(len(head))]
    lines = ["  ".join(c.rjust(w) for c, w in zip(row, widths)) for row in cells]
    if rows:
        r = rows[0]
        lines.insert(0, f"{r.model} ({r.runtime}, {r.device}), {r.questions} question(s) per image")
    return "\n".join(lines)
