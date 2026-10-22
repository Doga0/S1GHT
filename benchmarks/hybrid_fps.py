"""Does the decision model slow the detector down? Hybrid-mode frame rate with and without decisions.

    python benchmarks/hybrid_fps.py video.mp4 --model d1-omni-600m --runtime ollama
    python benchmarks/hybrid_fps.py video.mp4 --model fake-slow          # 0.5 s per decision, no model

Runs the same video three times through the Ultralytics tracker:
  1. detector only;
  2. `S1GHT.track` with the decision model answering in the background;
  3. (live sources excluded) the same, paced at the video's frame rate as a camera would deliver it.
and prints frames per second, the slowest frame-to-frame gap, and how many decisions were made.

On a CPU-only machine the decision model and the detector share cores, so (2) can be slower
than (1) through CPU contention even though the detector never waits on a decision; the
`slowest gap` column shows whether any frame was held up by a decision (it would be ~the
decision latency). Run it on a GPU machine for the numbers that matter.
"""

from __future__ import annotations

import argparse
import statistics
import time
from typing import Any

from s1ght import S1GHT, Choice, Noul
from s1ght.detectors.ultralytics import UltralyticsDetector
from s1ght.runtimes.base import Capabilities, RuntimeBase
from s1ght.schema.systemone import NoulAnswer, SystemOneResponse

ASK = {
    "person": {
        "sunglasses": Noul("Is this person wearing sunglasses?"),
        "top": Choice("What is this person wearing over their top?", ["fur coat", "dark coat", "jacket", "nothing"]),
    }
}


class SlowFake(RuntimeBase):
    """Answers every question after `delay` seconds, without a model."""

    caps = Capabilities(images=True, batch=True)

    def __init__(self, delay: float = 0.5) -> None:
        self.delay = delay

    def decide(self, req):
        return self.decide_batch([req])[0]

    def decide_batch(self, reqs):
        time.sleep(self.delay)
        return [SystemOneResponse("fake-slow", {q: NoulAnswer(0.5) for q in r.questions}, {}) for r in reqs]


def run(frames: Any) -> dict[str, float]:
    gaps, n, t0 = [], 0, time.perf_counter()
    last = t0
    for _ in frames:
        now = time.perf_counter()
        gaps.append(now - last)
        last, n = now, n + 1
    total = time.perf_counter() - t0
    gaps = gaps[1:]  # the first gap includes model warm-up
    return {
        "frames": n,
        "fps": n / total,
        "median_ms": statistics.median(gaps) * 1000 if gaps else 0.0,
        "slowest_ms": max(gaps) * 1000 if gaps else 0.0,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source")
    ap.add_argument("--model", default="d1-omni-600m", help="catalog name, URI, or fake-slow")
    ap.add_argument("--runtime", default=None, help="catalog runtime, e.g. ollama or native")
    ap.add_argument("--detector", default="yolo11n.pt")
    ap.add_argument("--budget", type=float, default=2.0)
    ap.add_argument("--timeout", type=float, default=300.0, help="server runtimes: seconds per request")
    args = ap.parse_args()

    if args.model == "fake-slow":
        model = S1GHT(SlowFake())
    else:
        kw = {"timeout": args.timeout} if args.runtime in ("ollama", "ollaya", "typesafe") else {}
        model = S1GHT(args.model, runtime=args.runtime, **kw)

    detector = UltralyticsDetector(args.detector, classes=list(ASK))
    list(detector.stream(args.source))  # warm-up (weights load, first inference)

    rows = {"detector only": run(detector.stream(args.source))}
    track = model.track(args.source, detector=detector, ask=ASK, stream=True, wait=False, budget=args.budget)
    rows["with decisions"] = run(track)
    rows["with decisions"]["decisions"] = model.scheduler.stats.decisions
    paced = model.track(
        args.source, detector=detector, ask=ASK, stream=True, wait=False, budget=args.budget, realtime=True
    )
    rows["with decisions, paced"] = run(paced)
    rows["with decisions, paced"]["decisions"] = model.scheduler.stats.decisions
    model.close()

    print(f"\n{args.source} | decisions: {model.name} | detector: {args.detector}")
    print(f"{'run':24s} {'frames':>6s} {'fps':>7s} {'median':>9s} {'slowest':>9s} {'decisions':>9s}")
    for name, r in rows.items():
        dec = str(int(r.get("decisions", 0))) if "decisions" in r else "-"
        print(
            f"{name:24s} {r['frames']:6.0f} {r['fps']:7.1f} {r['median_ms']:7.1f}ms {r['slowest_ms']:7.1f}ms {dec:>9s}"
        )


if __name__ == "__main__":
    main()
