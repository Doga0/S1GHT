"""Scheduler triggers, budget and errors; S1GHT.track end to end with a fake detector and runtime."""

import time

import numpy as np
import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, Choice, Noul
from s1ght.detectors.base import Detector, Frame
from s1ght.engine.scheduler import CHANGED, NEW, STALE, UNCERTAIN, Scheduler
from s1ght.runtimes.base import Capabilities, RuntimeBase
from s1ght.schema.systemone import NoulAnswer, SystemOneResponse

PERSON = {"red": Noul("Is this person wearing red?")}
RED, BLUE = (0, 0, 220), (220, 40, 0)


def make_frame(t: float, boxes: list[tuple[int, str, tuple, tuple[int, int, int]]], index: int = 0) -> Frame:
    """A 240x320 frame with filled boxes: (track id, class, xyxy, BGR color)."""
    img = np.full((240, 320, 3), 128, np.uint8)
    names = {0: "person", 1: "dog"}
    by_name = {v: k for k, v in names.items()}
    for _, _, (x1, y1, x2, y2), color in boxes:
        img[y1:y2, x1:x2] = color
        img[y1 + 5 : y1 + 10, x1:x2] = 255  # some texture for the signature
    return Frame(
        image=img,
        index=index,
        t=t,
        xyxy=np.array([b[2] for b in boxes], np.float32).reshape(-1, 4),
        conf=np.full(len(boxes), 0.9, np.float32),
        cls=np.array([by_name[b[1]] for b in boxes], np.int64),
        ids=np.array([b[0] for b in boxes], np.int64),
        names=names,
    )


class ColorRuntime(RuntimeBase):
    """Says yes to "red?" when the crop's center is red; optionally slow."""

    caps = Capabilities(images=True, batch=True)

    def __init__(self, delay: float = 0.0, p: float | None = None) -> None:
        self.delay, self.p, self.calls, self.sizes = delay, p, [], []

    def decide(self, req):
        return self.decide_batch([req])[0]

    def decide_batch(self, reqs):
        time.sleep(self.delay)
        self.calls.append(len(reqs))
        out = []
        for r in reqs:
            img = r.state["image"].to_array()
            self.sizes.append(img.shape[:2])
            b, _, rr = img[img.shape[0] // 2, img.shape[1] // 2]
            p = self.p if self.p is not None else (0.95 if rr > 150 and b < 100 else 0.05)
            out.append(SystemOneResponse("color", {q: NoulAnswer(p) for q in r.questions}, {}))
        return out


# --- triggers --------------------------------------------------------------------------------


def test_new_tracks_only_for_asked_classes() -> None:
    s = Scheduler(ColorRuntime(), {"person": PERSON})
    s.submit(make_frame(0.0, [(1, "person", (10, 10, 60, 100), RED), (2, "dog", (100, 10, 160, 100), BLUE)]))
    s.submit(Frame(np.zeros((10, 10, 3), np.uint8), 0, 0.0, np.zeros((0, 4)), np.zeros(0), np.zeros(0), None, {}))
    assert s.pending == 1 and s.stats.queued == 1
    assert s.step() == 1
    assert [(tid, q, round(a.noul, 2)) for tid, q, a, _ in s.drain()] == [(1, "red", 0.95)]
    assert s.stats.reasons["new"] == 1


def test_latest_frame_wins_and_no_double_asking() -> None:
    rt = ColorRuntime()
    s = Scheduler(rt, {"person": PERSON})
    for i in range(3):  # three frames before the worker gets to it
        s.submit(make_frame(i * 0.1, [(1, "person", (10, 10, 60, 100), RED)], i))
    assert s.pending == 1 and s.stats.replaced == 2
    jobs = s._take()
    assert jobs[0].t == pytest.approx(0.2)  # the newest crop
    s.submit(make_frame(0.3, [(1, "person", (10, 10, 60, 100), RED)]))
    assert s.pending == 0  # in flight: not queued again


def run(s: Scheduler, frame: Frame) -> int | None:
    """Submit one frame and return the reason of the job it queued (and answer it)."""
    s.submit(frame)
    jobs = list(s._pending.values())
    reason = jobs[0].reason if jobs else None
    s.step()
    s.drain()
    return reason


def test_triggers_in_order() -> None:
    s = Scheduler(ColorRuntime(), {"person": PERSON}, max_age=5.0, recheck=1.0, lost_after=10.0)
    box = (10, 10, 60, 100)
    assert run(s, make_frame(0.0, [(1, "person", box, RED)])) == NEW
    assert run(s, make_frame(0.5, [(1, "person", box, RED)])) is None  # same look, fresh answer
    assert run(s, make_frame(0.8, [(1, "person", box, BLUE)])) == CHANGED  # appearance changed
    assert run(s, make_frame(6.0, [(1, "person", box, BLUE)])) == STALE
    assert run(s, make_frame(6.5, [(1, "person", box, BLUE)])) is None
    assert run(s, make_frame(20.0, [(1, "person", box, BLUE)])) == NEW  # lost for > 10 s, then back


def test_uncertain_answers_are_rechecked() -> None:
    s = Scheduler(ColorRuntime(p=0.55), {"person": PERSON}, recheck=1.0, uncertain=0.3)
    box = (10, 10, 60, 100)
    assert run(s, make_frame(0.0, [(1, "person", box, RED)])) == NEW
    assert run(s, make_frame(0.5, [(1, "person", box, RED)])) is None
    assert run(s, make_frame(1.0, [(1, "person", box, RED)])) == UNCERTAIN


def test_priority_and_batch() -> None:
    rt = ColorRuntime()
    s = Scheduler(rt, {"person": PERSON}, batch=2, max_age=1.0, lost_after=10.0)
    old = [(i, "person", (10 + 30 * i, 10, 30 + 30 * i, 100), RED) for i in range(1, 4)]
    s.submit(make_frame(0.0, old))
    while s.step():
        pass
    s.submit(make_frame(2.0, [*old, (9, "person", (200, 120, 260, 220), BLUE)]))  # three stale, one new
    jobs = s._take()
    assert jobs[0].track_id == 9 and len(jobs) == 2


def test_crop_context_and_size_limits() -> None:
    rt = ColorRuntime()
    rt.caps = Capabilities(images=True, batch=True, image_px=(280, 1568))
    s = Scheduler(rt, {"person": PERSON}, context="crop+frame", pad=0.0)
    s.submit(make_frame(0.0, [(1, "person", (10, 10, 60, 110), RED)]))
    req = s._request(next(iter(s._pending.values())))
    assert set(req.state) == {"image", "frame"}
    assert min(req.state["image"].size) >= 280 and min(req.state["frame"].size) >= 280


def test_runtime_errors_do_not_kill_the_worker() -> None:
    class Broken(ColorRuntime):
        def decide_batch(self, reqs):
            raise RuntimeError("model crashed")

    s = Scheduler(Broken(), {"person": PERSON})
    s.submit(make_frame(0.0, [(1, "person", (10, 10, 60, 100), RED)]))
    assert s.step() == 0 and s.stats.errors == 1 and "crashed" in str(s.last_error)
    s.submit(make_frame(0.1, [(1, "person", (10, 10, 60, 100), RED)]))
    assert s.pending == 1  # the track can be asked again


def test_budget_limits_decision_rate() -> None:
    rt = ColorRuntime()
    s = Scheduler(rt, {"person": PERSON}, budget=10.0, batch=1, max_age=0.0)
    s.start()
    t0 = time.monotonic()
    i = 0
    while time.monotonic() - t0 < 0.6:
        s.submit(make_frame(i * 0.01, [(k, "person", (10 + 20 * k, 10, 25 + 20 * k, 100), RED) for k in range(8)], i))
        i += 1
        time.sleep(0.01)
    s.stop()
    assert 3 <= s.stats.decisions <= 8  # ~10/s for 0.6 s


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        Scheduler(ColorRuntime(), {})
    with pytest.raises(ValueError):
        Scheduler(ColorRuntime(), {"person": PERSON}, context="frame")


# --- S1GHT.track ------------------------------------------------------------------------------


class FakeDetector:
    """Track 1 (red) is visible for frames 0-29, track 2 (blue) for all 40; 10 fps."""

    def __init__(self) -> None:
        self.kwargs = None

    def stream(self, source, **kwargs):
        self.kwargs = kwargs
        for i in range(40):
            boxes = [(2, "person", (200, 60, 260, 200), BLUE)]
            if i < 30:
                boxes.insert(0, (1, "person", (20, 40, 90, 200), RED))
            yield make_frame(i / 10, boxes, i)


def test_track_end_to_end() -> None:
    det = FakeDetector()
    assert isinstance(det, Detector)
    model = S1GHT(ColorRuntime())
    results = model.track("video.mp4", detector=det, ask={"person": PERSON}, budget=100, half_life=0.0)
    assert len(results) == 41  # 40 frames + the final summary
    final = results[-1]
    assert final.answers[2]["red"].noul == pytest.approx(0.05)
    events = [e for r in results for e in r.events]
    started = [e for e in events if e.end is None]
    ended = [e for e in events if e.end is not None]
    assert [(e.track_id, e.question) for e in started] == [(1, "red")]
    assert len(ended) == 1 and ended[0].track_id == 1 and ended[0].end == pytest.approx(2.9)  # lost after frame 29
    assert any(1 in r.answers for r in results[:30])
    assert results[5].boxes.id is not None and len(results[5].boxes) == 2
    assert "submit" in results[0].speed
    assert model.scheduler.stats.decisions >= 2


def test_track_never_waits_for_the_decision_model() -> None:
    slow = ColorRuntime(delay=0.3)
    model = S1GHT(slow)
    gaps = []
    t = time.perf_counter()
    for _r in model.track("v.mp4", detector=FakeDetector(), ask={"person": PERSON}, stream=True, wait=False):
        now = time.perf_counter()
        gaps.append(now - t)
        t = now
    assert len(gaps) == 40
    assert max(gaps[1:]) < 0.1  # the detector loop never waits ~0.3 s for a decision
    assert sum(slow.calls) >= 1


def test_track_ask_from_yaml_and_errors(tmp_path) -> None:
    path = tmp_path / "ask.yaml"
    path.write_text('person:\n  red: {type: noul, instructions: "Red?"}\n', encoding="utf-8")
    out = S1GHT(FakeRuntime()).track("v", detector=FakeDetector(), ask=str(path), stream=True)
    assert next(iter(out)).frame == 0
    with pytest.raises(ValueError, match="ask"):
        S1GHT(FakeRuntime()).track("v", detector=FakeDetector(), ask={"red": Noul("?")})


def test_track_with_choice_questions() -> None:
    ask = {"person": {"top": Choice("What is this person wearing?", ["coat", "shirt"])}}
    results = S1GHT(FakeRuntime()).track("v", detector=FakeDetector(), ask=ask, budget=100)
    assert results[-1].answers[2]["top"].choice in {"coat", "shirt"}
