"""Scheduler: decides which tracks to ask about, and asks without ever blocking the detector.

The detector thread calls `submit(frame)` for every frame. For each detection whose class has
questions it decides whether to ask, and if so queues a crop; nothing waits on the decision
model. A worker thread takes the most urgent jobs, `batch` at a time and within `budget`
decisions per second, and calls `runtime.decide_batch`. Answers are collected for `drain()`.

A track is (re)asked when, in priority order:
  1. it is new, or came back after being lost for `lost_after` seconds;
  2. its appearance changed: the 32×32 signature moved more than `change`;
  3. its last answers are uncertain (some noul with |2p−1| < `uncertain`) and `recheck` seconds passed;
  4. its answers are stale: `max_age` seconds passed.
A newer crop of a track replaces its queued one (latest frame wins), and a track already being
answered is not queued again until its answer arrives.
"""

from __future__ import annotations

import heapq
import itertools
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from s1ght.detectors.base import Frame
from s1ght.engine.crops import crop, fit_to_limits, signature, signature_distance, with_frame
from s1ght.runtimes.base import Runtime
from s1ght.schema.assets import Image
from s1ght.schema.questions import QuestionSet
from s1ght.schema.systemone import Answer, NoulAnswer, SystemOneRequest

NEW, CHANGED, UNCERTAIN, STALE = 0, 1, 2, 3
REASONS = {NEW: "new", CHANGED: "changed", UNCERTAIN: "uncertain", STALE: "stale"}


@dataclass
class _TrackState:
    cls_name: str
    last_seen: float
    asked_at: float | None = None
    signature: np.ndarray | None = None
    answers: dict[str, Answer] = field(default_factory=dict)
    in_flight: bool = False


@dataclass
class _Job:
    track_id: int
    cls_name: str
    t: float
    image: np.ndarray
    signature: np.ndarray
    frame: np.ndarray | None
    reason: int


@dataclass
class SchedulerStats:
    frames: int = 0
    queued: int = 0
    replaced: int = 0
    decisions: int = 0
    batches: int = 0
    errors: int = 0
    reasons: dict[str, int] = field(default_factory=lambda: dict.fromkeys(REASONS.values(), 0))


class Scheduler:
    """Asks track-level questions in the background.

    Args:
        runtime: Decision runtime (any `Runtime`).
        ask: Detector class name → questions to ask about each object of that class.
        budget: Most decisions per second (one decision = one crop's questions).
        batch: Most crops per runtime call.
        pad: Box expansion on every side before cropping (fraction of width/height).
        context: "crop" (state `{"image": crop}`) or "crop+frame" (also a downscaled `frame`).
        max_age: Re-ask after this many seconds.
        change: Signature distance that counts as a new appearance.
        uncertain: Noul answers with |2p−1| below this are re-checked after `recheck` seconds.
        recheck: Seconds before an uncertain track is asked again.
        lost_after: Seconds unseen before a track counts as lost (and new when it returns).
        clock: Time source for rate limiting (tests).
    """

    def __init__(
        self,
        runtime: Runtime,
        ask: dict[str, QuestionSet],
        budget: float = 2.0,
        batch: int = 4,
        *,
        pad: float = 0.2,
        context: str = "crop",
        max_age: float = 5.0,
        change: float = 0.10,
        uncertain: float = 0.3,
        recheck: float = 1.0,
        lost_after: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if budget <= 0 or batch < 1:
            raise ValueError("budget must be > 0 and batch >= 1")
        if context not in ("crop", "crop+frame"):
            raise ValueError("context must be 'crop' or 'crop+frame'")
        if not ask:
            raise ValueError("ask needs at least one class with questions")
        self.runtime, self.ask, self.budget, self.batch = runtime, ask, budget, batch
        self.pad, self.context = pad, context
        self.max_age, self.change, self.uncertain, self.recheck, self.lost_after = (
            max_age, change, uncertain, recheck, lost_after,
        )  # fmt: skip
        self.clock = clock
        self.stats = SchedulerStats()
        self.last_error: BaseException | None = None
        self._tracks: dict[int, _TrackState] = {}
        self._pending: dict[int, _Job] = {}
        self._out: list[tuple[int, str, Answer, float]] = []
        self._lock = threading.Condition()
        self._next_slot = 0.0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._order = itertools.count()

    # ---------------------------------------------------------------- detector side

    def submit(self, frame: Frame) -> None:
        """Look at one frame's tracked detections and queue the ones worth asking about. Never blocks on decisions."""
        if frame.ids is None:
            return
        jobs: list[_Job] = []
        with self._lock:
            self.stats.frames += 1
            for i in range(len(frame)):
                name = frame.class_name(i)
                if name not in self.ask:
                    continue
                tid = int(frame.ids[i])
                img = crop(frame.image, frame.xyxy[i], self.pad)
                sig = signature(img)
                reason = self._reason(tid, name, sig, frame.t)
                if reason is not None:
                    context = frame.image if self.context == "crop+frame" else None
                    jobs.append(_Job(tid, name, frame.t, img, sig, context, reason))
            for job in jobs:
                if job.track_id in self._pending:
                    self.stats.replaced += 1
                    job.reason = min(job.reason, self._pending[job.track_id].reason)
                else:
                    self.stats.queued += 1
                self._pending[job.track_id] = job
            if jobs:
                self._lock.notify()

    def _reason(self, tid: int, name: str, sig: np.ndarray, t: float) -> int | None:
        s = self._tracks.get(tid)
        if s is None or t - s.last_seen > self.lost_after or s.cls_name != name:
            self._tracks[tid] = _TrackState(name, t)
            return NEW
        s.last_seen = t
        if s.in_flight:
            return None
        if s.asked_at is None:
            return NEW
        age = t - s.asked_at
        if s.signature is not None and signature_distance(sig, s.signature) > self.change:
            return CHANGED
        if age >= self.recheck and any(
            isinstance(a, NoulAnswer) and abs(2 * a.noul - 1) < self.uncertain for a in s.answers.values()
        ):
            return UNCERTAIN
        if age >= self.max_age:
            return STALE
        return None

    def drain(self) -> list[tuple[int, str, Answer, float]]:
        """Answers that arrived since the last call: (track id, question id, answer, frame time)."""
        with self._lock:
            out, self._out = self._out, []
        return out

    # ---------------------------------------------------------------- worker side

    def _take(self) -> list[_Job]:
        with self._lock:
            ranked = heapq.nsmallest(self.batch, self._pending.values(), key=lambda j: (j.reason, j.t))
            for j in ranked:
                del self._pending[j.track_id]
                s = self._tracks.get(j.track_id)
                if s is not None:
                    s.in_flight = True
            return ranked

    def _request(self, job: _Job) -> SystemOneRequest:
        image_px = self.runtime.caps.image_px
        img = Image(fit_to_limits(job.image, image_px))
        state: Any = {"image": img}
        if job.frame is not None:
            parts = with_frame(job.image, job.frame)
            state = {"image": img, "frame": Image(fit_to_limits(parts["frame"], image_px))}
        return SystemOneRequest(job.cls_name, state, self.ask[job.cls_name])

    def step(self) -> int:
        """Answer up to `batch` queued jobs in the calling thread; returns how many were answered."""
        jobs = self._take()
        if not jobs:
            return 0
        try:
            responses = self.runtime.decide_batch([self._request(j) for j in jobs])
        except Exception as e:  # noqa: BLE001 (any runtime failure must not kill the worker; see last_error)
            with self._lock:
                self.last_error = e
                self.stats.errors += 1
                for j in jobs:
                    if j.track_id in self._tracks:
                        self._tracks[j.track_id].in_flight = False
            return 0
        with self._lock:
            for job, resp in zip(jobs, responses, strict=True):
                s = self._tracks.get(job.track_id)
                if s is not None:
                    s.in_flight, s.asked_at, s.signature = False, job.t, job.signature
                    s.answers = dict(resp.answers)
                for qid, answer in resp.answers.items():
                    self._out.append((job.track_id, qid, answer, job.t))
                self.stats.reasons[REASONS[job.reason]] += 1
            self.stats.decisions += len(jobs)
            self.stats.batches += 1
        return len(jobs)

    def _run(self) -> None:
        while not self._stop.is_set():
            with self._lock:
                while not self._pending and not self._stop.is_set():
                    self._lock.wait(timeout=0.1)
            if self._stop.is_set():
                return
            wait = self._next_slot - self.clock()
            if wait > 0:
                self._stop.wait(wait)
                continue
            n = self.step()
            self._next_slot = self.clock() + n / self.budget

    def start(self) -> None:
        """Start the worker thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="s1ght-scheduler", daemon=True)
        self._thread.start()

    def stop(self, timeout: float | None = 10.0) -> None:
        """Stop the worker (an in-flight batch finishes first, up to `timeout`)."""
        self._stop.set()
        with self._lock:
            self._lock.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None

    def flush(self, timeout: float = 60.0) -> None:
        """Answer every queued job now (ignoring the budget), e.g. at the end of a video file."""
        deadline = time.monotonic() + timeout
        while self._pending and time.monotonic() < deadline:
            if self._thread is not None and self._thread.is_alive():
                time.sleep(0.01)
                continue
            if self.step() == 0 and self.last_error is not None:
                break

    def forget(self, track_id: int) -> None:
        """Drop a lost track and its queued job."""
        with self._lock:
            self._tracks.pop(track_id, None)
            self._pending.pop(track_id, None)

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)
