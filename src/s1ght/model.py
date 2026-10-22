"""`S1GHT`: the user-facing model, Ultralytics-style.

model = S1GHT("d1-3b")                         # native runtime, GPU if available else CPU
model = S1GHT("ollama://clef-flash")           # any catalog name or runtime URI
r = model.decide(state={"photo": Image("bus.jpg")},
                 questions={"has_people": Noul("Are people visible in `photo`?")})
r.answers["has_people"].noul

for res in model.predict("bus.jpg", questions={"has_people": Noul("Are people visible in `image`?")}):
    res.answers[None]["has_people"].noul
    res.save("out.jpg")
"""

from __future__ import annotations

import time
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import Any, Self

from s1ght.detectors.base import Detector, Frame
from s1ght.engine.crops import fit_to_limits
from s1ght.hub import RuntimeSpec, resolve
from s1ght.io.sources import SourceFrame, is_live, load_source
from s1ght.results import Boxes, Results
from s1ght.runtimes import load_runtime
from s1ght.runtimes.base import Runtime
from s1ght.schema.assets import Image
from s1ght.schema.questions import QuestionSet, load_questions, questions_from_wire
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse


class S1GHT:
    """A decision model behind one runtime.

    Args:
        model: Catalog name ("d1-3b"), runtime URI ("ollama://clef-flash", "typesafe://jev-preview",
            "hf://org/repo@rev"), HF repo id, local directory, or a ready `Runtime` object.
        runtime: For catalog names, which runtime to use ("native", "ollaya", ...).
        device: Native runtime only: "cuda", "cuda:1", "mps" or "cpu".
        calibration: Calibration profile path (P1 step 9).
        **kwargs: Passed to the runtime (e.g. `dtype` for native, `base_url`/`api_key` for servers).
    """

    def __init__(
        self,
        model: str | Runtime = "d1-3b",
        *,
        runtime: str | None = None,
        device: str | None = None,
        calibration: str | None = None,
        **kwargs: Any,
    ) -> None:
        if not isinstance(model, str):
            if not isinstance(model, Runtime):
                raise TypeError(f"model must be a name, URI or Runtime, got {type(model).__name__}")
            if runtime or device or calibration or kwargs:
                raise ValueError("runtime=, device=, calibration= and runtime options need a model name, not a Runtime")
            self.spec = RuntimeSpec(runtime="custom", model=str(getattr(model, "model", type(model).__name__)))
            self.runtime: Runtime = model
            return
        self.spec = resolve(model, runtime)
        if self.spec.runtime == "native":
            kwargs.update(device=device, calibration=calibration)
        elif device is not None or calibration is not None:
            raise ValueError(f"device= and calibration= apply to the native runtime, not {self.spec.runtime!r}")
        self.runtime = load_runtime(self.spec, **kwargs)

    @property
    def name(self) -> str:
        return self.spec.entry.name if self.spec.entry else self.spec.model

    @staticmethod
    def _questions(questions: QuestionSet | dict[str, dict[str, Any]]) -> QuestionSet:
        if any(isinstance(q, dict) for q in questions.values()):
            return questions_from_wire(questions)  # type: ignore[arg-type]
        return questions  # type: ignore[return-value]

    def decide(self, state: Any, questions: QuestionSet | dict[str, dict[str, Any]]) -> SystemOneResponse:
        """Answer TypeSafe-style questions about one state (text, JSON and/or `Image`s).

        `questions` maps ids to `Noul`/`Choice`/`Score`/`Box` objects, or to their wire dicts.
        """
        return self.runtime.decide(SystemOneRequest(self.name, state, self._questions(questions)))

    def predict(
        self,
        source: Any,
        questions: QuestionSet | dict[str, dict[str, Any]] | None = None,
        *,
        queries: list[str] | None = None,
        localizer: str = "tiles",
        stream: bool = False,
        context: Any = None,
        batch: int = 4,
        vid_stride: int = 1,
    ) -> list[Results] | Iterator[Results]:
        """Ask full-frame questions about every frame of `source`.

        Each frame is sent as the state `{"image": <frame>}` (plus `"context": context` when
        given), so questions can refer to it as `image`. Frames are resized to the runtime's
        size limits when it has them.

        Args:
            source: Anything `s1ght.io.sources.load_source` accepts: image, folder, glob, URL,
                video, camera index, stream URL, array, PIL image, or a list of these.
            questions: Question id → question (objects or wire dicts).
            queries: Standalone detection queries (P2); not available yet.
            localizer: Standalone localizer for `queries` (P2).
            stream: Return a generator instead of a list (use for videos and live sources).
            context: Extra state (text or JSON) sent with every frame.
            batch: Frames per runtime call for files; live sources use 1.
            vid_stride: Keep every n-th video frame.
        """
        if queries is not None:
            raise NotImplementedError(f"standalone detection (queries=, localizer={localizer!r}) arrives in P2")
        if not questions:
            raise ValueError("predict() needs questions=")
        if batch < 1:
            raise ValueError("batch must be >= 1")
        qs = self._questions(questions)
        frames = load_source(source, vid_stride=vid_stride)
        results = self._predict(frames, qs, context, 1 if is_live(source) else batch)
        return results if stream else list(results)

    def _predict(
        self, frames: Iterable[SourceFrame], questions: QuestionSet, context: Any, batch: int
    ) -> Iterator[Results]:
        caps = self.runtime.caps
        chunk: list[SourceFrame] = []
        for f in frames:
            chunk.append(f)
            if len(chunk) == batch:
                yield from self._decide_frames(chunk, questions, context, caps.image_px)
                chunk = []
        if chunk:
            yield from self._decide_frames(chunk, questions, context, caps.image_px)

    def _decide_frames(
        self, frames: list[SourceFrame], questions: QuestionSet, context: Any, image_px: tuple[int, int] | None
    ) -> list[Results]:
        t0 = time.perf_counter()
        reqs = []
        for f in frames:
            state: dict[str, Any] = {"image": Image(fit_to_limits(f.image, image_px))}
            if context is not None:
                state["context"] = context
            reqs.append(SystemOneRequest(self.name, state, questions))
        t1 = time.perf_counter()
        resps = self.runtime.decide_batch(reqs)
        t2 = time.perf_counter()
        n = len(frames)
        speed = {"preprocess": (t1 - t0) * 1000 / n, "decide": (t2 - t1) * 1000 / n}
        return [
            Results(
                orig_img=f.image,
                answers={None: dict(r.answers)},
                speed=dict(speed),
                path=f.path,
                frame=f.index,
                model=r.model or self.name,
                usage=dict(r.usage),
            )
            for f, r in zip(frames, resps, strict=True)
        ]

    def track(
        self,
        source: Any,
        *,
        ask: dict[str, QuestionSet] | dict[str, dict[str, dict[str, Any]]] | str,
        detector: str | Detector = "yolo11n.pt",
        stream: bool = False,
        budget: float = 2.0,
        batch: int = 4,
        pad: float = 0.2,
        context: str = "crop",
        half_life: float = 1.0,
        thresholds: tuple[float, float] = (0.6, 0.4),
        min_duration: float = 0.0,
        max_age: float = 5.0,
        lost_after: float = 1.0,
        realtime: bool = False,
        wait: bool = True,
        **detector_kwargs: Any,
    ) -> list[Results] | Iterator[Results]:
        """Hybrid mode: a detector tracks objects, the decision model answers questions per track.

        The detector never waits for decisions. A background scheduler picks which tracks to ask
        about (new, changed, uncertain or stale), sends their crops within `budget`, and answers
        are fused over time into per-track answers and `Event`s as they arrive.

        Args:
            source: Video file, camera index, stream URL, image or folder (anything the detector reads).
            ask: Detector class name → questions, or a YAML/JSON file of the same shape. Each crop is
                the state `{"image": crop}` (plus `frame` with `context="crop+frame"`).
            detector: Ultralytics weights ("yolo11n.pt", "yoloe-11s-seg.pt", ...) or a `Detector`.
            stream: Return a generator (use for long videos and live sources).
            budget: Most decisions per second.
            batch: Most crops per runtime call.
            pad: Box expansion before cropping, per side (fraction of width/height).
            context: "crop" or "crop+frame" (adds a downscaled full frame for relation questions).
            half_life: Seconds for the temporal average to halve an old answer's weight.
            thresholds: (on, off) noul probabilities for events (hysteresis).
            min_duration: Seconds above `on` before an event starts.
            max_age: Re-ask a track after this many seconds.
            lost_after: Seconds unseen before a track is lost (its events end).
            realtime: For video files, pace frames at the video's frame rate, as a camera would.
            wait: For finite sources, answer still-queued crops at the end and yield one last
                `Results` for the final frame with those answers and every open event closed.
            **detector_kwargs: For a weights name: `conf`, `tracker`, `imgsz`, `device`, `classes`
                (default: the classes in `ask`); anything else goes to `model.track` (e.g. `vid_stride`).
        """
        from s1ght.engine.fusion import Fusion
        from s1ght.engine.scheduler import Scheduler

        asked = self._ask(ask)
        det, stream_kwargs = self._detector(detector, asked, detector_kwargs)
        scheduler = Scheduler(
            self.runtime, asked, budget, batch, pad=pad, context=context, max_age=max_age, lost_after=lost_after
        )
        fusion = Fusion(half_life, on=thresholds[0], off=thresholds[1], min_duration=min_duration)
        self.scheduler = scheduler
        frames = det.stream(source, **stream_kwargs)
        results = self._track(frames, scheduler, fusion, realtime and not is_live(source), wait and not is_live(source))
        return results if stream else list(results)

    @staticmethod
    def _ask(ask: Any) -> dict[str, QuestionSet]:
        if isinstance(ask, (str, Path)):
            ask = load_questions(ask)
        if not isinstance(ask, dict) or not ask or not all(isinstance(v, dict) for v in ask.values()):
            raise ValueError("ask must map detector class names to questions, e.g. {'person': {...}}")
        return {cls: S1GHT._questions(qs) for cls, qs in ask.items()}

    @staticmethod
    def _detector(detector: Any, asked: dict[str, QuestionSet], kwargs: dict[str, Any]) -> tuple[Detector, dict]:
        if isinstance(detector, Detector):
            return detector, kwargs
        from s1ght.detectors.ultralytics import UltralyticsDetector

        ctor = {k: kwargs.pop(k) for k in ("conf", "tracker", "imgsz", "device", "classes") if k in kwargs}
        ctor.setdefault("classes", list(asked))
        return UltralyticsDetector(detector, **ctor), kwargs

    def _track(
        self, frames: Iterable[Frame], scheduler: Any, fusion: Any, realtime: bool, wait: bool
    ) -> Iterator[Results]:
        scheduler.start()
        last_seen: dict[int, float] = {}
        last: Results | None = None
        wall0: float | None = None
        try:
            for frame in frames:
                if realtime:
                    wall0 = wall0 if wall0 is not None else time.monotonic() - frame.t
                    ahead = frame.t - (time.monotonic() - wall0)
                    if ahead > 0:
                        time.sleep(ahead)
                t0 = time.perf_counter()
                scheduler.submit(frame)
                submit_ms = (time.perf_counter() - t0) * 1000

                events = [e for tid, qid, a, t in scheduler.drain() for e in fusion.update(tid, qid, a, t)]
                present = [] if frame.ids is None else [int(i) for i in frame.ids]
                for tid in present:
                    last_seen[tid] = frame.t
                for tid, seen in list(last_seen.items()):
                    if frame.t - seen > scheduler.lost_after:
                        events += fusion.close(tid, seen)
                        scheduler.forget(tid)
                        del last_seen[tid]
                last = Results(
                    orig_img=frame.image,
                    boxes=Boxes(frame.xyxy, frame.conf, frame.cls, frame.ids, dict(frame.names)),
                    answers={tid: fusion.state(tid) for tid in present if fusion.state(tid)},
                    events=events,
                    speed={**frame.speed, "submit": submit_ms},
                    path=frame.path,
                    frame=frame.index,
                    model=self.name,
                )
                yield last
            if wait and last is not None:
                scheduler.stop()
                scheduler.flush()
                end = max(last_seen.values(), default=0.0)
                events = [e for tid, qid, a, t in scheduler.drain() for e in fusion.update(tid, qid, a, t)]
                answers: dict[int | None, dict[str, Any]] = {
                    tid: a for tid, a in ((t, fusion.state(t)) for t in last_seen) if a
                }
                for tid in fusion.tracks():
                    events += fusion.close(tid, last_seen.get(tid, end)) if tid is not None else []
                yield Results(
                    orig_img=last.orig_img,
                    boxes=last.boxes,
                    answers=answers,
                    events=events,
                    speed={},
                    path=last.path,
                    frame=last.frame,
                    model=self.name,
                )
        finally:
            scheduler.stop()

    def calibrate(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("calibrate() arrives in P1 step 9")

    def serve(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("serve() arrives in P1 step 10")

    def close(self) -> None:
        """Free the runtime (model memory, HTTP sessions)."""
        self.runtime.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"S1GHT({self.name!r}, runtime={self.runtime!r})"
