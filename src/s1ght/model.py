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
from typing import Any, Self

from s1ght.engine.crops import fit_to_limits
from s1ght.hub import RuntimeSpec, resolve
from s1ght.io.sources import SourceFrame, is_live, load_source
from s1ght.results import Results
from s1ght.runtimes import load_runtime
from s1ght.runtimes.base import Runtime
from s1ght.schema.assets import Image
from s1ght.schema.questions import QuestionSet, questions_from_wire
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

    def track(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("track() arrives in P1 step 8")

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
