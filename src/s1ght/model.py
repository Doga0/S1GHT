"""`S1GHT`: the user-facing model, Ultralytics-style.

model = S1GHT("d1-3b")                         # native runtime, GPU if available else CPU
model = S1GHT("ollama://clef-flash")           # any catalog name or runtime URI
r = model.decide(state={"photo": Image("bus.jpg")},
                 questions={"has_people": Noul("Are people visible in `photo`?")})
r.answers["has_people"].noul
"""

from __future__ import annotations

from typing import Any, Self

from s1ght.hub import RuntimeSpec, resolve
from s1ght.runtimes import load_runtime
from s1ght.runtimes.base import Runtime
from s1ght.schema.questions import QuestionSet, questions_from_wire
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse


class S1GHT:
    """A decision model behind one runtime.

    Args:
        model: Catalog name ("d1-3b"), runtime URI ("ollama://clef-flash", "typesafe://jev-preview",
            "hf://org/repo@rev"), HF repo id or local directory.
        runtime: For catalog names, which runtime to use ("native", "ollaya", ...).
        device: Native runtime only: "cuda", "cuda:1", "mps" or "cpu".
        calibration: Calibration profile path (P1 step 9).
        **kwargs: Passed to the runtime (e.g. `dtype` for native, `base_url`/`api_key` for servers).
    """

    def __init__(
        self,
        model: str = "d1-3b",
        *,
        runtime: str | None = None,
        device: str | None = None,
        calibration: str | None = None,
        **kwargs: Any,
    ) -> None:
        self.spec: RuntimeSpec = resolve(model, runtime)
        if self.spec.runtime == "native":
            kwargs.update(device=device, calibration=calibration)
        elif device is not None or calibration is not None:
            raise ValueError(f"device= and calibration= apply to the native runtime, not {self.spec.runtime!r}")
        self.runtime: Runtime = load_runtime(self.spec, **kwargs)

    @property
    def name(self) -> str:
        return self.spec.entry.name if self.spec.entry else self.spec.model

    def decide(self, state: Any, questions: QuestionSet | dict[str, dict[str, Any]]) -> SystemOneResponse:
        """Answer TypeSafe-style questions about one state (text, JSON and/or `Image`s).

        `questions` maps ids to `Noul`/`Choice`/`Score`/`Box` objects, or to their wire dicts.
        """
        if any(isinstance(q, dict) for q in questions.values()):
            questions = questions_from_wire(questions)  # type: ignore[arg-type]
        return self.runtime.decide(SystemOneRequest(self.name, state, questions))  # type: ignore[arg-type]

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError("predict() arrives in P1 step 7")

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
