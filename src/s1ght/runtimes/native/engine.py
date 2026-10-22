"""NativeRuntime: decision models in-process, on CUDA, MPS or CPU.

Device and dtype come from the catalog's per-device table (`native: {dtype: {cuda, t4, cpu}}`):
CUDA GPUs with bf16 support (compute capability 8.0+, e.g. A100) use `cuda`, older ones such
as the T4 (7.5, no bf16) use `t4`, CPU uses `cpu`.

Every answer goes through S1GHT's readout: the family's probabilities become log-scores
(`RawScores`) and `scores_to_answers` turns them into TypeSafe answers, so `confidence` follows
TypeSafe's definition for every model; calibration profiles are applied on top
(`s1ght.engine.calibration.CalibratedRuntime`).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any

from s1ght.runtimes.base import (
    Capabilities,
    RawScores,
    RuntimeBase,
    answers_to_raw,
    scores_to_answers,
    validate_request,
)
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse, answer_from_wire

if TYPE_CHECKING:
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.native.families.base import FamilyAdapter, Item

CAPS = Capabilities(images=True, batch=True, prefix_cache=True, raw_scores=True, max_options=255)  # system_one families
DEFAULT_DTYPES = {"cuda": "bf16", "t4": "fp16", "cpu": "fp32"}


def select_device(device: str | None = None) -> str:
    """`device` if given, else "cuda" when available, then "mps", then "cpu"."""
    if device:
        return device
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    return "mps" if mps is not None and mps.is_available() else "cpu"


def select_dtype(
    device: str,
    table: dict[str, str] | None = None,
    capability: Callable[[str], tuple[int, int]] | None = None,
) -> str:
    """Dtype for `device` from a catalog table `{cuda, t4, cpu}`.

    Args:
        device: "cpu", "mps", "cuda" or "cuda:N".
        table: The catalog's `native.dtype` entry; missing keys use bf16 / fp16 / fp32.
        capability: Returns a CUDA device's compute capability; defaults to torch's.
    """
    table = {**DEFAULT_DTYPES, **(table or {})}
    if device.startswith("cuda"):
        if capability is None:
            import torch

            capability = torch.cuda.get_device_capability
        major, _ = capability(device)
        return table["cuda"] if major >= 8 else table["t4"]
    if device == "mps":
        return table.get("mps", "fp16")
    return table["cpu"]


def split_request(req: SystemOneRequest) -> Item:
    """A request as (state, wire questions, PIL images) for a family.

    Images become "[image N]" labels where they appeared, in `assets()` order, and are passed
    as a separate list. A state that is only an image stays as its label rather than None:
    d1-omni-600M answers image questions much worse with an empty text state.
    """
    body, _ = req.to_wire("ollama", embed_images=False)
    return body["state"], body["questions"], [img.to_pil() for img in req.assets().values()]


class NativeRuntime(RuntimeBase):
    """In-process runtime for one catalog model.

    Args:
        spec: A native `RuntimeSpec` from `s1ght.hub.resolve`.
        device: "cuda", "cuda:1", "mps" or "cpu"; picked automatically when None.
        dtype: "bf16", "fp16" or "fp32"; from the catalog's per-device table when None.
        family: A ready adapter (tests); built from `spec.family` when None.
        path: Local model directory; downloaded from `spec.source` into the S1GHT cache when None.
        **family_options: For option-letter families (`qwen_vl`, `semif_generic`): `batch`
            (prompts per forward), `flip` ("auto", "always", "never"), `margin`, `max_side`.
    """

    caps = CAPS

    def __init__(
        self,
        spec: RuntimeSpec,
        *,
        device: str | None = None,
        dtype: str | None = None,
        family: FamilyAdapter | None = None,
        path: str | None = None,
        **family_options: Any,
    ) -> None:
        if spec.runtime != "native":
            raise ValueError(f"NativeRuntime needs a native spec, got {spec.runtime!r}")
        self.spec = spec
        self.model = spec.entry.name if spec.entry else spec.model
        self.temperature: float | dict[str, float] = 1.0
        if family is None:
            from s1ght.runtimes.native.families import get_family

            family = get_family(spec.family or "semif_generic", **family_options)
            _require_torch()
        elif family_options:
            raise ValueError("family options apply when the family is built from the spec")
        self.family = family
        self.caps = getattr(family, "caps", CAPS)
        self.device = select_device(device)
        self.dtype = dtype or select_dtype(self.device, (spec.options or {}).get("dtype"))
        if path is None:
            if spec.source is None:
                raise ValueError(f"{spec.model}: no source to download and no local path")
            from s1ght.hub import download

            path = str(download(spec.source))
        self.path = path
        self.family.load(path, self.device, self.dtype)

    def _run(self, reqs: list[SystemOneRequest]) -> list[tuple[RawScores, int]]:
        for r in reqs:
            validate_request(self.caps, r)
        out = self.family.run([split_request(r) for r in reqs])
        return [
            (answers_to_raw(r.questions, {qid: answer_from_wire(a) for qid, a in answers.items()}), tokens)
            for r, (answers, tokens) in zip(reqs, out, strict=True)
        ]

    def raw(self, req: SystemOneRequest) -> RawScores:
        return self._run([req])[0][0]

    def raw_batch(self, reqs: list[SystemOneRequest]) -> list[RawScores]:
        """`raw` for several requests in one family call."""
        return [raw for raw, _ in self._run(reqs)]

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        return self.decide_batch([req])[0]

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        """Answer several requests in one family call (packed by the model's engine)."""
        return [
            SystemOneResponse(
                model=self.model,
                answers=dict(scores_to_answers(r.questions, raw, self.temperature)),
                usage={"input_tokens": tokens, "output_tokens": 0},
            )
            for r, (raw, tokens) in zip(reqs, self._run(reqs), strict=True)
        ]

    def close(self) -> None:
        self.family.close()

    def __repr__(self) -> str:
        return f"NativeRuntime(model={self.model!r}, device={self.device!r}, dtype={self.dtype!r})"


def _require_torch() -> Any:
    try:
        import torch
        import transformers  # noqa: F401
    except ImportError as e:
        raise ImportError("the native runtime needs torch and transformers: pip install 's1ght[native]'") from e
    return torch
