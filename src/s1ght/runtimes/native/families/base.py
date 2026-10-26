"""Family adapters: how one model family is loaded and read.

A family adapter hides everything model-specific (prompt template, image preprocessing,
answer readout) behind `run`, which takes wire-format questions and returns wire-format
answers with the tokens read.

Two kinds of family exist. `SystemOneFamily` wraps models that ship their own one-pass
decision engine (`system_one_batch`): d1 reads the state and images once and branches the
questions off that shared trunk. `semif_generic` and `qwen_vl` read general chat VLMs
through option letters (`s1ght.runtimes.letters`). A family may set `caps` when it differs
from the native default (letter families allow 26 options; their prefix cache depends on the family). Hooks for
`hidden_states` and `attention` come with the P2 localizers.
"""

from __future__ import annotations

import gc
from typing import TYPE_CHECKING, Any, Protocol

if TYPE_CHECKING:
    from PIL.Image import Image as PILImage

Item = tuple[Any, dict[str, dict[str, Any]], list["PILImage"]]
"""(state, {qid: wire question}, images): one request in a family's input format."""

DTYPES = ("fp32", "fp16", "bf16")


class FamilyAdapter(Protocol):
    """One model family in the native runtime."""

    name: str

    def load(self, path: str, device: str, dtype: str) -> None:
        """Load weights from a local directory onto `device` in `dtype` ("fp32", "fp16" or "bf16")."""
        ...

    def run(self, items: list[Item]) -> list[tuple[dict[str, dict[str, Any]], int]]:
        """Answer a batch: per item, ({qid: wire answer}, input tokens read)."""
        ...

    def close(self) -> None:
        """Free the model's memory."""
        ...


def torch_dtype(dtype: str) -> Any:
    import torch

    return {"fp32": torch.float32, "fp16": torch.float16, "bf16": torch.bfloat16}[dtype]


class SystemOneFamily:
    """Adapter for checkpoints whose remote code provides `system_one_batch(requests)`.

    `trust_remote_code=True` runs the code that ships with the checkpoint; the catalog pins
    every source to a commit, so that code is fixed and reviewable.
    """

    name = "system_one"

    def __init__(self) -> None:
        self.model: Any = None

    def load(self, path: str, device: str, dtype: str) -> None:
        from transformers import AutoModel

        if dtype not in DTYPES:
            raise ValueError(f"dtype must be one of {DTYPES}, got {dtype!r}")
        model = AutoModel.from_pretrained(path, trust_remote_code=True, dtype=torch_dtype(dtype))
        self.model = model.to(device).eval()
        if not hasattr(self.model, "system_one_batch"):
            raise TypeError(f"{type(model).__name__} from {path} has no system_one_batch; wrong family?")

    def run(self, items: list[Item]) -> list[tuple[dict[str, dict[str, Any]], int]]:
        import torch

        if self.model is None:
            raise RuntimeError("model is not loaded")
        with torch.inference_mode():
            out = self.model.system_one_batch([(state, qs, images or None) for state, qs, images in items])
        return [(r["answers"], int(r.get("usage", {}).get("input_tokens", 0))) for r in out]

    def close(self) -> None:
        self.model = None
        release()


def release() -> None:
    """Return freed model memory: run the garbage collector and empty the CUDA cache."""
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass
