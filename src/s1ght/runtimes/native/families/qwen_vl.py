"""Qwen2-VL, Qwen2.5-VL and Qwen3-VL (Qwen3-VL-4B-Instruct in the catalog), read through option letters.

Thinking stays off (`enable_thinking=False` in the chat template), so the first token is the
answer letter. Images are resized by S1GHT to multiples of 32 px (16 px patches merged 2x2)
before the processor sees them, so the processor does not resize again and a vLLM server
receiving the same pixels computes the same answer. Attention and hidden-state hooks for the
P2 localizers will live here.
"""

from __future__ import annotations

from typing import Any, ClassVar

from s1ght.runtimes.native.families.semif_generic import SemifGenericFamily


class QwenVlFamily(SemifGenericFamily):
    name = "qwen_vl"
    template_kwargs: ClassVar[dict[str, Any]] = {"enable_thinking": False}
