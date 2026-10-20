"""LiquidAI d1-omni-600M: LFM2.5-Encoder-350M trunk with vision and audio encoders and a decision head.

The checkpoint's engine applies the author's calibration to text-only questions and none to
image or audio questions. The card advises fp16 rather than bf16 on GPUs.
"""

from __future__ import annotations

from s1ght.runtimes.native.families.base import SystemOneFamily


class D1OmniFamily(SystemOneFamily):
    name = "d1_omni"
