"""LFM2-VL decision models: LiquidAI d1-3B (on LFM2.5-VL-3B).

The checkpoint's own engine renders the prompt, caps each image at 1024×1024 pixels, reads
the state and images once and scores every question at its answer slot (max over the
option's token forms, temperature 1.0, no calibration).
"""

from __future__ import annotations

from s1ght.runtimes.native.families.base import SystemOneFamily


class Lfm2VlFamily(SystemOneFamily):
    name = "lfm2_vl"
