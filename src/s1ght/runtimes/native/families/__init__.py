"""Native family adapters, by catalog family name."""

from __future__ import annotations

import importlib
from typing import Any

from s1ght.runtimes.native.families.base import FamilyAdapter

_FAMILIES = {
    "lfm2_vl": ("s1ght.runtimes.native.families.lfm2_vl", "Lfm2VlFamily"),
    "d1_omni": ("s1ght.runtimes.native.families.d1_omni", "D1OmniFamily"),
    "qwen_vl": ("s1ght.runtimes.native.families.qwen_vl", "QwenVlFamily"),
    "semif_generic": ("s1ght.runtimes.native.families.semif_generic", "SemifGenericFamily"),
}
LETTER_FAMILIES = frozenset({"qwen_vl", "semif_generic"})


def get_family(name: str, **options: Any) -> FamilyAdapter:
    """A new, unloaded adapter for `name`.

    Args:
        name: Catalog family name.
        **options: Letter families only: `batch`, `flip`, `margin`, `max_side`.
    """
    if name not in _FAMILIES:
        raise ValueError(f"unknown family {name!r}; expected one of {sorted(_FAMILIES)}")
    if options and name not in LETTER_FAMILIES:
        raise ValueError(f"the {name!r} family takes no options, got {sorted(options)}")
    module, cls = _FAMILIES[name]
    return getattr(importlib.import_module(module), cls)(**options)
