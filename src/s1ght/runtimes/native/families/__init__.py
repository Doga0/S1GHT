"""Native family adapters, by catalog family name."""

from __future__ import annotations

import importlib

from s1ght.runtimes.native.families.base import FamilyAdapter

_FAMILIES = {
    "lfm2_vl": ("s1ght.runtimes.native.families.lfm2_vl", "Lfm2VlFamily"),
    "d1_omni": ("s1ght.runtimes.native.families.d1_omni", "D1OmniFamily"),
}
PENDING = {"qwen_vl": "P1 step 11", "semif_generic": "P1 step 11"}


def get_family(name: str) -> FamilyAdapter:
    """A new, unloaded adapter for `name`."""
    if name in PENDING:
        raise NotImplementedError(f"the {name!r} family adapter arrives in {PENDING[name]}")
    if name not in _FAMILIES:
        raise ValueError(f"unknown family {name!r}; expected one of {sorted(_FAMILIES)}")
    module, cls = _FAMILIES[name]
    return getattr(importlib.import_module(module), cls)()
