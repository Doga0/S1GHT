"""S1GHT — Real-Time Visual Decisions.

Public names are imported lazily so that ``import s1ght`` stays free of heavy
dependencies (torch, transformers, ultralytics).
"""

from __future__ import annotations

import importlib
from typing import TYPE_CHECKING, Any

__version__ = "0.1.0"  # the single source of the package version (pyproject reads it)

_LAZY: dict[str, str] = {
    "S1GHT": "s1ght.model",
    "Image": "s1ght.schema.assets",
    "Noul": "s1ght.schema.questions",
    "Choice": "s1ght.schema.questions",
    "Score": "s1ght.schema.questions",
    "Box": "s1ght.schema.questions",
}

__all__ = ["S1GHT", "Box", "Choice", "Image", "Noul", "Score", "__version__"]

if TYPE_CHECKING:
    from s1ght.model import S1GHT
    from s1ght.schema.assets import Image
    from s1ght.schema.questions import Box, Choice, Noul, Score


def __getattr__(name: str) -> Any:
    """Resolve a public name on first access (PEP 562)."""
    module = _LAZY.get(name)
    if module is None:
        raise AttributeError(f"module 's1ght' has no attribute {name!r}")
    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(__all__))
