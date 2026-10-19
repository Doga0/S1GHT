import subprocess
import sys

import s1ght

HEAVY = ("torch", "transformers", "ultralytics")


def test_version() -> None:
    assert s1ght.__version__ == "0.1.0.dev0"


def test_import_is_light() -> None:
    """`import s1ght` must not pull in torch, transformers or ultralytics."""
    code = f"import sys, s1ght; print([m for m in {HEAVY!r} if m in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_unknown_attribute() -> None:
    try:
        s1ght.does_not_exist  # noqa: B018
    except AttributeError:
        return
    raise AssertionError("expected AttributeError")
