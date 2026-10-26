"""`scripts/check_dist.py` on hand-made distributions; the real build is checked in CI's `package` job."""

from __future__ import annotations

import io
import sys
import tarfile
import zipfile
from pathlib import Path

import pytest

import s1ght

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import check_dist

sys.path.pop(0)

V = s1ght.__version__
README = "See [docs](https://github.com/Doga0/S1GHT/blob/main/docs/README.md)."


def make_dist(tmp: Path, *, wheel_files=None, sdist_files=None, readme: str = README, version: str = V) -> Path:
    dist = tmp / "dist"
    dist.mkdir()
    meta = f"Metadata-Version: 2.4\nName: s1ght\nVersion: {version}\n\n{readme}"
    files = wheel_files if wheel_files is not None else list(check_dist.WHEEL_NEEDS)
    with zipfile.ZipFile(dist / f"s1ght-{version}-py3-none-any.whl", "w") as z:
        for f in files:
            z.writestr(f, "")
        z.writestr(f"s1ght-{version}.dist-info/METADATA", meta)
    with tarfile.open(dist / f"s1ght-{version}.tar.gz", "w:gz") as t:
        for f in sdist_files if sdist_files is not None else ["src/s1ght/__init__.py", "tests/conftest.py"]:
            data = b""
            info = tarfile.TarInfo(f"s1ght-{version}/{f}")
            info.size = len(data)
            t.addfile(info, io.BytesIO(data))
    return dist


def test_good_dist(tmp_path: Path) -> None:
    assert check_dist.check(make_dist(tmp_path), None, V) == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"wheel_files": ["s1ght/__init__.py"]}, "missing s1ght/registry.yaml"),
        ({"wheel_files": [*check_dist.WHEEL_NEEDS, "tests/conftest.py"]}, "outside the package"),
        ({"readme": "![banner](docs/a.svg) [x](docs/cli.md)"}, "relative links"),
        ({"sdist_files": ["src/s1ght/__init__.py", "tests/conftest.py", "design.md"]}, "design.md"),
        ({"sdist_files": ["src/s1ght/__init__.py"]}, "missing the tests"),
        ({"version": "9.9.9"}, "is not version"),
    ],
)
def test_bad_dist(tmp_path: Path, kwargs: dict, message: str) -> None:
    errors = check_dist.check(make_dist(tmp_path, **kwargs), None, V)
    assert any(message in e for e in errors), errors


def test_tag_must_match_version(tmp_path: Path) -> None:
    errors = check_dist.check(make_dist(tmp_path), "v0.0.1", V)
    assert any("does not match" in e for e in errors)
