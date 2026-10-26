"""Check built distributions before they are published.

    python -m build
    python scripts/check_dist.py dist/ [--tag v0.1.0]

- The wheel holds the package data the code reads at run time (`registry.yaml`, `cfg/default.yaml`,
  `py.typed`) and nothing from outside the package (tests, docs, notebooks, design notes).
- The sdist holds the tests and docs, but not the design notes or notebooks.
- The version in the file names and metadata matches `s1ght.__version__`, and the git tag when
  one is given (`v0.1.0` for version 0.1.0); a tagged release also needs a dated CHANGELOG section.
- The PyPI README has no relative links left.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from email.parser import Parser
from pathlib import Path

WHEEL_NEEDS = ("s1ght/__init__.py", "s1ght/registry.yaml", "s1ght/cfg/default.yaml", "s1ght/py.typed", "s1ght/cli.py")
NEVER = ("design.md", "handoff.md", ".ipynb", "/.github/", ".venv")
NAME = "s1ght-visual-decisions"  # the PyPI project name; the package imports as `s1ght`
FILE = NAME.replace("-", "_")  # how wheels and sdists spell it in file names


def version_from_init(root: Path) -> str:
    text = (root / "src" / "s1ght" / "__init__.py").read_text(encoding="utf-8")
    m = re.search(r'^__version__ = "([^"]+)"', text, re.MULTILINE)
    if not m:
        raise SystemExit("no __version__ in src/s1ght/__init__.py")
    return m.group(1)


def check(dist: Path, tag: str | None, version: str | None = None) -> list[str]:
    root = Path(__file__).resolve().parents[1]
    version = version or version_from_init(root)
    errors: list[str] = []
    if tag is not None:
        if tag.removeprefix("v") != version:
            errors.append(f"tag {tag} does not match __version__ {version}")
        changelog = (root / "CHANGELOG.md").read_text(encoding="utf-8")
        if not re.search(rf"^## \[{re.escape(version)}\] - \d{{4}}-\d{{2}}-\d{{2}}\s*$", changelog, re.MULTILINE):
            errors.append(f"CHANGELOG.md has no dated section '## [{version}] - YYYY-MM-DD'")

    wheels = sorted(dist.glob(f"{FILE}-*.whl"))
    sdists = sorted(dist.glob(f"{FILE}-*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        return [*errors, f"expected one wheel and one sdist in {dist}, found {len(wheels)} and {len(sdists)}"]
    wheel, sdist = wheels[0], sdists[0]
    for f in (wheel, sdist):
        if not f.name.startswith(f"{FILE}-{version}-") and f.name != f"{FILE}-{version}.tar.gz":
            errors.append(f"{f.name} is not version {version}")

    with zipfile.ZipFile(wheel) as z:
        names = z.namelist()
        meta_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        meta = Parser().parsestr(z.read(meta_name).decode("utf-8"))
    for need in WHEEL_NEEDS:
        if need not in names:
            errors.append(f"wheel is missing {need}")
    outside = [n for n in names if not n.startswith(("s1ght/", f"{FILE}-{version}.dist-info/"))]
    if outside:
        errors.append(f"wheel has files outside the package: {outside[:5]}")
    if meta["Name"] != NAME:
        errors.append(f"wheel metadata name {meta['Name']} != {NAME} (the PyPI trusted publisher is set up for {NAME})")
    if meta["Version"] != version:
        errors.append(f"wheel metadata version {meta['Version']} != {version}")
    readme = meta.get_payload()
    relative = re.findall(r'\]\((?!https?://|#)[^)]+\)|(?:src|srcset)="(?!https?://)[^"]+"', readme)
    if relative:
        errors.append(f"PyPI README still has relative links: {relative[:3]}")
    if "Doga0/S1GHT" not in readme:
        errors.append("PyPI README has no links to the repository")

    with tarfile.open(sdist) as t:
        snames = t.getnames()
    for bad in NEVER:
        hits = [n for n in [*names, *snames] if bad in n]
        if hits:
            errors.append(f"distribution contains {bad}: {hits[:3]}")
    if not any(n.endswith("/tests/conftest.py") for n in snames):
        errors.append("sdist is missing the tests")
    return errors


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dist", type=Path)
    ap.add_argument("--tag", default=None, help="git tag being released, e.g. v0.1.0")
    args = ap.parse_args()
    errors = check(args.dist, args.tag)
    for e in errors:
        print(f"error: {e}", file=sys.stderr)
    if errors:
        raise SystemExit(1)
    print(f"{args.dist}: ok")


if __name__ == "__main__":
    main()
