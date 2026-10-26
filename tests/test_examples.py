"""Examples and docs stay in step with the code: files load, scripts compile, the offline example runs, links resolve."""

from __future__ import annotations

import py_compile
import re
import subprocess
import sys
from importlib import resources
from pathlib import Path

import numpy as np
import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, Image, cli
from s1ght.schema.questions import Noul, load_questions

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), EXAMPLES / "README.md"]


def test_question_files() -> None:
    flat = load_questions(EXAMPLES / "questions.yaml")
    assert set(flat) == {"bus", "scene", "crowd"}
    crops = load_questions(EXAMPLES / "inspection.yaml")
    assert set(crops) == {"bag", "posture", "visible"}
    ask = load_questions(EXAMPLES / "ask.yaml")
    assert set(ask) == {"person", "car"} and isinstance(ask["person"]["no_helmet"], Noul)  # type: ignore[index]


def test_decide_and_predict_take_question_files() -> None:
    model = S1GHT(FakeRuntime())
    photo = Image(np.full((64, 64, 3), 120, np.uint8))
    r = model.decide({"image": photo, "camera": "gate 3"}, EXAMPLES / "inspection.yaml")
    assert set(r.answers) == {"bag", "posture", "visible"}
    with pytest.raises(ValueError, match="track"):
        model.decide({"image": photo}, str(EXAMPLES / "ask.yaml"))


def test_examples_and_docs_stay_visual() -> None:
    """S1GHT's scope is visual decisions: no text-only or audio use in examples, docs or the catalog."""
    banned = re.compile(r"(?i)\baudio\b|support ticket|charged twice|ticket\.yaml|decide_text")
    # the catalog through the installed package, so the test also runs against a built wheel
    files = [
        *DOCS,
        *EXAMPLES.glob("*.py"),
        *EXAMPLES.glob("*.yaml"),
        Path(str(resources.files("s1ght") / "registry.yaml")),
    ]
    hits = [f"{f.name}: {m.group(0)}" for f in files for m in banned.finditer(f.read_text(encoding="utf-8"))]
    allowed = {"models.md: audio"}  # the scope note says d1-omni's audio input is out of scope
    assert [h for h in hits if h not in allowed] == []


@pytest.mark.parametrize("script", sorted(EXAMPLES.glob("*.py")), ids=lambda p: p.name)
def test_example_compiles(script: Path) -> None:
    py_compile.compile(str(script), doraise=True)


def test_custom_runtime_example_runs() -> None:
    out = subprocess.run(
        [sys.executable, str(EXAMPLES / "custom_runtime.py")], capture_output=True, text=True, timeout=120, check=True
    ).stdout
    assert "predict image0: red=" in out and "color=red" in out and "color=blue" in out


def test_cli_doc_lists_every_key() -> None:
    text = (ROOT / "docs" / "cli.md").read_text(encoding="utf-8")
    missing = [k for k in cli.defaults() if f"`{k}`" not in text]
    assert not missing, f"docs/cli.md does not mention {missing}"


@pytest.mark.parametrize("doc", DOCS, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(doc: Path) -> None:
    for target in re.findall(r"\]\(([^)#\s]+)(?:#[^)]*)?\)", doc.read_text(encoding="utf-8")):
        if "://" in target:
            continue
        assert (doc.parent / target).exists(), f"{doc.name}: broken link {target}"
