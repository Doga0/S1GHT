"""Examples and docs stay in step with the code: files load, scripts compile, the offline example runs, links resolve."""

from __future__ import annotations

import py_compile
import re
import subprocess
import sys
from pathlib import Path

import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, cli
from s1ght.schema.questions import Noul, load_questions

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "examples"
DOCS = [ROOT / "README.md", *sorted((ROOT / "docs").glob("*.md")), EXAMPLES / "README.md"]


def test_question_files() -> None:
    flat = load_questions(EXAMPLES / "questions.yaml")
    assert set(flat) == {"bus", "scene", "crowd"}
    ticket = load_questions(EXAMPLES / "ticket.yaml")
    assert set(ticket) == {"refund", "team", "urgency"}
    ask = load_questions(EXAMPLES / "ask.yaml")
    assert set(ask) == {"person", "car"} and isinstance(ask["person"]["no_helmet"], Noul)  # type: ignore[index]


def test_decide_and_predict_take_question_files() -> None:
    model = S1GHT(FakeRuntime())
    r = model.decide("a red bus", EXAMPLES / "ticket.yaml")
    assert set(r.answers) == {"refund", "team", "urgency"}
    with pytest.raises(ValueError, match="track"):
        model.decide("x", str(EXAMPLES / "ask.yaml"))


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
