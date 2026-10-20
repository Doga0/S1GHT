"""System One request and response types, and their wire formats.

Two dialects of `/v1/systemone` are supported:

- "typesafe": images live in a top-level `assets` map keyed by reference string; the
  reference is written wherever the image belongs (state, instructions or criteria).
  Sent as multipart (`part`) or inline JSON (`data_base64` + `format`).
- "ollama": images are a top-level `images` list of base64 strings shared by every
  question. There are no references, so each image is written as "[image N]" (1-based,
  list order) where it appeared. Ollama has no Box question.
"""

from __future__ import annotations

import base64
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal, TypeAlias

from s1ght.schema.assets import Image
from s1ght.schema.questions import Box, QuestionSet, question_from_wire

Dialect = Literal["typesafe", "ollama"]


def _walk(obj: Any, on_image: Callable[[Image], Any], on_str: Callable[[str], Any] | None = None) -> Any:
    """Copy a JSON-like structure, mapping Image values and (optionally) strings. Keys are never touched."""
    if isinstance(obj, Image):
        return on_image(obj)
    if isinstance(obj, dict):
        return {k: _walk(v, on_image, on_str) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_walk(v, on_image, on_str) for v in obj]
    if on_str is not None and isinstance(obj, str):
        return on_str(obj)
    return obj


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


@dataclass
class SystemOneRequest:
    """A `/v1/systemone` request.

    Args:
        model: Model name sent on the wire.
        state: Content to evaluate: str, dict or list; `Image` objects may appear as values.
        questions: Question id → question.
        extra_assets: Images referenced only inside longer strings (e.g. f"... {img.ref}?").
            Images that appear as values in state/questions are collected automatically.
    """

    model: str
    state: Any
    questions: QuestionSet
    extra_assets: list[Image] = field(default_factory=list)

    def assets(self) -> dict[str, Image]:
        """All images in the request, keyed by reference, in first-seen order."""
        found: dict[str, Image] = {}

        def visit(img: Image) -> Image:
            found.setdefault(img.ref, img)
            return img

        _walk(self.state, visit)
        for q in self.questions.values():
            _walk(q.to_wire(), visit)
        for img in self.extra_assets:
            found.setdefault(img.ref, img)
        return found

    def map_images(self, fn: Callable[[Image], Image]) -> SystemOneRequest:
        """Copy of the request with every image replaced by `fn(image)`.

        `fn` is called once per image; keep the `ref` if the image may be referenced inside a
        longer string (`extra_assets`).
        """
        done: dict[str, Image] = {}

        def sub(img: Image) -> Image:
            if img.ref not in done:
                done[img.ref] = fn(img)
            return done[img.ref]

        return SystemOneRequest(
            model=self.model,
            state=_walk(self.state, sub),
            questions={qid: question_from_wire(_walk(q.to_wire(), sub)) for qid, q in self.questions.items()},
            extra_assets=[sub(img) for img in self.extra_assets],
        )

    @property
    def needs_images_flag(self) -> bool:
        """True when TypeSafe needs the `TypeSafe-Danger-Zone-Images: true` header."""
        return bool(self.assets()) or any(isinstance(q, Box) for q in self.questions.values())

    def to_wire(
        self, dialect: Dialect = "typesafe", *, inline: bool = False, embed_images: bool = True
    ) -> tuple[dict, dict[str, bytes]]:
        """Encode for the wire.

        Args:
            dialect: "typesafe" or "ollama".
            inline: TypeSafe only. True puts images in the JSON as `data_base64`; False
                (default) returns them as multipart parts named by their reference.
            embed_images: Ollama only. False writes the "[image N]" labels but leaves out the
                base64 `images` list, for in-process runtimes that take `assets()` directly.

        Returns:
            (JSON body, multipart parts). Parts are empty for inline and Ollama requests.
        """
        assets = self.assets()
        if dialect == "typesafe":

            def ref(img: Image) -> str:
                return img.ref

            body: dict[str, Any] = {
                "model": self.model,
                "state": _walk(self.state, ref),
                "questions": {qid: _walk(q.to_wire(), ref) for qid, q in self.questions.items()},
            }
            parts: dict[str, bytes] = {}
            if assets:
                entries: dict[str, dict[str, str]] = {}
                for r, img in assets.items():
                    if img.format == "webp":
                        raise ValueError("TypeSafe accepts PNG or JPEG images only; use format='jpeg' or 'png'")
                    if inline:
                        entries[r] = {"type": "image", "format": img.format, "data_base64": _b64(img.to_bytes())}
                    else:
                        entries[r] = {"type": "image", "part": r, "format": img.format}
                        parts[r] = img.to_bytes()
                body["assets"] = entries
            return body, parts

        if dialect == "ollama":
            boxes = [qid for qid, q in self.questions.items() if isinstance(q, Box)]
            if boxes:
                raise ValueError(f"Ollama does not support Box questions: {boxes}")
            labels = {r: f"[image {i}]" for i, r in enumerate(assets, 1)}
            inline_refs = [img.ref for img in self.extra_assets]

            def on_str(s: str) -> str:
                for r in inline_refs:
                    s = s.replace(r, labels[r])
                return s

            def label(img: Image) -> str:
                return labels[img.ref]

            body = {
                "model": self.model,
                "state": _walk(self.state, label, on_str),
                "questions": {qid: _walk(q.to_wire(), label, on_str) for qid, q in self.questions.items()},
            }
            if assets and embed_images:
                body["images"] = [_b64(img.to_bytes()) for img in assets.values()]
            return body, {}

        raise ValueError(f"unknown dialect {dialect!r}")

    @classmethod
    def from_wire(
        cls, data: dict[str, Any], parts: dict[str, bytes] | None = None, *, dialect: Dialect = "typesafe"
    ) -> SystemOneRequest:
        """Decode a wire request.

        Args:
            data: JSON body.
            parts: Multipart file parts by name (TypeSafe `part` assets).
            dialect: "typesafe" or "ollama".
        """
        if dialect == "ollama":
            return cls(
                model=data["model"],
                state=data["state"],
                questions={qid: question_from_wire(q) for qid, q in data["questions"].items()},
                extra_assets=[Image(base64.b64decode(s, validate=True)) for s in data.get("images", [])],
            )

        images: dict[str, Image] = {}
        for r, a in data.get("assets", {}).items():
            if a.get("type", "image") != "image":
                raise ValueError(f"asset {r!r}: unsupported type {a.get('type')!r}")
            if "part" in a:
                if parts is None or a["part"] not in parts:
                    raise ValueError(f"asset {r!r}: missing multipart part {a['part']!r}")
                images[r] = Image(parts[a["part"]], format=a.get("format"), ref=r)
            elif "data_base64" in a:
                if "format" not in a:
                    raise ValueError(f"asset {r!r}: data_base64 requires format")
                images[r] = Image(base64.b64decode(a["data_base64"], validate=True), format=a["format"], ref=r)
            else:
                raise ValueError(f"asset {r!r}: needs part or data_base64")

        placed: set[str] = set()

        def on_str(s: str) -> Any:
            img = images.get(s)
            if img is None:
                return s
            placed.add(s)
            return img

        state = _walk(data["state"], lambda img: img, on_str)
        questions = {qid: question_from_wire(_walk(q, lambda img: img, on_str)) for qid, q in data["questions"].items()}
        return cls(
            model=data["model"],
            state=state,
            questions=questions,
            extra_assets=[img for r, img in images.items() if r not in placed],
        )


@dataclass
class NoulAnswer:
    """Probability that the answer is yes."""

    noul: float

    @property
    def confidence(self) -> float:
        """|2p − 1|; TypeSafe sends no confidence for Noul, this is its suggested proxy."""
        return abs(2 * self.noul - 1)

    def to_wire(self) -> dict[str, Any]:
        return {"type": "noul", "noul": self.noul}


@dataclass
class ChoiceAnswer:
    """Most likely option, every option's probability and a confidence in [0, 1]."""

    choice: str
    probabilities: dict[str, float]
    confidence: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "choice",
            "choice": self.choice,
            "probabilities": self.probabilities,
            "confidence": self.confidence,
        }


@dataclass
class ScoreAnswer:
    """Probability-weighted level. `legend` and `probabilities` use int level keys (strings on the wire)."""

    score: float
    legend: dict[int, str]
    probabilities: dict[int, float]
    confidence: float

    def to_wire(self) -> dict[str, Any]:
        return {
            "type": "score",
            "score": self.score,
            "legend": {str(k): v for k, v in self.legend.items()},
            "probabilities": {str(k): v for k, v in self.probabilities.items()},
            "confidence": self.confidence,
        }


@dataclass
class BoxAnswer:
    """Four edges as fractions of the image: top, bottom (height), left, right (width)."""

    box: dict[str, float]
    confidence: float

    def to_wire(self) -> dict[str, Any]:
        return {"type": "box", "box": self.box, "confidence": self.confidence}


Answer: TypeAlias = NoulAnswer | ChoiceAnswer | ScoreAnswer | BoxAnswer


def answer_from_wire(data: dict[str, Any]) -> Answer:
    """Decode one wire answer by its `type`."""
    kind = data.get("type")
    if kind == "noul":
        return NoulAnswer(noul=data["noul"])
    if kind == "choice":
        return ChoiceAnswer(choice=data["choice"], probabilities=data["probabilities"], confidence=data["confidence"])
    if kind == "score":
        return ScoreAnswer(
            score=data["score"],
            legend={int(k): v for k, v in data["legend"].items()},
            probabilities={int(k): v for k, v in data["probabilities"].items()},
            confidence=data["confidence"],
        )
    if kind == "box":
        return BoxAnswer(box=data["box"], confidence=data["confidence"])
    raise ValueError(f"unknown answer type {kind!r}")


@dataclass
class SystemOneResponse:
    """A `/v1/systemone` response: one answer per question id, plus token usage."""

    model: str
    answers: dict[str, Answer]
    usage: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> SystemOneResponse:
        return cls(
            model=data["model"],
            answers={qid: answer_from_wire(a) for qid, a in data["answers"].items()},
            usage=dict(data.get("usage", {})),
        )

    def to_wire(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "answers": {qid: a.to_wire() for qid, a in self.answers.items()},
            "usage": self.usage,
        }
