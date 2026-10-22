"""Option-letter readout: any chat VLM as a decision model (the SemIf prompt contract).

Each Noul, Choice or Score question becomes one multiple-choice prompt: the state (text,
JSON and images) and a JSON object `{"criterion", "options": [{"letter", "description"}]}`.
The model's next-token log-probabilities of the option letters are the answer; nothing is
generated. Noul shows A = yes, B = no; Choice its options in `criteria` order; Score its
levels from lowest to highest.

Letter models prefer "A" (about 2.4 nat on Qwen3-VL-4B). When the first reading is close
(top-two gap below `margin`), the question is asked again with the options in other orders
and the readings are averaged. With up to `MAX_ROTATIONS` options every cyclic rotation is
asked, so each option is shown once at each letter and any per-letter bias cancels exactly
(two options: the reversed order). Longer lists are asked once more in reverse, which
cancels the bias between the ends but not for the middle options.

This module only builds prompts and reads letters; it imports neither torch nor httpx, so
the native families (`qwen_vl`, `semif_generic`) and the logprob server client
(`runtimes/server/logprob.py`) send the same prompt and the same pixels, and agree.
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
from PIL import Image as PILImage

from s1ght.runtimes.base import Capabilities
from s1ght.schema.assets import Image
from s1ght.schema.systemone import SystemOneRequest

LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
CAPS = Capabilities(images=True, batch=True, raw_scores=True, max_options=len(LETTERS))
FLIP_MARGIN = 6.0  # nat
MAX_ROTATIONS = 4  # options up to which every rotation is asked when debiasing
MAX_SIDE = 1024  # px, longer image side
MIN_PIXELS = 256 * 256  # smaller images are upscaled, as Qwen's processor would
ALIGN = 32  # Qwen3-VL: 16 px patches merged 2x2

Flip = Literal["auto", "always", "never"]

SYSTEM = (
    "You are a careful judge. You are given an input (text, JSON and/or images) and one "
    "multiple-choice question about it. Reply with the letter of the single best option "
    "and nothing else."
)


@dataclass(frozen=True)
class Prompt:
    """One question in one option order.

    Attributes:
        item: Index of the request in the batch.
        qid: Question id.
        order: `order[i]` is the option (in RawScores order) shown with letter `LETTERS[i]`.
        messages: Chat messages; images are `{"type": "image", "index": n}` parts (0-based
            into the request's image list), to be filled in by the runtime.
    """

    item: int
    qid: str
    order: tuple[int, ...]
    messages: list[dict[str, Any]]

    @property
    def letters(self) -> str:
        return LETTERS[: len(self.order)]


def split(req: SystemOneRequest) -> tuple[Any, dict[str, dict[str, Any]], list[Image]]:
    """(state, wire questions, images) with images replaced by "[image N]" labels, in `assets()` order."""
    body, _ = req.to_wire("ollama", embed_images=False)
    return body["state"], body["questions"], list(req.assets().values())


def options(question: dict[str, Any]) -> list[str]:
    """Option descriptions of a wire question, in RawScores order."""
    kind: str = question["type"]
    criteria: Any = question.get("criteria")
    if kind == "noul":
        c = criteria or {}
        return [_option("yes", c.get("true")), _option("no", c.get("false"))]
    if kind == "choice":
        return [_option(name, desc) for name, desc in criteria.items()]
    if kind == "score":
        return [f"level {i}: {_text(level)}" for i, level in enumerate(criteria)]
    raise ValueError(f"{kind} questions are not answered by option letters")


def _option(name: str, description: Any) -> str:
    return name if description is None else f"{name}: {_text(description)}"


def _text(content: Any) -> str:
    return content if isinstance(content, str) else json.dumps(content, ensure_ascii=False)


def build(item: int, qid: str, state: Any, question: dict[str, Any], n_images: int, order: Sequence[int]) -> Prompt:
    """The chat prompt for one question; `order` maps letters to options (see `Prompt`)."""
    opts = options(question)
    if sorted(order) != list(range(len(opts))):
        raise ValueError(f"order must be a permutation of {len(opts)} options")
    if len(opts) > len(LETTERS):
        raise ValueError(f"{qid!r} has {len(opts)} options; option letters allow {len(LETTERS)}")
    content: list[dict[str, Any]] = []
    for n in range(n_images):
        content += [{"type": "text", "text": f"[image {n + 1}]"}, {"type": "image", "index": n}]
    task = {
        "input": state,
        "criterion": question["instructions"],
        "options": [{"letter": LETTERS[i], "description": opts[o]} for i, o in enumerate(order)],
    }
    text = json.dumps(task, ensure_ascii=False, indent=1) + "\nAnswer with one letter."
    content.append({"type": "text", "text": text})
    messages: list[dict[str, Any]] = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": content}]
    return Prompt(item, qid, tuple(order), messages)


def extra_orders(n: int) -> list[tuple[int, ...]]:
    """Option orders asked besides the natural one when debiasing `n` options."""
    if n <= MAX_ROTATIONS:
        return [tuple((k + j) % n for j in range(n)) for k in range(1, n)]
    return [tuple(range(n - 1, -1, -1))]


def first_pass(items: Sequence[tuple[Any, dict[str, dict[str, Any]], int]], flip: Flip) -> list[Prompt]:
    """Prompts for every scorable question of every (state, questions, n_images) item.

    Box questions are skipped. With `flip="always"`, the extra orders are included now.
    """
    out = []
    for i, (state, questions, n_images) in enumerate(items):
        for qid, q in questions.items():
            if q["type"] == "box":
                continue
            n = len(options(q))
            out.append(build(i, qid, state, q, n_images, range(n)))
            if flip == "always":
                out += [build(i, qid, state, q, n_images, order) for order in extra_orders(n)]
    return out


def second_pass(
    items: Sequence[tuple[Any, dict[str, dict[str, Any]], int]],
    readings: dict[tuple[int, str], list[tuple[tuple[int, ...], np.ndarray]]],
    flip: Flip,
    margin: float = FLIP_MARGIN,
) -> list[Prompt]:
    """Prompts in the extra orders for questions whose first reading was closer than `margin` nat."""
    if flip != "auto":
        return []
    out = []
    for (i, qid), reads in readings.items():
        if len(reads) != 1:
            continue
        logp = np.sort(reads[0][1])[::-1]
        if logp[0] - logp[1] < margin:
            state, questions, n_images = items[i]
            out += [build(i, qid, state, questions[qid], n_images, order) for order in extra_orders(len(logp))]
    return out


def letter_logprobs(scores: dict[str, float], letters: str, floor: float | None = None) -> np.ndarray:
    """Log-softmax over `letters` from per-letter scores; missing letters get `floor`.

    `scores` maps a letter to its logit or log-probability (any constant shift cancels).
    """
    known = [scores[c] for c in letters if c in scores]
    if not known:
        raise ValueError(f"none of the option letters {letters!r} is among the model's top tokens")
    low = floor if floor is not None else min(known) - 1.0
    v = np.array([scores.get(c, low) for c in letters], dtype=np.float64)
    return v - (v.max() + math.log(np.exp(v - v.max()).sum()))


def reading(prompt: Prompt, letter_logp: np.ndarray) -> tuple[tuple[int, ...], np.ndarray]:
    """A prompt's letter log-probabilities in RawScores option order."""
    by_option = np.empty(len(prompt.order))
    by_option[list(prompt.order)] = letter_logp
    return prompt.order, by_option


def combine(reads: list[tuple[tuple[int, ...], np.ndarray]]) -> np.ndarray:
    """Average log-probabilities over option orders; the result is used as logits."""
    return np.mean([r for _, r in reads], axis=0)


def read_all(
    items: Sequence[tuple[Any, dict[str, dict[str, Any]], int]],
    score: Callable[[list[Prompt]], list[np.ndarray]],
    flip: Flip = "auto",
    margin: float = FLIP_MARGIN,
) -> list[dict[str, np.ndarray]]:
    """Logits per question for every item, asking in other option orders where needed.

    Args:
        items: (state, wire questions, number of images) per request.
        score: Runs prompts and returns each one's `letter_logprobs`.
        flip: "auto" (reverse when the first gap is below `margin`), "always" or "never".
        margin: Gap in nat under which "auto" asks again.
    """
    readings: dict[tuple[int, str], list[tuple[tuple[int, ...], np.ndarray]]] = {}
    prompts = first_pass(items, flip)
    for p, lp in zip(prompts, score(prompts) if prompts else [], strict=True):
        readings.setdefault((p.item, p.qid), []).append(reading(p, lp))
    extra = second_pass(items, readings, flip, margin)
    for p, lp in zip(extra, score(extra) if extra else [], strict=True):
        readings[(p.item, p.qid)].append(reading(p, lp))
    out: list[dict[str, np.ndarray]] = [{} for _ in items]
    for (i, qid), reads in readings.items():
        out[i][qid] = combine(reads)
    return out


def merge_token_scores(top: Sequence[tuple[str, float]]) -> dict[str, float]:
    """Per-letter log-probabilities from (token, logprob) pairs; "A" and " A" both count as A."""
    out: dict[str, float] = {}
    for token, logp in top:
        c = token.strip()
        if len(c) == 1 and c in LETTERS:
            out[c] = float(np.logaddexp(out[c], logp)) if c in out else float(logp)
    return out


def fit_image(img: Image | PILImage.Image, max_side: int = MAX_SIDE) -> PILImage.Image:
    """RGB image resized so its sides are multiples of 32, the longer one at most `max_side`,
    and its area at least 256x256. Native and server runtimes send these exact pixels."""
    pil = img.to_pil() if isinstance(img, Image) else img
    if pil.mode != "RGB":
        pil = pil.convert("RGB")
    w, h = pil.size
    scale = min(1.0, max_side / max(w, h))
    if w * h * scale * scale < MIN_PIXELS:
        scale = math.sqrt(MIN_PIXELS / (w * h))
    nw = max(ALIGN, round(w * scale / ALIGN) * ALIGN)
    nh = max(ALIGN, round(h * scale / ALIGN) * ALIGN)
    nw, nh = min(nw, max_side // ALIGN * ALIGN), min(nh, max_side // ALIGN * ALIGN)
    return pil if (nw, nh) == (w, h) else pil.resize((nw, nh), PILImage.Resampling.BICUBIC)
