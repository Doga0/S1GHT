"""Calibration: fit a temperature per question type and input kind on held-out data.

A raw logit difference is not a probability. For each (question type, input kind) pair, e.g.
"noul/crop" or "choice/full_frame", one temperature T is fitted by minimizing the negative
log-likelihood of softmax(logits / T) on labeled examples the model was not trained on.
Crops and full frames are different distributions and get separate temperatures.

Reported numbers are never measured on the data T was fitted on: with a "test" split the
profile is fitted on "calib" and evaluated on "test"; without one, on out-of-fold predictions
of a k-fold split of "calib" (folds by `group`, so crops of one image stay together).

Dataset format: JSON Lines, one example per line, paths relative to the file:

    {"image": "images/a.png", "box": [x1, y1, x2, y2], "pad": 0.5,
     "question": {"type": "noul", "instructions": "Is the person in `image` wearing a hard hat?"},
     "label": true, "split": "calib", "group": "a.png"}

`box` makes it a crop example (state `{"image": crop}`); without it the whole image is the
state; without `image` the `state` field (text/JSON) is used. Labels: true/false for noul, the
option name for choice, the level index for score.
"""

from __future__ import annotations

import datetime as dt
import itertools
import json
import math
import re
import warnings
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np

from s1ght.engine.crops import crop
from s1ght.runtimes.base import Capabilities, RawScores, Runtime, RuntimeBase, scores_to_answers
from s1ght.schema.assets import Image
from s1ght.schema.questions import Box, Choice, Noul, Question, Score, question_from_wire
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse

PROFILE_VERSION = 1
T_RANGE = (0.05, 20.0)
MIN_FIT = 30  # fewer examples than this per key draw a warning
ROLLING_ALIAS = re.compile(r"(^|[:\-_/])(latest|preview)$")


# --------------------------------------------------------------------------- data


@dataclass
class Example:
    """One labeled question about one state."""

    state: Any
    question: Question
    label: Any
    input: str  # "crop", "full_frame" or "text"
    split: str = "calib"
    group: str | None = None

    @property
    def key(self) -> str:
        return f"{_type(self.question)}/{self.input}"


def _type(q: Question) -> str:
    return {Noul: "noul", Choice: "choice", Score: "score", Box: "box"}[type(q)]


def label_index(q: Question, label: Any) -> int:
    """The option index a label stands for (`RawScores` order: noul is (yes, no))."""
    if isinstance(q, Noul):
        if not isinstance(label, bool):
            raise ValueError(f"noul labels must be true/false, got {label!r}")  # noqa: TRY004 (bad data, not a bug)
        return 0 if label else 1
    if isinstance(q, Choice):
        options = list(q.criteria)
        if label not in options:
            raise ValueError(f"label {label!r} is not one of {options}")
        return options.index(label)
    if isinstance(q, Score):
        if not isinstance(label, int) or not 0 <= label < len(q.criteria):
            raise ValueError(f"score labels are level indices 0..{len(q.criteria) - 1}, got {label!r}")
        return label
    raise TypeError("Box questions are not calibrated")


def load_examples(path: str | Path) -> list[Example]:
    """Read a JSON Lines dataset (see module docstring); crops are cut when the file is read."""
    from s1ght.io.sources import read_image

    path = Path(path)
    root = path.parent
    images: dict[str, np.ndarray] = {}
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        q = question_from_wire(row["question"])
        label_index(q, row["label"])  # validate early, with the line number below on failure
        if "image" in row:
            rel = row["image"]
            if rel not in images:
                images[rel] = read_image(root / rel)
            img = images[rel]
            if "box" in row:
                state, kind = {"image": Image(crop(img, row["box"], row.get("pad", 0.2)))}, "crop"
            else:
                state, kind = {"image": Image(img)}, "full_frame"
        elif "state" in row:
            state, kind = row["state"], "text"
        else:
            raise ValueError(f"{path}:{n}: needs image or state")
        out.append(Example(state, q, row["label"], kind, row.get("split", "calib"), row.get("group")))
    if not out:
        raise ValueError(f"{path} has no examples")
    return out


# --------------------------------------------------------------------------- scoring


@dataclass
class Scored:
    key: str
    logits: np.ndarray
    label: int
    split: str
    group: str | None


def collect(
    runtime: Runtime,
    examples: list[Example],
    *,
    batch: int = 8,
    progress: Callable[[int, int], None] | None = None,
) -> list[Scored]:
    """Raw scores for every example (uses `raw_batch` when the runtime has it)."""
    if not runtime.caps.raw_scores:
        raise ValueError("calibration needs a runtime with raw scores")
    out: list[Scored] = []
    raw_batch = getattr(runtime, "raw_batch", None)
    for start in range(0, len(examples), batch):
        chunk = examples[start : start + batch]
        reqs = [SystemOneRequest("calibration", e.state, {"q": e.question}) for e in chunk]
        raws = raw_batch(reqs) if raw_batch is not None else [runtime.raw(r) for r in reqs]
        for e, raw in zip(chunk, raws, strict=True):
            out.append(
                Scored(e.key, np.asarray(raw.scores["q"], float), label_index(e.question, e.label), e.split, e.group)
            )
        if progress is not None:
            progress(min(start + batch, len(examples)), len(examples))
    return out


def _log_softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=-1, keepdims=True)
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def nll(logits: list[np.ndarray], labels: list[int], t: float) -> float:
    return -float(np.mean([_log_softmax(z / t)[y] for z, y in zip(logits, labels, strict=True)]))


def fit_temperature(logits: list[np.ndarray], labels: list[int]) -> float:
    """The temperature minimizing NLL, by golden-section search over log T in [0.05, 20]."""
    if not logits:
        return 1.0
    lo, hi = math.log(T_RANGE[0]), math.log(T_RANGE[1])
    g = (math.sqrt(5) - 1) / 2
    a, b = hi - g * (hi - lo), lo + g * (hi - lo)
    fa, fb = nll(logits, labels, math.exp(a)), nll(logits, labels, math.exp(b))
    for _ in range(60):
        if fa < fb:
            hi, b, fb = b, a, fa
            a = hi - g * (hi - lo)
            fa = nll(logits, labels, math.exp(a))
        else:
            lo, a, fa = a, b, fb
            b = lo + g * (hi - lo)
            fb = nll(logits, labels, math.exp(b))
    return math.exp((lo + hi) / 2)


def metrics(logits: list[np.ndarray], labels: list[int], t: float = 1.0, bins: int = 10) -> dict[str, Any]:
    """Accuracy, top-label ECE (equal-width bins), Brier score, NLL and the reliability table."""
    probs = [np.exp(_log_softmax(z / t)) for z in logits]
    conf = np.array([p.max() for p in probs])
    correct = np.array([int(p.argmax() == y) for p, y in zip(probs, labels, strict=True)], float)
    brier = float(np.mean([np.sum((p - np.eye(len(p))[y]) ** 2) for p, y in zip(probs, labels, strict=True)]))
    table, ece = [], 0.0
    edges = np.linspace(0, 1, bins + 1)
    for i in range(bins):
        m = (conf > edges[i]) & (conf <= edges[i + 1]) if i else (conf >= 0) & (conf <= edges[1])
        if m.any():
            gap = abs(conf[m].mean() - correct[m].mean())
            ece += m.mean() * gap
            table.append({"lo": float(edges[i]), "hi": float(edges[i + 1]), "n": int(m.sum()),
                          "confidence": float(conf[m].mean()), "accuracy": float(correct[m].mean())})  # fmt: skip
    return {
        "n": len(labels),
        "accuracy": float(correct.mean()) if len(labels) else 0.0,
        "ece": float(ece),
        "brier": brier,
        "nll": nll(logits, labels, t) if labels else 0.0,
        "reliability": table,
    }


def _folds(items: list[Scored], k: int) -> list[list[int]]:
    groups: dict[str, list[int]] = {}
    for i, s in enumerate(items):
        groups.setdefault(s.group or f"#{i}", []).append(i)
    order = sorted(groups, key=lambda g: hash_str(g))
    folds: list[list[int]] = [[] for _ in range(min(k, len(order)))]
    for j, g in enumerate(order):
        folds[j % len(folds)].extend(groups[g])
    return folds


def hash_str(s: str) -> int:
    """A stable (not per-process) string hash for reproducible folds."""
    h = 2166136261
    for ch in s.encode():
        h = ((h ^ ch) * 16777619) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- profiles


@dataclass
class CalibrationProfile:
    """Temperatures for one model, keyed "<question type>/<input kind>", with how they were measured."""

    model: str
    runtime: str
    source: str
    temperatures: dict[str, float]
    metrics: dict[str, dict[str, Any]] = field(default_factory=dict)
    dataset: dict[str, Any] = field(default_factory=dict)
    created: str = ""
    version: int = PROFILE_VERSION

    def temperature(self, question: Question, input_kind: str) -> float:
        """T for a question; falls back to the same type on any input, then 1.0."""
        kind = _type(question)
        if f"{kind}/{input_kind}" in self.temperatures:
            return self.temperatures[f"{kind}/{input_kind}"]
        same = [t for k, t in self.temperatures.items() if k.startswith(f"{kind}/")]
        return float(np.exp(np.mean(np.log(same)))) if same else 1.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "s1ght_profile": self.version,
            "model": self.model,
            "runtime": self.runtime,
            "source": self.source,
            "created": self.created,
            "temperatures": self.temperatures,
            "metrics": self.metrics,
            "dataset": self.dataset,
        }

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=1), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> CalibrationProfile:
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        if d.get("s1ght_profile") != PROFILE_VERSION:
            raise ValueError(f"{path} is not a version-{PROFILE_VERSION} S1GHT calibration profile")
        return cls(
            model=d["model"],
            runtime=d["runtime"],
            source=d["source"],
            temperatures={k: float(v) for k, v in d["temperatures"].items()},
            metrics=d.get("metrics", {}),
            dataset=d.get("dataset", {}),
            created=d.get("created", ""),
        )

    def report(self) -> str:
        """A plain-text table of ECE / Brier / accuracy before and after, per key."""
        header = f"{'key':22s} {'n':>5s} {'T':>6s}  {'ECE before→after':>18s}  {'Brier before→after':>19s}  {'acc':>5s}"
        lines = [f"{self.model} ({self.runtime}, {self.source})", header]
        for key, m in sorted(self.metrics.items()):
            b, a = m["before"], m["after"]
            lines.append(
                f"{key:22s} {a['n']:5d} {self.temperatures[key]:6.2f}  {b['ece']:8.3f} → {a['ece']:6.3f}  "
                f"{b['brier']:9.3f} → {a['brier']:6.3f}  {a['accuracy']:5.2f}"
            )
        return "\n".join(lines)


def check_pinned(model: str, source: str) -> None:
    """Refuse to bind a profile to a rolling alias ("jev-latest", "clef:latest", "jev-preview")."""
    for name in (model, source):
        if ROLLING_ALIAS.search(name.strip().lower()):
            raise ValueError(
                f"{name!r} is a rolling alias; a calibration profile must be bound to a fixed model version"
            )


def calibrate(
    runtime: Runtime,
    examples: list[Example],
    *,
    model: str,
    runtime_name: str,
    source: str,
    folds: int = 5,
    batch: int = 8,
    dataset: dict[str, Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> CalibrationProfile:
    """Fit one temperature per question type and input kind, and measure ECE/Brier before and after.

    Args:
        runtime: A runtime with raw scores (native or server).
        examples: Labeled examples (`load_examples`); Box questions are skipped.
        model, runtime_name, source: What the profile is bound to (`source` should pin the version).
        folds: k for out-of-fold evaluation when there is no "test" split.
        batch: Requests per runtime call.
        dataset: Description stored in the profile (path, sha256, ...).
    """
    check_pinned(model, source)
    scored = collect(runtime, [e for e in examples if not isinstance(e.question, Box)], batch=batch, progress=progress)
    temps: dict[str, float] = {}
    report: dict[str, dict[str, Any]] = {}
    for key in sorted({s.key for s in scored}):
        items = [s for s in scored if s.key == key]
        calib = [s for s in items if s.split != "test"]
        test = [s for s in items if s.split == "test"]
        if not calib:
            continue
        if len(calib) < MIN_FIT:
            warnings.warn(f"{key}: only {len(calib)} examples to fit a temperature on; it may be far off", stacklevel=2)
        temps[key] = fit_temperature([s.logits for s in calib], [s.label for s in calib])
        if test:
            ev_logits, ev_labels = [s.logits for s in test], [s.label for s in test]
            ev_t = [temps[key]] * len(test)
            how = "test split"
        else:
            ev_logits, ev_labels, ev_t = [], [], []
            for fold in _folds(calib, folds):
                held = set(fold)
                fit = [s for i, s in enumerate(calib) if i not in held]
                t = fit_temperature([s.logits for s in fit], [s.label for s in fit]) if fit else 1.0
                for i in fold:
                    ev_logits.append(calib[i].logits)
                    ev_labels.append(calib[i].label)
                    ev_t.append(t)
            how = f"{min(folds, len(calib))}-fold out-of-fold"
        scaled = [z / t for z, t in zip(ev_logits, ev_t, strict=True)]
        report[key] = {
            "evaluated_on": how,
            "n_fit": len(calib),
            "before": metrics(ev_logits, ev_labels, 1.0),
            "after": metrics(scaled, ev_labels, 1.0),
        }
    if not temps:
        raise ValueError("no calibratable examples (noul, choice or score with a calib split)")
    return CalibrationProfile(
        model=model,
        runtime=runtime_name,
        source=source,
        temperatures=temps,
        metrics=report,
        dataset=dataset or {},
        created=dt.datetime.now(dt.UTC).isoformat(timespec="seconds"),
    )


# --------------------------------------------------------------------------- applying


def input_kind(req: SystemOneRequest, default: str) -> str:
    return default if req.assets() else "text"


class CalibratedRuntime(RuntimeBase):
    """Wraps a runtime so every noul/choice/score answer uses the profile's temperature.

    Args:
        inner: Any runtime with raw scores.
        profile: The calibration profile.
        input_kind: "full_frame" (predict/decide) or "crop" (track); requests without images use "text".
    """

    def __init__(self, inner: Runtime, profile: CalibrationProfile, input_kind: str = "full_frame") -> None:
        if not inner.caps.raw_scores:
            raise ValueError("calibration needs a runtime with raw scores")
        self.inner, self.profile, self.input_kind = inner, profile, input_kind
        self.caps: Capabilities = inner.caps
        self.model = getattr(inner, "model", profile.model)

    def with_input(self, kind: str) -> CalibratedRuntime:
        return CalibratedRuntime(self.inner, self.profile, kind)

    def _answers(self, req: SystemOneRequest, raw: RawScores) -> SystemOneResponse:
        kind = input_kind(req, self.input_kind)
        temps = {qid: self.profile.temperature(req.questions[qid], kind) for qid in raw.scores}
        answers = dict(scores_to_answers(req.questions, raw, temps))
        boxes: dict[str, Question] = {qid: q for qid, q in req.questions.items() if isinstance(q, Box)}
        if boxes:
            answers.update(self.inner.decide(replace(req, questions=boxes)).answers)
        return SystemOneResponse(model=str(self.model), answers=answers, usage={})

    def raw(self, req: SystemOneRequest) -> RawScores:
        return self.inner.raw(req)

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        return self._answers(req, self.inner.raw(req))

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        raw_batch = getattr(self.inner, "raw_batch", None)
        raws = raw_batch(reqs) if raw_batch is not None else [self.inner.raw(r) for r in reqs]
        return [self._answers(r, raw) for r, raw in zip(reqs, raws, strict=True)]

    def close(self) -> None:
        self.inner.close()

    def __repr__(self) -> str:
        return f"CalibratedRuntime({self.inner!r}, input={self.input_kind!r})"


def save_reliability(profile: CalibrationProfile, path: str | Path, size: int = 360) -> Path:
    """Reliability diagrams (accuracy vs confidence per bin, before and after) as one PNG row."""
    import cv2

    panels = []
    for key, m in sorted(profile.metrics.items()):
        img = np.full((size, size, 3), 255, np.uint8)
        pad, w = 40, size - 60
        org = (pad, size - pad)

        def pt(x: float, y: float, org: tuple[int, int] = org, w: int = w) -> tuple[int, int]:
            return int(org[0] + x * w), int(org[1] - y * w)

        cv2.rectangle(img, pt(0, 0), pt(1, 1), (200, 200, 200), 1)
        cv2.line(img, pt(0, 0), pt(1, 1), (160, 160, 160), 1, cv2.LINE_AA)
        for name, color in (("before", (60, 60, 230)), ("after", (200, 120, 30))):
            pts = [pt(b["confidence"], b["accuracy"]) for b in m[name]["reliability"]]
            for a, b in itertools.pairwise(pts):
                cv2.line(img, a, b, color, 2, cv2.LINE_AA)
            for p in pts:
                cv2.circle(img, p, 4, color, -1, cv2.LINE_AA)
        cv2.putText(img, key, (pad, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1, cv2.LINE_AA)
        txt = f"ECE {m['before']['ece']:.3f} -> {m['after']['ece']:.3f}"
        cv2.putText(img, txt, (pad, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)
        cv2.putText(img, "before", (size - 110, 46), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (60, 60, 230), 1, cv2.LINE_AA)
        cv2.putText(img, "after", (size - 110, 64), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 120, 30), 1, cv2.LINE_AA)
        panels.append(img)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.hstack(panels) if panels else np.full((size, size, 3), 255, np.uint8))
    return path
