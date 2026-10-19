"""Shared fixtures. `FakeRuntime` answers every question type deterministically, without GPU or network."""

from __future__ import annotations

import hashlib
import json

import numpy as np
import pytest

from s1ght.runtimes.base import Capabilities, RawScores, RuntimeBase, scores_to_answers, validate_request
from s1ght.schema.questions import Box, Choice, Noul, Score
from s1ght.schema.systemone import BoxAnswer, SystemOneRequest, SystemOneResponse


def _seed(*parts: object) -> int:
    key = json.dumps([str(p) for p in parts]).encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:8], "little")


class FakeRuntime(RuntimeBase):
    """Deterministic runtime: logits depend only on (model, state text, question id, option).

    Images in the state are keyed by their bytes, so the same picture gives the same answer.
    Every call is recorded in `calls` for scheduler tests.
    """

    def __init__(self, caps: Capabilities | None = None, model: str = "fake-1") -> None:
        self.caps = caps or Capabilities(images=True, box=True, batch=True, raw_scores=True)
        self.model = model
        self.calls: list[str] = []
        self.closed = False

    def _key(self, req: SystemOneRequest) -> str:
        body, _ = SystemOneRequest(req.model, req.state, {}).to_wire()
        key = json.dumps(body["state"], sort_keys=True)
        for ref, img in req.assets().items():
            key = key.replace(ref, hashlib.sha256(img.to_bytes()).hexdigest())
        return key

    def raw(self, req: SystemOneRequest) -> RawScores:
        validate_request(self.caps, req)
        key = self._key(req)
        scores = {}
        for qid, q in req.questions.items():
            if isinstance(q, Noul):
                n = 2
            elif isinstance(q, (Choice, Score)):
                n = len(q.criteria)
            else:
                continue
            scores[qid] = np.random.default_rng(_seed(self.model, key, qid)).normal(0, 2, n)
        return RawScores(scores)

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        self.calls.append("decide")
        answers = scores_to_answers(req.questions, self.raw(req))
        for qid, q in req.questions.items():
            if isinstance(q, Box):
                a, b = np.sort(
                    np.random.default_rng(_seed(self.model, self._key(req), qid)).uniform(0, 1, (2, 2)), axis=0
                )
                edges = {"top": a[0], "bottom": b[0], "left": a[1], "right": b[1]}
                answers[qid] = BoxAnswer(box={k: float(v) for k, v in edges.items()}, confidence=0.5)
        return SystemOneResponse(model=self.model, answers=answers, usage={"input_tokens": 0, "output_tokens": 0})

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        self.calls.append(f"batch:{len(reqs)}")
        out = [self.decide(r) for r in reqs]
        del self.calls[-len(reqs) :]
        return out

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def fake_runtime() -> FakeRuntime:
    return FakeRuntime()
