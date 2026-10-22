"""Plug your own decision back end into S1GHT. Runs offline, with no model download.

A runtime returns logits per question (`RawScores`); S1GHT turns them into TypeSafe answers,
calibrates them and runs `predict`, `track` and `serve` on top. This toy runtime "looks" at
the mean colour of each image.

    python examples/custom_runtime.py
"""

import numpy as np

from s1ght import S1GHT, Choice, Image, Noul
from s1ght.runtimes.base import Capabilities, RawScores, RuntimeBase, scores_to_answers, validate_request
from s1ght.schema.questions import Choice as ChoiceQ
from s1ght.schema.questions import Noul as NoulQ
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse

COLORS = {"red": (0, 0, 255), "green": (0, 255, 0), "blue": (255, 0, 0)}  # BGR


class ColorRuntime(RuntimeBase):
    """Answers Noul "is it red?" and Choice "which colour?" from the first image's mean colour."""

    caps = Capabilities(images=True, raw_scores=True, max_options=16)
    model = "color-1"

    def raw(self, req: SystemOneRequest) -> RawScores:
        validate_request(self.caps, req)
        images = list(req.assets().values())
        mean = images[0].to_array().reshape(-1, 3).mean(axis=0) if images else np.zeros(3)
        scores = {}
        for qid, q in req.questions.items():
            if isinstance(q, ChoiceQ):
                # closer colour, higher logit
                scores[qid] = np.array([-np.linalg.norm(mean - COLORS.get(o, (128,) * 3)) / 40 for o in q.criteria])
            elif isinstance(q, NoulQ):
                redness = (mean[2] - mean[:2].mean()) / 40
                scores[qid] = np.array([redness, -redness])  # (yes, no)
        return RawScores(scores)

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        return SystemOneResponse(self.model, scores_to_answers(req.questions, self.raw(req)), {"input_tokens": 0})


def swatch(bgr: tuple[int, int, int]) -> np.ndarray:
    img = np.zeros((64, 64, 3), np.uint8)
    img[:] = bgr
    return img


model = S1GHT(ColorRuntime())
questions = {"red": Noul("Is `image` red?"), "color": Choice("Which colour is `image`?", list(COLORS))}

r = model.decide({"image": Image(swatch((40, 30, 220)))}, questions)
print("decide:", r.to_wire()["answers"])

for res in model.predict([swatch(c) for c in COLORS.values()], questions=questions):
    a = res.answers[None]
    print(f"predict {res.path}: red={a['red'].noul:.2f} color={a['color'].choice}")
