"""Native runtime: engine logic with a fake family (offline), and real d1 checkpoints (slow / gpu)."""

import math
import subprocess
import sys
import time

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from s1ght import Box, Choice, Image, Noul, Score
from s1ght.hub import resolve
from s1ght.runtimes.base import CapabilityError, Runtime
from s1ght.runtimes.native.engine import NativeRuntime, select_dtype, split_request
from s1ght.schema.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest

QUESTIONS = {
    "red": Noul("Is the main shape red?"),
    "shape": Choice("What shape is it?", {"circle": None, "square": "Four equal sides", "triangle": None}),
    "size": Score("How large is the shape?", ["Tiny", "Medium", "Fills the image"]),
}


def drawing(shape: str = "circle", color: tuple[int, int, int] = (220, 20, 20)) -> PILImage.Image:
    canvas = PILImage.new("RGB", (384, 384), "white")
    draw = ImageDraw.Draw(canvas)
    (draw.ellipse if shape == "circle" else draw.rectangle)((96, 96, 288, 288), fill=color)
    return canvas


class FakeFamily:
    """Answers like d1's `system_one_batch` (confidence = p_max), from fixed probabilities."""

    name = "fake"

    def __init__(self) -> None:
        self.loaded: tuple | None = None
        self.items: list = []
        self.closed = False

    def load(self, path, device, dtype) -> None:
        self.loaded = (path, device, dtype)

    def run(self, items):
        self.items.extend(items)
        out = []
        for _, qs, images in items:
            answers = {}
            for qid, q in qs.items():
                if q["type"] == "noul":
                    answers[qid] = {"type": "noul", "noul": 0.9 if images else 0.2}
                elif q["type"] == "choice":
                    names = list(q["criteria"])
                    probs = dict(zip(names, [0.7] + [0.3 / (len(names) - 1)] * (len(names) - 1)))
                    answers[qid] = {"type": "choice", "choice": names[0], "confidence": 0.7, "probabilities": probs}
                else:
                    n = len(q["criteria"])
                    probs = {str(i): 1 / n for i in range(n)}
                    legend = {str(i): t for i, t in enumerate(q["criteria"])}
                    answers[qid] = {
                        "type": "score",
                        "score": (n - 1) / 2,
                        "confidence": 1 / n,
                        "probabilities": probs,
                        "legend": legend,
                    }
            out.append((answers, 123))
        return out

    def close(self) -> None:
        self.closed = True


def fake_runtime(**kw) -> tuple[NativeRuntime, FakeFamily]:
    fam = FakeFamily()
    return NativeRuntime(resolve("d1-3b"), family=fam, path="/models/d1", device=kw.pop("device", "cpu"), **kw), fam


# --- device and dtype ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("device", "capability", "expected"),
    [
        ("cpu", None, "fp32"),
        ("cuda", (8, 0), "bf16"),  # A100
        ("cuda:1", (9, 0), "bf16"),  # H100
        ("cuda", (7, 5), "fp16"),  # T4: no bf16
        ("mps", None, "fp16"),
    ],
)
def test_select_dtype(device, capability, expected) -> None:
    table = resolve("d1-3b").options["dtype"]
    assert select_dtype(device, table, lambda d: capability) == expected


def test_select_dtype_uses_catalog_table() -> None:
    omni = resolve("d1-omni-600m").options["dtype"]
    assert select_dtype("cuda", omni, lambda d: (8, 0)) == "fp16"  # the d1-omni card advises against bf16


# --- request conversion ----------------------------------------------------------------------


def test_split_request_text_only() -> None:
    state, qs, images = split_request(SystemOneRequest("m", {"ticket": "hi"}, {"red": QUESTIONS["red"]}))
    assert state == {"ticket": "hi"} and images == []
    assert qs == {"red": {"type": "noul", "instructions": "Is the main shape red?"}}


def test_split_request_images() -> None:
    a, b = Image(drawing()), Image(drawing("square"))
    state, _, images = split_request(SystemOneRequest("m", {"before": a, "after": b, "note": "x"}, {"r": Noul("?")}))
    assert state == {"before": "[image 1]", "after": "[image 2]", "note": "x"}
    assert [im.size for im in images] == [(384, 384), (384, 384)]
    assert np.array_equal(np.asarray(images[1]), np.asarray(drawing("square")))
    # an image as the whole state keeps its label (an empty text state hurts d1-omni-600M)
    state, _, images = split_request(SystemOneRequest("m", a, {"r": Noul("?")}))
    assert state == "[image 1]" and len(images) == 1


# --- runtime with a fake family --------------------------------------------------------------


def test_native_runtime_decide_uses_s1ght_readout() -> None:
    rt, fam = fake_runtime()
    assert isinstance(rt, Runtime)
    assert fam.loaded == ("/models/d1", "cpu", "fp32")
    resp = rt.decide(SystemOneRequest("x", Image(drawing()), QUESTIONS))
    assert resp.model == "d1-3b" and resp.usage == {"input_tokens": 123, "output_tokens": 0}

    noul, choice, score = (resp.answers[k] for k in ("red", "shape", "size"))
    assert isinstance(noul, NoulAnswer) and noul.noul == pytest.approx(0.9)
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "circle"
    # TypeSafe confidence, not the model's own p_max
    assert choice.confidence == pytest.approx((0.7 - 1 / 3) / (1 - 1 / 3))
    assert isinstance(score, ScoreAnswer) and score.score == pytest.approx(1.0) and score.confidence == pytest.approx(0)
    assert score.legend == {0: "Tiny", 1: "Medium", 2: "Fills the image"}


def test_native_runtime_batch_and_raw() -> None:
    rt, fam = fake_runtime()
    reqs = [
        SystemOneRequest("x", "text", {"red": QUESTIONS["red"]}),
        SystemOneRequest("x", Image(drawing()), QUESTIONS),
    ]
    out = rt.decide_batch(reqs)
    assert [r.answers["red"].noul for r in out] == pytest.approx([0.2, 0.9])
    assert len(fam.items) == 2  # one family call for the batch
    raw = rt.raw(reqs[1])
    assert raw.scores["red"] == pytest.approx(np.log([0.9, 0.1]))
    rt.close()
    assert fam.closed


def test_native_runtime_temperature_applies() -> None:
    rt, _ = fake_runtime()
    rt.temperature = {"red": 2.0}
    p = rt.decide(SystemOneRequest("x", Image(drawing()), {"red": QUESTIONS["red"]})).answers["red"].noul
    assert p == pytest.approx(1 / (1 + math.exp(-math.log(9) / 2)))


def test_native_runtime_rejects_box_and_calibration() -> None:
    rt, _ = fake_runtime()
    with pytest.raises(CapabilityError, match="box"):
        rt.decide(SystemOneRequest("x", Image(drawing()), {"b": Box("the circle")}))
    raws = rt.raw_batch([SystemOneRequest("x", "text", {"r": Noul("?")})] * 2)
    assert len(raws) == 2 and set(raws[0].scores) == {"r"}
    with pytest.raises(ValueError, match="native spec"):
        NativeRuntime(resolve("d1-3b", runtime="ollaya"), family=FakeFamily(), path="/x")


def test_import_still_light() -> None:
    code = "import sys, s1ght, s1ght.model, s1ght.runtimes.native.engine; print('torch' in sys.modules)"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "False"


# --- real checkpoints ------------------------------------------------------------------------

REAL = ["d1-3b", "d1-omni-600m"]


def _real_questions():
    return {
        "has_circle": Noul("Does the image show a red circle?"),
        "color": Choice("What color is the main shape?", {"red": None, "green": None, "blue": None}),
    }


def _check_real(model) -> dict:
    """A red circle and a blue square must be told apart; returns timings and answers."""
    t0 = time.perf_counter()
    red = model.decide(Image(drawing("circle", (220, 20, 20))), _real_questions())
    t1 = time.perf_counter()
    blue = model.decide(
        {"photo": Image(drawing("square", (20, 40, 220))), "note": "Look at `photo`."}, _real_questions()
    )
    t2 = time.perf_counter()
    text = model.decide(
        "I was charged twice this month, please refund one.", {"refund": Noul("Is a refund requested?")}
    )
    assert red.answers["has_circle"].noul > 0.5 > blue.answers["has_circle"].noul
    assert red.answers["color"].choice == "red" and blue.answers["color"].choice == "blue"
    assert text.answers["refund"].noul > 0.5
    return {"red": red, "blue": blue, "text": text, "first_s": t1 - t0, "warm_s": t2 - t1}


@pytest.mark.slow
@pytest.mark.network
@pytest.mark.parametrize("name", REAL)
def test_real_cpu_fp32(name) -> None:
    """Downloads the pinned checkpoint and answers on CPU in fp32 (d1-3b: 6.3 GB, ~12.5 GB RAM)."""
    pytest.importorskip("torch")
    from s1ght import S1GHT

    with S1GHT(name, device="cpu") as model:
        assert model.runtime.dtype == "fp32"
        r = _check_real(model)
    print(
        f"\n{name} cpu fp32: first {r['first_s']:.1f}s warm {r['warm_s']:.1f}s",
        {k: v.to_wire()["answers"] for k, v in r.items() if k in ("red", "blue")},
    )


@pytest.mark.gpu
@pytest.mark.network
@pytest.mark.parametrize("name", REAL)
def test_real_gpu_matches_fp32(name) -> None:
    """On a GPU, the catalog dtype (A100: bf16, T4: fp16) gives the same answers as fp32."""
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("no CUDA device")
    from s1ght import S1GHT

    results = {}
    for dtype in (None, "fp32"):
        with S1GHT(name, device="cuda", dtype=dtype) as model:
            results[model.runtime.dtype] = _check_real(model)
    (fast_dtype, fast), (_, ref) = results.items()
    for key in ("red", "blue"):
        for qid in ("has_circle",):
            assert fast[key].answers[qid].noul == pytest.approx(ref[key].answers[qid].noul, abs=0.05)
        assert fast[key].answers["color"].choice == ref[key].answers["color"].choice
    gpu = torch.cuda.get_device_name()
    print(f"\n{name} {gpu} {fast_dtype}: warm {fast['warm_s'] * 1000:.0f} ms, fp32: {ref['warm_s'] * 1000:.0f} ms")
