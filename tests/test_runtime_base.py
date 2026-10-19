import io
import math

import numpy as np
import pytest
from PIL import Image as PILImage

from conftest import FakeRuntime
from s1ght import Box, Choice, Image, Noul, Score
from s1ght.runtimes.base import (
    Capabilities,
    CapabilityError,
    RawScores,
    Runtime,
    RuntimeBase,
    confidence,
    require,
    score_to_answer,
    scores_to_answers,
    validate_request,
)
from s1ght.schema.systemone import (
    BoxAnswer,
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

QUESTIONS = {
    "rain": Noul("Is it raining in `photo`?"),
    "sky": Choice("What is the sky like?", ["clear", "cloudy", "overcast"]),
    "wind": Score("How windy is it?", ["Calm", "Breezy", "Stormy"]),
    "umbrella": Box("the umbrella in `photo`"),
}


def png(color: tuple[int, int, int], size: tuple[int, int] = (300, 300)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


def request(color: tuple[int, int, int] = (10, 20, 30), questions=None) -> SystemOneRequest:
    return SystemOneRequest("fake-1", {"photo": Image(png(color))}, questions or QUESTIONS)


def assert_valid(question, answer) -> None:
    """Answer has the type and value ranges TypeSafe defines for the question."""
    if isinstance(question, Noul):
        assert isinstance(answer, NoulAnswer) and 0 <= answer.noul <= 1
    elif isinstance(question, Choice):
        assert isinstance(answer, ChoiceAnswer)
        assert set(answer.probabilities) == set(question.criteria)
        assert math.isclose(sum(answer.probabilities.values()), 1.0)
        assert answer.choice == max(answer.probabilities, key=answer.probabilities.get)
        assert 0 <= answer.confidence <= 1
    elif isinstance(question, Score):
        assert isinstance(answer, ScoreAnswer)
        assert set(answer.probabilities) == set(range(len(question.criteria)))
        assert math.isclose(sum(answer.probabilities.values()), 1.0)
        assert 0 <= answer.score <= len(question.criteria) - 1
        assert answer.legend == dict(enumerate(question.criteria))
        assert 0 <= answer.confidence <= 1
    elif isinstance(question, Box):
        assert isinstance(answer, BoxAnswer)
        b = answer.box
        assert 0 <= b["left"] < b["right"] <= 1 and 0 <= b["top"] < b["bottom"] <= 1
    else:
        raise TypeError(type(question))


# --- readout ---------------------------------------------------------------------------------


def test_noul_readout_is_sigmoid_of_logit_difference() -> None:
    a = score_to_answer(Noul("?"), np.array([2.0, -1.0]))
    assert a.noul == pytest.approx(1 / (1 + math.exp(-3.0)))


def test_choice_readout_and_typesafe_confidence() -> None:
    a = score_to_answer(Choice("?", ["a", "b", "c", "d"]), np.log([0.7, 0.1, 0.1, 0.1]))
    assert a.choice == "a"
    assert a.probabilities["a"] == pytest.approx(0.7)
    assert a.confidence == pytest.approx((0.7 - 0.25) / 0.75)


def test_score_readout_expected_level() -> None:
    a = score_to_answer(Score("?", ["lo", "mid", "hi"]), np.log([0.05, 0.5, 0.45]))
    assert a.score == pytest.approx(0.5 + 2 * 0.45)
    assert a.legend == {0: "lo", 1: "mid", 2: "hi"}


def test_temperature_softens() -> None:
    q = Noul("?")
    sharp = score_to_answer(q, np.array([2.0, 0.0])).noul
    soft = score_to_answer(q, np.array([2.0, 0.0]), temperature=4.0).noul
    assert 0.5 < soft < sharp
    per_q = scores_to_answers({"x": q}, RawScores({"x": np.array([2.0, 0.0])}), {"x": 4.0})
    assert per_q["x"].noul == pytest.approx(soft)


def test_confidence_bounds() -> None:
    assert confidence(np.full(4, 0.25)) == pytest.approx(0.0)
    assert confidence(np.array([1.0, 0.0, 0.0])) == pytest.approx(1.0)


@pytest.mark.parametrize(
    ("question", "logits", "err"),
    [
        (Noul("?"), [1.0, 2.0, 3.0], ValueError),
        (Choice("?", ["a", "b"]), [1.0], ValueError),
        (Score("?", ["a", "b"]), [1.0, 2.0, 3.0], ValueError),
        (Box("x"), [1.0], TypeError),
    ],
)
def test_readout_shape_errors(question, logits, err) -> None:
    with pytest.raises(err):
        score_to_answer(question, np.array(logits))
    with pytest.raises(ValueError):
        score_to_answer(Noul("?"), np.zeros(2), temperature=0)


# --- capabilities ----------------------------------------------------------------------------


def test_require() -> None:
    caps = Capabilities(images=True)
    require(caps, {"images"}, "x")
    with pytest.raises(CapabilityError, match="attention"):
        require(caps, frozenset({"images", "attention"}), "the attention localizer")
    with pytest.raises(ValueError, match="unknown"):
        require(caps, {"telepathy"}, "x")


def test_validate_request() -> None:
    text_only = Capabilities()
    validate_request(text_only, SystemOneRequest("m", "text", {"q": Noul("?")}))
    with pytest.raises(CapabilityError, match="images"):
        validate_request(text_only, request(questions={"q": Noul("?")}))
    with pytest.raises(CapabilityError, match="box"):
        validate_request(Capabilities(images=True), request())
    with pytest.raises(ValueError, match="options"):
        validate_request(Capabilities(max_options=2), SystemOneRequest("m", "t", {"c": Choice("?", ["a", "b", "c"])}))
    small = SystemOneRequest("m", Image(png((0, 0, 0), (100, 400))), {"q": Noul("?")})
    with pytest.raises(ValueError, match="280-1568"):
        validate_request(Capabilities(images=True, image_px=(280, 1568)), small)


# --- runtime protocol + FakeRuntime ---------------------------------------------------------


def test_fake_runtime_answers_every_question_type(fake_runtime: FakeRuntime) -> None:
    assert isinstance(fake_runtime, Runtime)
    resp = fake_runtime.decide(request())
    assert isinstance(resp, SystemOneResponse)
    assert set(resp.answers) == set(QUESTIONS)
    for qid, q in QUESTIONS.items():
        assert_valid(q, resp.answers[qid])
    assert resp.to_wire()["answers"]["umbrella"]["type"] == "box"


def test_fake_runtime_is_deterministic(fake_runtime: FakeRuntime) -> None:
    a = fake_runtime.decide(request((10, 20, 30))).to_wire()
    b = FakeRuntime().decide(request((10, 20, 30))).to_wire()  # new Image objects, new refs
    c = fake_runtime.decide(request((200, 20, 30))).to_wire()
    assert a == b
    assert a != c


def test_fake_runtime_batch_and_raw(fake_runtime: FakeRuntime) -> None:
    reqs = [request((i, i, i)) for i in range(3)]
    out = fake_runtime.decide_batch(reqs)
    assert [r.to_wire() for r in out] == [fake_runtime.decide(r).to_wire() for r in reqs]
    assert fake_runtime.calls == ["batch:3", "decide", "decide", "decide"]
    raw = fake_runtime.raw(reqs[0])
    assert set(raw.scores) == {"rain", "sky", "wind"}
    assert raw.scores["sky"].shape == (3,)


def test_fake_runtime_respects_caps() -> None:
    rt = FakeRuntime(Capabilities(images=True))
    with pytest.raises(CapabilityError):
        rt.decide(request())


def test_runtime_base_defaults() -> None:
    class Plain(RuntimeBase):
        pass

    with Plain() as rt:
        assert isinstance(rt, Runtime)
        with pytest.raises(NotImplementedError):
            rt.raw(request())
        with pytest.raises(NotImplementedError):
            rt.decide(request())
