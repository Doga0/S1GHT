import math

import pytest

from s1ght.engine.fusion import Fusion
from s1ght.schema.systemone import BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer


def logit(p: float) -> float:
    return math.log(p / (1 - p))


def test_noul_average_in_logit_space() -> None:
    f = Fusion(half_life=1.0)
    f.update(1, "q", NoulAnswer(0.9), 0.0)
    f.update(1, "q", NoulAnswer(0.1), 1.0)  # one half-life later: equal weights
    assert f.state(1)["q"].noul == pytest.approx(0.5)
    f2 = Fusion(half_life=0)
    f2.update(1, "q", NoulAnswer(0.9), 0.0)
    f2.update(1, "q", NoulAnswer(0.2), 0.0)
    assert f2.state(1)["q"].noul == pytest.approx(0.2)  # no smoothing


def test_hysteresis_events() -> None:
    f = Fusion(half_life=0, on=0.6, off=0.4)
    seq = [(0.0, 0.3), (1.0, 0.7), (2.0, 0.55), (3.0, 0.95), (4.0, 0.45), (5.0, 0.35), (6.0, 0.65)]
    events = [(t, e) for t, p in seq for e in f.update(7, "no_helmet", NoulAnswer(p), t)]
    assert [(t, e.start, e.end) for t, e in events] == [(1.0, 1.0, None), (5.0, 1.0, 5.0), (6.0, 6.0, None)]
    started, ended = events[0][1], events[1][1]
    assert started.end is None, "events are snapshots, not live objects"
    assert ended.peak == pytest.approx(0.95) and ended.track_id == 7 and ended.duration == pytest.approx(4.0)


def test_min_duration() -> None:
    f = Fusion(half_life=0, on=0.6, off=0.4, min_duration=1.5)
    assert f.update(1, "q", NoulAnswer(0.8), 0.0) == []
    assert f.update(1, "q", NoulAnswer(0.8), 1.0) == []
    (e,) = f.update(1, "q", NoulAnswer(0.8), 1.5)
    assert e.start == 0.0
    f.update(2, "q", NoulAnswer(0.8), 0.0)
    assert f.update(2, "q", NoulAnswer(0.5), 1.0) == [] and f.update(2, "q", NoulAnswer(0.8), 2.0) == []  # reset


def test_choice_and_score_average() -> None:
    f = Fusion(half_life=1.0)
    f.update(1, "c", ChoiceAnswer("a", {"a": 0.8, "b": 0.2}, 0.6), 0.0)
    f.update(1, "c", ChoiceAnswer("b", {"a": 0.2, "b": 0.8}, 0.6), 2.0)  # weight 0.75 on the newer
    c = f.state(1)["c"]
    assert c.choice == "b" and c.probabilities["b"] == pytest.approx(0.65) and c.confidence == pytest.approx(0.3)

    legend = {0: "low", 1: "mid", 2: "high"}
    f.update(1, "s", ScoreAnswer(0.0, legend, {0: 1.0, 1: 0.0, 2: 0.0}, 1.0), 0.0)
    f.update(1, "s", ScoreAnswer(2.0, legend, {0: 0.0, 1: 0.0, 2: 1.0}, 1.0), 1.0)
    s = f.state(1)["s"]
    assert s.score == pytest.approx(1.0) and s.legend == legend

    box = BoxAnswer({"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.6}, 0.9)
    f.update(1, "b", box, 0.0)
    assert f.state(1)["b"] is box


def test_close_ends_open_events() -> None:
    f = Fusion(half_life=0)
    f.update(3, "q", NoulAnswer(0.9), 1.0)
    f.update(3, "other", NoulAnswer(0.1), 1.0)
    (e,) = f.close(3, 2.5)
    assert (e.question, e.start, e.end) == ("q", 1.0, 2.5)
    assert f.state(3) == {} and 3 not in f.tracks()
    assert f.close(3, 3.0) == []


def test_bad_arguments() -> None:
    with pytest.raises(ValueError):
        Fusion(on=0.4, off=0.6)
    with pytest.raises(ValueError):
        Fusion(half_life=-1)
