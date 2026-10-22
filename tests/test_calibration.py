"""Step 9: temperature fitting, metrics, profiles, CalibratedRuntime and S1GHT.calibrate."""

import json
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, Box, Choice, Image, Noul, Score
from s1ght.engine.calibration import (
    CalibratedRuntime,
    CalibrationProfile,
    Example,
    calibrate,
    check_pinned,
    fit_temperature,
    label_index,
    load_examples,
    metrics,
    save_reliability,
)
from s1ght.runtimes.base import Capabilities, RawScores, RuntimeBase
from s1ght.schema.systemone import BoxAnswer, SystemOneResponse

NOUL = Noul("Is the person in `image` wearing a hard hat?")


def overconfident(n: int, true_t: float = 3.0, seed: int = 0) -> tuple[list[np.ndarray], list[int]]:
    """Binary logits whose labels follow softmax(logits / true_t): the model is overconfident by true_t."""
    rng = np.random.default_rng(seed)
    logits, labels = [], []
    for _ in range(n):
        d = rng.normal(0, 6)
        z = np.array([d / 2, -d / 2])
        p_yes = 1 / (1 + math.exp(-d / true_t))
        logits.append(z)
        labels.append(0 if rng.random() < p_yes else 1)
    return logits, labels


# --- fitting and metrics ---------------------------------------------------------------------


def test_fit_recovers_known_temperature() -> None:
    logits, labels = overconfident(4000, true_t=3.0)
    assert fit_temperature(logits, labels) == pytest.approx(3.0, rel=0.1)
    assert fit_temperature([], []) == 1.0


def test_temperature_reduces_ece_and_brier() -> None:
    logits, labels = overconfident(2000, true_t=3.0)
    t = fit_temperature(logits, labels)
    before, after = metrics(logits, labels, 1.0), metrics(logits, labels, t)
    assert after["ece"] < before["ece"] / 2 and after["brier"] < before["brier"] and after["nll"] < before["nll"]
    assert before["accuracy"] == after["accuracy"]  # scaling never changes the argmax
    assert sum(b["n"] for b in after["reliability"]) == 2000


def test_metrics_perfect_and_multiclass() -> None:
    m = metrics([np.array([10.0, -10.0, -10.0]), np.array([-10.0, 10.0, -10.0])], [0, 1])
    assert m["accuracy"] == 1.0 and m["ece"] < 1e-6 and m["brier"] < 1e-6


def test_label_index() -> None:
    assert label_index(NOUL, True) == 0 and label_index(NOUL, False) == 1
    assert label_index(Choice("?", ["a", "b", "c"]), "c") == 2
    assert label_index(Score("?", ["lo", "mid", "hi"]), 1) == 1
    for q, bad in ((NOUL, "yes"), (Choice("?", ["a", "b"]), "z"), (Score("?", ["lo", "hi"]), 2)):
        with pytest.raises(ValueError):
            label_index(q, bad)
    with pytest.raises(TypeError):
        label_index(Box("x"), True)


@pytest.mark.parametrize("name", ["jev-latest", "clef:latest", "jev-preview", "org/model-latest"])
def test_rolling_aliases_are_refused(name) -> None:
    with pytest.raises(ValueError, match="rolling alias"):
        check_pinned(name, "x")
    check_pinned("jev-1.14.0", "hf://LiquidAI/d1-3B@051bcc464b01b9f92942b364d9586b0ef5912432")


# --- calibrate() on a runtime ------------------------------------------------------------------


class OverconfidentRuntime(RuntimeBase):
    """Logits ±4 for 'helmet' crops (bright top half) with 20% of labels flipped: badly overconfident."""

    caps = Capabilities(images=True, batch=True, raw_scores=True)
    model = "overconfident-1"

    def raw(self, req):
        if isinstance(req.state, str):  # text examples
            return RawScores({qid: np.array([-1.0, 1.0]) for qid in req.questions})
        img = req.state["image"].to_array()
        yes = img[: img.shape[0] // 2].mean() > 128
        return RawScores({qid: np.array([4.0, -4.0] if yes else [-4.0, 4.0]) for qid in req.questions})

    def decide(self, req):
        raise AssertionError("calibration must use raw scores")


def write_dataset(tmp_path: Path, n: int = 80, with_test: bool = True, noise: float = 0.2) -> Path:
    rng = np.random.default_rng(1)
    rows = []
    for i in range(n):
        hat = bool(i % 2)
        img = np.zeros((64, 64, 3), np.uint8)
        img[:32] = 220 if hat else 30
        name = f"img{i}.png"
        cv2.imwrite(str(tmp_path / name), img)
        label = hat if rng.random() > noise else not hat
        split = ("test" if i % 4 >= 2 else "calib") if with_test else "calib"
        rows.append({"image": name, "box": [0, 0, 64, 64], "pad": 0.0, "question": NOUL.to_wire(),
                     "label": label, "split": split, "group": f"photo{i // 2}"})  # fmt: skip
    rows.append({"image": "img0.png", "question": NOUL.to_wire(), "label": False, "split": "calib"})  # full frame
    rows.append({"state": "A worker without a helmet.", "question": NOUL.to_wire(), "label": False})  # text
    path = tmp_path / "manifest.jsonl"
    path.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    return path


def test_load_examples(tmp_path: Path) -> None:
    examples = load_examples(write_dataset(tmp_path, n=4))
    assert [e.input for e in examples] == ["crop"] * 4 + ["full_frame", "text"]
    assert [e.key for e in examples][-2:] == ["noul/full_frame", "noul/text"]
    assert isinstance(examples[0].state["image"], Image) and examples[0].state["image"].size == (64, 64)
    bad = tmp_path / "bad.jsonl"
    bad.write_text(json.dumps({"image": "img0.png", "question": NOUL.to_wire(), "label": "maybe"}) + "\n")
    with pytest.raises(ValueError):
        load_examples(bad)


def test_calibrate_fits_on_calib_and_reports_on_test(tmp_path: Path) -> None:
    examples = load_examples(write_dataset(tmp_path, n=200))
    profile = calibrate(OverconfidentRuntime(), examples, model="overconfident-1", runtime_name="custom", source="v1")
    m = profile.metrics["noul/crop"]
    assert m["evaluated_on"] == "test split" and m["after"]["n"] == 100 and m["n_fit"] == 100
    assert profile.temperatures["noul/crop"] > 1.5  # overconfident → T > 1
    assert m["after"]["ece"] < m["before"]["ece"]
    assert "noul/full_frame" in profile.temperatures and "noul/text" in profile.temperatures


def test_calibrate_without_test_uses_out_of_fold(tmp_path: Path) -> None:
    examples = load_examples(write_dataset(tmp_path, n=100, with_test=False))
    profile = calibrate(OverconfidentRuntime(), examples, model="m", runtime_name="custom", source="v1", folds=5)
    assert profile.metrics["noul/crop"]["evaluated_on"] == "5-fold out-of-fold"
    assert profile.metrics["noul/crop"]["after"]["n"] == 100


def test_small_keys_warn(tmp_path: Path) -> None:
    examples = load_examples(write_dataset(tmp_path, n=100))
    with pytest.warns(UserWarning, match="noul/full_frame: only 1 examples"):
        calibrate(OverconfidentRuntime(), examples, model="m", runtime_name="custom", source="v1")


def test_calibrate_needs_raw_scores() -> None:
    class NoRaw(RuntimeBase):
        caps = Capabilities(images=True)

    with pytest.raises(ValueError, match="raw scores"):
        calibrate(NoRaw(), [Example("x", NOUL, True, "text")], model="m", runtime_name="r", source="v")


# --- profiles and applying them --------------------------------------------------------------


def test_profile_roundtrip_and_fallback(tmp_path: Path) -> None:
    p = CalibrationProfile("m", "native", "hf://a/b@c", {"noul/crop": 2.0, "noul/full_frame": 0.5, "choice/crop": 1.5})
    loaded = CalibrationProfile.load(p.save(tmp_path / "p.json"))
    assert loaded.temperatures == p.temperatures and loaded.model == "m"
    assert loaded.temperature(NOUL, "crop") == 2.0
    assert loaded.temperature(NOUL, "text") == pytest.approx(1.0)  # geometric mean of 2.0 and 0.5
    assert loaded.temperature(Score("?", ["a", "b"]), "crop") == 1.0
    (tmp_path / "other.json").write_text('{"temperatures": {}}')
    with pytest.raises(ValueError, match="not a version"):
        CalibrationProfile.load(tmp_path / "other.json")


def test_calibrated_runtime_applies_temperature_by_input() -> None:
    profile = CalibrationProfile("fake-1", "custom", "v", {"noul/crop": 4.0, "noul/full_frame": 1.0})
    inner = FakeRuntime()
    req_img = SystemOneResponse  # placeholder for type checkers
    from s1ght.schema.systemone import SystemOneRequest

    req = SystemOneRequest("m", {"image": Image(np.zeros((32, 32, 3), np.uint8))}, {"q": NOUL})
    raw = inner.raw(req).scores["q"]
    p_raw = 1 / (1 + math.exp(-(raw[0] - raw[1])))
    full = CalibratedRuntime(inner, profile).decide(req).answers["q"].noul
    crop = CalibratedRuntime(inner, profile).with_input("crop").decide(req).answers["q"].noul
    assert full == pytest.approx(p_raw)
    assert crop == pytest.approx(1 / (1 + math.exp(-(raw[0] - raw[1]) / 4)))
    assert abs(crop - 0.5) < abs(full - 0.5)  # T > 1 pulls toward 0.5
    assert req_img is SystemOneResponse


def test_calibrated_runtime_passes_box_questions_through() -> None:
    from s1ght.schema.systemone import SystemOneRequest

    profile = CalibrationProfile("fake-1", "custom", "v", {"noul/full_frame": 2.0})
    req = SystemOneRequest("m", {"image": Image(np.zeros((32, 32, 3), np.uint8))}, {"q": NOUL, "where": Box("the hat")})
    out = CalibratedRuntime(FakeRuntime(), profile).decide_batch([req])[0]
    assert isinstance(out.answers["where"], BoxAnswer) and set(out.answers) == {"q", "where"}


def test_s1ght_calibrate_and_apply(tmp_path: Path) -> None:
    data = write_dataset(tmp_path, n=40)
    model = S1GHT(OverconfidentRuntime())
    profile = model.calibrate(data, save=tmp_path / "profile.json")
    assert (tmp_path / "profile.json").exists() and profile.dataset["examples"] == 42
    assert profile.dataset["splits"] == {"calib": 22, "test": 20}
    assert isinstance(model.runtime, CalibratedRuntime) and model.profile is profile
    from s1ght.schema.systemone import SystemOneRequest

    req = SystemOneRequest("m", {"image": Image(np.full((64, 64, 3), 220, np.uint8))}, {"q": NOUL})
    out = model.runtime.with_input("crop").decide(req).answers["q"].noul
    assert 0.5 < out < 1 / (1 + math.exp(-8))  # softer than the raw 8-nat margin

    again = S1GHT(OverconfidentRuntime(), calibration=tmp_path / "profile.json")
    assert again.profile.temperatures == profile.temperatures
    with pytest.raises(ValueError, match="fitted for"):
        S1GHT(FakeRuntime(), calibration=tmp_path / "profile.json")
    png = save_reliability(profile, tmp_path / "rel.png")
    assert cv2.imread(str(png)).shape[0] == 360
