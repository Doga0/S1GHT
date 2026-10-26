"""`s1ght` command line: argument parsing and every mode, on FakeRuntime (no model, GPU or network)."""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from conftest import FakeRuntime
from s1ght import S1GHT, Image, cli
from s1ght.schema.questions import load_questions
from s1ght.schema.systemone import SystemOneRequest
from test_scheduler import FakeDetector

QUESTIONS = 'bus: {type: noul, instructions: "Is there a bus in `image`?"}\n'
ASK = 'person:\n  red: {type: noul, instructions: "Red?"}\n'


@pytest.fixture
def fake(monkeypatch) -> list[dict]:
    """Every mode gets an S1GHT on FakeRuntime; the configs it was built from are recorded."""
    seen: list[dict] = []

    def load(cfg):
        seen.append(cfg)
        return S1GHT(FakeRuntime(model="fake"))

    monkeypatch.setattr(cli, "load_model", load)
    return seen


@pytest.fixture
def files(tmp_path):
    for i, color in enumerate([(0, 0, 255), (255, 0, 0)]):
        cv2.imwrite(str(tmp_path / f"img{i}.jpg"), np.full((64, 96, 3), color, np.uint8))
    (tmp_path / "q.yaml").write_text(QUESTIONS, encoding="utf-8")
    (tmp_path / "ask.yaml").write_text(ASK, encoding="utf-8")
    return tmp_path


def test_parse_args_defaults_and_values() -> None:
    mode, cfg = cli.parse_args(["track", "source=0", "show=True", "thresholds=[0.7, 0.3]", "model=1", "budget=4"])
    assert mode == "track"
    assert cfg["source"] == 0 and cfg["show"] is True and cfg["thresholds"] == [0.7, 0.3]
    assert cfg["model"] == "1" and cfg["budget"] == 4
    assert cfg["detector"] == "yolo11n.pt" and cfg["lost_after"] == 1.0  # from cfg/default.yaml
    assert "port" not in cfg  # only the mode's own keys

    _, cfg = cli.parse_args(["serve", "--port=8000", "api-key=k", "calibration=none"])
    assert cfg["port"] == 8000 and cfg["api_key"] == "k" and cfg["calibration"] is None
    assert cli.parse_args([]) == ("help", {})


@pytest.mark.parametrize(
    ("argv", "message"),
    [
        (["predikt"], "did you mean 'predict'"),
        (["predict", "sourse=a.jpg"], "did you mean 'source'"),
        (["serve", "ask=a.yaml"], "serve has no key 'ask'"),
        (["decide", "model"], "expected key=value"),
    ],
)
def test_usage_errors(argv, message, capsys) -> None:
    assert cli.main(argv) == 2
    assert message in capsys.readouterr().err


def test_every_default_key_belongs_to_a_mode() -> None:
    used = {k for keys in cli.MODE_KEYS.values() for k in keys}
    assert set(cli.defaults()) == used


def test_decide_image(fake, files, capsys) -> None:
    img = files / "img0.jpg"
    assert cli.main(["decide", f"source={img}", f"questions={files / 'q.yaml'}", "state=at a bus stop"]) == 0
    out = json.loads(capsys.readouterr().out)
    assert out["model"] == "fake" and out["answers"]["bus"]["type"] == "noul"


def test_decide_json_context_next_to_the_image(fake, files, capsys) -> None:
    img = files / "img1.jpg"
    args = ["decide", f"source={img}", 'state={"camera": "gate 3"}', f"questions={files / 'q.yaml'}"]
    assert cli.main(args) == 0
    questions = load_questions(files / "q.yaml")
    state = {"image": Image(str(img)), "context": {"camera": "gate 3"}}
    direct = FakeRuntime(model="fake").decide(SystemOneRequest("x", state, questions))  # type: ignore[arg-type]
    assert json.loads(capsys.readouterr().out)["answers"]["bus"]["noul"] == pytest.approx(direct.answers["bus"].noul)


def test_decide_needs_an_image(fake, files, capsys) -> None:
    assert cli.main(["decide", f"questions={files / 'q.yaml'}"]) == 2
    assert "source=" in capsys.readouterr().err
    assert cli.main(["decide", "state=a red bus", f"questions={files / 'q.yaml'}"]) == 2  # text alone is not enough
    assert cli.main(["decide", f"source={files / 'img0.jpg'}", f"questions={files / 'ask.yaml'}"]) == 2


def test_predict_saves_and_prints(fake, files, capsys) -> None:
    out_dir = files / "out"
    code = cli.main(["predict", f"source={files}/*.jpg", f"questions={files / 'q.yaml'}", "save=True",
                     f"save_dir={out_dir}"])  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "img0.jpg: bus=" in text and "img1.jpg: bus=" in text and "2 frame(s)" in text
    assert sorted(p.name for p in out_dir.iterdir()) == ["img0.jpg", "img1.jpg"]


def test_predict_json_lines(fake, files, capsys) -> None:
    assert cli.main(["predict", f"source={files / 'img1.jpg'}", f"questions={files / 'q.yaml'}", "json=True"]) == 0
    lines = capsys.readouterr().out.strip().splitlines()
    assert len(lines) == 1 and "bus" in json.loads(lines[0])["answers"]["frame"]


def test_save_dir_is_numbered(tmp_path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / "runs" / "predict").mkdir(parents=True)
    (tmp_path / "runs" / "predict2").mkdir()
    out = cli.output_for("predict", {"save": True, "show": False, "save_dir": None, "vid_stride": 1})
    assert out is not None and out.save_dir is not None and out.save_dir.name == "predict3"


class ClipDetector(FakeDetector):
    def stream(self, source, **kwargs):
        for f in super().stream(source, **kwargs):
            f.path = str(source)
            yield f


def test_track_events_and_video(fake, files, monkeypatch, capsys) -> None:
    det = ClipDetector()
    monkeypatch.setattr(S1GHT, "_detector", staticmethod(lambda d, asked, kw: (det, kw)))
    monkeypatch.setattr(cli, "video_fps", lambda path: 10.0)
    out_dir = files / "track"
    code = cli.main(["track", "source=clip.mp4", f"ask={files / 'ask.yaml'}", "budget=100", "half_life=0",
                     "save=True", f"save_dir={out_dir}", "vid_stride=2"])  # fmt: skip
    assert code == 0
    text = capsys.readouterr().out
    assert "41 frame(s)" in text and "event change(s)" in text
    assert det.kwargs["vid_stride"] == 2 and det.kwargs["conf"] == 0.25
    video = out_dir / "clip.mp4"
    cap = cv2.VideoCapture(str(video))
    assert cap.get(cv2.CAP_PROP_FRAME_COUNT) == 40  # the final summary repeats frame 39 and is not written
    cap.release()


def test_calibrate(fake, tmp_path, capsys) -> None:
    rows = [{"state": f"case {i}", "question": {"type": "noul", "instructions": "Unsafe?"}, "label": bool(i % 2)}
            for i in range(40)]  # fmt: skip
    data = tmp_path / "set"
    data.mkdir()
    (data / "manifest.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    profile = tmp_path / "p.json"
    assert cli.main(["calibrate", f"data={data}", f"save={profile}", "folds=4"]) == 0
    out = capsys.readouterr().out
    assert "noul/text" in out and f"calibration={profile}" in out
    assert json.loads(profile.read_text(encoding="utf-8"))["model"] == "fake"
    assert profile.with_suffix(".png").is_file()


def test_serve(fake, monkeypatch, capsys) -> None:
    calls = []
    monkeypatch.setattr(S1GHT, "serve", lambda self, host, port, api_key=None: calls.append((host, port, api_key)))
    assert cli.main(["serve", "port=9000", "host=0.0.0.0", "api_key=k"]) == 0
    assert calls == [("0.0.0.0", 9000, "k")]
    assert "TYPESAFE_BASE_URL=http://0.0.0.0:9000" in capsys.readouterr().out


def test_model_keys_reach_s1ght(monkeypatch) -> None:
    built = {}

    class Spy:
        def __init__(self, model, **kw):
            built.update(model=model, **kw)

    monkeypatch.setattr("s1ght.model.S1GHT", Spy)
    cli.load_model({"model": "ollama://x", "runtime": None, "device": None, "calibration": "p.json", "timeout": 30})
    assert built == {"model": "ollama://x", "runtime": None, "calibration": "p.json", "timeout": 30.0}


def test_models_version_help(capsys) -> None:
    assert cli.main(["models"]) == 0
    assert "d1-omni-600m" in capsys.readouterr().out
    assert cli.main(["version"]) == 0
    assert capsys.readouterr().out.strip() == __import__("s1ght").__version__
    assert cli.main(["help"]) == 0
    assert "track: model=d1-3b" in capsys.readouterr().out


def test_runtime_errors_exit_1(fake, files, capsys) -> None:
    assert cli.main(["predict", f"source={files / 'missing.jpg'}", f"questions={files / 'q.yaml'}"]) == 1
    assert "missing.jpg" in capsys.readouterr().err
