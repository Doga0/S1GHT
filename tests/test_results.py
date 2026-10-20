"""Step 7: sources, Results, plotting, crops.fit_to_limits and S1GHT.predict."""

import json
import threading
import time
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image as PILImage

from conftest import FakeRuntime
from s1ght import S1GHT, Choice, Image, Noul, Score
from s1ght.engine.crops import fit_to_limits
from s1ght.io import sources
from s1ght.io.plotting import describe, draw
from s1ght.io.sources import LatestFrameReader, is_live, load_source, read_image
from s1ght.results import Boxes, Event, Results
from s1ght.runtimes.base import Capabilities
from s1ght.schema.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneResponse

QUESTIONS = {
    "people": Noul("Are people visible in `image`?"),
    "scene": Choice("What is the scene?", ["street", "indoor", "nature"]),
    "busy": Score("How busy is it?", ["Empty", "Some", "Crowded"]),
}


def frame(color=(40, 120, 200), size=(480, 640)) -> np.ndarray:
    img = np.zeros((*size, 3), np.uint8)
    img[:] = color
    cv2.circle(img, (size[1] // 2, size[0] // 2), min(size) // 4, (0, 0, 255), -1)
    return img


def write_video(path: Path, n: int = 6, size=(96, 128)) -> Path:
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (size[1], size[0]))
    for i in range(n):
        vw.write(frame((i * 40 % 255, 80, 80), size))
    vw.release()
    return path


# --- sources ---------------------------------------------------------------------------------


def test_load_source_kinds(tmp_path: Path) -> None:
    img = frame()
    cv2.imwrite(str(tmp_path / "a.png"), img)
    cv2.imwrite(str(tmp_path / "b.jpg"), img)
    (tmp_path / "notes.txt").write_text("x")
    uni = tmp_path / "çğü.png"
    cv2.imencode(".png", img)[1].tofile(str(uni))

    assert np.array_equal(next(load_source(tmp_path / "a.png")).image, img)
    assert [Path(f.path).name for f in load_source(tmp_path)] == ["a.png", "b.jpg", "çğü.png"]
    assert len(list(load_source(str(tmp_path / "*.png")))) == 2
    assert read_image(uni).shape == img.shape  # non-ASCII path on Windows

    rgb = PILImage.fromarray(img[:, :, ::-1].copy())
    items = list(load_source([img, rgb, Image(img, format="png"), cv2.imencode(".png", img)[1].tobytes()]))
    assert [f.path for f in items] == ["image0", "image1", "image2", "image3"]
    for f in items:
        assert np.array_equal(f.image, img)
    gray = next(load_source(img[:, :, 0])).image
    assert gray.shape == img.shape


def test_load_source_errors(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        list(load_source(tmp_path / "missing.jpg"))
    with pytest.raises(FileNotFoundError, match="no images"):
        list(load_source(tmp_path))
    with pytest.raises(TypeError):
        list(load_source(3.5))
    with pytest.raises(ValueError):
        list(load_source(frame(), vid_stride=0))


def test_video_file_frames_and_stride(tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.avi", n=6)
    frames = list(load_source(video))
    assert [f.index for f in frames] == [0, 1, 2, 3, 4, 5]
    assert frames[2].t == pytest.approx(0.2) and frames[0].image.shape == (96, 128, 3)
    assert [f.index for f in load_source(video, vid_stride=2)] == [0, 2, 4]


@pytest.mark.parametrize(
    ("source", "live"),
    [
        (0, True),
        ("1", True),
        ("rtsp://cam/stream", True),
        ("https://cam.local/mjpeg", True),
        ("https://ultralytics.com/images/bus.jpg", False),
        ("https://x.org/a.mp4", False),
        ("bus.jpg", False),
        (True, False),
        (np.zeros((2, 2, 3), np.uint8), False),
    ],
)
def test_is_live(source, live) -> None:
    assert is_live(source) is live


class FakeCapture:
    """A camera that produces numbered frames faster than the consumer reads them."""

    def __init__(self, n: int, delay: float = 0.002) -> None:
        self.n, self.delay, self.i, self.released = n, delay, 0, False

    def isOpened(self) -> bool:
        return True

    def read(self):
        if self.i >= self.n:
            return False, None
        time.sleep(self.delay)
        img = np.full((4, 4, 3), self.i % 256, np.uint8)
        self.i += 1
        return True, img

    def release(self) -> None:
        self.released = True


def test_live_source_latest_frame_wins(monkeypatch) -> None:
    cap = FakeCapture(200)
    monkeypatch.setattr(sources, "open_capture", lambda target: cap)
    seen = []
    for f in load_source(0):
        seen.append(f.index)
        time.sleep(0.02)  # a slow consumer
    assert seen == sorted(seen) and len(seen) < 60  # frames were skipped, never reordered
    assert seen[-1] == 199  # the last frame is still delivered
    assert cap.released


def test_live_source_buffered_keeps_every_frame(monkeypatch) -> None:
    monkeypatch.setattr(sources, "open_capture", lambda target: FakeCapture(30, 0))
    assert [f.index for f in load_source("rtsp://cam", buffer=True)] == list(range(30))


def test_live_reader_close_stops_thread(monkeypatch) -> None:
    monkeypatch.setattr(sources, "open_capture", lambda target: FakeCapture(10**9, 0.001))
    reader = LatestFrameReader(0)
    it = iter(reader)
    assert next(it).path == "camera0"
    it.close()
    assert not reader._thread.is_alive() and threading.active_count() >= 1


# --- crops -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("hw", "expected"),
    [
        ((480, 640), (480, 640)),  # fits: unchanged
        ((2000, 3000), (1045, 1568)),  # too big: long side to 1568
        ((100, 200), (280, 560)),  # too small: short side to 280
        ((100, 3000), (280, 1568)),  # too long and thin: scale down, pad the short side
    ],
)
def test_fit_to_limits(hw, expected) -> None:
    out = fit_to_limits(np.zeros((*hw, 3), np.uint8), (280, 1568))
    assert out.shape[:2] == expected
    assert 280 <= min(out.shape[:2]) and max(out.shape[:2]) <= 1568
    assert fit_to_limits(np.zeros((10, 10, 3), np.uint8), None).shape == (10, 10, 3)


# --- Results and plotting --------------------------------------------------------------------


def make_results() -> Results:
    boxes = Boxes(
        xyxy=np.array([[10, 20, 200, 300], [300, 50, 600, 400]], np.float32),
        conf=np.array([0.9, 0.55], np.float32),
        cls=np.array([0, 2]),
        id=np.array([7, 9]),
        names={0: "person", 2: "car"},
    )
    noul = NoulAnswer(0.94)
    choice = ChoiceAnswer("street", {"street": 0.8, "indoor": 0.1, "nature": 0.1}, 0.7)
    score = ScoreAnswer(1.4, {0: "Empty", 1: "Some", 2: "Crowded"}, {0: 0.1, 1: 0.4, 2: 0.5}, 0.25)
    return Results(
        orig_img=frame(),
        boxes=boxes,
        answers={None: {"scene": choice, "busy": score}, 7: {"no_helmet": noul}},
        events=[Event(7, "no_helmet", 1.0, 2.2, 0.94)],
        speed={"decide": 12.0},
        path="bus.jpg",
        model="fake-1",
        usage={"input_tokens": 100, "output_tokens": 0},
    )


def test_results_serialization() -> None:
    r = make_results()
    data = json.loads(r.to_json())
    assert data["answers"]["frame"]["scene"] == {
        "type": "choice",
        "choice": "street",
        "probabilities": {"street": 0.8, "indoor": 0.1, "nature": 0.1},
        "confidence": 0.7,
    }
    assert data["answers"]["7"]["no_helmet"] == {"type": "noul", "noul": 0.94}
    assert data["boxes"][1] == {"xyxy": [300.0, 50.0, 600.0, 400.0], "conf": 0.55, "cls": 2, "name": "car", "id": 9}
    assert data["events"] == [{"track_id": 7, "question": "no_helmet", "start": 1.0, "end": 2.2, "peak": 0.94}]

    responses = r.to_systemone()
    assert [list(x.answers) for x in responses] == [["scene", "busy"], ["no_helmet"]]
    for resp in responses:  # valid TypeSafe responses: they round-trip through the wire format
        assert SystemOneResponse.from_wire(json.loads(json.dumps(resp.to_wire()))).to_wire() == resp.to_wire()
    assert str(r.events[0]) == "track 7: no_helmet p=0.94, 1.2s"
    assert "boxes=2" in repr(r)


def test_plot_draws_and_keeps_original(tmp_path: Path) -> None:
    r = make_results()
    before = r.orig_img.copy()
    out = r.plot()
    assert out.shape == before.shape and not np.array_equal(out, before)
    assert np.array_equal(r.orig_img, before)
    assert np.array_equal(draw(r, boxes=False, panel=False), before)
    saved = r.save(tmp_path / "sub" / "out.jpg")
    assert cv2.imread(str(saved)).shape == before.shape


def test_plot_tiny_image_does_not_crash() -> None:
    r = make_results()
    r.orig_img = np.zeros((20, 30, 3), np.uint8)
    assert r.plot().shape == (20, 30, 3)


def test_describe() -> None:
    assert describe("x", NoulAnswer(0.2))[0] == "x: no 0.20"
    assert describe("busy", make_results().answers[None]["busy"])[0] == "busy: 1.40 (Some)"


# --- S1GHT.predict ---------------------------------------------------------------------------


def test_predict_full_frame(tmp_path: Path) -> None:
    for i in range(5):
        cv2.imwrite(str(tmp_path / f"{i}.jpg"), frame((i * 50, 100, 100)))
    rt = FakeRuntime()
    model = S1GHT(rt)
    results = model.predict(tmp_path, questions=QUESTIONS, batch=2)
    assert len(results) == 5 and rt.calls == ["batch:2", "batch:2", "batch:1"]
    r = results[0]
    assert set(r.answers) == {None} and set(r.answers[None]) == set(QUESTIONS)
    assert r.path.endswith("0.jpg") and r.model == "fake-1" and len(r.boxes) == 0
    assert set(r.speed) == {"preprocess", "decide"}
    data = json.loads(r.to_json())
    assert data["answers"]["frame"]["people"]["type"] == "noul"
    assert len(r.to_systemone()) == 1
    # same picture → same answers (the fake runtime keys on image bytes)
    again = model.predict(tmp_path / "0.jpg", questions=QUESTIONS)[0]
    assert again.answers[None]["scene"] == r.answers[None]["scene"]


def test_predict_stream_and_context() -> None:
    rt = FakeRuntime()
    gen = S1GHT(rt).predict(
        [frame(), frame((1, 2, 3))],
        questions={"q": {"type": "noul", "instructions": "Red?"}},
        stream=True,
        context="Loading dock camera",
    )
    assert not isinstance(gen, list)
    out = list(gen)
    assert len(out) == 2 and isinstance(out[0].answers[None]["q"], NoulAnswer)


def test_predict_fits_image_to_runtime_limits() -> None:
    seen = []

    class Recording(FakeRuntime):
        def decide_batch(self, reqs):
            seen.extend(next(iter(r.assets().values())).size for r in reqs)
            return super().decide_batch(reqs)

    rt = Recording(Capabilities(images=True, batch=True, raw_scores=True, image_px=(280, 1568)))
    S1GHT(rt).predict(frame(size=(2000, 3000)), questions={"q": Noul("?")})
    assert seen == [(1568, 1045)]


def test_predict_argument_errors() -> None:
    m = S1GHT(FakeRuntime())
    with pytest.raises(NotImplementedError, match="P2"):
        m.predict(frame(), queries=["a person"])
    with pytest.raises(ValueError, match="questions"):
        m.predict(frame())
    with pytest.raises(TypeError):
        S1GHT(object())
    with pytest.raises(ValueError):
        S1GHT(FakeRuntime(), device="cpu")


@pytest.mark.network
def test_predict_bus_live_ollama(tmp_path: Path) -> None:
    """`predict("bus.jpg")` against a local Ollama with d1-omni-600M; skipped when unavailable."""
    import httpx

    from s1ght.runtimes.server.systemone import SystemOneError

    try:
        httpx.get("http://127.0.0.1:11434/api/version", timeout=2).raise_for_status()
    except httpx.HTTPError:
        pytest.skip("no Ollama server")
    model = S1GHT("d1-omni-600m", runtime="ollama")
    questions = {
        "people": Noul("Are people visible in `image`?"),
        "vehicle": Choice("Which vehicle is the largest in `image`?", {"bus": None, "car": None, "bicycle": None}),
    }
    try:
        r = model.predict("https://ultralytics.com/images/bus.jpg", questions=questions)[0]
    except SystemOneError as e:
        if e.status == 404:
            pytest.skip(str(e))
        raise
    assert r.answers[None]["people"].noul > 0.5
    assert r.answers[None]["vehicle"].choice == "bus"
    r.save(tmp_path / "bus_decisions.jpg")
