import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

from s1ght import Box, Choice, Image, Noul, Score
from s1ght.schema.questions import load_questions, question_from_wire
from s1ght.schema.systemone import (
    ChoiceAnswer,
    NoulAnswer,
    ScoreAnswer,
    SystemOneRequest,
    SystemOneResponse,
)

DATA = Path(__file__).parent / "data"


def load(name: str) -> list[dict]:
    return json.loads((DATA / name).read_text(encoding="utf-8"))


def png_bytes(color: tuple[int, int, int] = (30, 30, 200), size: tuple[int, int] = (4, 3)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


# --- questions -------------------------------------------------------------------------------


def test_choice_list_becomes_dict() -> None:
    q = Choice("What is this person doing?", ["standing", "bending", "on a ladder", "other"])
    assert q.criteria == {"standing": None, "bending": None, "on a ladder": None, "other": None}


@pytest.mark.parametrize(
    "make",
    [
        lambda: Noul(""),
        lambda: Noul("ok?", criteria={"yes": "x"}),
        lambda: Choice("pick", ["only"]),
        lambda: Choice("pick", ["a", "a"]),
        lambda: Choice("pick", [f"o{i}" for i in range(256)]),
        lambda: Score("rate", ["low"]),
        lambda: Score("rate", [str(i) for i in range(11)]),
    ],
)
def test_question_validation(make) -> None:
    with pytest.raises(ValueError):
        make()


def test_choice_limits_accepted() -> None:
    assert len(Choice("pick", [f"o{i}" for i in range(255)]).criteria) == 255
    assert len(Score("rate", [str(i) for i in range(10)]).criteria) == 10


@pytest.mark.parametrize(
    "q",
    [
        Noul("Is it raining?"),
        Noul("Is it raining?", criteria={"true": "Visible rain", "false": "Dry"}),
        Choice("Weather?", {"sun": None, "rain": "Wet ground"}),
        Score("How cloudy?", ["Clear", "Some", "Overcast"]),
        Box("the umbrella in `photo`"),
    ],
)
def test_question_wire_roundtrip(q) -> None:
    assert question_from_wire(q.to_wire()) == q


def test_unknown_question_type() -> None:
    with pytest.raises(ValueError, match="unknown question type"):
        question_from_wire({"type": "rank", "instructions": "x"})
    with pytest.raises(ValueError, match="invalid box"):
        question_from_wire({"type": "box", "instructions": "x", "criteria": ["a"]})


def test_load_questions_yaml_nested(tmp_path: Path) -> None:
    path = tmp_path / "q.yaml"
    path.write_text(
        """
person:
  no_helmet: {type: noul, instructions: "Is this person NOT wearing a hard hat?"}
  posture:
    type: choice
    instructions: "What is this person doing?"
    criteria: {standing: null, bending: null, on a ladder: null, other: null}
""",
        encoding="utf-8",
    )
    ask = load_questions(path)
    assert set(ask) == {"person"}
    assert ask["person"]["no_helmet"] == Noul("Is this person NOT wearing a hard hat?")
    assert ask["person"]["posture"] == Choice(
        "What is this person doing?", ["standing", "bending", "on a ladder", "other"]
    )


def test_load_questions_flat_dict() -> None:
    qs = load_questions({"rain": {"type": "noul", "instructions": "Is it raining?"}})
    assert qs == {"rain": Noul("Is it raining?")}


# --- assets ----------------------------------------------------------------------------------


def test_image_from_bgr_array_keeps_channel_order() -> None:
    bgr = np.zeros((3, 4, 3), np.uint8)
    bgr[..., 0] = 255  # pure blue in BGR
    img = Image(bgr, format="png")
    assert img.to_array() is bgr
    assert img.size == (4, 3)
    decoded = PILImage.open(io.BytesIO(img.to_bytes()))
    assert decoded.getpixel((0, 0)) == (0, 0, 255)
    assert np.array_equal(Image(img.to_bytes()).to_array(), bgr)


def test_image_encoded_input_passes_through(tmp_path: Path) -> None:
    data = png_bytes()
    path = tmp_path / "x.png"
    path.write_bytes(data)
    for img in (Image(path), Image(data)):
        assert img.format == "png"
        assert img.to_bytes() == data
        assert img.size == (4, 3)


def test_image_reencodes_on_format_change() -> None:
    img = Image(png_bytes(), format="jpg")
    assert img.to_bytes()[:2] == b"\xff\xd8"
    assert img.format == "jpg"


def test_image_from_pil_and_refs_unique() -> None:
    a, b = Image(PILImage.new("RGB", (2, 2))), Image(PILImage.new("L", (2, 2)))
    assert a.format == "jpeg" and a.ref != b.ref
    assert b.to_array().shape == (2, 2, 3)


def test_image_rejects_bad_input() -> None:
    with pytest.raises(ValueError):
        Image(np.zeros((2, 2, 4), np.uint8))
    with pytest.raises(ValueError):
        Image(png_bytes(), format="gif")
    with pytest.raises(FileNotFoundError):
        Image("does/not/exist.jpg")


# --- requests --------------------------------------------------------------------------------


@pytest.mark.parametrize("wire", load("typesafe_requests.json"))
def test_typesafe_request_roundtrip(wire: dict) -> None:
    req = SystemOneRequest.from_wire(wire)
    body, parts = req.to_wire(inline=True)
    assert body == wire
    assert parts == {}


def test_typesafe_request_images_placed() -> None:
    wire = load("typesafe_requests.json")[2]
    req = SystemOneRequest.from_wire(wire)
    assert isinstance(req.state["frame"], Image)
    helmet = req.questions["headgear"].criteria["hard_hat"]["examples"][0]
    assert isinstance(helmet, Image) and helmet.ref == "e8820b6c-helmet"
    # referenced only mid-sentence: kept as text, image carried in extra_assets
    assert [img.ref for img in req.extra_assets] == ["5b07d2f3-ref"]
    assert set(req.assets()) == {"c41e9a20-frame", "5b07d2f3-ref", "e8820b6c-helmet"}
    assert req.needs_images_flag


def test_typesafe_request_multipart() -> None:
    photo = Image(png_bytes())
    req = SystemOneRequest(
        model="jev-preview",
        state={"photo": photo},
        questions={"visible": Noul("Is there a mug in `photo`?"), "mug": Box("the mug in `photo`")},
    )
    body, parts = req.to_wire()
    assert body["state"] == {"photo": photo.ref}
    assert body["assets"] == {photo.ref: {"type": "image", "part": photo.ref, "format": "png"}}
    assert parts == {photo.ref: photo.to_bytes()}
    again = SystemOneRequest.from_wire(body, parts)
    assert again.state["photo"].to_bytes() == photo.to_bytes()


def test_typesafe_request_multipart_part_name() -> None:
    wire = {
        "model": "jev-preview",
        "state": {"photo": "r1"},
        "assets": {"r1": {"type": "image", "part": "kitchen-photo"}},
        "questions": {"mug": {"type": "box", "instructions": "the mug in `photo`"}},
    }
    req = SystemOneRequest.from_wire(wire, {"kitchen-photo": png_bytes()})
    assert req.state["photo"].format == "png"
    with pytest.raises(ValueError, match="missing multipart part"):
        SystemOneRequest.from_wire(wire)


def test_typesafe_keeps_jpg_spelling() -> None:
    jpeg = Image(png_bytes(), format="jpg").to_bytes()
    req = SystemOneRequest("jev-latest", Image(jpeg, format="jpg", ref="r"), {"q": Noul("Is it red?")})
    body, _ = req.to_wire(inline=True)
    assert body["state"] == "r"
    assert body["assets"]["r"] == {"type": "image", "format": "jpg", "data_base64": base64.b64encode(jpeg).decode()}


def test_text_request_needs_no_flag() -> None:
    req = SystemOneRequest("jev-latest", "hello", {"q": Noul("Is it a greeting?")})
    assert not req.needs_images_flag
    assert "assets" not in req.to_wire()[0]


@pytest.mark.parametrize("wire", load("ollama_requests.json"))
def test_ollama_request_roundtrip(wire: dict) -> None:
    req = SystemOneRequest.from_wire(wire, dialect="ollama")
    assert req.to_wire("ollama") == (wire, {})


def test_ollama_images_field_and_labels() -> None:
    a, b = Image(png_bytes((1, 2, 3))), Image(png_bytes((4, 5, 6)))
    req = SystemOneRequest(
        model="clef-flash",
        state={"before": a, "after": b},
        questions={"changed": Noul(f"Is {b.ref} different from `before`?")},
        extra_assets=[b],
    )
    body, parts = req.to_wire("ollama")
    assert "assets" not in body and parts == {}
    assert body["images"] == [base64.b64encode(a.to_bytes()).decode(), base64.b64encode(b.to_bytes()).decode()]
    assert body["state"] == {"before": "[image 1]", "after": "[image 2]"}
    assert body["questions"]["changed"]["instructions"] == "Is [image 2] different from `before`?"


def test_ollama_rejects_box() -> None:
    req = SystemOneRequest("clef", Image(png_bytes()), {"where": Box("the cup")})
    with pytest.raises(ValueError, match="Box"):
        req.to_wire("ollama")


# --- responses -------------------------------------------------------------------------------


@pytest.mark.parametrize("wire", load("typesafe_responses.json") + load("ollama_responses.json"))
def test_response_roundtrip(wire: dict) -> None:
    assert SystemOneResponse.from_wire(wire).to_wire() == wire


def test_response_types() -> None:
    resp = SystemOneResponse.from_wire(load("typesafe_responses.json")[0])
    noul, choice, score = (resp.answers[k] for k in ("is_hazard", "area", "severity"))
    assert isinstance(noul, NoulAnswer) and noul.confidence == pytest.approx(0.94)
    assert isinstance(choice, ChoiceAnswer) and choice.choice == "warehouse"
    assert isinstance(score, ScoreAnswer)
    assert score.legend == {0: "Minor", 1: "Moderate", 2: "Severe"}
    assert score.probabilities[1] == 0.5
    assert resp.usage == {"input_tokens": 412, "output_tokens": 31}
