import base64
import email
import io
import json
import os

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from s1ght import Box, Choice, Image, Noul, Score
from s1ght.runtimes.base import CapabilityError, Runtime, scores_to_answers
from s1ght.runtimes.server.systemone import SystemOneClient, SystemOneError
from s1ght.schema.systemone import ChoiceAnswer, NoulAnswer, SystemOneRequest

ANSWERS = {
    "rain": {"type": "noul", "noul": 0.8},
    "sky": {
        "type": "choice",
        "choice": "cloudy",
        "probabilities": {"clear": 0.1, "cloudy": 0.7, "overcast": 0.2},
        "confidence": 0.55,
    },
    "wind": {
        "type": "score",
        "score": 0.6,
        "legend": {"0": "Calm", "1": "Breezy"},
        "probabilities": {"0": 0.4, "1": 0.6},
        "confidence": 0.2,
    },
    "umbrella": {"type": "box", "box": {"top": 0.1, "bottom": 0.5, "left": 0.2, "right": 0.6}, "confidence": 0.9},
}
QUESTIONS = {
    "rain": Noul("Is it raining in `photo`?"),
    "sky": Choice("What is the sky like?", ["clear", "cloudy", "overcast"]),
    "wind": Score("How windy is it?", ["Calm", "Breezy"]),
}


def jpeg(size: tuple[int, int] = (320, 320)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, (90, 140, 200)).save(buf, "JPEG")
    return buf.getvalue()


class Recorder:
    """MockTransport handler that records requests and replays queued responses."""

    def __init__(self, *responses: httpx.Response) -> None:
        self.requests: list[httpx.Request] = []
        self.responses = list(responses)

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        return self.responses.pop(0)


def ok(qids, model: str = "m", **extra) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "model": model,
            "answers": {q: ANSWERS[q] for q in qids},
            "usage": {"input_tokens": 9, "output_tokens": 0},
        }
        | extra,
    )


def client(provider: str, rec: Recorder, **kw) -> SystemOneClient:
    c = SystemOneClient(provider, kw.pop("model", "m"), transport=httpx.MockTransport(rec), backoff=0, **kw)
    c._sleep = lambda s: None
    return c


def multipart(request: httpx.Request) -> dict[str, tuple[str, bytes]]:
    msg = email.message_from_bytes(
        b"Content-Type: " + request.headers["content-type"].encode() + b"\r\n\r\n" + request.read()
    )
    return {
        p.get_param("name", header="content-disposition"): (p.get_content_type(), p.get_payload(decode=True))
        for p in msg.get_payload()
    }


# --- TypeSafe --------------------------------------------------------------------------------


def test_typesafe_text_request(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_API_KEY", "sk-test")
    rec = Recorder(ok(["rain", "sky", "wind"], "jev-1.13.0"))
    c = client("typesafe", rec, model="jev-latest")
    assert isinstance(c, Runtime)
    resp = c.decide(SystemOneRequest("ignored", "It is pouring outside.", QUESTIONS))

    req = rec.requests[0]
    assert str(req.url) == "https://api.typesafe.ai/v1/systemone"
    assert req.headers["authorization"] == "Bearer sk-test"
    assert "typesafe-danger-zone-images" not in req.headers
    body = json.loads(req.read())
    assert body["model"] == "jev-latest" and body["state"] == "It is pouring outside."
    assert body["questions"]["sky"]["criteria"] == {"clear": None, "cloudy": None, "overcast": None}
    assert isinstance(resp.answers["rain"], NoulAnswer) and resp.model == "jev-1.13.0"


def test_typesafe_image_request_is_multipart(monkeypatch) -> None:
    monkeypatch.setenv("TYPESAFE_BASE_URL", "http://localhost:11436/")
    photo = Image(jpeg(), ref="ref-1")
    rec = Recorder(ok(["rain", "umbrella"]))
    c = client("typesafe", rec, model="jev-preview")
    resp = c.decide(
        SystemOneRequest("m", {"photo": photo}, {"rain": QUESTIONS["rain"], "umbrella": Box("the umbrella in `photo`")})
    )

    req = rec.requests[0]
    assert str(req.url) == "http://localhost:11436/v1/systemone"
    assert req.headers["typesafe-danger-zone-images"] == "true"
    parts = multipart(req)
    ctype, request_json = parts["request"]
    assert ctype == "application/json"
    body = json.loads(request_json)
    assert body["state"] == {"photo": "ref-1"}
    assert body["assets"] == {"ref-1": {"type": "image", "part": "ref-1", "format": "jpeg"}}
    assert parts["ref-1"] == ("image/jpeg", photo.to_bytes())
    assert resp.answers["umbrella"].box["right"] == 0.6


def test_typesafe_rejects_out_of_range_image_before_sending() -> None:
    rec = Recorder()
    c = client("typesafe", rec)
    with pytest.raises(ValueError, match="280-1568"):
        c.decide(SystemOneRequest("m", Image(jpeg((100, 400))), {"rain": QUESTIONS["rain"]}))
    assert rec.requests == []


# --- Ollama / Ollaya -------------------------------------------------------------------------


def test_ollama_image_request(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_HOST", "0.0.0.0:11434")
    monkeypatch.delenv("OLLAMA_API_KEY", raising=False)
    photo = Image(jpeg())
    rec = Recorder(ok(["rain", "sky"], "clef-flash"))
    c = client("ollama", rec, model="clef-flash")
    c.decide(
        SystemOneRequest("m", "A street camera frame.", {"rain": QUESTIONS["rain"], "sky": QUESTIONS["sky"]}, [photo])
    )

    req = rec.requests[0]
    assert str(req.url) == "http://0.0.0.0:11434/v1/systemone"
    assert "authorization" not in req.headers
    body = json.loads(req.read())
    assert body["images"] == [base64.b64encode(photo.to_bytes()).decode()]
    assert "assets" not in body


def test_ollama_has_no_box() -> None:
    rec = Recorder()
    with pytest.raises(CapabilityError, match="box"):
        client("ollama", rec).decide(SystemOneRequest("m", Image(jpeg()), {"u": Box("the umbrella")}))
    assert rec.requests == []


def test_ollama_choice_limit() -> None:
    q = {"c": Choice("pick", [f"o{i}" for i in range(27)])}
    with pytest.raises(ValueError, match="26"):
        client("ollama", Recorder()).decide(SystemOneRequest("m", "x", q))


def test_ollaya_uses_api_decide_and_png() -> None:
    rec = Recorder(ok(["rain"], "d1:3b-vision", routing=None, done_reason="decide", total_duration=123))
    c = client("ollaya", rec, model="d1:3b-vision")
    resp = c.decide(SystemOneRequest("m", "Camera frame.", {"rain": QUESTIONS["rain"]}, [Image(jpeg())]))

    req = rec.requests[0]
    assert str(req.url) == "http://127.0.0.1:11435/api/decide"
    sent = base64.b64decode(json.loads(req.read())["images"][0])
    assert sent.startswith(b"\x89PNG")
    assert resp.answers["rain"].noul == 0.8


# --- errors, retries, raw --------------------------------------------------------------------


def test_error_body_is_parsed() -> None:
    err = {"error": 'model "x" not found, try pulling it first', "code": "MODEL_NOT_FOUND"}
    c = client("ollaya", Recorder(httpx.Response(404, json=err)))
    with pytest.raises(SystemOneError) as e:
        c.decide(SystemOneRequest("m", "x", {"rain": QUESTIONS["rain"]}))
    assert e.value.status == 404 and e.value.code == "MODEL_NOT_FOUND"
    assert "try pulling it first" in str(e.value)


def test_fastapi_style_error() -> None:
    c = client(
        "typesafe",
        Recorder(httpx.Response(422, json={"detail": [{"loc": ["body", "state"], "msg": "Field required"}]})),
    )
    with pytest.raises(SystemOneError, match="Field required"):
        c.decide(SystemOneRequest("m", "x", {"rain": QUESTIONS["rain"]}))


def test_retries_then_succeeds() -> None:
    rec = Recorder(httpx.Response(503, headers={"Retry-After": "1"}), httpx.Response(529), ok(["rain"]))
    c = client("typesafe", rec)
    slept: list[float] = []
    c._sleep = slept.append
    assert c.decide(SystemOneRequest("m", "x", {"rain": QUESTIONS["rain"]})).answers["rain"].noul == 0.8
    assert len(rec.requests) == 3 and slept == [1.0, 0.0]


def test_no_retry_on_client_error() -> None:
    rec = Recorder(httpx.Response(401, json={"error": "bad key"}), ok(["rain"]))
    with pytest.raises(SystemOneError):
        client("typesafe", rec).decide(SystemOneRequest("m", "x", {"rain": QUESTIONS["rain"]}))
    assert len(rec.requests) == 1


def test_raw_scores_recover_provider_probabilities() -> None:
    c = client("typesafe", Recorder(ok(["rain", "sky", "wind"])))
    raw = c.raw(SystemOneRequest("m", "x", QUESTIONS))
    assert set(raw.scores) == {"rain", "sky", "wind"}
    again = scores_to_answers(QUESTIONS, raw)
    assert again["rain"].noul == pytest.approx(0.8)
    assert isinstance(again["sky"], ChoiceAnswer) and again["sky"].probabilities["cloudy"] == pytest.approx(0.7)
    assert again["wind"].score == pytest.approx(0.6)
    assert np.all(np.isfinite(raw.scores["sky"]))


def test_unknown_provider() -> None:
    with pytest.raises(ValueError, match="unknown provider"):
        SystemOneClient("vllm", "m")  # type: ignore[arg-type]


# --- live servers (network) ------------------------------------------------------------------

LIVE = {
    # provider: (env var for the model, default model, minimum server version for image decisions)
    "ollama": ("S1GHT_OLLAMA_MODEL", "hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0", (0, 35, 1)),
    "ollaya": ("S1GHT_OLLAYA_MODEL", "d1:3b-vision", (0, 7, 5)),
}


def drawing(shape: str, color: tuple[int, int, int]) -> Image:
    canvas = PILImage.new("RGB", (448, 448), "white")
    draw = ImageDraw.Draw(canvas)
    (draw.ellipse if shape == "circle" else draw.rectangle)((112, 112, 336, 336), fill=color)
    return Image(canvas, format="jpeg")


@pytest.mark.network
@pytest.mark.parametrize("provider", sorted(LIVE))
def test_live_single_image(provider: str) -> None:
    """Single-image noul + choice against a local server; a red circle and a blue square must be told apart.

    Models: Ollama `ollama pull hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0` (~640 MB);
    Ollaya `ollaya pull d1:3b-vision` (3.5 GB). Override with S1GHT_OLLAMA_MODEL / S1GHT_OLLAYA_MODEL.
    """
    env, default, min_version = LIVE[provider]
    model = os.environ.get(env, default)
    c = SystemOneClient(provider, model, timeout=600)
    try:
        version = httpx.get(f"{c.base_url}/api/version", timeout=2).json()["version"]
    except httpx.HTTPError:
        pytest.skip(f"no {provider} server at {c.base_url}")
    if tuple(int(x) for x in version.split("-")[0].split(".")[:3]) < min_version:
        pytest.skip(f"{provider} {version} is too old for image decisions")

    questions = {
        "has_circle": Noul("Does the image show a red circle?"),
        "color": Choice("What color is the main shape?", {"red": None, "green": None, "blue": None}),
    }
    try:
        red = c.decide(
            SystemOneRequest(model, "A picture drawn for a test.", questions, [drawing("circle", (220, 20, 20))])
        )
        blue = c.decide(
            SystemOneRequest(model, "A picture drawn for a test.", questions, [drawing("square", (20, 40, 220))])
        )
    except SystemOneError as e:
        if e.status == 404:
            pytest.skip(f"model {model} not pulled on {provider}: {e}")
        raise
    finally:
        c.close()

    assert red.answers["has_circle"].noul > 0.5 > blue.answers["has_circle"].noul
    assert red.answers["color"].choice == "red"
    assert blue.answers["color"].choice == "blue"
