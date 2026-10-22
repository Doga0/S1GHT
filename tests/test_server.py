"""`s1ght serve`: TypeSafe and Ollama dialects, errors, and the official TypeSafe SDK against a live server."""

from __future__ import annotations

import base64
import gzip
import io
import json
import socket
import threading
import time

import pytest
from PIL import Image as PILImage

pytest.importorskip("fastapi")
from fastapi.testclient import TestClient

from conftest import FakeRuntime
from s1ght import S1GHT, Image
from s1ght.runtimes.base import Capabilities
from s1ght.runtimes.server.systemone import SystemOneClient, SystemOneError
from s1ght.schema.questions import Box, Choice, Noul
from s1ght.schema.systemone import SystemOneRequest
from s1ght.server import create_app, ollama_request

QUESTIONS = {
    "bus": {"type": "noul", "instructions": "Is there a bus in `photo`?"},
    "color": {"type": "choice", "instructions": "What color is it?", "criteria": {"red": None, "blue": None}},
}


def png(size: tuple[int, int] = (320, 320), color: tuple[int, int, int] = (200, 30, 30)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, color).save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def runtime() -> FakeRuntime:
    return FakeRuntime(model="fake-served")


@pytest.fixture
def client(runtime: FakeRuntime) -> TestClient:
    return TestClient(create_app(S1GHT(runtime)))


def expected(runtime: FakeRuntime, state: object, questions: dict) -> dict:
    """What the runtime answers for this state when called directly."""
    from s1ght.schema.questions import questions_from_wire

    return runtime.decide(SystemOneRequest("x", state, questions_from_wire(questions))).to_wire()["answers"]


def test_text_request(client: TestClient, runtime: FakeRuntime) -> None:
    r = client.post("/v1/systemone", json={"model": "jev-latest", "state": "a red bus", "questions": QUESTIONS})
    assert r.status_code == 200
    body = r.json()
    assert body["model"] == "fake-served"  # the served model, whatever the request asked for
    assert body["answers"] == expected(runtime, "a red bus", QUESTIONS)
    assert body["usage"] == {"input_tokens": 0, "output_tokens": 0}


def test_multipart_image_request_matches_direct_call(client: TestClient, runtime: FakeRuntime) -> None:
    data = png()
    request = {
        "model": "m",
        "state": {"photo": "ref-1"},
        "assets": {"ref-1": {"type": "image", "part": "photo-part"}},
        "questions": QUESTIONS,
    }
    r = client.post(
        "/v1/systemone",
        headers={"TypeSafe-Danger-Zone-Images": "true"},
        files={"request": (None, json.dumps(request), "application/json"), "photo-part": ("p", data, "image/png")},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answers"] == expected(runtime, {"photo": Image(data)}, QUESTIONS)


def test_base64_assets_and_gzip(client: TestClient, runtime: FakeRuntime) -> None:
    data = png(color=(10, 10, 220))
    request = {
        "model": "m",
        "state": {"photo": "a"},
        "assets": {"a": {"type": "image", "format": "png", "data_base64": base64.b64encode(data).decode()}},
        "questions": QUESTIONS,
    }
    r = client.post(
        "/v1/systemone",
        content=gzip.compress(json.dumps(request).encode()),
        headers={"Content-Type": "application/json", "Content-Encoding": "gzip"},
    )
    assert r.status_code == 200, r.text
    assert r.json()["answers"] == expected(runtime, {"photo": Image(data)}, QUESTIONS)


def test_our_typesafe_client_round_trip(client: TestClient, runtime: FakeRuntime) -> None:
    """SystemOneClient(provider="typesafe") pointed at the server gets the runtime's own answers."""
    import httpx

    def handler(request: httpx.Request) -> httpx.Response:
        r = client.request(request.method, request.url.path, headers=dict(request.headers), content=request.read())
        return httpx.Response(r.status_code, headers=r.headers, content=r.content)

    remote = SystemOneClient("typesafe", "anything", base_url="http://s1ght", transport=httpx.MockTransport(handler))
    photo = Image(png(), ref="p1")
    req = SystemOneRequest("m", {"photo": photo}, {"bus": Noul("Is there a bus in `photo`?")})
    assert remote.decide(req).answers == runtime.decide(req).answers


def test_ollama_dialect_puts_images_back(client: TestClient, runtime: FakeRuntime) -> None:
    data = png()
    body = {
        "model": "d1",
        "state": {"photo": "[image 1]", "note": "compare with [image 1]"},
        "questions": {"bus": QUESTIONS["bus"]},
        "images": [base64.b64encode(data).decode()],
    }
    req = ollama_request(body)
    assert isinstance(req.state["photo"], Image)
    assert req.state["note"] == f"compare with {req.state['photo'].ref}"
    assert [i.ref for i in req.extra_assets] == [req.state["photo"].ref]

    for path in ("/v1/systemone", "/api/decide"):
        r = client.post(path, json=body)
        assert r.status_code == 200, r.text
        assert set(r.json()["answers"]) == {"bus"}


def test_ollama_label_out_of_range() -> None:
    with pytest.raises(ValueError, match="only 1 images"):
        ollama_request({"state": "[image 2]", "questions": {}, "images": [base64.b64encode(png()).decode()]})


def test_errors(runtime: FakeRuntime) -> None:
    runtime.caps = Capabilities(images=True, box=False, raw_scores=True)
    client = TestClient(create_app(S1GHT(runtime), api_key="secret"))
    auth = {"Authorization": "Bearer secret"}

    r = client.post("/v1/systemone", json={"state": "x", "questions": QUESTIONS})
    assert r.status_code == 401 and r.json()["error"]["type"] == "authentication_error"

    r = client.post("/v1/systemone", content=b"{nope", headers={**auth, "Content-Type": "application/json"})
    assert r.status_code == 400

    r = client.post("/v1/systemone", json={"state": "x"}, headers=auth)
    assert r.status_code == 422 and "questions" in r.json()["error"]["message"]

    r = client.post("/v1/systemone", json={"state": "x", "questions": {"q": {"type": "nope"}}}, headers=auth)
    assert r.status_code == 422

    box = {"where": Box("the bus").to_wire()}
    r = client.post("/v1/systemone", json={"state": "x", "questions": box}, headers=auth)
    assert r.status_code == 422 and "box" in r.json()["error"]["message"]

    r = client.post(
        "/v1/systemone", content=b"x", headers={**auth, "Content-Type": "application/json", "Content-Encoding": "br"}
    )
    assert r.status_code == 415


def test_images_resized_to_runtime_limits(runtime: FakeRuntime) -> None:
    seen: list[tuple[int, int]] = []
    decide = runtime.decide

    def spy(req: SystemOneRequest):
        seen.extend(img.size for img in req.assets().values())
        return decide(req)

    runtime.caps = Capabilities(images=True, raw_scores=True, image_px=(280, 1568))
    runtime.decide = spy  # type: ignore[method-assign]
    client = TestClient(create_app(S1GHT(runtime)))
    request = {
        "model": "m",
        "state": {"photo": "a"},
        "assets": {"a": {"format": "png", "data_base64": base64.b64encode(png((3000, 400))).decode()}},
        "questions": {"bus": QUESTIONS["bus"]},
    }
    assert client.post("/v1/systemone", json=request).status_code == 200
    w, h = seen[0]
    assert w <= 1568 and h >= 280


def test_models_and_health(client: TestClient) -> None:
    models = client.get("/v1/models").json()["models"]
    assert models[0]["name"] == "fake-served" and len(models[0]["release_date"]) == 10
    assert client.get("/health").json()["status"] == "ok"


def test_choice_question_object_on_the_wire(client: TestClient) -> None:
    q = {"c": Choice("Which?", ["a", "b", "c"]).to_wire()}
    r = client.post("/v1/systemone", json={"state": "s", "questions": q})
    assert r.json()["answers"]["c"]["type"] == "choice"


# --- the official TypeSafe SDK against a live server -----------------------------------------


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def live(runtime: FakeRuntime):
    uvicorn = pytest.importorskip("uvicorn")
    port = free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(S1GHT(runtime)), port=port, log_level="warning"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.monotonic() + 10
    while not server.started and time.monotonic() < deadline:
        time.sleep(0.02)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    thread.join(5)


def test_typesafe_sdk_connects_with_base_url(live: str, runtime: FakeRuntime, monkeypatch) -> None:
    ts = pytest.importorskip("typesafe_sdk")
    monkeypatch.setenv("TYPESAFE_BASE_URL", live)
    monkeypatch.setenv("TYPESAFE_API_KEY", "unused")
    data = png()
    with ts.TypeSafeClient() as sdk:
        assert sdk.models.list().models[0].name == "fake-served"
        photo = ts.Image(data)
        resp = sdk.system_one(
            model="jev-latest",
            state={"photo": photo},
            questions={
                "bus": ts.Noul(instructions="Is there a bus in `photo`?"),
                "color": ts.Choice(instructions="What color is it?", criteria={"red": None, "blue": None}),
            },
        )
    assert resp.model == "fake-served"
    direct = runtime.decide(
        SystemOneRequest(
            "x",
            {"photo": Image(data)},
            {"bus": Noul("Is there a bus in IMG?"), "color": Choice("What color is it?", ["red", "blue"])},
        )
    )
    assert resp.choices["color"].choice == direct.answers["color"].choice  # type: ignore[union-attr]
    assert resp.nouls["bus"].noul == pytest.approx(direct.answers["bus"].noul)  # type: ignore[union-attr]


def test_our_client_reports_server_errors(live: str, runtime: FakeRuntime) -> None:
    runtime.caps = Capabilities(images=True, raw_scores=True)  # no Box
    remote = SystemOneClient("typesafe", "m", base_url=live, max_retries=0)
    with pytest.raises(SystemOneError) as e:
        remote.decide(SystemOneRequest("m", "x", {"where": Box("the bus")}))
    assert e.value.status == 422
