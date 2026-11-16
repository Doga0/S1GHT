"""Option-letter readout: prompts, debiasing and the logprob server client (against a fake server)."""

from __future__ import annotations

import base64
import io
import json

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage

from s1ght import Box, Choice, Image, Noul, Score
from s1ght.runtimes import letters
from s1ght.runtimes.base import CapabilityError, Runtime
from s1ght.runtimes.server.logprob import LogprobClient
from s1ght.runtimes.server.systemone import SystemOneError
from s1ght.schema.systemone import ChoiceAnswer, NoulAnswer, ScoreAnswer, SystemOneRequest

QUESTIONS = {
    "bus": Noul("Is there a bus in `photo`?", criteria={"true": "a bus is visible"}),
    "color": Choice("What color is the bus?", {"red": None, "blue": "light or dark blue", "green": None}),
    "busy": Score("How busy is the street?", ["empty", "a few people", "crowded"]),
}
TRUE = {"bus": [3.0, 1.0], "color": [0.0, 2.0, -1.0], "busy": [0.5, 1.5, 0.0]}  # by option, RawScores order
A_BIAS = 2.4


def png(size: tuple[int, int] = (300, 200)) -> bytes:
    buf = io.BytesIO()
    PILImage.new("RGB", size, (30, 60, 200)).save(buf, "PNG")
    return buf.getvalue()


def biased_letters(prompt: letters.Prompt) -> dict[str, float]:
    """A model that sees TRUE logits but adds A_BIAS to whatever option is shown as "A"."""
    logits = TRUE[prompt.qid]
    return {c: logits[o] + (A_BIAS if i == 0 else 0.0) for i, (c, o) in enumerate(zip(prompt.letters, prompt.order))}


# --- prompts ---------------------------------------------------------------------------------


def test_options_and_prompt() -> None:
    wire = {qid: q.to_wire() for qid, q in QUESTIONS.items()}
    assert letters.options(wire["bus"]) == ["yes: a bus is visible", "no"]
    assert letters.options(wire["color"]) == ["red", "blue: light or dark blue", "green"]
    assert letters.options(wire["busy"])[2] == "level 2: crowded"
    with pytest.raises(ValueError, match="box"):
        letters.options(Box("the bus").to_wire())

    p = letters.build(0, "color", {"photo": "[image 1]"}, wire["color"], 1, [2, 1, 0])
    system, user = p.messages
    assert system["role"] == "system" and "letter" in system["content"]
    assert user["content"][:2] == [{"type": "text", "text": "[image 1]"}, {"type": "image", "index": 0}]
    task = json.loads(user["content"][2]["text"].rsplit("\n", 1)[0])
    assert task["input"] == {"photo": "[image 1]"} and task["criterion"] == "What color is the bus?"
    assert [o["letter"] + o["description"] for o in task["options"]] == ["Agreen", "Bblue: light or dark blue", "Cred"]
    assert p.letters == "ABC"
    with pytest.raises(ValueError, match="permutation"):
        letters.build(0, "color", "s", wire["color"], 0, [0, 0, 1])
    many = Choice("?", [f"o{i}" for i in range(27)]).to_wire()
    with pytest.raises(ValueError, match="26"):
        letters.build(0, "q", "s", many, 0, range(27))


def test_letter_logprobs_and_tokens() -> None:
    lp = letters.letter_logprobs({"A": 2.0, "B": 0.0}, "ABC", floor=-5.0)
    assert np.exp(lp).sum() == pytest.approx(1.0) and lp[0] > lp[1] > lp[2]
    missing = letters.letter_logprobs({"B": -1.0}, "AB")  # default floor: 1 nat below the lowest seen
    assert missing[1] - missing[0] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="top tokens"):
        letters.letter_logprobs({"Z": 0.0}, "AB")
    merged = letters.merge_token_scores([("A", -0.5), (" A", -1.0), ("B", -2.0), ("a", -0.1), ("The", -3.0)])
    assert set(merged) == {"A", "B"} and merged["A"] == pytest.approx(np.logaddexp(-0.5, -1.0))


def test_flip_cancels_position_bias() -> None:
    items = [("state", {qid: q.to_wire() for qid, q in QUESTIONS.items()}, 0)]
    calls: list[letters.Prompt] = []

    def score(prompts):
        calls.extend(prompts)
        return [letters.letter_logprobs(biased_letters(p), p.letters) for p in prompts]

    never = letters.read_all(items, score, "never")[0]
    assert never["bus"][0] - never["bus"][1] == pytest.approx(2.0 + A_BIAS)  # biased toward "yes"
    calls.clear()
    auto = letters.read_all(items, score, "auto")[0]
    assert len(calls) == 8  # every gap is under 6 nat: 2 orders for the Noul, 3 rotations for the others
    assert auto["bus"][0] - auto["bus"][1] == pytest.approx(2.0)  # binary: the bias cancels exactly
    for qid in ("color", "busy"):  # all rotations: the bias cancels exactly for any number of options
        assert auto[qid] - auto[qid][0] == pytest.approx(np.array(TRUE[qid]) - TRUE[qid][0])
    assert letters.extra_orders(3) == [(1, 2, 0), (2, 0, 1)]
    assert letters.extra_orders(5) == [(4, 3, 2, 1, 0)]

    calls.clear()
    letters.read_all(items, score, "auto", margin=0.3)
    assert len(calls) == 3  # every first gap is above 0.3 nat
    calls.clear()
    letters.read_all([("s", {"where": Box("x").to_wire()}, 0)], score, "always")
    assert calls == []


def test_fit_image() -> None:
    assert letters.fit_image(Image(png((3000, 1000)))).size == (1024, 352)
    small = letters.fit_image(PILImage.new("L", (100, 50)))
    assert small.mode == "RGB" and small.size[0] % 32 == 0 and small.size[0] * small.size[1] >= 256 * 256 * 0.9
    same = PILImage.new("RGB", (320, 256))
    assert letters.fit_image(same) is same


# --- logprob server client -------------------------------------------------------------------


class FakeServer:
    """An OpenAI-compatible endpoint answering from `biased_letters`, recording request bodies."""

    def __init__(self, top: int = 20) -> None:
        self.bodies: list[dict] = []
        self.top = top

    def __call__(self, request: httpx.Request) -> httpx.Response:
        body = json.loads(request.read())
        self.bodies.append(body)
        text = body["messages"][1]["content"][-1]["text"]
        task = json.loads(text.rsplit("\n", 1)[0])
        qid = next(q for q, w in QUESTIONS.items() if w.to_wire()["instructions"] == task["criterion"])
        shown = [o["description"] for o in task["options"]]
        opts = letters.options(QUESTIONS[qid].to_wire())
        prompt = letters.Prompt(0, qid, tuple(opts.index(d) for d in shown), [])
        scores = biased_letters(prompt)
        z = np.logaddexp.reduce(list(scores.values()))
        top = sorted(((f" {c}" if c == "B" else c, v - z) for c, v in scores.items()), key=lambda t: -t[1])
        top = top[: self.top] + [("The", -9.0)]
        content = [{"token": top[0][0], "logprob": top[0][1], "top_logprobs": [
            {"token": t, "logprob": lp} for t, lp in top]}]  # fmt: skip
        return httpx.Response(
            200, json={"choices": [{"logprobs": {"content": content}}], "usage": {"prompt_tokens": 50}}
        )


def test_logprob_client_answers() -> None:
    server = FakeServer()
    client = LogprobClient("Qwen/Qwen3-VL-4B-Instruct", base_url="http://vllm", transport=httpx.MockTransport(server))
    assert isinstance(client, Runtime)
    assert client.caps.max_options == 26 and not client.caps.box
    photo = Image(png((2000, 1000)))
    remote = client.decide(SystemOneRequest("m", {"photo": photo}, QUESTIONS))
    assert isinstance(remote.answers["bus"], NoulAnswer)
    assert remote.answers["bus"].noul == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert isinstance(remote.answers["color"], ChoiceAnswer) and remote.answers["color"].choice == "blue"
    assert isinstance(remote.answers["busy"], ScoreAnswer)
    assert remote.usage["input_tokens"] == 400
    with pytest.raises(ValueError, match="26"):
        client.decide(SystemOneRequest("m", "x", {"c": Choice("?", [f"o{i}" for i in range(27)])}))

    body = server.bodies[0]
    assert body["max_tokens"] == 1 and body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0.0
    url = body["messages"][1]["content"][1]["image_url"]["url"]
    sent = PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert url.startswith("data:image/png;base64,") and sent.size == (1024, 512)
    # the prompt `letters.build` renders
    p = letters.build(0, "bus", {"photo": "[image 1]"}, QUESTIONS["bus"].to_wire(), 1, [0, 1])
    assert body["messages"][1]["content"][-1] == p.messages[1]["content"][-1]
    assert body["messages"][0] == p.messages[0]


def test_logprob_client_flip_and_missing_letters() -> None:
    server = FakeServer(top=1)  # only the top letter is shown; the others get the floor
    client = LogprobClient("m", flip="never", transport=httpx.MockTransport(server))
    r = client.decide(SystemOneRequest("m", "a street", {"color": QUESTIONS["color"]}))
    assert len(server.bodies) == 1
    probs = r.answers["color"].probabilities  # type: ignore[union-attr]
    assert probs["red"] > 0.5  # "red" is shown as A and gets the bias; the rest sit at the floor
    assert probs["blue"] == pytest.approx(probs["green"])


def test_logprob_client_errors() -> None:
    def no_logprobs(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"message": {"content": "A"}}]})

    client = LogprobClient("m", transport=httpx.MockTransport(no_logprobs))
    with pytest.raises(SystemOneError, match="top_logprobs"):
        client.decide(SystemOneRequest("m", "x", {"bus": QUESTIONS["bus"]}))
    with pytest.raises(CapabilityError):
        client.decide(SystemOneRequest("m", "x", {"where": Box("the bus")}))

    def bad(request: httpx.Request) -> httpx.Response:
        return httpx.Response(400, json={"error": {"message": "image too large", "type": "BadRequestError"}})

    client = LogprobClient("m", transport=httpx.MockTransport(bad))
    with pytest.raises(SystemOneError, match="image too large"):
        client.decide(SystemOneRequest("m", "x", {"bus": QUESTIONS["bus"]}))
    with pytest.raises(ValueError, match="server"):
        LogprobClient("m", server="tgi")
