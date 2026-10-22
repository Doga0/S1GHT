"""Option-letter readout: prompts, debiasing, the native letter family and the logprob server client.

The native family is tested with its forward pass replaced, so no torch model is loaded; a
`slow` test compares real Qwen3-VL-4B weights against a running vLLM/llama-server when
`S1GHT_LOGPROB_URL` is set.
"""

from __future__ import annotations

import base64
import io
import json
import os

import httpx
import numpy as np
import pytest
from PIL import Image as PILImage

from s1ght import S1GHT, Box, Choice, Image, Noul, Score
from s1ght.runtimes import letters
from s1ght.runtimes.base import CapabilityError, Runtime
from s1ght.runtimes.native.families.semif_generic import SemifGenericFamily
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


# --- native letter family (forward pass replaced) --------------------------------------------


class FakeLetterFamily(SemifGenericFamily):
    """`forward` returns logits for a 26-letter vocabulary from `biased_letters`."""

    def __init__(self, **kw) -> None:
        super().__init__(**kw)
        self.model = object()
        self.letter_ids = {c: [i] for i, c in enumerate(letters.LETTERS)}
        self.seen: list[tuple[letters.Prompt, list]] = []

    def load(self, path, device, dtype) -> None:
        pass

    def forward(self, prompts, images):
        rows = np.full((len(prompts), 26), -30.0)
        for row, p in zip(rows, prompts, strict=True):
            for c, v in biased_letters(p).items():
                row[letters.LETTERS.index(c)] = v
            self.seen.append((p, [images[p.item][i] for i in range(len(images[p.item]))]))
        return rows, [100] * len(prompts)


def native(family: SemifGenericFamily) -> S1GHT:
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.native.engine import NativeRuntime

    spec = RuntimeSpec("native", "Qwen/Qwen3-VL-4B-Instruct", family="qwen_vl")
    return S1GHT(NativeRuntime(spec, family=family, path=".", device="cpu", dtype="fp32"))


def test_native_letter_family_answers() -> None:
    family = FakeLetterFamily(batch=4)
    model = native(family)
    assert model.runtime.caps.max_options == 26 and not model.runtime.caps.box
    photo = Image(png((2000, 1000)))
    r = model.decide({"photo": photo}, QUESTIONS)
    assert isinstance(r.answers["bus"], NoulAnswer)
    assert r.answers["bus"].noul == pytest.approx(1 / (1 + np.exp(-2.0)))
    assert isinstance(r.answers["color"], ChoiceAnswer) and r.answers["color"].choice == "blue"
    assert isinstance(r.answers["busy"], ScoreAnswer)
    assert r.usage["input_tokens"] == 800  # 2 + 3 + 3 prompts x 100 tokens
    assert family.seen[0][1][0].size == (1024, 512)  # S1GHT's resize, before the processor
    with pytest.raises(CapabilityError):
        model.decide("x", {"where": Box("the bus")})
    with pytest.raises(ValueError, match="26"):
        model.decide("x", {"c": Choice("?", [f"o{i}" for i in range(27)])})


def test_family_options() -> None:
    from s1ght.runtimes.native.families import get_family

    fam = get_family("qwen_vl", flip="never", batch=2)
    assert fam.name == "qwen_vl" and fam.flip == "never" and fam.template_kwargs == {"enable_thinking": False}
    with pytest.raises(ValueError, match="no options"):
        get_family("d1_omni", flip="never")
    with pytest.raises(ValueError, match="flip"):
        get_family("semif_generic", flip="sometimes")


def test_letter_token_ids() -> None:
    from s1ght.runtimes.native.families.semif_generic import letter_token_ids

    class Tok:
        def encode(self, text, add_special_tokens=False):
            return [ord(text.strip()) + (1000 if text.startswith(" ") else 0)]

    ids = letter_token_ids(Tok())
    assert ids["A"] == [65, 1065]

    class Split:
        def encode(self, text, add_special_tokens=False):
            return [1, 2]

    with pytest.raises(ValueError, match="single token"):
        letter_token_ids(Split())


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


def test_logprob_client_matches_native_family() -> None:
    server = FakeServer()
    client = LogprobClient("Qwen/Qwen3-VL-4B-Instruct", base_url="http://vllm", transport=httpx.MockTransport(server))
    assert isinstance(client, Runtime)
    photo = Image(png((2000, 1000)))
    req = SystemOneRequest("m", {"photo": photo}, QUESTIONS)
    remote = client.decide(req)
    local = native(FakeLetterFamily()).runtime.decide(req)
    assert remote.answers["bus"].noul == pytest.approx(local.answers["bus"].noul)  # type: ignore[union-attr]
    assert remote.answers["color"].probabilities == pytest.approx(local.answers["color"].probabilities)  # type: ignore[union-attr]
    assert remote.answers["busy"].score == pytest.approx(local.answers["busy"].score)  # type: ignore[union-attr]
    assert remote.usage["input_tokens"] == 400

    body = server.bodies[0]
    assert body["max_tokens"] == 1 and body["logprobs"] is True and body["top_logprobs"] == 20
    assert body["chat_template_kwargs"] == {"enable_thinking": False} and body["temperature"] == 0.0
    url = body["messages"][1]["content"][1]["image_url"]["url"]
    sent = PILImage.open(io.BytesIO(base64.b64decode(url.split(",", 1)[1])))
    assert url.startswith("data:image/png;base64,") and sent.size == (1024, 512)
    # the same prompt text the native family renders
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


# --- live: Qwen3-VL-4B native vs a logprob server --------------------------------------------


@pytest.mark.slow
@pytest.mark.network
def test_qwen3_vl_native_matches_server() -> None:
    """Needs the native extra, ~9 GB of weights and a vLLM or llama-server at S1GHT_LOGPROB_URL."""
    url = os.environ.get("S1GHT_LOGPROB_URL")
    if not url:
        pytest.skip("set S1GHT_LOGPROB_URL to a server running Qwen3-VL-4B-Instruct")
    pytest.importorskip("torch")
    import ultralytics

    bus = Image(os.path.join(os.path.dirname(ultralytics.__file__), "assets", "bus.jpg"))
    qs = {"bus": Noul("Is there a bus in `photo`?"), "where": Choice("Where is this?", ["street", "beach", "kitchen"])}
    server = S1GHT(
        f"vllm://{url.split('://')[-1]}/{os.environ.get('S1GHT_LOGPROB_MODEL', 'Qwen/Qwen3-VL-4B-Instruct')}"
    )
    with S1GHT("qwen3-vl-4b") as local:
        a, b = local.decide({"photo": bus}, qs), server.decide({"photo": bus}, qs)
    assert a.answers["where"].choice == b.answers["where"].choice  # type: ignore[union-attr]
    assert a.answers["bus"].noul == pytest.approx(b.answers["bus"].noul, abs=0.05)  # type: ignore[union-attr]
