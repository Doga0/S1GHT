"""OpenAI-compatible logprob client: vLLM and llama-server as decision back ends.

Each question is sent as one `POST /v1/chat/completions` with `max_tokens=1`, `logprobs` and
`top_logprobs`, and the answer is read from the first token's top log-probabilities, with the
same prompt, images and option-order debiasing as the native letter families
(`s1ght.runtimes.letters`). Requests run concurrently.

    vllm serve Qwen/Qwen3-VL-4B-Instruct --max-model-len 8192
    model = S1GHT("vllm://localhost:8000/Qwen/Qwen3-VL-4B-Instruct")

    llama-server -hf Qwen/Qwen3-VL-4B-Instruct-GGUF:Q8_0 --port 8080
    model = S1GHT("llama-server://localhost:8080")

Thinking is turned off through `chat_template_kwargs` (Qwen3). Only letters among the top
`top_logprobs` tokens are seen; a missing letter is put 1 nat below the least likely one shown.
"""

from __future__ import annotations

import base64
import io
import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any

import httpx
import numpy as np

from s1ght.runtimes import letters
from s1ght.runtimes.base import RawScores, RuntimeBase, scores_to_answers, validate_request
from s1ght.runtimes.server.systemone import RETRY_STATUS, SystemOneError, _error, _normalize_url
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse

DEFAULT_URLS = {"vllm": "http://127.0.0.1:8000", "llama_server": "http://127.0.0.1:8080"}


class LogprobClient(RuntimeBase):
    """Runtime backed by an OpenAI-compatible chat server that returns top log-probabilities.

    Args:
        model: Served model id (vLLM: the HF repo id it serves). llama-server ignores it.
        server: "vllm" or "llama_server"; picks the default address.
        base_url: Server URL; defaults to `S1GHT_LOGPROB_URL`, then the server's usual address.
        api_key: Bearer token (`vllm serve --api-key`); defaults to `S1GHT_LOGPROB_API_KEY`.
        flip: Option-order debiasing: "auto", "always" or "never".
        margin: Gap in nat under which "auto" asks again in other option orders.
        max_side: Longer image side in pixels (same default as the native families).
        top_logprobs: Top tokens to request (vLLM allows 20 unless started with --max-logprobs).
        concurrency: Requests in flight.
        chat_template_kwargs: Sent with every request; `{"enable_thinking": False}` by default.
        timeout: Seconds per HTTP request.
        max_retries: Retries on 408, 429 and 5xx.
        transport: Optional httpx transport (tests).
    """

    caps = letters.CAPS

    def __init__(
        self,
        model: str,
        *,
        server: str = "vllm",
        base_url: str | None = None,
        api_key: str | None = None,
        flip: letters.Flip = "auto",
        margin: float = letters.FLIP_MARGIN,
        max_side: int = letters.MAX_SIDE,
        top_logprobs: int = 20,
        concurrency: int = 4,
        chat_template_kwargs: dict[str, Any] | None = None,
        timeout: float = 120.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if server not in DEFAULT_URLS:
            raise ValueError(f"server must be one of {sorted(DEFAULT_URLS)}, got {server!r}")
        if flip not in ("auto", "always", "never"):
            raise ValueError(f"flip must be 'auto', 'always' or 'never', got {flip!r}")
        self.model, self.server = model, server
        self.flip, self.margin, self.max_side, self.top_logprobs = flip, margin, max_side, top_logprobs
        self.concurrency = max(1, concurrency)
        self.chat_template_kwargs = {"enable_thinking": False} if chat_template_kwargs is None else chat_template_kwargs
        self.base_url = _normalize_url(base_url or os.environ.get("S1GHT_LOGPROB_URL") or DEFAULT_URLS[server])
        api_key = api_key if api_key is not None else os.environ.get("S1GHT_LOGPROB_API_KEY")
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.max_retries, self.backoff = max_retries, backoff
        self._sleep: Callable[[float], None] = time.sleep
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    # --- one prompt ---------------------------------------------------------------------------

    def body(self, prompt: letters.Prompt, images: list[str]) -> dict[str, Any]:
        """The chat completion request for one prompt; `images` are data URLs."""
        messages = []
        for m in prompt.messages:
            content = m["content"]
            if isinstance(content, list):
                content = [
                    {"type": "image_url", "image_url": {"url": images[p["index"]]}} if p["type"] == "image" else p
                    for p in content
                ]
            messages.append({"role": m["role"], "content": content})
        out: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "max_tokens": 1,
            "temperature": 0.0,
            "logprobs": True,
            "top_logprobs": self.top_logprobs,
        }
        if self.chat_template_kwargs:
            out["chat_template_kwargs"] = self.chat_template_kwargs
        return out

    def _post(self, body: dict[str, Any]) -> dict[str, Any]:
        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post("/v1/chat/completions", json=body)
            except httpx.TimeoutException as e:
                raise SystemOneError(
                    0, f"no answer from {self.base_url} in time; pass a larger timeout=", "TIMEOUT"
                ) from e
            except httpx.TransportError as e:
                raise SystemOneError(0, f"cannot reach {self.base_url}: {e}", "UNREACHABLE") from e
            if resp.status_code < 400:
                return resp.json()
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise _error(resp)
            self._sleep(self.backoff * 2**attempt)
        raise AssertionError("unreachable")

    def _score(self, prompt: letters.Prompt, images: list[str]) -> tuple[np.ndarray, int]:
        data = self._post(self.body(prompt, images))
        try:
            first = data["choices"][0]["logprobs"]["content"][0]
            top = [(t["token"], float(t["logprob"])) for t in first["top_logprobs"]]
        except (KeyError, IndexError, TypeError) as e:
            raise SystemOneError(
                200, f"response has no top_logprobs ({e}); is this an OpenAI-compatible server?"
            ) from e
        if all(t != first["token"] for t, _ in top):  # the sampled token is usually listed already
            top.append((first["token"], float(first["logprob"])))
        floor = min(lp for _, lp in top) - 1.0
        logp = letters.letter_logprobs(letters.merge_token_scores(top), prompt.letters, floor)
        return logp, int((data.get("usage") or {}).get("prompt_tokens", 0))

    # --- runtime ------------------------------------------------------------------------------

    def _read(self, reqs: list[SystemOneRequest]) -> list[tuple[dict[str, np.ndarray], int]]:
        items, urls = [], []
        for r in reqs:
            validate_request(self.caps, r)
            state, questions, images = letters.split(r)
            items.append((state, questions, len(images)))
            urls.append([_data_url(letters.fit_image(img, self.max_side)) for img in images])
        tokens = [0] * len(reqs)

        def score(prompts: list[letters.Prompt]) -> list[np.ndarray]:
            with ThreadPoolExecutor(self.concurrency) as pool:
                results = list(pool.map(lambda p: self._score(p, urls[p.item]), prompts))
            for p, (_, n) in zip(prompts, results, strict=True):
                tokens[p.item] += n
            return [lp for lp, _ in results]

        reads = letters.read_all(items, score, self.flip, self.margin)
        return list(zip(reads, tokens, strict=True))

    def raw(self, req: SystemOneRequest) -> RawScores:
        return RawScores(self._read([req])[0][0])

    def raw_batch(self, reqs: list[SystemOneRequest]) -> list[RawScores]:
        """`raw` for several requests, their prompts sent concurrently."""
        return [RawScores(r) for r, _ in self._read(reqs)]

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        return self.decide_batch([req])[0]

    def decide_batch(self, reqs: list[SystemOneRequest]) -> list[SystemOneResponse]:
        return [
            SystemOneResponse(
                model=self.model,
                answers=dict(scores_to_answers(r.questions, RawScores(scores))),
                usage={"input_tokens": n, "output_tokens": 0},
            )
            for r, (scores, n) in zip(reqs, self._read(reqs), strict=True)
        ]

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:
        return f"LogprobClient(server={self.server!r}, model={self.model!r}, base_url={self.base_url!r})"


def _data_url(pil: Any) -> str:
    """PNG data URL: lossless, so the server decodes exactly the pixels the native family sees."""
    buf = io.BytesIO()
    pil.save(buf, "PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")
