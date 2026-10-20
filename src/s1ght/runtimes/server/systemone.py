"""`/v1/systemone` client for TypeSafe, Ollama and Ollaya.

One client, three provider presets that differ in where images go:

- **typesafe**: `POST {base}/v1/systemone`; images in `assets`, sent as multipart parts with
  the `TypeSafe-Danger-Zone-Images: true` header. Box questions need `jev-preview`/`jev-1.14.0`.
- **ollama**: `POST {base}/v1/systemone`; images as a top-level base64 `images` list (Clef and
  Clef Flash read images). No Box; Choice takes at most 26 options.
- **ollaya**: `POST {base}/api/decide`, because Ollaya's `/v1/systemone` is TypeSafe-identical
  and takes no images. Images go in `images` like Ollama and are re-encoded to PNG, the only
  format its vision runners read. The response is TypeSafe's plus extra fields, which are ignored.

Providers return calibrated probabilities. `raw()` turns them back into log-probabilities so
S1GHT's own calibration can be fitted on top.
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

import httpx

from s1ght.runtimes.base import Capabilities, RawScores, RuntimeBase, answers_to_raw, validate_request
from s1ght.schema.assets import Image
from s1ght.schema.systemone import Dialect, SystemOneRequest, SystemOneResponse

Provider = Literal["typesafe", "ollama", "ollaya"]

RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 529})


@dataclass(frozen=True)
class ProviderSpec:
    """How to reach one `/v1/systemone` provider."""

    base_url: str
    path: str
    dialect: Dialect
    caps: Capabilities
    base_url_env: str
    api_key_env: str
    image_format: str | None = None  # re-encode every image to this format before sending


PROVIDERS: dict[str, ProviderSpec] = {
    "typesafe": ProviderSpec(
        base_url="https://api.typesafe.ai",
        path="/v1/systemone",
        dialect="typesafe",
        caps=Capabilities(images=True, box=True, raw_scores=True, max_options=255, image_px=(280, 1568)),
        base_url_env="TYPESAFE_BASE_URL",
        api_key_env="TYPESAFE_API_KEY",
    ),
    "ollama": ProviderSpec(
        base_url="http://127.0.0.1:11434",
        path="/v1/systemone",
        dialect="ollama",
        caps=Capabilities(images=True, raw_scores=True, max_options=26),
        base_url_env="OLLAMA_HOST",
        api_key_env="OLLAMA_API_KEY",
    ),
    "ollaya": ProviderSpec(
        base_url="http://127.0.0.1:11435",
        path="/api/decide",
        dialect="ollama",
        caps=Capabilities(images=True, raw_scores=True, max_options=255),
        base_url_env="OLLAYA_HOST",
        api_key_env="OLLAYA_API_KEY",
        image_format="png",
    ),
}


class SystemOneError(RuntimeError):
    """A provider rejected a request or failed to answer it.

    Attributes:
        status: HTTP status code, or 0 when no response arrived (codes TIMEOUT, UNREACHABLE).
        code: Provider error code when given (Ollaya: `MODEL_NOT_FOUND`, ...), else None.
        body: Decoded JSON error body, or the raw text.
    """

    def __init__(self, status: int, message: str, code: str | None = None, body: Any = None) -> None:
        head = f"HTTP {status}{f' {code}' if code else ''}" if status else (code or "error")
        super().__init__(f"{head}: {message}")
        self.status, self.code, self.body = status, code, body


def _normalize_url(url: str) -> str:
    url = url.strip().rstrip("/")
    return url if "://" in url else f"http://{url}"


def _error(resp: httpx.Response) -> SystemOneError:
    try:
        body: Any = resp.json()
    except ValueError:
        return SystemOneError(resp.status_code, resp.text[:500] or resp.reason_phrase, body=resp.text)
    message, code = resp.reason_phrase, None
    if isinstance(body, dict):
        err = body.get("error")
        if isinstance(err, str):
            message, code = err, body.get("code")
        elif isinstance(err, dict):
            message, code = str(err.get("message", err)), err.get("code") or err.get("type")
        elif "detail" in body:
            message = json.dumps(body["detail"])
    return SystemOneError(resp.status_code, message, code, body)


class SystemOneClient(RuntimeBase):
    """Runtime backed by a `/v1/systemone` provider.

    Args:
        provider: "typesafe", "ollama" or "ollaya".
        model: Model name on that provider, e.g. "jev-preview", "clef-flash", "d1:3b-vision".
            Every request is sent with this model.
        base_url: Server URL; defaults to the provider's env var (`TYPESAFE_BASE_URL`,
            `OLLAMA_HOST`, `OLLAYA_HOST`), then its standard address.
        api_key: Bearer token; defaults to `TYPESAFE_API_KEY` / `OLLAMA_API_KEY` / `OLLAYA_API_KEY`.
        timeout: Seconds per HTTP request.
        max_retries: Retries on 408, 429 and 5xx, honouring `Retry-After`.
        backoff: First retry delay in seconds; doubles each retry.
        transport: Optional httpx transport (tests use `httpx.MockTransport`).
    """

    def __init__(
        self,
        provider: Provider,
        model: str,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_retries: int = 2,
        backoff: float = 0.5,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        if provider not in PROVIDERS:
            raise ValueError(f"unknown provider {provider!r}; expected one of {sorted(PROVIDERS)}")
        self.provider = provider
        self.spec = PROVIDERS[provider]
        self.caps = self.spec.caps
        self.model = model
        self.base_url = _normalize_url(base_url or os.environ.get(self.spec.base_url_env) or self.spec.base_url)
        api_key = api_key if api_key is not None else os.environ.get(self.spec.api_key_env)
        headers = {"Authorization": f"Bearer {api_key}"} if api_key else {}
        self.max_retries, self.backoff = max_retries, backoff
        self._sleep: Callable[[float], None] = time.sleep
        self._http = httpx.Client(base_url=self.base_url, headers=headers, timeout=timeout, transport=transport)

    def _prepare(self, req: SystemOneRequest) -> SystemOneRequest:
        req = replace(req, model=self.model)
        validate_request(self.caps, req)
        fmt = self.spec.image_format
        if fmt is not None:
            req = req.map_images(lambda img: img if img.format == fmt else Image(img.to_pil(), format=fmt, ref=img.ref))
        return req

    def _post(self, req: SystemOneRequest) -> httpx.Response:
        body, parts = req.to_wire(self.spec.dialect)
        headers = {}
        if self.spec.dialect == "typesafe" and req.needs_images_flag:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
        if parts:
            files: dict[str, tuple[str | None, bytes, str]] = {
                "request": (None, json.dumps(body).encode(), "application/json")
            }
            for ref, data in parts.items():
                fmt = body["assets"][ref]["format"]
                files[ref] = (ref, data, "image/png" if fmt == "png" else "image/jpeg")
            kwargs: dict[str, Any] = {"files": files}
        else:
            kwargs = {"json": body}

        for attempt in range(self.max_retries + 1):
            try:
                resp = self._http.post(self.spec.path, headers=headers, **kwargs)
            except httpx.TimeoutException as e:
                timeout = self._http.timeout.read
                raise SystemOneError(
                    0, f"no answer from {self.base_url} within {timeout}s; pass a larger timeout=", "TIMEOUT"
                ) from e
            except httpx.TransportError as e:
                raise SystemOneError(0, f"cannot reach {self.base_url}: {e}", "UNREACHABLE") from e
            if resp.status_code < 400:
                return resp
            if resp.status_code not in RETRY_STATUS or attempt == self.max_retries:
                raise _error(resp)
            retry_after = resp.headers.get("Retry-After")
            delay = float(retry_after) if retry_after and retry_after.isdigit() else self.backoff * 2**attempt
            self._sleep(delay)
        raise AssertionError("unreachable")

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        """Send one request and decode the provider's answer."""
        resp = self._post(self._prepare(req))
        try:
            return SystemOneResponse.from_wire(resp.json())
        except (ValueError, KeyError) as e:
            raise SystemOneError(resp.status_code, f"malformed response: {e}", body=resp.text) from e

    def raw(self, req: SystemOneRequest) -> RawScores:
        """Log-probabilities of the provider's answers, in `RawScores` option order (Box questions skipped)."""
        return answers_to_raw(req.questions, self.decide(req).answers)

    def close(self) -> None:
        self._http.close()

    def __repr__(self) -> str:
        return f"SystemOneClient(provider={self.provider!r}, model={self.model!r}, base_url={self.base_url!r})"
