"""`s1ght serve`: any S1GHT model behind a `/v1/systemone` endpoint that takes images.

    s1ght serve model=d1-omni-600m port=11436
    TYPESAFE_BASE_URL=http://127.0.0.1:11436 python my_typesafe_app.py

Routes:

- `POST /v1/systemone`: the TypeSafe API. JSON with base64 `assets`, or multipart with the JSON
  in a part named `request` and one file part per asset. Requests with a top-level `images`
  list (Ollama's dialect) are accepted too. The `TypeSafe-Danger-Zone-Images` header is not
  required: images are always on here.
- `POST /api/decide`: Ollaya's image route (Ollama dialect), so Ollaya clients can point here.
- `GET /v1/models`, `GET /health`.

Every request is answered by the one model this server was started with, whatever its `model`
field says; the response's `model` names the served model. Images outside the runtime's size
limits are resized to fit instead of rejected. Bodies may be gzip or zstd compressed (zstd needs
Python 3.14+ or `backports.zstd`). Errors are JSON: `{"error": {"type", "message"}}` with
TypeSafe's status codes (401, 413, 415, 422; 400 for unreadable bodies).
"""

# no rom __future__ import annotations: FastAPI must resolve the handlers' Request annotation,
# which is imported inside create_app

import base64
import datetime
import json
import re
import secrets
import threading
from collections.abc import AsyncIterator
from typing import TYPE_CHECKING, Any

from s1ght.engine.crops import fit_to_limits
from s1ght.runtimes.base import CapabilityError
from s1ght.schema.assets import Image
from s1ght.schema.questions import question_from_wire
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse

if TYPE_CHECKING:
    from fastapi import FastAPI

    from s1ght.model import S1GHT

MAX_BODY = 32 * 1024 * 1024  # TypeSafe's request limit
_LABEL = re.compile(r"\[image (\d+)\]")


class RequestError(Exception):
    """A request the server refuses; carries the HTTP status and TypeSafe-style error type."""

    def __init__(self, status: int, kind: str, message: str) -> None:
        super().__init__(message)
        self.status, self.kind = status, kind


def ollama_request(body: dict[str, Any]) -> SystemOneRequest:
    """Decode an Ollama-dialect request, putting each image back where its "[image N]" label was.

    A value that is exactly a label becomes that `Image`; a label inside longer text becomes the
    image's reference, with the image kept in `extra_assets`.
    """
    images = [Image(base64.b64decode(s, validate=True)) for s in body.get("images") or []]
    inline: dict[str, Image] = {}

    def image(n: str) -> Image:
        i = int(n) - 1
        if not 0 <= i < len(images):
            raise ValueError(f"[image {n}] is referenced but only {len(images)} images were sent")
        return images[i]

    def walk(obj: Any) -> Any:
        if isinstance(obj, dict):
            return {k: walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [walk(v) for v in obj]
        if isinstance(obj, str):
            whole = _LABEL.fullmatch(obj.strip())
            if whole:
                return image(whole.group(1))

            def ref(m: re.Match[str]) -> str:
                img = image(m.group(1))
                inline[img.ref] = img
                return img.ref

            return _LABEL.sub(ref, obj)
        return obj

    questions = {qid: question_from_wire(walk(q)) for qid, q in body["questions"].items()}
    state = walk(body["state"])
    req = SystemOneRequest(str(body.get("model", "")), state, questions)
    placed = req.assets()
    req.extra_assets = [img for img in images if img.ref in inline or img.ref not in placed]
    return req


def parse_request(body: dict[str, Any], parts: dict[str, bytes] | None = None) -> SystemOneRequest:
    """Decode a TypeSafe or Ollama wire request; raises `RequestError` (422) on invalid input."""
    if not isinstance(body, dict):
        raise RequestError(422, "invalid_request_error", "request body must be a JSON object")
    missing = [k for k in ("state", "questions") if k not in body]
    if missing:
        raise RequestError(422, "invalid_request_error", f"missing required field(s): {', '.join(missing)}")
    if not isinstance(body["questions"], dict) or not body["questions"]:
        raise RequestError(422, "invalid_request_error", "questions must be a non-empty object")
    try:
        if "images" in body:
            return ollama_request(body)
        return SystemOneRequest.from_wire({"model": "", **body}, parts)
    except (ValueError, KeyError, TypeError) as e:
        raise RequestError(422, "invalid_request_error", str(e)) from None


def decompress(body: bytes, encoding: str | None) -> bytes:
    """Undo `Content-Encoding` (gzip or zstd, one layer; the TypeSafe SDK sends zstd when it can)."""
    enc = (encoding or "identity").strip().lower()
    if enc == "identity":
        return body
    if "," in enc:
        raise RequestError(415, "unsupported_media_type", "only one Content-Encoding layer is accepted")
    try:
        if enc in ("gzip", "x-gzip"):
            import zlib

            d = zlib.decompressobj(wbits=31)
            out = d.decompress(body, MAX_BODY + 1)
        elif enc == "zstd":
            try:
                from compression.zstd import decompress as unzstd  # type: ignore[import-not-found]
            except ImportError:
                try:
                    from backports.zstd import decompress as unzstd  # type: ignore[import-not-found,import-untyped,no-redef]
                except ImportError:
                    raise RequestError(
                        415, "unsupported_media_type", "zstd bodies need Python 3.14+ or `pip install backports.zstd`"
                    ) from None
            out = unzstd(body)
        else:
            raise RequestError(415, "unsupported_media_type", f"unsupported Content-Encoding {encoding!r}")
    except RequestError:
        raise
    except Exception as e:  # noqa: BLE001 - zlib and zstd raise their own error types
        raise RequestError(400, "invalid_request_error", f"malformed {enc} body: {e}") from None
    if len(out) > MAX_BODY:
        raise RequestError(413, "request_too_large", f"decompressed body is over {MAX_BODY // 2**20} MiB")
    return out


def fit_images(req: SystemOneRequest, image_px: tuple[int, int] | None) -> SystemOneRequest:
    """Resize images whose sides fall outside `image_px` (min, max); others pass through untouched."""
    if image_px is None:
        return req
    lo, hi = image_px

    def fit(img: Image) -> Image:
        w, h = img.size
        if lo <= w <= hi and lo <= h <= hi:
            return img
        return Image(fit_to_limits(img.to_array(), image_px), format=img.format, ref=img.ref)

    return req.map_images(fit)


class Server:
    """The request handling behind `create_app`, without FastAPI (unit-testable).

    One lock serializes runtime calls: runtimes are only required to be safe from one thread.
    """

    def __init__(self, model: "S1GHT", *, api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key
        self.lock = threading.Lock()

    def authorize(self, header: str | None) -> None:
        if self.api_key is None:
            return
        scheme, _, token = (header or "").partition(" ")
        if scheme.lower() != "bearer" or not secrets.compare_digest(token.strip(), self.api_key):
            raise RequestError(401, "authentication_error", "missing or invalid API key")

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        runtime = self.model.runtime
        req = fit_images(req, runtime.caps.image_px)
        req.model = self.model.name
        try:
            with self.lock:
                resp = runtime.decide(req)
        except CapabilityError as e:
            raise RequestError(422, "invalid_request_error", str(e)) from None
        except ValueError as e:
            raise RequestError(422, "invalid_request_error", str(e)) from None
        resp.model = self.model.name
        # TypeSafe clients require both counts
        resp.usage = {**resp.usage, "input_tokens": int(resp.usage.get("input_tokens", 0)),
                      "output_tokens": int(resp.usage.get("output_tokens", 0))}  # fmt: skip
        return resp


def create_app(model: "S1GHT", *, api_key: str | None = None) -> "FastAPI":
    """A FastAPI app serving `model` (needs the `serve` extra: `pip install s1ght-visual-decisions[serve]`).

    Args:
        model: The loaded model; every request is answered by it.
        api_key: If given, requests must send `Authorization: Bearer <api_key>`.
    """
    try:
        from fastapi import FastAPI, Request
        from fastapi.concurrency import run_in_threadpool
        from fastapi.responses import JSONResponse
        from starlette.datastructures import UploadFile
        from starlette.formparsers import MultiPartException, MultiPartParser
    except ImportError as e:
        raise ImportError("s1ght serve needs FastAPI: pip install 's1ght-visual-decisions[serve]'") from e

    server = Server(model, api_key=api_key)
    started = datetime.date.today().isoformat()
    app = FastAPI(title="S1GHT", summary=f"/v1/systemone with images, served by {model.name}")
    app.state.s1ght = server

    @app.exception_handler(RequestError)
    async def request_error(_: Request, e: RequestError) -> JSONResponse:
        return JSONResponse({"error": {"type": e.kind, "message": str(e)}}, status_code=e.status)

    async def read(request: Request) -> tuple[dict[str, Any], dict[str, bytes] | None]:
        server.authorize(request.headers.get("authorization"))
        size = request.headers.get("content-length")
        if size is not None and size.isdigit() and int(size) > MAX_BODY:
            raise RequestError(413, "request_too_large", f"request body is over {MAX_BODY // 2**20} MiB")
        body = await request.body()
        body = decompress(body, request.headers.get("content-encoding"))
        try:
            if request.headers.get("content-type", "").startswith("multipart/form-data"):

                async def stream() -> AsyncIterator[bytes]:
                    yield body

                parser = MultiPartParser(request.headers, stream(), max_part_size=MAX_BODY)  # type: ignore[arg-type]
                raw, parts = None, {}
                for name, value in (await parser.parse()).multi_items():
                    data = await value.read() if isinstance(value, UploadFile) else value.encode()
                    if name == "request":
                        raw = data
                    else:
                        parts[name] = data
                if raw is None:
                    raise RequestError(400, "invalid_request_error", "multipart request has no 'request' part")
                return json.loads(raw), parts
            return json.loads(body), None
        except (json.JSONDecodeError, UnicodeDecodeError) as e:
            raise RequestError(400, "invalid_request_error", f"body is not valid JSON: {e}") from None
        except MultiPartException as e:
            raise RequestError(400, "invalid_request_error", f"malformed multipart body: {e}") from None

    @app.post("/v1/systemone")
    async def systemone(request: Request) -> JSONResponse:
        body, parts = await read(request)
        resp = await run_in_threadpool(server.decide, parse_request(body, parts))
        return JSONResponse(resp.to_wire())

    @app.post("/api/decide")
    async def decide(request: Request) -> JSONResponse:
        body, _ = await read(request)
        if isinstance(body, dict):
            body.setdefault("images", [])
        resp = await run_in_threadpool(server.decide, parse_request(body))
        return JSONResponse(resp.to_wire())

    @app.get("/v1/models")
    async def models() -> dict[str, Any]:
        # TypeSafe's shape; release_date is required by its SDK, so the server's start date stands in
        about = f"{model.name} served by S1GHT ({model.spec.runtime} runtime); answers any model name."
        return {"models": [{"name": model.name, "description": about, "release_date": started}]}

    @app.get("/health")
    async def health() -> dict[str, Any]:
        caps = model.runtime.caps
        return {"status": "ok", "model": model.name, "runtime": model.spec.runtime, "box": caps.box}

    return app
