"""Image assets: one picture that can go anywhere text can go in a System One request."""

from __future__ import annotations

import io
import uuid
from pathlib import Path
from typing import TypeAlias

import numpy as np
from PIL import Image as PILImage

ImageSource: TypeAlias = str | Path | bytes | np.ndarray | PILImage.Image

FORMATS = ("png", "jpg", "jpeg", "webp")
_PIL_FORMAT = {"png": "PNG", "jpg": "JPEG", "jpeg": "JPEG", "webp": "WEBP"}


def _family(fmt: str | None) -> str | None:
    return "jpeg" if fmt == "jpg" else fmt


def sniff_format(data: bytes) -> str | None:
    """Return "png", "jpeg" or "webp" from magic bytes, or None."""
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "png"
    if data.startswith(b"\xff\xd8"):
        return "jpeg"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "webp"
    return None


class Image:
    """An image in a request.

    Args:
        source: File path, encoded bytes, numpy array (BGR or grayscale, uint8) or PIL image.
        format: Wire format: "png", "jpg"/"jpeg" or "webp". None infers it from encoded
            input (path or bytes) and uses "jpeg" for arrays and PIL images.
        ref: Reference string written into the request; a new UUID by default.

    Encoded input that already matches `format` is sent as is, without re-encoding.
    """

    def __init__(self, source: ImageSource, *, format: str | None = None, ref: str | None = None) -> None:
        self._path: Path | None = None
        self._bytes: bytes | None = None
        self._array: np.ndarray | None = None
        self._pil: PILImage.Image | None = None

        if isinstance(source, (str, Path)):
            self._path = Path(source)
            if not self._path.is_file():
                raise FileNotFoundError(f"image not found: {self._path}")
        elif isinstance(source, (bytes, bytearray)):
            self._bytes = bytes(source)
        elif isinstance(source, np.ndarray):
            if source.dtype != np.uint8 or not (source.ndim == 2 or (source.ndim == 3 and source.shape[2] == 3)):
                raise ValueError(f"expected uint8 HxW or HxWx3 BGR array, got {source.dtype} {source.shape}")
            self._array = source
        elif isinstance(source, PILImage.Image):
            self._pil = source
        else:
            raise TypeError(f"unsupported image source: {type(source).__name__}")

        self._is_encoded = self._path is not None or self._bytes is not None
        if format is None:
            format = (sniff_format(self._encoded()) if self._is_encoded else None) or "jpeg"
        format = format.lower()
        if format not in FORMATS:
            raise ValueError(f"format must be one of {FORMATS}, got {format!r}")
        self.format: str = format
        self.ref: str = ref if ref is not None else str(uuid.uuid4())
        if not self.ref.strip():
            raise ValueError("ref must not be empty")

    def _encoded(self) -> bytes:
        if self._bytes is None and self._path is not None:
            self._bytes = self._path.read_bytes()
        return self._bytes or b""

    def to_pil(self) -> PILImage.Image:
        """Decoded image as PIL (RGB or L)."""
        if self._pil is None:
            if self._array is not None:
                arr = self._array if self._array.ndim == 2 else np.ascontiguousarray(self._array[:, :, ::-1])
                self._pil = PILImage.fromarray(arr)
            else:
                self._pil = PILImage.open(io.BytesIO(self._encoded()))
                self._pil.load()
        return self._pil

    def to_bytes(self) -> bytes:
        """Image encoded in `self.format`."""
        if self._is_encoded:
            data = self._encoded()
            if _family(sniff_format(data)) == _family(self.format):
                return data
        pil = self.to_pil()
        if _PIL_FORMAT[self.format] == "JPEG" and pil.mode not in ("RGB", "L"):
            pil = pil.convert("RGB")
        buf = io.BytesIO()
        pil.save(buf, _PIL_FORMAT[self.format])
        return buf.getvalue()

    def to_array(self) -> np.ndarray:
        """Decoded image as a uint8 BGR array (HxWx3); an array source is returned without copying."""
        if self._array is not None and self._array.ndim == 3:
            return self._array
        pil = self.to_pil()
        if pil.mode != "RGB":
            pil = pil.convert("RGB")
        return np.ascontiguousarray(np.asarray(pil)[:, :, ::-1])

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) in pixels."""
        if self._array is not None:
            return self._array.shape[1], self._array.shape[0]
        if self._pil is not None:
            return self._pil.size
        with PILImage.open(io.BytesIO(self._encoded())) as im:
            return im.size

    def __repr__(self) -> str:
        return f"Image(ref={self.ref!r}, format={self.format!r})"
