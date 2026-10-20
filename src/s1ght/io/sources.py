"""Frame sources: images, folders, globs, URLs, videos, cameras and network streams.

`load_source(source)` yields `SourceFrame`s (BGR uint8). Files are read frame by frame.
Live sources (camera index, rtsp://, rtmp://, tcp://, or http(s) URLs that are not images)
are read on a background thread that keeps only the newest frame, so a slow consumer
always gets the latest picture instead of a growing backlog ("latest frame wins").
"""

from __future__ import annotations

import glob
import threading
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image as PILImage

from s1ght.schema.assets import Image

IMG_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"})
VID_SUFFIXES = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v", ".mpg", ".mpeg", ".wmv"})
STREAM_SCHEMES = ("rtsp://", "rtmp://", "tcp://", "udp://")

# Replaced in tests with a fake capture.
open_capture: Callable[[Any], Any] = cv2.VideoCapture


@dataclass
class SourceFrame:
    """One frame from a source.

    Attributes:
        image: BGR uint8 array.
        path: File, URL or a generated name ("image0", "camera0").
        index: Frame index within that source (0 for still images).
        t: Seconds: position in a video file, or time since a live stream started.
    """

    image: np.ndarray
    path: str
    index: int = 0
    t: float = 0.0


def decode_image(data: bytes) -> np.ndarray:
    """Decode encoded image bytes to BGR."""
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("could not decode image")
    return img


def read_image(path: str | Path) -> np.ndarray:
    """Read an image file to BGR (works with non-ASCII paths on Windows)."""
    return decode_image(np.fromfile(str(path), np.uint8).tobytes())


def _as_bgr(obj: Any) -> np.ndarray:
    if isinstance(obj, np.ndarray):
        if obj.dtype != np.uint8:
            raise ValueError(f"expected a uint8 image array, got {obj.dtype}")
        if obj.ndim == 2:
            return cv2.cvtColor(obj, cv2.COLOR_GRAY2BGR)
        if obj.ndim == 3 and obj.shape[2] == 4:
            return cv2.cvtColor(obj, cv2.COLOR_BGRA2BGR)
        if obj.ndim == 3 and obj.shape[2] == 3:
            return obj
        raise ValueError(f"expected HxW, HxWx3 or HxWx4 array, got {obj.shape}")
    if isinstance(obj, PILImage.Image):
        return np.ascontiguousarray(np.asarray(obj.convert("RGB"))[:, :, ::-1])
    if isinstance(obj, Image):
        return obj.to_array()
    if isinstance(obj, (bytes, bytearray)):
        return decode_image(bytes(obj))
    raise TypeError(f"unsupported source: {type(obj).__name__}")


def is_live(source: Any) -> bool:
    """True for camera indices and stream URLs, which are read with latest-frame-wins."""
    if isinstance(source, bool) or not isinstance(source, (str, Path, int)):
        return False
    if isinstance(source, int):
        return True
    s = str(source)
    if s.isdigit():
        return True
    if s.lower().startswith(STREAM_SCHEMES):
        return True
    return s.lower().startswith(("http://", "https://")) and Path(s.split("?")[0]).suffix.lower() not in (
        IMG_SUFFIXES | VID_SUFFIXES
    )


def _expand(source: str | Path) -> list[str]:
    s = str(source)
    if s.lower().startswith(("http://", "https://")):
        return [s]
    if any(ch in s for ch in "*?["):
        files = sorted(glob.glob(s, recursive=True))
    elif Path(s).is_dir():
        files = sorted(str(p) for p in Path(s).iterdir())
    else:
        return [s]
    files = [f for f in files if Path(f).suffix.lower() in IMG_SUFFIXES | VID_SUFFIXES]
    if not files:
        raise FileNotFoundError(f"no images or videos in {s}")
    return files


def load_source(source: Any, *, vid_stride: int = 1, buffer: bool = False) -> Iterator[SourceFrame]:
    """Yield frames from `source`.

    Args:
        source: Path, folder, glob, http(s) image URL, video file, camera index (0 or "0"),
            stream URL, numpy array (BGR), PIL image, `s1ght.Image`, encoded bytes, or a list of these.
        vid_stride: Keep every n-th frame of a video file.
        buffer: Live sources only. True yields every frame in order (may fall behind);
            False (default) always yields the newest frame.
    """
    if vid_stride < 1:
        raise ValueError("vid_stride must be >= 1")
    if isinstance(source, (list, tuple)):
        for i, item in enumerate(source):
            for f in load_source(item, vid_stride=vid_stride, buffer=buffer):
                if not isinstance(item, (str, Path)):
                    f.path = f"image{i}"
                yield f
        return
    if not isinstance(source, (str, Path, int)):
        yield SourceFrame(_as_bgr(source), "image0")
        return
    if is_live(source):
        yield from _live(source, buffer)
        return
    for path in _expand(str(source)):
        low = path.lower()
        if low.startswith(("http://", "https://")) and Path(low.split("?")[0]).suffix in IMG_SUFFIXES:
            import httpx

            resp = httpx.get(path, follow_redirects=True, timeout=30)
            resp.raise_for_status()
            yield SourceFrame(decode_image(resp.content), path)
        elif Path(low).suffix in VID_SUFFIXES or low.startswith(("http://", "https://")):
            yield from _video(path, vid_stride)
        else:
            if not Path(path).is_file():
                raise FileNotFoundError(f"source not found: {path}")
            yield SourceFrame(read_image(path), path)


def _video(path: str, stride: int) -> Iterator[SourceFrame]:
    cap = open_capture(path)
    if not cap.isOpened():
        raise OSError(f"could not open video {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    try:
        i = 0
        while True:
            ok = cap.grab()
            if not ok:
                return
            if i % stride == 0:
                ok, frame = cap.retrieve()
                if not ok:
                    return
                yield SourceFrame(frame, path, i, i / fps)
            i += 1
    finally:
        cap.release()


class LatestFrameReader:
    """Reads a live capture on a thread, keeping only the newest frame."""

    def __init__(self, source: Any, buffer: bool = False) -> None:
        target = int(source) if str(source).isdigit() else source
        self.name = f"camera{target}" if isinstance(target, int) else str(source)
        self.cap = open_capture(target)
        if not self.cap.isOpened():
            raise OSError(f"could not open stream {source}")
        self.buffer = buffer
        self._cond = threading.Condition()
        self._frames: list[tuple[int, float, np.ndarray]] = []
        self._done = False
        self._t0 = time.monotonic()
        self._thread = threading.Thread(target=self._read, name=f"s1ght-source-{self.name}", daemon=True)
        self._thread.start()

    def _read(self) -> None:
        i = 0
        try:
            while not self._done:
                ok, frame = self.cap.read()
                if not ok:
                    break
                with self._cond:
                    item = (i, time.monotonic() - self._t0, frame)
                    if self.buffer:
                        self._frames.append(item)
                    else:
                        self._frames[:] = [item]
                    self._cond.notify()
                i += 1
        finally:
            with self._cond:
                self._done = True
                self._cond.notify_all()

    def __iter__(self) -> Iterator[SourceFrame]:
        try:
            while True:
                with self._cond:
                    while not self._frames and not self._done:
                        self._cond.wait(timeout=1.0)
                    if not self._frames:
                        return
                    i, t, frame = self._frames.pop(0)
                yield SourceFrame(frame, self.name, i, t)
        finally:
            self.close()

    def close(self) -> None:
        self._done = True
        self._thread.join(timeout=2.0)
        self.cap.release()


def _live(source: Any, buffer: bool) -> Iterator[SourceFrame]:
    yield from LatestFrameReader(source, buffer)
