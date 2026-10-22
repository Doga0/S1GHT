"""The Ultralytics adapter on a real model (downloads yolo11n.pt, ~5 MB, and Ultralytics' bus.jpg)."""

import numpy as np
import pytest

pytestmark = [pytest.mark.network, pytest.mark.slow]


def test_ultralytics_tracks_people(tmp_path, monkeypatch) -> None:
    pytest.importorskip("ultralytics")
    import cv2
    import httpx

    from s1ght.detectors.base import Detector, Frame
    from s1ght.detectors.ultralytics import UltralyticsDetector
    from s1ght.io.sources import decode_image

    monkeypatch.chdir(tmp_path)  # Ultralytics downloads weights into the working directory
    img = decode_image(httpx.get("https://ultralytics.com/images/bus.jpg", follow_redirects=True, timeout=30).content)
    video = str(tmp_path / "still.avi")
    vw = cv2.VideoWriter(video, cv2.VideoWriter_fourcc(*"MJPG"), 5, (img.shape[1], img.shape[0]))
    for _ in range(5):
        vw.write(img)
    vw.release()

    det = UltralyticsDetector("yolo11n.pt", classes=["person"])
    assert isinstance(det, Detector) and det.class_ids == [0]
    frames = list(det.stream(video))
    assert len(frames) == 5 and all(isinstance(f, Frame) for f in frames)
    last = frames[-1]
    assert len(last) >= 3 and set(last.cls.tolist()) == {0}
    assert last.ids is not None and len(set(last.ids.tolist())) == len(last)
    assert last.t == pytest.approx(4 / 5)  # frame index / video fps
    assert np.array_equal(last.image.shape, img.shape)
    with pytest.raises(ValueError, match="not classes"):
        UltralyticsDetector("yolo11n.pt", classes=["unicorn"])
