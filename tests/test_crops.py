import cv2
import numpy as np
import pytest

from s1ght.engine.crops import crop, expand_box, signature, signature_distance, with_frame


def test_expand_box_pads_and_clips() -> None:
    assert expand_box((100, 100, 200, 300), (480, 640), pad=0.2) == (80, 60, 220, 340)
    assert expand_box((0, 0, 50, 50), (40, 40), pad=0.5) == (0, 0, 40, 40)  # clipped to the image
    assert expand_box((10, 10, 10, 10), (40, 40), pad=0.2) == (10, 10, 11, 11)  # never empty


def test_crop_is_a_copy() -> None:
    img = np.zeros((100, 100, 3), np.uint8)
    c = crop(img, (10, 10, 30, 30), pad=0.0)
    assert c.shape == (20, 20, 3)
    c[:] = 255
    assert img.max() == 0


def test_signature_distance() -> None:
    rng = np.random.default_rng(0)

    def blocks() -> np.ndarray:  # 12x8 random blocks, 10 px each: structure that survives the 32x32 thumbnail
        return np.kron(rng.integers(0, 200, (12, 8, 3), np.uint8), np.ones((10, 10, 1), np.uint8))

    a, other = blocks(), blocks()
    brighter = a + np.uint8(30)
    assert signature(a).shape == (32, 32, 3)
    assert signature_distance(signature(a), signature(a)) == 0
    assert signature_distance(signature(a), signature(brighter)) < 0.01  # brightness alone is not a change
    assert signature_distance(signature(a), signature(other)) > 0.15
    # same gray level, different color, over half the crop (an object padded by 0.2 per side):
    # above the scheduler's default change threshold of 0.10
    red, blue = np.zeros((40, 40, 3), np.uint8), np.zeros((40, 40, 3), np.uint8)
    red[6:34, 6:34], blue[6:34, 6:34] = (0, 0, 200), (130, 77, 0)  # both gray ≈ 60
    gray = cv2.cvtColor(red, cv2.COLOR_BGR2GRAY).mean(), cv2.cvtColor(blue, cv2.COLOR_BGR2GRAY).mean()
    assert abs(gray[0] - gray[1]) < 1
    assert signature_distance(signature(red), signature(blue)) > 0.10


def test_with_frame_downscales() -> None:
    parts = with_frame(np.zeros((50, 40, 3), np.uint8), np.zeros((1080, 1920, 3), np.uint8), max_side=512)
    assert parts["image"].shape == (50, 40, 3) and parts["frame"].shape == (288, 512, 3)


@pytest.mark.parametrize("pad", [0.0, 0.2, 1.0])
def test_expand_box_contains_original(pad) -> None:
    x1, y1, x2, y2 = expand_box((30.5, 40.2, 70.7, 90.9), (200, 200), pad)
    assert x1 <= 30.5 and y1 <= 40.2 and x2 >= 70.7 and y2 >= 90.9
