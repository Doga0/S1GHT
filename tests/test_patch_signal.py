"""benchmarks/patch_signal.py (P2.0): metric helpers offline, and an end-to-end run on a tiny random Qwen3-VL.

The end-to-end test builds the model from the real processor files (~12 MB, downloaded once from
the pinned catalog revision, or S1GHT_QWEN_PROCESSOR=<dir>) and a synthetic YOLO-seg dataset.
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PILImage

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "benchmarks"))
import patch_signal as ps

REPO, REVISION = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"


def test_auc() -> None:
    labels = np.array([False, False, True, True])
    assert ps.auc(np.array([0.1, 0.2, 0.8, 0.9]), labels) == 1.0
    assert ps.auc(np.array([0.9, 0.8, 0.2, 0.1]), labels) == 0.0
    assert ps.auc(np.array([0.5, 0.5, 0.5, 0.5]), labels) == 0.5  # ties count half
    assert ps.auc(np.array([0.1, 0.4, 0.4, 0.9]), labels) == pytest.approx(0.875)
    assert np.isnan(ps.auc(np.array([0.1, 0.2]), np.array([True, True])))


def test_components_and_matching() -> None:
    on = np.zeros((4, 6), bool)
    on[0:2, 0:2] = True  # one object
    on[3, 5] = True  # another, not touching
    boxes = ps.components(on.reshape(-1), (4, 6))
    assert sorted(boxes) == [(0.0, 0.0, 64.0, 64.0), (160.0, 96.0, 192.0, 128.0)]
    truth = [(0, 0, 60, 60), (100, 0, 140, 40)]
    assert ps.match(boxes, truth, 0.5) == 1
    assert ps.iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_coverage() -> None:
    mask = np.zeros((64, 96), bool)
    mask[:32, :48] = True
    assert ps.coverage(mask, (2, 3)).tolist() == [1.0, 0.5, 0.0, 0.0, 0.0, 0.0]


def synthetic_dataset(root: Path, n: int = 4) -> Path:
    """Images with a red rectangle (class 0) and sometimes a blue one (class 3, held out from the probes), YOLO-seg labels."""
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    rng = np.random.default_rng(0)
    for k in range(n):
        img = PILImage.new("RGB", (320, 256), "white")
        lines = []
        for cls, color in [(0, (220, 20, 20)), (3, (20, 20, 220))][: 1 + k % 2]:
            x0, y0 = rng.integers(0, 160), rng.integers(0, 128)
            x1, y1 = x0 + 128, y0 + 96
            img.paste(color, (int(x0), int(y0), int(x1), int(y1)))
            poly = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
            lines.append(f"{cls} " + " ".join(f"{x / 320:.4f} {y / 256:.4f}" for x, y in poly))
        img.save(root / "images" / "val" / f"{k}.jpg")
        (root / "labels" / "val" / f"{k}.txt").write_text("\n".join(lines))
    return root


def test_load_samples(tmp_path: Path) -> None:
    samples = ps.load_samples(synthetic_dataset(tmp_path), 10, 640)
    assert len(samples) == 4
    s = samples[1]
    assert set(s.masks) == {0, 3} and s.image.size == (320, 256)
    assert s.masks[0].sum() == pytest.approx(128 * 96, rel=0.05)
    assert len(s.boxes[0]) == 1


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch = pytest.importorskip("torch")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    src = os.environ.get("S1GHT_QWEN_PROCESSOR")
    if not src:
        from huggingface_hub import snapshot_download

        src = snapshot_download(REPO, revision=REVISION, allow_patterns=["*.json", "*.txt"])
    out = tmp_path_factory.mktemp("tiny-qwen3-vl")
    for name in os.listdir(src):
        if name.endswith((".json", ".txt")) and name != "config.json":
            shutil.copy(os.path.join(src, name), out)
    config = Qwen3VLConfig.from_pretrained(src)
    text, vision = config.text_config, config.vision_config
    for key, value in {
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "head_dim": 16,
    }.items():
        setattr(text, key, value)
    text.rope_parameters = {**text.rope_parameters, "mrope_section": [2, 3, 3]}
    for key, value in {
        "depth": 2,
        "hidden_size": 32,
        "intermediate_size": 64,
        "num_heads": 2,
        "out_hidden_size": 64,
    }.items():
        setattr(vision, key, value)
    vision.deepstack_visual_indexes = [0]
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(config).save_pretrained(out)
    return str(out)


@pytest.mark.network
def test_end_to_end_tiny_model(tiny_qwen: str, tmp_path: Path) -> None:
    data = synthetic_dataset(tmp_path / "data")
    out = tmp_path / "report.json"
    report = ps.main(
        [
            *("--model", tiny_qwen, "--data", str(data), "--images", "4", "--layers", "0", "1", "2"),
            *("--present", "2", "--absent", "1", "--tiles", "2", "--steps", "5", "--rank", "4"),
            *("--device", "cpu", "--dtype", "fp32", "--out", str(out)),
        ]
    )
    best = report["best"]
    for name in ("cosine", "cosine_centered", "attention_mean", "attention_top5", "objectness", "bilinear", "tiles_2"):
        assert name in best, name
        assert 0.0 <= best[name]["test_auc"] <= 1.0 or np.isnan(best[name]["test_auc"])
    assert best["tiles_2"]["layer"] == -1
    # class 3 is held out from the probes: graded with the seen probe's layer and threshold
    assert best["bilinear_unseen"]["layer"] == best["bilinear"]["layer"]
    assert best["bilinear_unseen"]["threshold"] == best["bilinear"]["threshold"]
    assert set(report["per_layer"]["cosine"]) == {0, 1, 2}
    assert "attention_mean" not in report["per_layer"] or 0 not in report["per_layer"]["attention_mean"]
    assert out.exists() and "| method |" in ps.markdown(report)
