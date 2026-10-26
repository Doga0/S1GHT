"""Prefix cache (`runtimes/native/prefix.py`) on qwen_vl: same answers as the plain path, with the images encoded once per request.

Runs a tiny random Qwen3-VL on CPU built from the real processor files (~12 MB, downloaded once
from the pinned catalog revision; or set S1GHT_QWEN_PROCESSOR to a local copy).
"""

import os
import shutil

import numpy as np
import pytest
from PIL import Image as PILImage
from PIL import ImageDraw

from s1ght import Choice, Noul, Score

pytestmark = pytest.mark.network

REPO, REVISION = "Qwen/Qwen3-VL-4B-Instruct", "ebb281ec70b05090aa6165b016eac8ec08e71b17"
PROCESSOR_FILES = ["*.json", "*.txt"]


@pytest.fixture(scope="module")
def tiny_qwen(tmp_path_factory: pytest.TempPathFactory) -> str:
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration

    src = os.environ.get("S1GHT_QWEN_PROCESSOR")
    if not src:
        from huggingface_hub import snapshot_download

        src = snapshot_download(REPO, revision=REVISION, allow_patterns=PROCESSOR_FILES)
    out = tmp_path_factory.mktemp("tiny-qwen3-vl")
    for name in os.listdir(src):
        if name.endswith((".json", ".txt")) and name != "config.json":
            shutil.copy(os.path.join(src, name), out)
    config = Qwen3VLConfig.from_pretrained(src)
    text, vision = config.text_config, config.vision_config
    text.update(
        {
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 16,
        }
    )
    text.rope_parameters = {**text.rope_parameters, "mrope_section": [2, 3, 3]}
    vision.update({"depth": 2, "hidden_size": 32, "intermediate_size": 64, "num_heads": 2, "out_hidden_size": 64})
    vision.deepstack_visual_indexes = [0]
    torch.manual_seed(0)
    Qwen3VLForConditionalGeneration(config).save_pretrained(out)
    return str(out)


def drawing(shape: str, size: tuple[int, int] = (320, 256)) -> PILImage.Image:
    canvas = PILImage.new("RGB", size, "white")
    draw = ImageDraw.Draw(canvas)
    (draw.ellipse if shape == "circle" else draw.rectangle)((60, 60, 200, 200), fill=(220, 20, 20))
    return canvas


def wire(questions: dict) -> dict:
    return {qid: q.to_wire() for qid, q in questions.items()}


ITEMS = [
    # three questions about one image: one prefix, every prompt branches off it
    (
        "[image 1]",
        wire(
            {
                "red": Noul("Is the shape red?"),
                "shape": Choice("Which shape?", ["circle", "square", "triangle"]),
                "size": Score("How large?", ["small", "medium", "large", "huge"]),
            }
        ),
        [drawing("circle")],
    ),
    # one question, two images: one prompt per pass takes the plain path; "always" shares a prefix
    (
        {"left": "[image 1]", "right": "[image 2]"},
        wire({"same": Noul("Same shape?")}),
        [drawing("circle"), drawing("square", (256, 320))],
    ),
    # no images: always the plain path
    ("a red circle", wire({"red": Noul("Is it red?"), "kind": Choice("Kind?", ["shape", "text"])}), []),
]


@pytest.fixture(scope="module")
def family(tiny_qwen: str):
    from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily

    fam = QwenVlFamily(flip="never", batch=2)
    fam.load(tiny_qwen, "cpu", "fp32")
    yield fam
    fam.close()


def spy_encode(fam) -> list:
    calls: list = []
    encode = fam.prefix.encode
    fam.prefix.encode = lambda *a: calls.append(1) or encode(*a)
    return calls


def test_logits_match_plain_path(family) -> None:
    """Every prompt in every option order: the raw letter logits off the prefix equal the full read.

    Compared before debiasing: averaging all option orders of a random model gives ~uniform
    answers whatever the positions, which would hide a wrong prefix."""
    from s1ght.runtimes import letters

    images = [[letters.fit_image(img) for img in imgs] for _, _, imgs in ITEMS]
    prompts = letters.first_pass([(s, qs, len(imgs)) for s, qs, imgs in ITEMS], "always")
    calls = spy_encode(family)
    try:
        cached = family.logits(prompts, images, {})
    finally:
        del family.prefix.encode
    plain = family.plain_logits(prompts, images)
    assert len(calls) == 2  # items 0 and 1; item 2 has no image
    for p, (row, n), (ref, ref_n) in zip(prompts, cached, plain, strict=True):
        assert n == ref_n, (p.item, p.qid)
        np.testing.assert_allclose(row, ref, atol=1e-4, rtol=1e-4, err_msg=f"{p.item} {p.qid} {p.order}")


def test_prefix_kept_for_second_pass(family) -> None:
    from s1ght.runtimes import letters

    images = [[letters.fit_image(img) for img in imgs] for _, _, imgs in ITEMS]
    prompts = letters.first_pass([(s, qs, len(imgs)) for s, qs, imgs in ITEMS], "always")
    store: dict = {}
    calls = spy_encode(family)
    try:
        family.logits(prompts, images, store)
        assert set(store) == {0, 1}
        # the second pass asks item 0 again (one prompt): its prefix is reused, item 1's is freed
        family.logits([p for p in prompts if p.item == 0][:1], images, store)
    finally:
        del family.prefix.encode
    assert len(calls) == 2
    assert set(store) == {0}


def test_run_matches_without_cache(family, tiny_qwen: str) -> None:
    from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily

    plain = QwenVlFamily(flip="never", batch=2, prefix_cache=False)
    plain.load(tiny_qwen, "cpu", "fp32")
    try:
        expected = plain.run(ITEMS)
    finally:
        plain.close()
    for (got, tokens), (want, want_tokens) in zip(family.run(ITEMS), expected, strict=True):
        assert tokens == want_tokens
        for qid, x in got.items():
            y = want[qid]
            if x["type"] == "noul":
                assert x["noul"] == pytest.approx(y["noul"], abs=1e-4)
            else:
                assert x["probabilities"] == pytest.approx(y["probabilities"], abs=1e-4)


def test_caps() -> None:
    from s1ght.runtimes.native.families.qwen_vl import QwenVlFamily
    from s1ght.runtimes.native.families.semif_generic import SemifGenericFamily

    assert QwenVlFamily().caps.prefix_cache
    assert not QwenVlFamily(prefix_cache=False).caps.prefix_cache
    assert not SemifGenericFamily().caps.prefix_cache  # no layout: always the plain path
