"""State pairs: same class, only the state differs. Builds a calibration set and calibrates a model on it.

    python benchmarks/state_pairs.py build out/hardhat --images 150
    python benchmarks/state_pairs.py calibrate out/hardhat/manifest.jsonl --model d1-omni-600m --runtime ollama

`build` makes a hard-hat set from Voxel51/hard-hat-detection (CC0, 5,000 worksite photos with
`helmet` and `head` boxes): every head box becomes a crop, labeled true if it wears a hard hat.
Only the label file and the chosen images are downloaded, from a pinned dataset revision.
Images are split half and half into "calib" and "test" by photo, so crops of one photo never
end up on both sides. Classes are balanced per split.

`calibrate` fits a temperature per question type and input kind on "calib", reports ECE and
Brier before and after on "test", and writes the profile JSON and a reliability diagram.

The design doc's state-pair set will collect several domains (safety, home, traffic, retail);
hard hats are the first, because a public, permissively licensed set exists for them.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

REPO = "Voxel51/hard-hat-detection"
REVISION = "d11e7d6b69ba1a22422db3230c0415d39a699d7c"
# crops are padded, so other people can be in view: the label is about the one in the middle
QUESTION = {"type": "noul", "instructions": "Is the person in the center of `image` wearing a hard hat?"}
MIN_SIDE = 20  # px: smaller heads are too small to judge


def build(out: Path, images: int, per_image: int, pad: float, seed: int) -> Path:
    from huggingface_hub import hf_hub_download

    out.mkdir(parents=True, exist_ok=True)
    labels = hf_hub_download(REPO, "samples.json", repo_type="dataset", revision=REVISION, local_dir=out)
    samples = json.loads(Path(labels).read_text(encoding="utf-8"))["samples"]
    rng = random.Random(seed)
    rng.shuffle(samples)

    def bare_heads(s: dict) -> int:
        w, h = s["metadata"]["width"], s["metadata"]["height"]
        return sum(
            d["label"] == "head" and min(d["bounding_box"][2] * w, d["bounding_box"][3] * h) >= MIN_SIDE
            for d in s["ground_truth"]["detections"]
        )

    # bare heads are the rarer state (5.8k vs 19k): use photos that have them first
    samples.sort(key=lambda s: bare_heads(s) == 0)

    rows, used = [], 0
    for s in samples:
        if used >= images:
            break
        w, h = s["metadata"]["width"], s["metadata"]["height"]
        heads = []
        for d in s["ground_truth"]["detections"]:
            if d["label"] not in ("helmet", "head"):
                continue
            x, y, bw, bh = d["bounding_box"]
            if min(bw * w, bh * h) < MIN_SIDE:
                continue
            heads.append((d["label"] == "helmet", [x * w, y * h, (x + bw) * w, (y + bh) * h]))
        with_hat = [b for b in heads if b[0]]
        without = [b for b in heads if not b[0]]
        if not without and not with_hat:
            continue
        rng.shuffle(with_hat)
        rng.shuffle(without)
        k = max(1, per_image // 2)
        chosen = with_hat[:k] + without[:k]  # keep both states where a photo has them
        split = "calib" if used % 2 == 0 else "test"
        hf_hub_download(REPO, s["filepath"], repo_type="dataset", revision=REVISION, local_dir=out)
        for label, box in chosen:
            rows.append({"image": s["filepath"], "box": [round(v, 1) for v in box], "pad": pad,
                         "question": QUESTION, "label": label, "split": split, "group": s["filepath"]})  # fmt: skip
        used += 1

    # balance yes/no per split
    balanced = []
    for split in ("calib", "test"):
        yes = [r for r in rows if r["split"] == split and r["label"]]
        no = [r for r in rows if r["split"] == split and not r["label"]]
        n = min(len(yes), len(no))
        balanced += yes[:n] + no[:n]
    manifest = out / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in balanced), encoding="utf-8")
    for split in ("calib", "test"):
        n = sum(r["split"] == split for r in balanced)
        print(f"{split}: {n} crops ({n // 2} with a hard hat, {n // 2} without)")
    print(f"wrote {manifest} from {used} photos")
    return manifest


def run_calibration(manifest: Path, model: str, runtime: str | None, batch: int, timeout: float) -> None:
    from s1ght import S1GHT
    from s1ght.engine.calibration import save_reliability

    kw = {"timeout": timeout} if runtime in ("ollama", "ollaya", "typesafe") else {}
    m = S1GHT(model, runtime=runtime, **kw)
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        rate = done / (time.perf_counter() - t0)
        print(f"\r{done}/{total} examples, {rate:.2f}/s", end="", flush=True)

    tag = f"{model}-{runtime or 'default'}".replace("/", "_").replace(":", "_")
    profile = m.calibrate(manifest, save=manifest.parent / f"profile-{tag}.json", batch=batch, progress=progress)
    print(f"\n{time.perf_counter() - t0:.0f}s\n")
    print(profile.report())
    png = save_reliability(profile, manifest.parent / f"reliability-{tag}.png")
    print(f"\nprofile: {manifest.parent / f'profile-{tag}.json'}\nreliability diagram: {png}")
    m.close()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="download a subset and write manifest.jsonl")
    b.add_argument("out", type=Path)
    b.add_argument("--images", type=int, default=150, help="photos to use")
    b.add_argument("--per-image", type=int, default=4, help="most crops per photo (half each state)")
    b.add_argument("--pad", type=float, default=1.0, help="box expansion per side; heads need their surroundings")
    b.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("calibrate", help="fit and report a calibration profile")
    c.add_argument("manifest", type=Path)
    c.add_argument("--model", default="d1-omni-600m")
    c.add_argument("--runtime", default=None)
    c.add_argument("--batch", type=int, default=8)
    c.add_argument("--timeout", type=float, default=300.0)
    args = ap.parse_args()
    if args.cmd == "build":
        build(args.out, args.images, args.per_image, args.pad, args.seed)
    else:
        run_calibration(args.manifest, args.model, args.runtime, args.batch, args.timeout)


if __name__ == "__main__":
    main()
