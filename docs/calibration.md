# Calibration

A model's raw readout is not a probability you can put a threshold on. Small decision models are often overconfident: on hard-hat crops, d1-omni-600M said "0.99" far more often than it was right. Calibration fits one temperature `T` per question type and input kind on labeled examples the model has not seen, and divides the logits by it.

```python
model = S1GHT("d1-omni-600m")
profile = model.calibrate("hardhat/manifest.jsonl", save="profiles/d1-omni-hardhat.json")
print(profile.report())

# later
model = S1GHT("d1-omni-600m", calibration="profiles/d1-omni-hardhat.json")
```

```
d1-omni-600m (ollama, hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0)
key                        n      T    ECE before→after   Brier before→after    acc
noul/crop                128   5.07     0.291 →  0.117      0.652 →  0.489   0.62
```

## Dataset

JSON Lines, one labeled question per line; image paths are relative to the file.

```json
{"image": "images/a.jpg", "box": [412, 80, 470, 150], "pad": 1.0, "question": {"type": "noul", "instructions": "Is the person in the center of `image` wearing a hard hat?"}, "label": true, "split": "calib", "group": "a.jpg"}
{"image": "images/b.jpg", "question": {"type": "choice", "instructions": "Where was `image` taken?", "criteria": {"street": null, "site": null}}, "label": "site", "split": "test"}
{"state": "The parcel arrived late and damaged.", "question": {"type": "score", "instructions": "How serious is it?", "criteria": ["none", "minor", "major"]}, "label": 2}
```

| Field | |
| --- | --- |
| `image` | image file; with `box` (pixel `x1, y1, x2, y2`, expanded by `pad` per side) the example is a **crop**, without it a **full frame** |
| `state` | text or JSON, for **text** examples without an image |
| `question` | one question in wire form |
| `label` | `true`/`false` for Noul, the option name for Choice, the level index for Score |
| `split` | `"calib"` (default) to fit on, `"test"` to report on |
| `group` | examples that belong together (crops of one photo) stay in the same fold |

## What is fitted

- **Keys.** One temperature per `question type / input kind`: `noul/crop`, `choice/full_frame`, `score/text`, ... Crops and full frames are different distributions. A key missing from the profile falls back to the same question type with another input kind, then to `T = 1`. Keys fitted on fewer than 30 examples warn.
- **Fit.** `T` minimizes the negative log-likelihood of `softmax(logits / T)` (golden-section search over `log T` in [0.05, 20]).
- **Honest numbers.** Reported metrics never use the data `T` was fitted on: with a `test` split, `T` is fitted on `calib` and measured on `test`; without one, k-fold (`folds=5`) out-of-fold predictions are used, folded by `group`.
- **Metrics.** Accuracy, top-label ECE (10 bins), Brier (multi-class sum, 0–2), NLL and a reliability table, before and after. `s1ght.engine.calibration.save_reliability(profile, "rel.png")` draws the reliability diagrams.

## Where profiles apply

`calibration=` works with every runtime that exposes raw scores: native, Ollama, Ollaya, TypeSafe, vLLM, llama-server. Profiles record the model, the runtime and its exact source (an HF commit or a server tag), and refuse rolling aliases such as `jev-latest` or `clef:latest`, whose weights can change under the profile. A profile fitted for one model is refused by another.

`track()` uses the `crop` temperatures, `decide()` and `predict()` the `full_frame` ones, and requests without images the `text` ones.

**Fit on your domain.** A temperature fitted on hard hats says little about kitchens. The catalog ships no profiles for this reason; fit one on a few hundred examples of the questions you will actually ask.

## Hard-hat benchmark

`benchmarks/state_pairs.py` builds a balanced crop set from the CC0 [Voxel51/hard-hat-detection](https://huggingface.co/datasets/Voxel51/hard-hat-detection) dataset (pinned revision; only the chosen images are downloaded) and calibrates a model on it:

```bash
python benchmarks/state_pairs.py build out/hardhat --images 150
python benchmarks/state_pairs.py calibrate out/hardhat/manifest.jsonl --model d1-omni-600m --runtime ollama
```

| Runtime | T | ECE before → after | Brier before → after | Accuracy |
| --- | ---: | ---: | ---: | ---: |
| d1-omni-600M, Ollama GGUF Q8_0 | 5.07 | 0.291 → 0.117 | 0.652 → 0.489 | 0.62 |
| d1-omni-600M, native fp32 | 5.15 | 0.300 → 0.056 | 0.679 → 0.489 | 0.56 |

Calibration does not change which answer wins (accuracy stays); it makes the probabilities honest about how often that answer is right.

## Command line

```bash
s1ght calibrate model=d1-omni-600m runtime=ollama data=out/hardhat/ save=profiles/hardhat.json
s1ght predict model=d1-omni-600m runtime=ollama calibration=profiles/hardhat.json source=... questions=...
```
