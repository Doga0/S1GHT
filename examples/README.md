# Examples

| File | Shows | Needs |
| --- | --- | --- |
| [decide_image.py](decide_image.py) | one decision about one image: Noul, Choice, Score | a model (native or `--runtime ollama`) |
| [decide_crops.py](decide_crops.py) | the same questions about several objects in one photo, one crop each, with questions from [inspection.yaml](inspection.yaml) | a model |
| [predict_folder.py](predict_folder.py) | full-frame questions over a folder, video or camera; annotated frames saved | a model |
| [track_video.py](track_video.py) | hybrid mode: YOLO tracks, per-track questions from [ask.yaml](ask.yaml), events | a model and `.[ultralytics]` |
| [typesafe_client.py](typesafe_client.py) | the official TypeSafe SDK against `s1ght serve` | `.[serve]`, `typesafe-sdk`, a running `s1ght serve` |
| [custom_runtime.py](custom_runtime.py) | plugging your own back end into S1GHT | nothing: runs offline |
| [questions.yaml](questions.yaml) | full-frame questions for `decide` / `predict` | |
| [ask.yaml](ask.yaml) | per-class questions for `track` | |
| [inspection.yaml](inspection.yaml) | questions about the object in the center of a crop | |

Every script takes `--model` and `--runtime`. The default model is `d1-omni-600m`:

- **native** (default): `pip install "s1ght-visual-decisions[native]"`; downloads 2.4 GB once, runs on GPU or CPU.
- **Ollama**: `ollama pull hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0` (0.8 GB), then add `--runtime ollama`. No torch needed.

The same files work from the command line:

```bash
s1ght decide  model=d1-omni-600m source=bus.jpg questions=examples/questions.yaml
s1ght predict model=d1-omni-600m source="photos/*.jpg" questions=examples/questions.yaml save=True
s1ght track   model=d1-omni-600m source=0 ask=examples/ask.yaml show=True
```
