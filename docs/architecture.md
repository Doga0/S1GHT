# Architecture and extending S1GHT

```
source ──► "where" layer ──► scheduler ──► decision runtime ──► fusion ──► Results
           detector (hybrid)              native | server
           localizer (standalone, P2)        │
                                       calibration profile
```

The two modes differ only in the "where" layer. Questions, runtimes, calibration, fusion and `Results` are shared.

## Package layout

| Module | |
| --- | --- |
| `s1ght/model.py` | `S1GHT`: `decide`, `predict`, `track`, `calibrate`, `serve` |
| `s1ght/schema/` | `Noul`, `Choice`, `Score`, `Box`; `Image`; requests and answers with their TypeSafe and Ollama wire forms |
| `s1ght/hub.py`, `registry.yaml` | names and URIs → `RuntimeSpec`; the catalog; downloads |
| `s1ght/runtimes/base.py` | the `Runtime` protocol, `Capabilities`, `RawScores`, logits → answers |
| `s1ght/runtimes/native/` | `NativeRuntime` and model families (`lfm2_vl`, `d1_omni`, `qwen_vl`, `semif_generic`) |
| `s1ght/runtimes/letters.py` | the option-letter prompt and readout shared by letter families and logprob servers |
| `s1ght/runtimes/server/` | `/v1/systemone` client (TypeSafe, Ollama, Ollaya) and logprob client (vLLM, llama-server) |
| `s1ght/engine/` | crops, scheduler, fusion, calibration |
| `s1ght/detectors/` | the `Detector` protocol and the Ultralytics adapter |
| `s1ght/io/` | sources (latest frame wins) and drawing |
| `s1ght/results.py` | `Results`, `Boxes`, `Event` |
| `s1ght/cli.py`, `s1ght/server.py` | the `s1ght` command and `s1ght serve` |

Dependency rules keep `import s1ght` light: `schema`, `engine` and `runtimes/base` never import torch, httpx or Ultralytics; native code imports torch lazily; the server client imports httpx but not torch. CI checks that `import s1ght` works without torch.

## Everything but the forward pass is S1GHT's

Servers limit what a decision model can do (only top-N log-probabilities, no hidden states, one forward per runner on Ollaya). So preprocessing, batching, option-order debiasing, calibration and fusion live in S1GHT, and runtimes are interchangeable back ends. Every runtime returns **raw scores** (logits per option); answers, TypeSafe confidences and calibration are computed from them the same way for all.

## Writing a runtime

A runtime answers System One requests. Anything with these members works, and `S1GHT(runtime_object)` wraps it:

```python
from s1ght.runtimes.base import Capabilities, RawScores, RuntimeBase, scores_to_answers, validate_request
from s1ght.schema.systemone import SystemOneRequest, SystemOneResponse


class MyRuntime(RuntimeBase):
    caps = Capabilities(images=True, raw_scores=True, max_options=26)

    def raw(self, req: SystemOneRequest) -> RawScores:
        validate_request(self.caps, req)  # refuse what caps says you cannot do
        ...  # one logit array per question, RawScores order:
        return RawScores({...})  # Noul (yes, no); Choice criteria order; Score low→high

    def decide(self, req: SystemOneRequest) -> SystemOneResponse:
        return SystemOneResponse(model="mine", answers=scores_to_answers(req.questions, self.raw(req)))
```

`RuntimeBase` provides `decide_batch` (one by one), `close` and the context manager; override `decide_batch` if your back end batches. Runtimes are called from one worker thread at a time. `examples/custom_runtime.py` is a complete, runnable example.

`Capabilities` declares what a runtime supports (`images`, `box`, `batch`, `prefix_cache`, `attention`, `hidden_states`, `raw_scores`, `max_options`, `image_px`). Features check them at setup time and fail with `CapabilityError` instead of degrading silently.

## Writing a native family

A family adapter loads one model family and answers a batch of requests given as `(state, wire questions, PIL images)` with images replaced by `[image N]` labels:

```python
class MyFamily:
    name = "my_family"
    caps = Capabilities(images=True, batch=True, raw_scores=True)  # optional; the native default otherwise

    def load(self, path: str, device: str, dtype: str) -> None: ...
    def run(self, items) -> list[tuple[dict[str, dict], int]]: ...  # per item: ({qid: wire answer}, tokens read)
    def close(self) -> None: ...
```

Register it in `s1ght/runtimes/native/families/__init__.py` and in `FAMILIES` in `hub.py`, and give catalog entries `family: my_family`. Chat VLMs usually need no new family: subclass `SemifGenericFamily` and set `template_kwargs`, as `QwenVlFamily` does.

## Writing a detector

`track(detector=...)` takes any object with `stream(source, **kwargs)` that yields `s1ght.detectors.base.Frame` (image, `xyxy`, `conf`, `cls`, `ids`, `names`, time `t`, `index`, `path`).

## Adding a catalog entry

```yaml
my-model:
  source: hf://org/repo@<40-char commit>
  family: qwen_vl
  runtimes:
    native: {dtype: {cuda: bf16, t4: fp16, cpu: fp32}}
    vllm: {model: org/repo}
  calibration: null
  license: apache-2.0
```

Pin a full commit hash: it fixes the weights and any remote code, and calibration profiles refuse rolling names.

## Tests

`pytest` runs the offline suite (no GPU, network or model downloads; `tests/conftest.py` has a deterministic `FakeRuntime`). Tests marked `gpu`, `network` or `slow` are opt-in: `pytest -m network`, and so on. `benchmarks/native_gpu.ipynb` runs the GPU checks on Colab or Kaggle.
