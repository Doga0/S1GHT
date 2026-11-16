# Architecture and extending S1GHT

```
source ──► "where" layer ──► scheduler ──► decision runtime ──► fusion ──► Results
           detector (hybrid)              server | in-process
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
| `s1ght/runtimes/letters.py` | the option-letter prompt and readout for chat VLMs, shared by every runtime that reads letters |
| `s1ght/runtimes/server/` | `/v1/systemone` client (TypeSafe, Ollama, Ollaya) and logprob client (vLLM, llama-server) |
| `s1ght/engine/` | crops, scheduler, fusion, calibration |
| `s1ght/detectors/` | the `Detector` protocol and the Ultralytics adapter |
| `s1ght/io/` | sources (latest frame wins) and drawing |
| `s1ght/results.py` | `Results`, `Boxes`, `Event` |
| `s1ght/cli.py`, `s1ght/server.py` | the `s1ght` command and `s1ght serve` |

Dependency rules keep `import s1ght` light: `schema`, `engine` and `runtimes/base` never import httpx or Ultralytics; the server clients import httpx, and in-process runtimes import their engine lazily. CI checks that `import s1ght` works without torch.

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

## Writing a detector

`track(detector=...)` takes any object with `stream(source, **kwargs)` that yields `s1ght.detectors.base.Frame` (image, `xyxy`, `conf`, `cls`, `ids`, `names`, time `t`, `index`, `path`).

## Adding a catalog entry

```yaml
my-model:
  source: hf://org/repo@<40-char commit>
  family: qwen_vl
  runtimes:
    vllm: {model: org/repo}
    llama_server: {model: repo}
  calibration: null
  license: apache-2.0
```

`family` says how the model is prompted and read: `lfm2_vl` and `d1_omni` use the model's own decision engine, `qwen_vl` and `semif_generic` use option letters. Pin a full commit hash: it fixes the weights, and calibration profiles refuse rolling names.

## Tests

`pytest` runs the offline suite (no GPU, network or model downloads; `tests/conftest.py` has a deterministic `FakeRuntime`). Tests marked `gpu`, `network` or `slow` are opt-in: `pytest -m network`, and so on.
