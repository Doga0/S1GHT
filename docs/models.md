# Models and runtimes

A **model** is a set of weights; a **runtime** is what runs them. The same model often has several runtimes, and every runtime answers the same TypeSafe questions.

## The catalog

`s1ght models` prints the packaged catalog (`src/s1ght/registry.yaml`):

| Name | Weights | Runtimes (first is the default) | Read through | License |
| --- | --- | --- | --- | --- |
| `d1-3b` | LiquidAI/d1-3B | ollaya, llama_cpp | the model's own `system_one` engine | LFM Open License 1.0 |
| `d1-omni-600m` | LiquidAI/d1-omni-600M | ollama, llama_cpp | the model's own `system_one` engine | LFM Open License 1.0 |
| `qwen3-vl-4b` | Qwen/Qwen3-VL-4B-Instruct | vllm, llama_server | option-letter logits | Apache-2.0 |

S1GHT uses every model for visual decisions: images, with optional text context next to them. Other inputs some checkpoints accept (d1-omni-600M also reads audio) are outside S1GHT's scope.

Every source is pinned to a Hugging Face commit, which fixes the content of every file. The LFM Open License 1.0 does not cover commercial use by organizations with more than US$10M annual revenue. `llama_cpp` (running GGUF files in-process) is listed but not implemented yet.

## Choosing a runtime

```python
S1GHT("d1-3b")  # the entry's first runtime: ollaya
S1GHT("d1-3b", runtime="ollaya")  # another runtime of the same entry
S1GHT("ollama://clef-flash")  # a URI: no catalog needed
```

| Runtime | How to name it | Runs | Notes |
| --- | --- | --- | --- |
| `ollama` | `runtime="ollama"`, `ollama://<tag>` | Ollama's `/v1/systemone` (images as base64) | `OLLAMA_HOST` (default `127.0.0.1:11434`); no Box; ≤ 26 options |
| `ollaya` | `runtime="ollaya"`, `ollaya://<tag>` | Ollaya's `/api/decide` | `OLLAYA_HOST` (default `127.0.0.1:11435`); images re-encoded to PNG |
| `typesafe` | `typesafe://<model>` | TypeSafe's hosted `/v1/systemone` | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`; images 280–1568 px per side; Box on `jev-preview` |
| `vllm` | `runtime="vllm"`, `vllm://host:port/<model>` | OpenAI-compatible chat with top log-probabilities | `vllm serve ... --max-logprobs 20` |
| `llama_server` | `runtime="llama_server"`, `llama-server://host:port` | llama.cpp's server, same client | `llama-server -m model.gguf --mmproj mmproj.gguf` |

Server runtimes take `base_url=`, `api_key=` and `timeout=`:

```python
S1GHT("d1-omni-600m", runtime="ollama", base_url="http://gpu-box:11434", timeout=120)
```

## Measuring speed

`s1ght bench` times a model by image size and batch size on your hardware ([command line](cli.md)); `benchmarks/latency.py` does the same for several models and prints Markdown:

```bash
s1ght bench model=d1-omni-600m runtime=ollama sizes=[256,640] batches=[1,4]
python benchmarks/latency.py d1-omni-600m:ollama qwen3-vl-4b:llama_server
```

d1-omni-600M on a laptop CPU (Intel Core i7-13700H, 32 GB), one Noul per image, warm median ms per image:

| Longer side | Ollama Q8_0, batch 1 | Ollama Q8_0, batch 16 |
| ---: | ---: | ---: |
| 128 px | 1290 | 1275 |
| 256 px | 1198 | 1251 |
| 384 px | 1726 | 1764 |
| 640 px | 3439 | 3637 |

The client takes 0.5 GB; the model lives in Ollama's process. Batching barely helps on a CPU, since the server client sends a batch's requests one after another (#14). Below 384 px the cost hardly changes with image size, so a smaller frame does not make d1-omni much faster on CPU; most of the time is the fixed cost of the vision encoder and decision trunk.

## Models outside the catalog

Any model a server runtime serves can be named by URI, without a catalog entry:

```python
S1GHT("ollama://hf.co/someone/my-decision-model-GGUF:Q8_0")
S1GHT("vllm://localhost:8000/Qwen/Qwen2.5-VL-3B-Instruct")
```

These models are **uncalibrated**: their probabilities should be calibrated before you threshold them.

## Option-letter models (vLLM, llama-server)

General chat VLMs have no decision head. S1GHT turns each question into a multiple-choice prompt (the state, the images, and the options lettered A, B, C, ...) and reads the next-token probabilities of the letters. Nothing is generated, so one question costs one forward pass. Thinking is turned off for Qwen3.

Letter models prefer "A" (about +2.4 nat on Qwen3-VL-4B). When a reading is close (top-two gap under 6 nat), S1GHT asks again with the options in other orders and averages: every cyclic rotation for up to 4 options, which cancels a per-letter bias exactly, else the reversed order.

```python
S1GHT("qwen3-vl-4b", flip="auto")  # default; "always" or "never"
S1GHT("qwen3-vl-4b", margin=4.0, max_side=768)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `flip` | `"auto"` | when to ask in other option orders |
| `margin` | `6.0` | gap in nat under which `"auto"` asks again |
| `max_side` | `1024` | longer image side in pixels |
| `concurrency` | `4` | requests in flight (vLLM, llama-server) |

S1GHT resizes images itself (sides multiples of 32 px) before any runtime sees them, so every runtime reads the same pixels. In 0.1, vLLM (fp16, T4) and llama-server (Q8_0, CPU) each gave the same answers as the in-process reference to every question checked, with matching prompt token counts.

Limits: at most 26 options per question, and no Box.
