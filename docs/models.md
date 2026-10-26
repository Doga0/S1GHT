# Models and runtimes

A **model** is a set of weights; a **runtime** is what runs them. The same model often has several runtimes, and every runtime answers the same TypeSafe questions.

## The catalog

`s1ght models` prints the packaged catalog (`src/s1ght/registry.yaml`):

| Name | Weights | Runtimes (first is the default) | Read through | License |
| --- | --- | --- | --- | --- |
| `d1-3b` | LiquidAI/d1-3B | native, ollaya, llama_cpp | the model's own `system_one` engine | LFM Open License 1.0 |
| `d1-omni-600m` | LiquidAI/d1-omni-600M | native, ollama, llama_cpp | the model's own `system_one` engine | LFM Open License 1.0 |
| `qwen3-vl-4b` | Qwen/Qwen3-VL-4B-Instruct | native, vllm, llama_server | option-letter logits | Apache-2.0 |

S1GHT uses every model for visual decisions: images, with optional text context next to them. Other inputs some checkpoints accept (d1-omni-600M also reads audio) are outside S1GHT's scope.

Every source is pinned to a Hugging Face commit, which fixes the content of every file (including any `trust_remote_code` code). The LFM Open License 1.0 does not cover commercial use by organizations with more than US$10M annual revenue. `llama_cpp` (running GGUF files in-process) is listed but not implemented yet.

## Choosing a runtime

```python
S1GHT("d1-3b")  # the entry's first runtime: native
S1GHT("d1-3b", runtime="ollaya")  # another runtime of the same entry
S1GHT("ollama://clef-flash")  # a URI: no catalog needed
```

| Runtime | How to name it | Runs | Notes |
| --- | --- | --- | --- |
| `native` | catalog name, `hf://org/repo@rev`, `"org/repo"`, a local folder | transformers in this process, on CUDA, MPS or CPU | needs the `native` extra |
| `ollama` | `runtime="ollama"`, `ollama://<tag>` | Ollama's `/v1/systemone` (images as base64) | `OLLAMA_HOST` (default `127.0.0.1:11434`); no Box; ≤ 26 options |
| `ollaya` | `runtime="ollaya"`, `ollaya://<tag>` | Ollaya's `/api/decide` | `OLLAYA_HOST` (default `127.0.0.1:11435`); images re-encoded to PNG |
| `typesafe` | `typesafe://<model>` | TypeSafe's hosted `/v1/systemone` | `TYPESAFE_API_KEY`, `TYPESAFE_BASE_URL`; images 280–1568 px per side; Box on `jev-preview` |
| `vllm` | `runtime="vllm"`, `vllm://host:port/<model>` | OpenAI-compatible chat with top log-probabilities | `vllm serve ... --max-logprobs 20` |
| `llama_server` | `runtime="llama_server"`, `llama-server://host:port` | llama.cpp's server, same client | `llama-server -m model.gguf --mmproj mmproj.gguf` |

Server runtimes take `base_url=`, `api_key=` and `timeout=`:

```python
S1GHT("d1-omni-600m", runtime="ollama", base_url="http://gpu-box:11434", timeout=120)
```

## Native: devices and dtypes

```python
S1GHT("d1-3b")  # cuda if available, then mps, then cpu
S1GHT("d1-3b", device="cuda:1")
S1GHT("d1-3b", dtype="fp32")  # override the catalog's choice
```

The catalog picks the dtype per device: **bf16** on GPUs with compute capability 8.0+ (A100, RTX 30xx/40xx), **fp16** on older GPUs such as the T4, **fp32** on CPU. d1-omni-600M uses fp16 on every GPU, as its model card advises.

Measured warm latency, one request (Tesla T4):

| Model | dtype | latency |
| --- | --- | --- |
| d1-omni-600m | fp16 | 58 ms (fp32: 127 ms) |
| d1-3b | fp16 | 143 ms (fp32: 471 ms) |
| qwen3-vl-4b | fp16 | 2.7–3.9 s for three questions about one photo (no prefix cache yet); vLLM 1.4–2.7 s |

Weights download into `$S1GHT_CACHE` (default `~/.cache/s1ght`).

## Models outside the catalog

```python
S1GHT("Qwen/Qwen2.5-VL-3B-Instruct")  # any HF repo id
S1GHT("./my-finetune")  # a local folder with config.json
```

The family is guessed from `config.json`: Qwen2/2.5/3-VL → `qwen_vl`, LFM2-VL → `lfm2_vl`, d1-omni → `d1_omni`, anything else that loads with `AutoModelForImageTextToText` → `semif_generic`. These models are **uncalibrated**: S1GHT warns, and their probabilities should be calibrated before you threshold them.

## Option-letter models (`qwen_vl`, `semif_generic`, vLLM, llama-server)

General chat VLMs have no decision head. S1GHT turns each question into a multiple-choice prompt (the state, the images, and the options lettered A, B, C, ...) and reads the next-token probabilities of the letters. Nothing is generated, so one question costs one forward pass. Thinking is turned off for Qwen3.

Letter models prefer "A" (about +2.4 nat on Qwen3-VL-4B). When a reading is close (top-two gap under 6 nat), S1GHT asks again with the options in other orders and averages: every cyclic rotation for up to 4 options, which cancels a per-letter bias exactly, else the reversed order.

```python
S1GHT("qwen3-vl-4b", flip="auto")  # default; "always" or "never"
S1GHT("qwen3-vl-4b", margin=4.0, batch=8, max_side=768)
```

| Option | Default | Meaning |
| --- | --- | --- |
| `flip` | `"auto"` | when to ask in other option orders |
| `margin` | `6.0` | gap in nat under which `"auto"` asks again |
| `batch` | `4` | prompts per forward pass (native) |
| `max_side` | `1024` | longer image side in pixels |
| `concurrency` | `4` | requests in flight (vLLM, llama-server) |

S1GHT resizes images itself (sides multiples of 32 px) before any runtime sees them, so native and server paths read the same pixels and agree: on a T4, native fp16 and vLLM gave the same answers to every question; on a CPU, native fp32 and llama-server (Q8_0) did too, with matching prompt token counts.

Limits: at most 26 options per question, and no Box.
