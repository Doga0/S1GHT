# Changelog

All notable changes to S1GHT. Versions follow [semantic versioning](https://semver.org/); before 1.0, minor versions may change the API.

## [Unreleased]

## [0.1.0] - 2026-10-26

The first release: the shared core and hybrid mode (P1).

### Added

- **Questions and answers in the TypeSafe `/v1/systemone` schema:** `Noul`, `Choice`, `Score`, `Box`, with images anywhere in the state or questions. Wire formats for TypeSafe (assets, multipart) and Ollama (`images`).
- **Runtimes behind one `Runtime` protocol:**
  - native in-process models: d1-3B, d1-omni-600M, and Qwen2/2.5/3-VL or any chat VLM through option letters, on CUDA, MPS or CPU, with per-device dtypes;
  - the `/v1/systemone` client for TypeSafe, Ollama and Ollaya;
  - an OpenAI-compatible logprob client for vLLM and llama-server.
- **Model catalog** (`registry.yaml`) pinned to Hugging Face commits, and name/URI resolution (`hf://`, `ollama://`, `ollaya://`, `typesafe://`, `vllm://`, `llama-server://`).
- **`S1GHT`:**
  - `decide()` for one request;
  - `predict()` for full-frame questions over images, folders, videos, cameras and streams, with the latest frame winning on live sources;
  - `track()` for hybrid mode: Ultralytics detection and tracking, a non-blocking scheduler with a decisions-per-second budget, temporal fusion and events.
- **`Results`:** `plot()`, `save()`, `to_systemone()`, `to_json()`.
- **Calibration:** per question type and input kind, with held-out ECE/Brier/NLL reports, profiles bound to model and source, and a hard-hat state-pair benchmark.
- **Option-letter readout with order debiasing:** cyclic rotations for up to 4 options. Native and server paths send the same prompts and pixels.
- **`s1ght` command line:** `decide`, `predict`, `track`, `calibrate`, `serve`, `bench`, `models`.
- **`s1ght bench`:** latency by image size and batch size (`benchmarks/latency.py` for several models at once), with measured CPU and GPU (T4, A100, RTX PRO 6000) tables in `docs/models.md`.
- **`s1ght serve`:** any model as `/v1/systemone` with images. The official TypeSafe SDK connects with `TYPESAFE_BASE_URL`.
- **Documentation and runnable examples**, all visual: images with optional text context. Text-only and audio use are out of scope.
- **Releases:** the version lives only in `s1ght.__version__`, and a `v*` tag publishes to PyPI after the wheel's tests pass.
