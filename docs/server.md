# Server: `s1ght serve`

`s1ght serve` puts any S1GHT model behind TypeSafe's `/v1/systemone` API, with images. Existing TypeSafe code then runs on a local model by changing one environment variable.

```bash
pip install "s1ght[serve]"
s1ght serve model=d1-omni-600m port=11436
```

```python
# unchanged TypeSafe SDK code (pip install typesafe-sdk)
# TYPESAFE_BASE_URL=http://127.0.0.1:11436  TYPESAFE_API_KEY=local
from pathlib import Path

from typesafe_sdk import Image, Noul, TypeSafeClient

with TypeSafeClient() as client:
    r = client.system_one(
        model="jev-latest",
        state={"photo": Image(Path("bus.jpg"))},
        questions={"bus": Noul(instructions="Is there a bus in `photo`?")},
    )
    print(r.model, r.nouls["bus"].noul)  # d1-omni-600m 0.9999
```

From Python: `S1GHT("d1-omni-600m").serve(host="127.0.0.1", port=11436, api_key=None)`. Port 11436 sits next to Ollama (11434) and Ollaya (11435).

## Routes

| Route | |
| --- | --- |
| `POST /v1/systemone` | TypeSafe requests: JSON with base64 `assets`, or multipart with the JSON in a part named `request` and one file part per asset. Requests with a top-level `images` list (Ollama's dialect) are accepted too. |
| `POST /api/decide` | Ollaya's image route (Ollama dialect), so `S1GHT("ollaya://...", base_url="http://host:11436")` and Ollaya clients work |
| `GET /v1/models` | TypeSafe's model list: the served model |
| `GET /health` | `{"status": "ok", "model", "runtime", "box"}` |

## Behaviour

- **One model.** Every request is answered by the served model, whatever its `model` field says, so apps can keep `model="jev-latest"`. The response's `model` names the served model.
- **Images.** The `TypeSafe-Danger-Zone-Images` header is accepted but not required. Images outside the runtime's size limits are resized to fit rather than rejected. In the Ollama dialect, `[image N]` labels are turned back into the images they stand for.
- **Compression.** Bodies may be gzip or zstd compressed (the TypeSafe SDK sends zstd when available; zstd needs Python 3.14+ or `pip install backports.zstd`).
- **Errors.** JSON `{"error": {"type", "message"}}` with TypeSafe's status codes: 401 (bad API key), 413 (over 32 MiB), 415 (unsupported encoding), 422 (invalid request or a question the runtime cannot answer, e.g. Box), 400 (unreadable body).
- **Auth.** With `api_key=...`, requests must send `Authorization: Bearer <key>`. Set one whenever you bind to anything but `127.0.0.1`.
- **Concurrency.** Requests are answered one at a time, in the order they arrive.

## Measured

d1-omni-600M on a CPU laptop, official `typesafe-sdk` 0.7.5: Ollama runtime 20–43 s per photo with two questions, native 10–21 s, with the same answers as calling S1GHT directly.
