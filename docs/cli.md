# Command line

```bash
s1ght MODE key=value ...
```

Like Ultralytics: a mode, then `key=value` pairs. Values are read as YAML (`show=True`, `budget=4`, `thresholds=[0.7,0.3]`, `state={"camera": 3}`); names, paths and URIs stay text. `none` or an empty value means "not set". Dashes work too: `--port=8000`, `api-key=...`. A mistyped mode or key gets a suggestion. Defaults live in `src/s1ght/cfg/default.yaml`.

Exit codes: `0` success, `1` runtime error (missing file, unreachable server, ...), `2` bad command line.

## Modes

| Mode | Does |
| --- | --- |
| `decide` | one request; prints the TypeSafe JSON response |
| `predict` | full-frame questions over images, folders, videos or cameras |
| `track` | hybrid mode: detector + per-track decisions and events |
| `calibrate` | fit a calibration profile and print its report |
| `serve` | serve the model as `/v1/systemone` with images |
| `models` | list the catalog |
| `version`, `help` | |

## Keys

**Model** (every mode except `models`, `version`, `help`):

| Key | Default | |
| --- | --- | --- |
| `model` | `d1-3b` | catalog name, runtime URI, HF repo id or local folder |
| `runtime` | | for catalog names: `native`, `ollama`, `ollaya`, `vllm`, `llama_server` |
| `device` | | native only: `cuda`, `cuda:1`, `mps`, `cpu` |
| `calibration` | | profile JSON from `calibrate` |
| `timeout` | | server runtimes: seconds per request |

**decide**: `source` (an image, sent as `image`), `questions` (YAML/JSON file), `state` (optional text or JSON context sent next to the image as `context`, e.g. a camera name).

**predict**:

| Key | Default | |
| --- | --- | --- |
| `source` | | image, folder, glob, URL, video, camera index, stream URL |
| `questions` | | YAML/JSON file |
| `context` | | extra text or JSON state per frame |
| `batch` | `4` | frames per runtime call |
| `vid_stride` | `1` | keep every n-th video frame |
| `save` | `False` | write annotated images/videos |
| `save_dir` | `runs/predict` | numbered when it exists (`runs/predict2`, ...) |
| `show` | `False` | show results in a window; `q` or Esc stops |
| `json` | `False` | print one `Results.to_json()` line per frame instead of a summary |

**track**: `source`, `ask` (YAML/JSON file: class → questions), and

| Key | Default | |
| --- | --- | --- |
| `detector` | `yolo11n.pt` | Ultralytics weights |
| `conf` | `0.25` | detector confidence |
| `tracker` | `bytetrack.yaml` | |
| `imgsz` | `640` | |
| `detector_device` | | the detector's device |
| `budget` | `2.0` | decisions per second |
| `batch` | `4` | crops per runtime call |
| `pad` | `0.2` | box expansion per side |
| `context` | `crop` | `crop` or `crop+frame` |
| `half_life` | `1.0` | seconds |
| `thresholds` | `[0.6, 0.4]` | Noul on/off for events |
| `min_duration` | `0.0` | seconds above *on* before an event starts |
| `max_age` | `5.0` | re-ask after this many seconds |
| `lost_after` | `1.0` | seconds unseen before a track is lost |
| `realtime` | `False` | pace video files at their frame rate |
| `vid_stride`, `save`, `save_dir`, `show`, `json` | | as for `predict` |

Without `json=True`, `track` prints events as they start and end, then a summary.

**calibrate**: `data` (a manifest file, or a folder with `manifest.jsonl`), `folds` (`5`), `batch` (`4`), `save` (profile path; default `runs/calibrate/<model>-<runtime>.json`), `save_dir`. A reliability diagram is written next to the profile.

**serve**: `host` (`127.0.0.1`), `port` (`11436`), `api_key` (none: no authentication).

## Examples

```bash
s1ght decide model=d1-omni-600m runtime=ollama source=bus.jpg questions=examples/questions.yaml
s1ght decide model=qwen3-vl-4b source=gate.jpg state="camera: gate 3, forklifts only" questions=examples/inspection.yaml
s1ght predict model=d1-omni-600m source="photos/*.jpg" questions=examples/questions.yaml save=True
s1ght track model=d1-omni-600m source=0 ask=examples/ask.yaml show=True budget=4
s1ght track model=d1-omni-600m source=site.mp4 ask=examples/ask.yaml save=True json=True > events.jsonl
s1ght calibrate model=d1-omni-600m runtime=ollama data=out/hardhat/ save=hardhat.json
s1ght serve model=d1-omni-600m runtime=ollama port=11436 api_key=secret
```
