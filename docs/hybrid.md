# Hybrid mode: `track`

A detector finds and tracks objects; the decision model answers questions about each track. The two never wait for each other.

```python
from s1ght import S1GHT, Noul, Choice

model = S1GHT("d1-omni-600m")
ask = {
    "person": {
        "no_helmet": Noul("Is the person in `image` NOT wearing a hard hat?"),
        "posture": Choice("What is the person in `image` doing?", ["standing", "bending", "on a ladder", "other"]),
    },
    "truck": {"loaded": Noul("Is the truck in `image` carrying a load?")},
}
for r in model.track("site.mp4", detector="yolo11n.pt", ask=ask, budget=4, stream=True):
    for tid, answers in r.answers.items():
        print(tid, answers["no_helmet"].noul if "no_helmet" in answers else answers)
    for e in r.events:
        print(e)  # "track 7: no_helmet p=0.91, ongoing" / "track 7: no_helmet p=0.91, 3.2s"
```

Needs the `ultralytics` extra (AGPL-3.0). `ask` maps detector class names to questions; only those classes are detected (`classes=` overrides it).

## How it works

```
frames ──► detector + tracker ──► Results (every frame, at detector speed)
                │ submit(frame): never blocks
                ▼
           scheduler ── crops of tracks worth asking ──► worker thread ──► runtime.decide_batch
                ▲                                                              │
                └──────────── fusion: smoothed answers + events ◄──────────────┘
```

1. **Detector.** Ultralytics `model.track(stream=True, persist=True)`; S1GHT consumes its output and does not patch Ultralytics.
2. **Scheduler.** For every tracked object of an asked class it decides whether to ask now, in priority order:
   1. **new**: a new track, or one back after being lost for `lost_after` seconds;
   2. **changed**: its appearance (a 32×32 colour signature) moved more than 0.10;
   3. **uncertain**: some Noul answer has `|2p − 1| < 0.3` and 1 s passed;
   4. **stale**: `max_age` seconds since it was last asked.

   A newer crop of a track replaces its queued one (latest frame wins). A worker thread takes the most urgent crops, `batch` at a time and at most `budget` decisions per second.
3. **Crops.** Each box is expanded by `pad` per side and cropped. With `context="crop+frame"` the state also holds a downscaled full frame as `frame`, for relation questions (*"is the person in `image` inside the fenced area in `frame`?"*).
4. **Fusion.** Answers are averaged over time per track with a half-life: Noul in logit space, Choice and Score as probability vectors, Box as the latest answer. A Noul turns **on** when the average passes `thresholds[0]` (for at least `min_duration` seconds) and **off** below `thresholds[1]`; each change is an `Event`.

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | | video file, camera index, stream URL, image or folder |
| `ask` | | class → questions, or a YAML/JSON file of that shape |
| `detector` | `"yolo11n.pt"` | Ultralytics weights (`yolo11n.pt`, `yolo11s.pt`, `yoloe-11s-seg.pt`, ...) or a `Detector` object |
| `stream` | `False` | return a generator |
| `budget` | `2.0` | most decisions per second |
| `batch` | `4` | most crops per runtime call |
| `pad` | `0.2` | box expansion per side, as a fraction of its size |
| `context` | `"crop"` | `"crop"` or `"crop+frame"` |
| `half_life` | `1.0` | seconds for an old answer's weight to halve; `0` keeps only the latest |
| `thresholds` | `(0.6, 0.4)` | Noul on/off probabilities for events |
| `min_duration` | `0.0` | seconds above *on* before an event starts |
| `max_age` | `5.0` | re-ask a track after this many seconds |
| `lost_after` | `1.0` | seconds unseen before a track is lost and its events end |
| `realtime` | `False` | pace a video file at its frame rate, as a camera would |
| `wait` | `True` | for files, answer still-queued crops at the end and yield one last `Results` with every event closed |
| `conf`, `tracker`, `imgsz`, `device`, `classes` | Ultralytics defaults | detector settings; other keywords (e.g. `vid_stride`) go to `model.track` |

## Results

`r.answers` maps track id → `{question id: smoothed answer}` for tracks visible in the frame that have answers. `r.boxes.id` holds the track ids. `r.events` lists events that started (`end is None`) or ended on this frame. `r.plot()` draws boxes with each track's answers.

`model.scheduler.stats` counts frames, queued and replaced crops, decisions, batches, errors and the reasons crops were asked (`new`, `changed`, `uncertain`, `stale`).

## Sizing the budget

The detector's frame rate is independent of the decision model, but they share the machine. Measured on a CPU-only laptop with YOLO11n and d1-omni-600M through Ollama, the detector dropped from 14.5 to 4–6 fps while the decision model ran, because both used the same cores; it never stalled. On a GPU the two contend far less. Start with a budget a little below what the runtime sustains (`decide_batch` calls per second × `batch`) and watch `stats.replaced`: a high count means tracks are waiting and their crops are being replaced by newer ones.

## Custom detectors

Any object with `stream(source, **kwargs)` yielding `s1ght.detectors.base.Frame` objects (image, `xyxy`, `conf`, `cls`, `ids`, `names`, time `t`) works as `detector=`.

## Command line

```bash
s1ght track model=d1-omni-600m source=0 ask=examples/ask.yaml show=True
s1ght track model=d1-omni-600m source=site.mp4 ask=ask.yaml budget=4 save=True      # writes runs/track/site.mp4
s1ght track model=d1-omni-600m source=site.mp4 ask=ask.yaml json=True > track.jsonl
```
