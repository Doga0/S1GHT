# Images, folders and video: `predict`

`predict()` asks the same full-frame questions about every frame of a source.

```python
results = model.predict("bus.jpg", questions="examples/questions.yaml")  # list of Results
for r in model.predict("site.mp4", questions=qs, stream=True, vid_stride=10):  # generator
    print(r.frame, r.answers[None]["bus"].noul)
```

Each frame is sent as the state `{"image": <frame>}`; with `context="..."` (text or JSON) the state is `{"image": <frame>, "context": ...}`. Frames are resized to the runtime's size limits when it has them (TypeSafe: sides of 280–1568 px).

## Arguments

| Argument | Default | Meaning |
| --- | --- | --- |
| `source` | | see below |
| `questions` | | question objects, wire dicts or a YAML/JSON file |
| `stream` | `False` | return a generator instead of a list; use it for videos and live sources |
| `context` | `None` | extra state sent with every frame |
| `batch` | `4` | frames per runtime call (live sources use 1) |
| `vid_stride` | `1` | keep every n-th video frame |

## Sources

| Source | Example |
| --- | --- |
| image file | `"bus.jpg"` |
| folder or glob | `"photos/"`, `"photos/**/*.png"` |
| image URL | `"https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"` |
| video file | `"site.mp4"` |
| camera | `0` or `"0"` |
| stream | `"rtsp://user:pass@cam/stream"`, `"http://cam/mjpeg"` |
| in memory | a BGR numpy array, a PIL image, an `s1ght.Image`, encoded bytes, or a list of these |

Live sources (cameras and streams) are read on a background thread that keeps only the newest frame: a slow model always gets the latest picture instead of a growing backlog.

## Results

| Attribute | |
| --- | --- |
| `r.answers` | `{None: {qid: answer}}` for `predict`; `{track_id: {...}}` for `track` |
| `r.orig_img` | the frame, BGR uint8 |
| `r.boxes` | detections (`xyxy`, `conf`, `cls`, `id`, `names`); empty for `predict` |
| `r.events` | events that started or ended on this frame (`track`) |
| `r.speed` | milliseconds per stage (`preprocess`, `decide`, ...) |
| `r.path`, `r.frame` | where the frame came from and its index |
| `r.usage` | tokens read for this frame |

| Method | |
| --- | --- |
| `r.plot(boxes=True, labels=True, panel=True, font_scale=None)` | a copy of the frame with boxes, per-track decisions and a full-frame panel drawn |
| `r.save("out.jpg")` | write `plot()` to a file |
| `r.to_systemone()` | TypeSafe responses: the full frame first, then tracks by id |
| `r.to_dict()`, `r.to_json(indent=None)` | answers in wire form plus boxes, events, speed and usage |

## Command line

```bash
s1ght predict model=d1-omni-600m source="photos/*.jpg" questions=examples/questions.yaml save=True
s1ght predict model=d1-omni-600m source=site.mp4 questions=q.yaml vid_stride=10 json=True > answers.jsonl
```

`save=True` writes annotated images (videos for video and live sources) to `runs/predict`, `runs/predict2`, ...; `show=True` opens a window (needs `opencv-python` rather than the headless build).
