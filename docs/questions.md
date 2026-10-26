# Questions and answers

S1GHT speaks the TypeSafe `/v1/systemone` schema. A request is a **state** (what to look at) and a set of named **questions**; the response has one typed **answer** per question.

```python
from s1ght import S1GHT, Image, Noul, Choice, Score, Box

model = S1GHT("d1-omni-600m")
r = model.decide(state={"photo": Image("bus.jpg"), "note": "Camera 3, 08:14"}, questions={...})
r.answers  # {question id: answer}
r.usage  # {"input_tokens": ..., "output_tokens": 0}
```

## State

S1GHT is for visual decisions, so the state holds one or more images:

- **Images**: `Image` objects anywhere a value can go: `{"photo": Image("bus.jpg")}`, `{"before": Image(a), "after": Image(b)}`.
- **Context** (optional): text or JSON next to the images that helps the question, such as a camera name, a zone or the time: `{"photo": Image("gate.jpg"), "camera": "gate 3", "zone": "forklifts only"}`.

The schema itself also accepts states without images, but S1GHT's examples, defaults and calibration are built for images.

Questions refer to parts of the state by name, conventionally in backticks: ``"Is there a bus in `photo`?"``. In `predict()` and `track()` the frame or crop is always the key `image` (plus `frame` with `track(context="crop+frame")`), so questions there say `` `image` ``.

`Image` accepts a file path, encoded bytes, a BGR numpy array (as OpenCV gives you) or a PIL image. Encoded files are sent as they are; arrays are encoded to JPEG unless you pass `format="png"`.

## Question types

### Noul: yes or no

```python
Noul("Is there a bus in `photo`?")
Noul("Is the worker safe?", criteria={"true": "wears a hard hat and vest", "false": "anything missing"})
```

Answer: `NoulAnswer(noul=0.97)`, the probability of yes. `answer.confidence` is `|2p − 1|`, TypeSafe's suggested proxy.

### Choice: one of several options

```python
Choice("What is the person doing?", ["standing", "bending", "on a ladder", "other"])
Choice(
    "Which vehicle is in `photo`?",
    {"forklift": None, "truck": "a road truck or lorry", "van": None, "none": "no vehicle"},
)
```

2 to 255 options (26 for option-letter models such as Qwen3-VL, see [models](models.md)). Descriptions are optional; use `None` for an undescribed option.

Answer: `ChoiceAnswer(choice="standing", probabilities={...}, confidence=0.83)`. Confidence follows TypeSafe: `(p_max − 1/n) / (1 − 1/n)`, so 0 is a uniform guess and 1 is certain.

### Score: an ordered level

```python
Score("How crowded is the platform?", ["empty", "a few people", "busy", "packed"])
```

2 to 10 levels, lowest first. Answer: `ScoreAnswer(score=1.7, legend={0: "empty", ...}, probabilities={0: 0.05, ...}, confidence=...)`. `score` is the probability-weighted level, so it can fall between levels.

### Box: where is one target

```python
Box("the bus in `photo`")
```

Answer: `BoxAnswer(box={"top": 0.21, "bottom": 0.78, "left": 0.05, "right": 0.97}, confidence=...)`, edges as fractions of the image. Box questions run only on runtimes that support them (TypeSafe `jev-preview`); others refuse them before sending. There is no "not found" answer, so pair a Box with a Noul that asks whether the target is there.

## Questions in YAML or JSON

Every question has a wire form, which is also the file format:

```yaml
# examples/questions.yaml: a flat map for decide() and predict()
bus: {type: noul, instructions: "Is there a bus in `image`?"}
scene:
  type: choice
  instructions: "Where was `image` taken?"
  criteria: {city street: null, football pitch: null, kitchen: null}
crowd:
  type: score
  instructions: "How many people are in `image`?"
  criteria: ["none", "one or two", "three to ten", "more than ten"]
```

```yaml
# examples/ask.yaml: keyed by detector class, for track(ask=...)
person:
  no_helmet: {type: noul, instructions: "Is the person in `image` NOT wearing a hard hat?"}
```

```python
model.decide(state, "examples/questions.yaml")
model.predict("photos/", questions="examples/questions.yaml")
model.track("site.mp4", ask="examples/ask.yaml")

from s1ght.schema.questions import load_questions

load_questions("examples/questions.yaml")  # {"bus": Noul(...), ...}
```

## Answers on the wire

`r.to_wire()` gives the TypeSafe JSON response; `SystemOneResponse.from_wire(d)` reads one back. In `predict()`/`track()` results, `r.to_systemone()` gives one response per answered track (and one for the full frame), and `r.to_json()` adds boxes, events and timings.

```json
{"model": "d1-omni-600m",
 "answers": {"bus": {"type": "noul", "noul": 0.9999},
             "scene": {"type": "choice", "choice": "city street",
                       "probabilities": {"city street": 0.99, "football pitch": 0.0, "kitchen": 0.01},
                       "confidence": 0.99}},
 "usage": {"input_tokens": 3614, "output_tokens": 0}}
```

## Writing good questions

- Name the image: ``"... in `image`?"`` is answered better than a bare question, especially by small models.
- For crops that include surroundings (`pad`), say which object you mean: *"the person in the center of `image`"*.
- Ask about state, not location: decision models are strong at *"is it X"* and weak at *"where"* or *"how many"* over a large scene. Let the detector find objects and ask about each one ([hybrid mode](hybrid.md)).
- Probabilities from an uncalibrated model are often overconfident. Calibrate before you threshold ([calibration](calibration.md)).
