"""Ask the same questions about several objects in one photo, one crop each, without a detector.

The boxes default to the four people in Ultralytics' bus.jpg; pass your own as JSON. Each crop is
the state `{"image": crop}` plus the camera name as context, and all crops go to the runtime in one
`decide_batch` call.

python examples/decide_crops.py --runtime ollama
python examples/decide_crops.py --image site.jpg --boxes "[[10, 20, 200, 400], [300, 40, 420, 380]]"
"""

import argparse
import json
import tempfile
from pathlib import Path

import httpx

from s1ght import S1GHT, Image
from s1ght.engine.crops import crop
from s1ght.io.sources import read_image
from s1ght.schema.questions import load_questions
from s1ght.schema.systemone import SystemOneRequest

BUS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"
BUS_PEOPLE = [[49, 399, 245, 903], [221, 406, 344, 857], [670, 380, 810, 875], [0, 551, 67, 873]]

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="d1-omni-600m")
ap.add_argument("--runtime", default=None)
ap.add_argument("--image", default=None, help="a photo (default: Ultralytics' bus.jpg)")
ap.add_argument("--boxes", default=None, help="JSON list of [x1, y1, x2, y2] pixel boxes")
ap.add_argument("--pad", type=float, default=0.2, help="box expansion per side before cropping")
ap.add_argument("--camera", default="street camera 1", help="context sent next to each crop")
args = ap.parse_args()

if args.image is None:
    args.image = str(Path(tempfile.gettempdir()) / "s1ght-bus.jpg")
    if not Path(args.image).is_file():
        Path(args.image).write_bytes(httpx.get(BUS, follow_redirects=True, timeout=30).content)
boxes = json.loads(args.boxes) if args.boxes else BUS_PEOPLE

frame = read_image(args.image)
questions = load_questions(Path(__file__).with_name("inspection.yaml"))
with S1GHT(args.model, runtime=args.runtime) as model:
    requests = [
        SystemOneRequest(model.name, {"image": Image(crop(frame, box, args.pad)), "camera": args.camera}, questions)  # type: ignore[arg-type]
        for box in boxes
    ]
    for box, r in zip(boxes, model.runtime.decide_batch(requests), strict=True):
        a = r.answers
        print(
            f"{box!s:24s} bag={a['bag'].noul:.2f}  posture={a['posture'].choice:8s}  visible={a['visible'].score:.1f}/3"
        )
