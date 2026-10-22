"""One decision about one image, in the TypeSafe format.

python examples/decide_image.py                                   # d1-omni-600m, native
python examples/decide_image.py --runtime ollama                  # same model in Ollama
python examples/decide_image.py --model qwen3-vl-4b --image my.jpg
"""

import argparse
import tempfile
from pathlib import Path

import httpx

from s1ght import S1GHT, Choice, Image, Noul, Score

BUS = "https://github.com/ultralytics/assets/releases/download/v0.0.0/bus.jpg"

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="d1-omni-600m")
ap.add_argument("--runtime", default=None, help="native, ollama, ollaya, vllm, ... (default: the catalog's first)")
ap.add_argument("--image", default=None, help="an image file (default: Ultralytics' bus.jpg)")
args = ap.parse_args()

if args.image is None:
    args.image = str(Path(tempfile.gettempdir()) / "s1ght-bus.jpg")
    if not Path(args.image).is_file():
        Path(args.image).write_bytes(httpx.get(BUS, follow_redirects=True, timeout=30).content)

model = S1GHT(args.model, runtime=args.runtime)
response = model.decide(
    state={"photo": Image(args.image)},
    questions={
        "bus": Noul("Is there a bus in `photo`?"),
        "scene": Choice("Where was `photo` taken?", ["city street", "football pitch", "kitchen", "beach"]),
        "crowd": Score("How many people are in `photo`?", ["none", "one or two", "three to ten", "more than ten"]),
    },
)

a = response.answers
print(f"bus:   p(yes) = {a['bus'].noul:.3f}")
print(f"scene: {a['scene'].choice} (confidence {a['scene'].confidence:.2f})")
for option, p in a["scene"].probabilities.items():
    print(f"         {option:15s} {p:.3f}")
print(f"crowd: level {a['crowd'].score:.2f} of 0-3")
print(f"\n{response.usage['input_tokens']} input tokens, answered by {response.model}")
model.close()
