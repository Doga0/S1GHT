"""Full-frame questions over a folder, a video or a camera; annotated frames are saved.

python examples/predict_folder.py photos/ --runtime ollama
python examples/predict_folder.py site.mp4 --vid-stride 15
python examples/predict_folder.py 0                     # webcam, newest frame each time (Ctrl+C to stop)
"""

import argparse
from pathlib import Path

from s1ght import S1GHT

ap = argparse.ArgumentParser()
ap.add_argument("source", help="image, folder, glob, URL, video, camera index or stream URL")
ap.add_argument("--model", default="d1-omni-600m")
ap.add_argument("--runtime", default=None)
ap.add_argument("--questions", default=str(Path(__file__).with_name("questions.yaml")))
ap.add_argument("--vid-stride", type=int, default=1)
ap.add_argument("--out", default="runs/examples/predict")
args = ap.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
out = Path(args.out)
with S1GHT(args.model, runtime=args.runtime) as model:
    try:
        for r in model.predict(source, questions=args.questions, stream=True, vid_stride=args.vid_stride):
            a = r.answers[None]
            print(f"{Path(r.path).name} #{r.frame}: bus={a['bus'].noul:.2f} scene={a['scene'].choice} "
                  f"crowd={a['crowd'].score:.1f}  ({sum(r.speed.values()):.0f} ms)")  # fmt: skip
            r.save(out / f"{Path(r.path).stem}_{r.frame:05d}.jpg")
    except KeyboardInterrupt:
        pass
print(f"annotated frames in {out}")
