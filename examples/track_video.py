"""Hybrid mode: YOLO tracks people and cars, the decision model answers per track, events print as they happen.

Needs the ultralytics extra: pip install "s1ght[ultralytics]"

    python examples/track_video.py 0 --show                 # webcam
    python examples/track_video.py site.mp4 --save site_out.mp4 --budget 4
"""

import argparse
from pathlib import Path

import cv2

from s1ght import S1GHT

ap = argparse.ArgumentParser()
ap.add_argument("source", help="video file, camera index or stream URL")
ap.add_argument("--model", default="d1-omni-600m")
ap.add_argument("--runtime", default=None)
ap.add_argument("--ask", default=str(Path(__file__).with_name("ask.yaml")))
ap.add_argument("--detector", default="yolo11n.pt")
ap.add_argument("--budget", type=float, default=2.0, help="decisions per second")
ap.add_argument("--show", action="store_true", help="show frames (needs opencv-python, not the headless build)")
ap.add_argument("--save", default=None, help="write the annotated video here")
args = ap.parse_args()

source = int(args.source) if args.source.isdigit() else args.source
writer = None
with S1GHT(args.model, runtime=args.runtime) as model:
    results = model.track(source, ask=args.ask, detector=args.detector, budget=args.budget, stream=True)
    try:
        for r in results:
            for e in r.events:
                print(f"[{r.frame:5d}] {e}")
            frame = r.plot()
            if args.save:
                if writer is None:
                    h, w = frame.shape[:2]
                    writer = cv2.VideoWriter(args.save, cv2.VideoWriter.fourcc(*"mp4v"), 30, (w, h))
                writer.write(frame)
            if args.show:
                cv2.imshow("s1ght", frame)
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
    except KeyboardInterrupt:
        pass
    finally:
        if writer is not None:
            writer.release()
        if args.show:
            cv2.destroyAllWindows()
    stats = model.scheduler.stats
    print(f"{stats.frames} frames, {stats.decisions} decisions in {stats.batches} batches, asked because: "
          f"{dict(stats.reasons)}")  # fmt: skip
