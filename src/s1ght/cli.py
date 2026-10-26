"""`s1ght` command line, Ultralytics-style: `s1ght MODE key=value ...`.

    s1ght decide    model=d1-3b source=bus.jpg questions=q.yaml
    s1ght predict   model=d1-3b source=photos/ questions=q.yaml save=True
    s1ght track     model=d1-3b source=0 ask=ask.yaml show=True
    s1ght calibrate model=d1-3b data=state_pairs/manifest.jsonl
    s1ght serve     model=d1-3b port=11436
    s1ght models | version | help

Defaults come from `s1ght/cfg/default.yaml`. Values are read as YAML (`show=True`,
`thresholds=[0.7,0.3]`); names, paths and URIs stay strings.
"""

from __future__ import annotations

import difflib
import json
import sys
import time
from collections.abc import Callable, Iterable
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

MODEL_KEYS = ("model", "runtime", "device", "calibration", "timeout")
MODE_KEYS: dict[str, tuple[str, ...]] = {
    "decide": (*MODEL_KEYS, *["source", "questions", "state"]),
    "predict": (
        *MODEL_KEYS,
        *["source", "questions", "context", "batch", "vid_stride", "show", "save", "save_dir", "json"],
    ),
    "track": (
        *MODEL_KEYS,
        *[
            "source",
            "ask",
            "detector",
            "conf",
            "tracker",
            "imgsz",
            "detector_device",
            "budget",
            "batch",
            "pad",
            "context",
            "half_life",
            "thresholds",
        ],
        *["min_duration", "max_age", "lost_after", "realtime", "vid_stride", "show", "save", "save_dir", "json"],
    ),
    "calibrate": (*MODEL_KEYS, *["data", "folds", "batch", "save", "save_dir"]),
    "serve": (*MODEL_KEYS, "host", "port", "api_key"),
    "models": (),
    "version": (),
    "help": (),
}
# kept as text even when they look like YAML numbers or booleans ("0" for source is the exception)
STRING_KEYS = frozenset(
    {"model", "runtime", "device", "calibration", "source", "questions", "ask", "state", "detector", "tracker",
     "detector_device", "data", "save_dir", "host", "api_key"}
)  # fmt: skip


class UsageError(Exception):
    """Bad command line: printed with a hint, exit code 2."""


def defaults() -> dict[str, Any]:
    text = resources.files("s1ght").joinpath("cfg/default.yaml").read_text(encoding="utf-8")
    return yaml.safe_load(text)


def parse_args(argv: list[str]) -> tuple[str, dict[str, Any]]:
    """Split `MODE key=value ...` into the mode and its full settings (defaults filled in)."""
    if not argv or argv[0] in ("-h", "--help"):
        return "help", {}
    mode, args = argv[0], argv[1:]
    if mode not in MODE_KEYS:
        close = difflib.get_close_matches(mode, MODE_KEYS, n=1)
        raise UsageError(f"unknown mode {mode!r}" + (f"; did you mean {close[0]!r}?" if close else ""))
    allowed = MODE_KEYS[mode]
    cfg = {k: v for k, v in defaults().items() if k in allowed}
    for arg in args:
        key, sep, value = arg.partition("=")
        key = key.strip().lstrip("-").replace("-", "_")
        if not sep or not key:
            raise UsageError(f"expected key=value, got {arg!r}")
        if key not in allowed:
            close = difflib.get_close_matches(key, allowed, n=1)
            hint = f"; did you mean {close[0]!r}?" if close else f"; {mode} takes: {', '.join(allowed) or 'nothing'}"
            raise UsageError(f"{mode} has no key {key!r}{hint}")
        cfg[key] = parse_value(key, value)
    return mode, cfg


def parse_value(key: str, value: str) -> Any:
    if value.lower() in ("none", "null", ""):
        return None
    if key == "source" and value.isdigit():
        return int(value)
    if key in STRING_KEYS:
        return value
    try:
        return yaml.safe_load(value)
    except yaml.YAMLError:
        return value


def require(cfg: dict[str, Any], *keys: str, mode: str) -> None:
    missing = [k for k in keys if cfg.get(k) is None]
    if missing:
        raise UsageError(f"{mode} needs {', '.join(f'{k}=...' for k in missing)}")


def load_model(cfg: dict[str, Any]) -> Any:
    from s1ght.model import S1GHT

    kwargs: dict[str, Any] = {}
    if cfg.get("device") is not None:
        kwargs["device"] = cfg["device"]
    if cfg.get("timeout") is not None:
        kwargs["timeout"] = float(cfg["timeout"])
    return S1GHT(cfg["model"], runtime=cfg.get("runtime"), calibration=cfg.get("calibration"), **kwargs)


def increment(path: Path) -> Path:
    """`path` if it does not exist, else `path2`, `path3`, ..."""
    if not path.exists():
        return path
    n = 2
    while path.with_name(f"{path.name}{n}").exists():
        n += 1
    return path.with_name(f"{path.name}{n}")


def format_answer(a: Any) -> str:
    from s1ght.schema.systemone import BoxAnswer, ChoiceAnswer, NoulAnswer, ScoreAnswer

    if isinstance(a, NoulAnswer):
        return f"{a.noul:.2f}"
    if isinstance(a, ChoiceAnswer):
        return f"{a.choice} ({a.confidence:.2f})"
    if isinstance(a, ScoreAnswer):
        return f"{a.score:.2f}"
    if isinstance(a, BoxAnswer):
        b = a.box
        return f"box[{b['left']:.2f},{b['top']:.2f},{b['right']:.2f},{b['bottom']:.2f}]"
    return str(a)


def summary(answers: dict[str, Any]) -> str:
    return "  ".join(f"{qid}={format_answer(a)}" for qid, a in answers.items())


class Output:
    """Saves (`save=True`) and shows (`show=True`) annotated results.

    Image sources become `<save_dir>/<name>.jpg`; videos and live sources one `<name>.mp4` each.
    A result that repeats the previous frame (the final summary of `track`) is not written again.
    """

    def __init__(self, save_dir: Path | None, show: bool, fps: Callable[[str], float]) -> None:
        self.save_dir, self.show, self.fps = save_dir, show, fps
        self.writers: dict[str, Any] = {}
        self.last: tuple[str, int] | None = None
        self.saved: set[Path] = set()

    def __call__(self, r: Any) -> bool:
        """Handle one result; False when the user closed the window (q / Esc)."""
        import cv2

        if self.last == (r.path, r.frame):
            return True
        self.last = (r.path, r.frame)
        frame = r.plot()
        if self.save_dir is not None:
            from s1ght.io.sources import VID_SUFFIXES, is_live

            name = Path(str(r.path)).stem or "image"
            if is_live(r.path) or Path(str(r.path)).suffix.lower() in VID_SUFFIXES:
                w = self.writers.get(r.path)
                if w is None:
                    out = self.save_dir / f"{name}.mp4"
                    h, wd = frame.shape[:2]
                    w = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), self.fps(r.path), (wd, h))
                    self.writers[r.path] = w
                    self.saved.add(out)
                w.write(frame)
            else:
                out = self.save_dir / (f"{name}.jpg" if r.frame == 0 else f"{name}_{r.frame}.jpg")
                if out in self.saved:  # same name from another folder or source
                    out = out.with_name(f"{out.stem}_{len(self.saved)}.jpg")
                if not cv2.imwrite(str(out), frame):
                    raise OSError(f"could not write {out}")
                self.saved.add(out)
        if self.show:
            try:
                cv2.imshow("s1ght", frame)
                key = cv2.waitKey(1) & 0xFF
            except cv2.error as e:
                raise UsageError("show=True needs a GUI build of OpenCV: pip install opencv-python") from e
            if key in (ord("q"), 27):
                return False
        return True

    def close(self) -> None:
        for w in self.writers.values():
            w.release()
        if self.show:
            import cv2

            try:
                cv2.destroyAllWindows()
            except cv2.error:
                pass


def video_fps(path: str) -> float:
    import cv2

    from s1ght.io.sources import is_live

    if is_live(path):
        return 30.0
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cap.release()
    return float(fps)


def output_for(mode: str, cfg: dict[str, Any]) -> Output | None:
    if not cfg["save"] and not cfg["show"]:
        return None
    save_dir = None
    if cfg["save"]:
        save_dir = Path(cfg["save_dir"]) if cfg["save_dir"] else increment(Path("runs") / mode)
        save_dir.mkdir(parents=True, exist_ok=True)
    stride = int(cfg.get("vid_stride") or 1)
    return Output(save_dir, bool(cfg["show"]), lambda p: video_fps(p) / stride)


def consume(results: Iterable[Any], out: Output | None, each: Callable[[Any], None]) -> int:
    n = 0
    try:
        for r in results:
            n += 1
            each(r)
            if out is not None and not out(r):
                break
    finally:
        if out is not None:
            out.close()
    return n


# --- modes -----------------------------------------------------------------------------------


def run_decide(cfg: dict[str, Any]) -> None:
    from s1ght.schema.assets import Image
    from s1ght.schema.questions import load_questions

    require(cfg, "source", "questions", mode="decide")
    questions = load_questions(cfg["questions"])
    if any(not hasattr(q, "to_wire") for q in questions.values()):
        raise UsageError(f"{cfg['questions']} is keyed by detector class; use it with track ask=")
    context: Any = cfg["state"]
    if isinstance(context, str):
        try:
            context = json.loads(context)
        except json.JSONDecodeError:
            pass
    state: dict[str, Any] = {"image": Image(str(cfg["source"]))}
    if context is not None:
        state["context"] = context
    with load_model(cfg) as model:
        print(json.dumps(model.decide(state, questions).to_wire(), indent=2))


def run_predict(cfg: dict[str, Any]) -> None:
    require(cfg, "source", "questions", mode="predict")
    from s1ght.schema.questions import load_questions

    questions = load_questions(cfg["questions"])
    if any(not hasattr(q, "to_wire") for q in questions.values()):
        raise UsageError(f"{cfg['questions']} is keyed by detector class; use it with track ask=")
    out = output_for("predict", cfg)
    t0 = time.perf_counter()
    with load_model(cfg) as model:
        results = model.predict(
            cfg["source"], questions, stream=True, context=cfg["context"],
            batch=int(cfg["batch"]), vid_stride=int(cfg["vid_stride"]),
        )  # fmt: skip

        def each(r: Any) -> None:
            if cfg["json"]:
                print(r.to_json())
            else:
                ms = sum(r.speed.values())
                where = r.path if r.frame == 0 else f"{r.path} #{r.frame}"
                print(f"{where}: {summary(r.answers.get(None, {}))}  {ms:.0f}ms")

        n = consume(results, out, each)
    if not cfg["json"]:
        print(f"{n} frame(s) in {time.perf_counter() - t0:.1f}s")
    if out is not None and out.save_dir is not None:
        print(f"saved to {out.save_dir}")


def run_track(cfg: dict[str, Any]) -> None:
    require(cfg, "source", "ask", mode="track")
    out = output_for("track", cfg)
    det = {k: cfg[k] for k in ("conf", "tracker", "imgsz") if cfg[k] is not None}
    if cfg["detector_device"] is not None:
        det["device"] = cfg["detector_device"]
    if int(cfg["vid_stride"]) != 1:
        det["vid_stride"] = int(cfg["vid_stride"])
    thresholds = tuple(cfg["thresholds"])
    if len(thresholds) != 2:
        raise UsageError("thresholds takes two values: [on, off]")
    t0 = time.perf_counter()
    events = 0
    with load_model(cfg) as model:
        results = model.track(
            cfg["source"], ask=cfg["ask"], detector=cfg["detector"], stream=True,
            budget=float(cfg["budget"]), batch=int(cfg["batch"]), pad=float(cfg["pad"]),
            context=cfg["context"] or "crop", half_life=float(cfg["half_life"]),
            thresholds=(float(thresholds[0]), float(thresholds[1])), min_duration=float(cfg["min_duration"]),
            max_age=float(cfg["max_age"]), lost_after=float(cfg["lost_after"]), realtime=bool(cfg["realtime"]),
            **det,
        )  # fmt: skip

        def each(r: Any) -> None:
            nonlocal events
            if cfg["json"]:
                print(r.to_json())
                return
            for e in r.events:
                events += 1
                state = "started" if e.end is None else "ended"
                print(f"[frame {r.frame}] {e} ({state})")

        n = consume(results, out, each)
    if not cfg["json"]:
        dt = time.perf_counter() - t0
        print(f"{n} frame(s) in {dt:.1f}s ({n / dt if dt else 0:.1f} fps), {events} event change(s)")
    if out is not None and out.save_dir is not None:
        print(f"saved to {out.save_dir}")


def run_calibrate(cfg: dict[str, Any]) -> None:
    from s1ght.engine.calibration import save_reliability

    require(cfg, "data", mode="calibrate")
    data = Path(cfg["data"])
    if data.is_dir():
        data = data / "manifest.jsonl"
    t0 = time.perf_counter()

    def progress(done: int, total: int) -> None:
        print(f"\r{done}/{total} examples", end="", file=sys.stderr, flush=True)

    with load_model(cfg) as model:
        if isinstance(cfg["save"], str):
            path = Path(cfg["save"])
        else:
            save_dir = Path(cfg["save_dir"]) if cfg["save_dir"] else increment(Path("runs") / "calibrate")
            tag = f"{model.name}-{model.spec.runtime}".replace("/", "_").replace(":", "_")
            path = save_dir / f"{tag}.json"
        profile = model.calibrate(
            data, save=path, folds=int(cfg["folds"]), batch=int(cfg["batch"]), apply=False, progress=progress
        )
    print(f"\n{time.perf_counter() - t0:.0f}s\n", file=sys.stderr)
    print(profile.report())
    png = save_reliability(profile, path.with_suffix(".png"))
    print(f"\nprofile: {path}\nreliability diagram: {png}")
    print(f"use it with: calibration={path}")


def run_serve(cfg: dict[str, Any]) -> None:
    model = load_model(cfg)
    print(f"serving {model.name} ({model.spec.runtime}) on http://{cfg['host']}:{cfg['port']}/v1/systemone")
    print(f"TypeSafe clients: TYPESAFE_BASE_URL=http://{cfg['host']}:{cfg['port']}", flush=True)
    try:
        model.serve(cfg["host"], int(cfg["port"]), api_key=cfg["api_key"])
    finally:
        model.close()


def run_models(cfg: dict[str, Any]) -> None:
    from s1ght.hub import load_registry

    rows = [(e.name, ", ".join(e.runtimes), e.family, e.license) for e in load_registry().values()]
    head = ("name", "runtimes", "family", "license")
    widths = [max(len(r[i]) for r in [head, *rows]) for i in range(len(head))]
    for r in [head, *rows]:
        print("  ".join(c.ljust(w) for c, w in zip(r, widths)).rstrip())


def run_version(cfg: dict[str, Any]) -> None:
    from s1ght import __version__

    print(__version__)


def run_help(cfg: dict[str, Any]) -> None:
    print(__doc__.strip())
    d = defaults()
    print()
    for mode, keys in MODE_KEYS.items():
        if keys:
            print(f"{mode}: " + " ".join(f"{k}={'' if d.get(k) is None else d[k]}" for k in keys))


MODES: dict[str, Callable[[dict[str, Any]], None]] = {
    "decide": run_decide,
    "predict": run_predict,
    "track": run_track,
    "calibrate": run_calibrate,
    "serve": run_serve,
    "models": run_models,
    "version": run_version,
    "help": run_help,
}


def main(argv: list[str] | None = None) -> int:
    """Entry point of the `s1ght` command; returns the exit code."""
    argv = sys.argv[1:] if argv is None else argv
    try:
        mode, cfg = parse_args(argv)
        MODES[mode](cfg)
    except UsageError as e:
        print(f"s1ght: {e}\nrun 's1ght help' for modes and keys", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        return 130
    except (FileNotFoundError, KeyError, ValueError, ImportError, OSError, RuntimeError) as e:
        msg = e.args[0] if isinstance(e, KeyError) and e.args else e
        print(f"s1ght: error: {msg}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
