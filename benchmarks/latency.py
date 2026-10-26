"""Latency table for several models and runtimes, as Markdown (for docs/models.md and the design doc).

    python benchmarks/latency.py d1-omni-600m:native d1-omni-600m:ollama --sizes 128 256 384 640 --batches 1 4 16
    python benchmarks/latency.py qwen3-vl-4b:native --device cuda --out results/t4.json

Each target is `model[:runtime]`. One model is loaded at a time and closed before the next, so
memory numbers are per model. See `s1ght.bench` for what is measured.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

from s1ght import S1GHT
from s1ght.bench import BenchRow, bench


def hardware(device: str | None) -> str:
    if device and device.startswith("cuda"):
        import torch

        return torch.cuda.get_device_name(device)
    return platform.processor() or platform.machine()


def markdown(rows: list[BenchRow], hw: str) -> str:
    sizes = sorted({r.size for r in rows})
    batches = sorted({r.batch for r in rows})
    lines = [f"Hardware: {hw}. Cells: ms per image (images/s), warm median.", ""]
    for (model, runtime, device), group in _groups(rows).items():
        lines.append(f"**{model}** ({runtime}, {device}), {group[0].questions} question(s) per image")
        lines.append("")
        lines.append("| size | " + " | ".join(f"batch {b}" for b in batches) + " |")
        lines.append("| ---: | " + " | ".join("---:" for _ in batches) + " |")
        for s in sizes:
            cells = []
            for b in batches:
                r = next((x for x in group if x.size == s and x.batch == b), None)
                cells.append("" if r is None else f"{r.ms_per_image:g} ({r.images_per_s:g}/s)")
            lines.append(f"| {s} | " + " | ".join(cells) + " |")
        peak = [r.peak_mem_mb for r in group if r.peak_mem_mb is not None]
        if peak:
            lines.append("")
            lines.append(f"Peak memory: {max(peak):.0f} MB")
        lines.append("")
    return "\n".join(lines)


def _groups(rows: list[BenchRow]) -> dict[tuple[str, str, str], list[BenchRow]]:
    out: dict[tuple[str, str, str], list[BenchRow]] = {}
    for r in rows:
        out.setdefault((r.model, r.runtime, r.device), []).append(r)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="+", help="model[:runtime], e.g. d1-omni-600m:ollama")
    ap.add_argument("--sizes", type=int, nargs="+", default=[128, 256, 384, 640])
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--max-time", type=float, default=60.0)
    ap.add_argument("--device", default=None, help="native runtimes: cuda, cpu, ...")
    ap.add_argument("--out", type=Path, default=None, help="also write the rows as JSON")
    args = ap.parse_args()

    rows: list[BenchRow] = []
    for target in args.targets:
        name, _, runtime = target.partition(":")
        kwargs = {"device": args.device} if args.device and runtime in ("", "native") else {}
        with S1GHT(name, runtime=runtime or None, **kwargs) as model:
            print(f"{target}: {model.runtime!r}", file=sys.stderr, flush=True)
            rows += bench(
                model, sizes=args.sizes, batches=args.batches, runs=args.runs, warmup=args.warmup,
                max_time=args.max_time,
                progress=lambda r: print(f"  {r.size:4d} px x{r.batch:<3d} {r.ms_per_image} ms/image",
                                         file=sys.stderr, flush=True),
            )  # fmt: skip
    print(markdown(rows, hardware(args.device)))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps([r.to_dict() for r in rows], indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
