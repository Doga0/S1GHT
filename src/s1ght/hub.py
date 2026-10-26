"""Model names, URIs and downloads.

`resolve(name)` turns what the user passed to `S1GHT(...)` into a `RuntimeSpec`, in this order:

1. A short name from the packaged catalog (`registry.yaml`), e.g. "d1-3b".
2. A URI: `hf://org/repo[@rev][/file]`, `ollama://<tag>`, `ollaya://<tag>`, `typesafe://<model>`,
   `vllm://host:port/<model>`, `llama-server://host:port[/<model>]`.
3. A Hugging Face repo id not in the catalog ("org/repo") or a local model directory: the
   family is guessed from `config.json`; unknown architectures fall back to `semif_generic`
   (generic option-letter readout) with an "uncalibrated" warning.

Native weights are downloaded with `huggingface_hub` into `$S1GHT_CACHE` (default
`~/.cache/s1ght`). Server runtimes download their own models (`ollama pull`, ...).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import warnings
from dataclasses import dataclass, field, replace
from importlib import resources
from pathlib import Path
from typing import Any

import yaml

SERVER_RUNTIMES = ("ollama", "ollaya", "typesafe", "vllm", "llama_server")
RUNTIMES = ("native", "llama_cpp", *SERVER_RUNTIMES)
FAMILIES = ("lfm2_vl", "d1_omni", "qwen_vl", "semif_generic")

# config.json `model_type` → family adapter
_MODEL_TYPE_FAMILY = {
    "lfm2_vl": "lfm2_vl",
    "d1_omni": "d1_omni",
    "qwen2_vl": "qwen_vl",
    "qwen2_5_vl": "qwen_vl",
    "qwen3_vl": "qwen_vl",
}
_COMMIT = re.compile(r"^[0-9a-f]{40}$")


class UncalibratedWarning(UserWarning):
    """A model is used without a calibration profile; its probabilities are the raw readout."""


def cache_dir() -> Path:
    """Root of S1GHT's download cache: `$S1GHT_CACHE`, else `~/.cache/s1ght`."""
    return Path(os.environ.get("S1GHT_CACHE") or Path.home() / ".cache" / "s1ght")


@dataclass(frozen=True)
class HFRef:
    """A Hugging Face repo, optionally pinned to a revision and narrowed to one file."""

    repo: str
    revision: str | None = None
    path: str | None = None

    @property
    def pinned(self) -> bool:
        """True when `revision` is a full commit hash (immutable content)."""
        return bool(self.revision and _COMMIT.match(self.revision))

    @classmethod
    def parse(cls, uri: str) -> HFRef:
        """Parse `hf://org/repo[@revision][/path/to/file]` (the `hf://` prefix is optional)."""
        rest = uri.removeprefix("hf://")
        parts = rest.split("/")
        if len(parts) < 2 or not parts[0] or not parts[1]:
            raise ValueError(f"expected hf://org/repo[@rev][/file], got {uri!r}")
        name, _, revision = parts[1].partition("@")
        path = "/".join(parts[2:]) or None
        return cls(repo=f"{parts[0]}/{name}", revision=revision or None, path=path)

    def __str__(self) -> str:
        rev = f"@{self.revision}" if self.revision else ""
        path = f"/{self.path}" if self.path else ""
        return f"hf://{self.repo}{rev}{path}"


@dataclass(frozen=True)
class ModelEntry:
    """One catalog entry from `registry.yaml`."""

    name: str
    source: HFRef
    family: str
    runtimes: dict[str, Any]
    calibration: str | None = None
    license: str = ""
    notes: str = ""

    @property
    def default_runtime(self) -> str:
        return next(iter(self.runtimes))


@dataclass(frozen=True)
class RuntimeSpec:
    """Everything needed to build a runtime for one model.

    Attributes:
        runtime: "native", "llama_cpp", "ollama", "ollaya", "typesafe", "vllm" or "llama_server".
        model: Model name for that runtime: HF repo id (native), server tag or served model id.
        source: Weights to download for native/llama_cpp; None for server runtimes.
        family: Native adapter family, or None when the runtime reads answers itself.
        base_url: Server URL from the URI (vllm), else None (client defaults / env vars apply).
        options: Runtime options from the catalog (e.g. native dtypes, llama_cpp files).
        entry: Catalog entry when the name came from the catalog.
        calibrated: Whether a calibration profile exists for this model.
    """

    runtime: str
    model: str
    source: HFRef | None = None
    family: str | None = None
    base_url: str | None = None
    options: dict[str, Any] = field(default_factory=dict)
    entry: ModelEntry | None = None
    calibrated: bool = False


def load_registry(path: str | Path | None = None) -> dict[str, ModelEntry]:
    """Load the model catalog (the packaged `registry.yaml` by default)."""
    if path is None:
        text = resources.files("s1ght").joinpath("registry.yaml").read_text(encoding="utf-8")
    else:
        text = Path(path).read_text(encoding="utf-8")
    entries: dict[str, ModelEntry] = {}
    for name, e in (yaml.safe_load(text) or {}).items():
        family, runtimes = e["family"], e.get("runtimes") or {}
        if family not in FAMILIES:
            raise ValueError(f"{name}: unknown family {family!r}; expected one of {FAMILIES}")
        unknown = set(runtimes) - set(RUNTIMES)
        if not runtimes or unknown:
            raise ValueError(f"{name}: runtimes must be a non-empty subset of {RUNTIMES}, got {list(runtimes)}")
        entries[name] = ModelEntry(
            name=name,
            source=HFRef.parse(e["source"]),
            family=family,
            runtimes=runtimes,
            calibration=e.get("calibration"),
            license=str(e.get("license", "")),
            notes=str(e.get("notes", "")),
        )
    return entries


def parse_uri(uri: str) -> RuntimeSpec:
    """Parse a runtime URI without consulting the catalog (see module docstring for the forms)."""
    scheme, sep, rest = uri.partition("://")
    if not sep or not rest:
        raise ValueError(f"not a model URI: {uri!r}")
    if scheme == "hf":
        ref = HFRef.parse(uri)
        return RuntimeSpec(runtime="native", model=ref.repo, source=ref)
    if scheme in ("ollama", "ollaya", "typesafe"):
        return RuntimeSpec(runtime=scheme, model=rest)
    if scheme == "vllm":
        host, _, model = rest.partition("/")
        if not host or not model:
            raise ValueError(f"expected vllm://host:port/model, got {uri!r}")
        return RuntimeSpec(runtime="vllm", model=model, base_url=f"http://{host}")
    if scheme == "llama-server":
        host, _, model = rest.partition("/")
        if not host:
            raise ValueError(f"expected llama-server://host:port[/model], got {uri!r}")
        return RuntimeSpec(runtime="llama_server", model=model or "default", base_url=f"http://{host}")
    raise ValueError(
        f"unknown scheme {scheme!r} in {uri!r}; expected hf, ollama, ollaya, typesafe, vllm or llama-server"
    )


def family_from_config(config: dict[str, Any]) -> str | None:
    """Family adapter for a transformers `config.json`, or None if the architecture is unknown."""
    family = _MODEL_TYPE_FAMILY.get(str(config.get("model_type", "")))
    if family is None:
        for arch in config.get("architectures") or []:
            if arch.startswith(("Qwen2VL", "Qwen2_5_VL", "Qwen3VL")):
                return "qwen_vl"
            if arch.startswith("Lfm2Vl"):
                return "lfm2_vl"
    return family


def _fetch_config(ref: HFRef) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(ref.repo, "config.json", revision=ref.revision, cache_dir=cache_dir() / "hub")
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _guessed(spec: RuntimeSpec, config: dict[str, Any]) -> RuntimeSpec:
    family = family_from_config(config)
    if family is None:
        family = "semif_generic"
        warnings.warn(
            f"{spec.model}: architecture {config.get('model_type') or config.get('architectures')} is not in the "
            "catalog; using the generic option-letter readout. Answers are uncalibrated.",
            UncalibratedWarning,
            stacklevel=3,
        )
    else:
        warnings.warn(
            f"{spec.model} is not in the catalog; answers are uncalibrated.", UncalibratedWarning, stacklevel=3
        )
    return RuntimeSpec(runtime=spec.runtime, model=spec.model, source=spec.source, family=family)


def resolve(name: str, runtime: str | None = None, *, registry: dict[str, ModelEntry] | None = None) -> RuntimeSpec:
    """Resolve a catalog name, URI, HF repo id or local directory into a `RuntimeSpec`.

    Args:
        name: E.g. "d1-3b", "ollama://clef-flash", "vllm://localhost:8000/Qwen/Qwen3-VL-4B-Instruct",
            "org/repo" or a local path.
        runtime: For catalog names, which of the entry's runtimes to use (default: its first one).
        registry: Catalog to use; the packaged one by default.
    """
    catalog = registry if registry is not None else load_registry()
    entry = catalog.get(name) or catalog.get(name.lower())
    if entry is not None:
        return _from_entry(entry, runtime)
    if runtime is not None:
        raise ValueError(f"runtime= only applies to catalog names; put the runtime in the URI instead of {name!r}")

    if "://" in name:
        spec = parse_uri(name)
        if spec.runtime == "native":
            return _guessed(spec, _fetch_config(spec.source))  # type: ignore[arg-type]
        return spec

    local = Path(name)
    if local.is_dir():
        config = local / "config.json"
        if not config.is_file():
            raise FileNotFoundError(f"{local} has no config.json")
        spec = RuntimeSpec(runtime="native", model=str(local.resolve()))
        return _guessed(spec, json.loads(config.read_text(encoding="utf-8")))

    if re.fullmatch(r"[\w.-]+/[\w.-]+(@[\w.-]+)?", name):
        ref = HFRef.parse(name)
        return _guessed(RuntimeSpec(runtime="native", model=ref.repo, source=ref), _fetch_config(ref))

    close = difflib.get_close_matches(name.lower(), list(catalog), n=3)
    hint = f" Did you mean {', '.join(close)}?" if close else f" Catalog: {', '.join(sorted(catalog))}."
    raise KeyError(f"unknown model {name!r}.{hint}")


def _from_entry(entry: ModelEntry, runtime: str | None) -> RuntimeSpec:
    runtime = runtime or entry.default_runtime
    if runtime not in entry.runtimes:
        raise ValueError(f"{entry.name} has no {runtime!r} runtime; available: {list(entry.runtimes)}")
    opts = entry.runtimes[runtime]
    spec = RuntimeSpec(runtime, str(opts), entry=entry, calibrated=entry.calibration is not None)
    if runtime == "native":
        return replace(spec, model=entry.source.repo, source=entry.source, family=entry.family, options=opts or {})
    if runtime == "llama_cpp":
        return replace(spec, model=entry.source.repo, source=HFRef.parse(opts["gguf"]), options=opts)
    if runtime in ("vllm", "llama_server"):
        return replace(spec, model=opts["model"], family=entry.family, base_url=opts.get("base_url"))
    return spec


def download(ref: HFRef | str, *, cache: str | Path | None = None, allow_patterns: list[str] | None = None) -> Path:
    """Download a repo snapshot (or one file when `ref.path` is set) into the S1GHT cache.

    Args:
        ref: `HFRef` or `hf://` URI. Unpinned refs download the current revision and warn.
        cache: Cache root; defaults to `cache_dir()`. Files go under `<cache>/hub`.
        allow_patterns: Glob patterns limiting a snapshot download (e.g. ["*.json"]).

    Returns:
        Local snapshot directory, or the file path when `ref.path` is set.
    """
    from huggingface_hub import hf_hub_download, snapshot_download

    ref = HFRef.parse(ref) if isinstance(ref, str) else ref
    if not ref.pinned:
        warnings.warn(f"{ref} is not pinned to a commit; its content can change", stacklevel=2)
    hub = Path(cache or cache_dir()) / "hub"
    if ref.path:
        return Path(hf_hub_download(ref.repo, ref.path, revision=ref.revision, cache_dir=hub))
    return Path(snapshot_download(ref.repo, revision=ref.revision, cache_dir=hub, allow_patterns=allow_patterns))
