import json
from pathlib import Path

import pytest

from s1ght import hub
from s1ght.hub import HFRef, RuntimeSpec, UncalibratedWarning, download, load_registry, parse_uri, resolve
from s1ght.runtimes import load_runtime
from s1ght.runtimes.server.systemone import SystemOneClient

D1 = "LiquidAI/d1-3B"


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test tries to reach Hugging Face."""

    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(hub, "_fetch_config", boom)


# --- catalog ---------------------------------------------------------------------------------


def test_registry_entries_are_pinned_and_known() -> None:
    reg = load_registry()
    assert {"d1-3b", "qwen3-vl-4b", "d1-omni-600m"} <= set(reg)
    for e in reg.values():
        assert e.source.pinned, e.name
        assert e.family in hub.FAMILIES
        for opts in e.runtimes.values():
            if isinstance(opts, dict):
                for v in opts.values():
                    if isinstance(v, str) and v.startswith("hf://"):
                        assert HFRef.parse(v).pinned, (e.name, v)


def test_resolve_d1_3b_defaults_to_native(no_network) -> None:
    spec = resolve("d1-3b")
    assert spec.runtime == "native" and spec.model == D1
    assert spec.source == HFRef(D1, "051bcc464b01b9f92942b364d9586b0ef5912432")
    assert spec.family == "lfm2_vl"
    assert spec.options["dtype"] == {"cuda": "bf16", "t4": "fp16", "cpu": "fp32"}
    assert spec.calibrated is False
    assert resolve("D1-3B").model == D1


def test_resolve_qwen3_vl_4b(no_network) -> None:
    spec = resolve("qwen3-vl-4b")
    assert (spec.runtime, spec.model, spec.family) == ("native", "Qwen/Qwen3-VL-4B-Instruct", "qwen_vl")
    vllm = resolve("qwen3-vl-4b", runtime="vllm")
    assert (vllm.runtime, vllm.model, vllm.family) == ("vllm", "Qwen/Qwen3-VL-4B-Instruct", "qwen_vl")


def test_resolve_catalog_server_and_cpu_runtimes(no_network) -> None:
    assert resolve("d1-3b", runtime="ollaya") == RuntimeSpec(
        "ollaya", "d1:3b-vision", entry=load_registry()["d1-3b"], calibrated=False
    )
    assert resolve("d1-omni-600m", runtime="ollama").model == "hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0"
    cpu = resolve("d1-3b", runtime="llama_cpp")
    assert cpu.source.path == "d1-3B-Q8_0.gguf" and cpu.source.repo == "LiquidAI/d1-3B-GGUF"
    assert HFRef.parse(cpu.options["mmproj"]).path == "mmproj-d1-3B-Q8_0.gguf"
    with pytest.raises(ValueError, match="no 'ollama' runtime"):
        resolve("d1-3b", runtime="ollama")


def test_unknown_name_suggests(no_network) -> None:
    with pytest.raises(KeyError, match="d1-3b"):
        resolve("d1-3")
    with pytest.raises(ValueError, match="only applies to catalog names"):
        resolve("ollama://x", runtime="ollaya")


def test_registry_validation(tmp_path: Path) -> None:
    bad = tmp_path / "r.yaml"
    bad.write_text("m:\n  source: hf://a/b\n  family: magic\n  runtimes: {native: {}}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown family"):
        load_registry(bad)
    bad.write_text("m:\n  source: hf://a/b\n  family: qwen_vl\n  runtimes: {tgi: x}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="runtimes"):
        load_registry(bad)


# --- URIs ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "ref"),
    [
        ("hf://org/repo", HFRef("org/repo")),
        ("hf://org/repo@main", HFRef("org/repo", "main")),
        ("hf://org/repo@" + "a" * 40 + "/sub/w.gguf", HFRef("org/repo", "a" * 40, "sub/w.gguf")),
        ("org/repo", HFRef("org/repo")),
    ],
)
def test_hfref_parse_roundtrip(uri: str, ref: HFRef) -> None:
    assert HFRef.parse(uri) == ref
    assert HFRef.parse(str(ref)) == ref
    assert ref.pinned == (ref.revision == "a" * 40)


@pytest.mark.parametrize(
    ("uri", "runtime", "model", "base_url"),
    [
        ("ollama://clef-flash", "ollama", "clef-flash", None),
        ("ollama://hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0", "ollama", "hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0", None),
        ("ollaya://d1:3b-vision", "ollaya", "d1:3b-vision", None),
        ("typesafe://jev-preview", "typesafe", "jev-preview", None),
        (
            "vllm://localhost:8000/Qwen/Qwen3-VL-4B-Instruct",
            "vllm",
            "Qwen/Qwen3-VL-4B-Instruct",
            "http://localhost:8000",
        ),
    ],
)
def test_parse_server_uris(uri: str, runtime: str, model: str, base_url: str | None) -> None:
    spec = parse_uri(uri)
    assert (spec.runtime, spec.model, spec.base_url, spec.source) == (runtime, model, base_url, None)
    assert resolve(uri) == spec


@pytest.mark.parametrize("uri", ["ftp://x/y", "vllm://localhost:8000", "ollama://", "hf://onlyorg", "clef-flash"])
def test_bad_uris(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_uri(uri)


# --- models outside the catalog --------------------------------------------------------------


def test_unknown_hf_repo_guesses_family(monkeypatch) -> None:
    monkeypatch.setattr(hub, "_fetch_config", lambda ref: {"model_type": "qwen3_vl"})
    with pytest.warns(UncalibratedWarning):
        spec = resolve("someone/qwen3-vl-finetune")
    assert (spec.runtime, spec.family, spec.calibrated) == ("native", "qwen_vl", False)
    with pytest.warns(UncalibratedWarning):
        assert resolve("hf://someone/qwen3-vl-finetune@abc").source == HFRef("someone/qwen3-vl-finetune", "abc")


def test_unrecognized_architecture_falls_back(monkeypatch) -> None:
    monkeypatch.setattr(hub, "_fetch_config", lambda ref: {"model_type": "llava", "architectures": ["LlavaForX"]})
    with pytest.warns(UncalibratedWarning, match="generic option-letter readout"):
        assert resolve("someone/llava").family == "semif_generic"


def test_local_directory(tmp_path: Path, no_network) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Lfm2VlForConditionalGeneration"]}))
    with pytest.warns(UncalibratedWarning):
        spec = resolve(str(tmp_path))
    assert (spec.runtime, spec.family, spec.source) == ("native", "lfm2_vl", None)


# --- download --------------------------------------------------------------------------------


def test_download_goes_to_s1ght_cache(tmp_path: Path, monkeypatch) -> None:
    import huggingface_hub

    calls = []
    monkeypatch.setenv("S1GHT_CACHE", str(tmp_path))
    monkeypatch.setattr(
        huggingface_hub, "snapshot_download", lambda repo, **kw: calls.append((repo, kw)) or str(tmp_path / "snap")
    )
    monkeypatch.setattr(
        huggingface_hub, "hf_hub_download", lambda repo, f, **kw: calls.append((repo, f, kw)) or str(tmp_path / f)
    )

    spec = resolve("d1-3b")
    assert download(spec.source) == tmp_path / "snap"
    repo, kw = calls[0]
    assert repo == D1 and kw["revision"] == spec.source.revision and kw["cache_dir"] == tmp_path / "hub"

    gguf = resolve("d1-3b", runtime="llama_cpp").source
    assert download(gguf, cache=tmp_path / "other") == tmp_path / "d1-3B-Q8_0.gguf"
    assert calls[1][:2] == ("LiquidAI/d1-3B-GGUF", "d1-3B-Q8_0.gguf")
    assert calls[1][2]["cache_dir"] == tmp_path / "other" / "hub"

    with pytest.warns(UserWarning, match="not pinned"):
        download("hf://org/repo")


@pytest.mark.network
def test_download_real_file(tmp_path: Path) -> None:
    """Fetch one small pinned file from Hugging Face into a fresh cache."""
    ref = resolve("d1-omni-600m").source
    path = download(HFRef(ref.repo, ref.revision, "config.json"), cache=tmp_path)
    assert path.is_file() and tmp_path in path.parents
    assert json.loads(path.read_text())["model_type"] == "d1_omni"


# --- load_runtime ----------------------------------------------------------------------------


def test_load_runtime_server(no_network) -> None:
    rt = load_runtime("d1-3b", runtime="ollaya", base_url="http://127.0.0.1:9")
    assert isinstance(rt, SystemOneClient) and rt.model == "d1:3b-vision" and rt.base_url == "http://127.0.0.1:9"
    rt.close()
    rt = load_runtime("typesafe://jev-preview", api_key="k")
    assert rt.provider == "typesafe" and rt.model == "jev-preview"
    rt.close()


def test_load_runtime_logprob_servers(no_network) -> None:
    from s1ght.runtimes.server.logprob import LogprobClient

    rt = load_runtime("vllm://localhost:8000/Qwen/Qwen3-VL-4B-Instruct")
    assert isinstance(rt, LogprobClient) and rt.server == "vllm" and rt.base_url == "http://localhost:8000"
    assert rt.model == "Qwen/Qwen3-VL-4B-Instruct"
    rt.close()
    rt = load_runtime("llama-server://127.0.0.1:8081")
    assert rt.server == "llama_server" and rt.base_url == "http://127.0.0.1:8081"  # type: ignore[attr-defined]
    rt.close()
    rt = load_runtime("qwen3-vl-4b", runtime="llama_server")
    assert rt.base_url == "http://127.0.0.1:8080" and rt.model == "Qwen3-VL-4B-Instruct"  # type: ignore[attr-defined]
    rt.close()
    with pytest.raises(ValueError, match="llama-server://host:port"):
        resolve("llama-server:///model")
    with pytest.raises(NotImplementedError, match="llama_cpp"):
        load_runtime("d1-3b", runtime="llama_cpp")
