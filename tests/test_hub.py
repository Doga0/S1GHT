import json
from pathlib import Path

import pytest

from s1ght import hub
from s1ght.hub import HFRef, RuntimeSpec, download, load_registry, parse_uri, resolve
from s1ght.runtimes import load_runtime
from s1ght.runtimes.server.systemone import SystemOneClient

D1 = "LiquidAI/d1-3B"


@pytest.fixture
def no_network(monkeypatch):
    """Fail loudly if a test tries to reach Hugging Face."""

    import huggingface_hub

    def boom(*a, **k):
        raise AssertionError("network access in an offline test")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", boom)
    monkeypatch.setattr(huggingface_hub, "snapshot_download", boom)


# --- catalog ---------------------------------------------------------------------------------


def test_registry_entries_are_pinned_and_known() -> None:
    reg = load_registry()
    assert {"d1-3b", "d1-omni-600m", "d1-2b", "h2o-lightning-4b"} <= set(reg)
    assert "qwen3-vl-4b" not in reg  # a general VLM, not a decision model
    for e in reg.values():
        assert e.source.pinned, e.name
        assert e.family in hub.FAMILIES
        for opts in e.runtimes.values():
            if isinstance(opts, dict):
                for v in opts.values():
                    if isinstance(v, str) and v.startswith("hf://"):
                        assert HFRef.parse(v).pinned, (e.name, v)


def test_resolve_d1_3b_defaults_to_first_runtime(no_network) -> None:
    spec = resolve("d1-3b")
    assert (spec.runtime, spec.model, spec.calibrated) == ("ollaya", "d1:3b-vision", False)
    assert resolve("D1-3B") == spec
    assert load_registry()["d1-3b"].source == HFRef(D1, "051bcc464b01b9f92942b364d9586b0ef5912432")


def test_resolve_catalog_server_and_cpu_runtimes(no_network) -> None:
    assert resolve("d1-3b", runtime="ollaya") == RuntimeSpec(
        "ollaya", "d1:3b-vision", entry=load_registry()["d1-3b"], calibrated=False
    )
    assert resolve("d1-omni-600m", runtime="ollama").model == "hf.co/LiquidAI/d1-omni-600M-GGUF:Q8_0"
    cpu = resolve("d1-3b", runtime="llama_cpp")
    assert cpu.family == "lfm2_vl" and cpu.source.path == "d1-3B-Q8_0.gguf" and cpu.source.repo == "LiquidAI/d1-3B-GGUF"
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
    bad.write_text("m:\n  source: hf://a/b\n  family: magic\n  runtimes: {ollama: x}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown family"):
        load_registry(bad)
    for runtimes in ("{tgi: x}", "{native: {}}"):
        bad.write_text(f"m:\n  source: hf://a/b\n  family: qwen_vl\n  runtimes: {runtimes}\n", encoding="utf-8")
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
            "systemone://127.0.0.1:8741/h2oai/h2o-lightning-4b",
            "systemone",
            "h2oai/h2o-lightning-4b",
            "http://127.0.0.1:8741",
        ),
        ("systemone://gpu-box:8741", "systemone", "default", "http://gpu-box:8741"),
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


@pytest.mark.parametrize("name", ["hf://someone/qwen3-vl-finetune@abc", "someone/llava"])
def test_hf_repos_need_a_runtime(name: str, no_network) -> None:
    with pytest.raises(ValueError, match="transformers runtime was removed"):
        resolve(name)


def test_local_directory_needs_a_runtime(tmp_path: Path, no_network) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"architectures": ["Lfm2VlForConditionalGeneration"]}))
    with pytest.raises(ValueError, match="transformers runtime was removed"):
        resolve(str(tmp_path))


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

    source = load_registry()["d1-3b"].source
    assert download(source) == tmp_path / "snap"
    repo, kw = calls[0]
    assert repo == D1 and kw["revision"] == source.revision and kw["cache_dir"] == tmp_path / "hub"

    gguf = resolve("d1-3b", runtime="llama_cpp").source
    assert download(gguf, cache=tmp_path / "other") == tmp_path / "d1-3B-Q8_0.gguf"
    assert calls[1][:2] == ("LiquidAI/d1-3B-GGUF", "d1-3B-Q8_0.gguf")
    assert calls[1][2]["cache_dir"] == tmp_path / "other" / "hub"

    with pytest.warns(UserWarning, match="not pinned"):
        download("hf://org/repo")


@pytest.mark.network
def test_download_real_file(tmp_path: Path) -> None:
    """Fetch one small pinned file from Hugging Face into a fresh cache."""
    ref = load_registry()["d1-omni-600m"].source
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


def test_box_models_in_the_catalog(no_network) -> None:
    h2o = load_runtime("h2o-lightning-4b")  # the model's shim: Box answers
    assert isinstance(h2o, SystemOneClient) and h2o.provider == "systemone" and h2o.caps.box and not h2o.caps.boxes
    assert h2o.base_url == "http://127.0.0.1:8741" and h2o.model == "h2oai/h2o-lightning-4b"
    h2o.close()
    plain = load_runtime("h2o-lightning-4b", runtime="llama_server")  # llama-server has no box question
    assert isinstance(plain, SystemOneClient) and plain.provider == "llama_server" and not plain.caps.box
    plain.close()
    d1 = load_runtime("d1-2b")
    assert isinstance(d1, SystemOneClient) and d1.provider == "llama_server" and d1.model == "d1-2B"
    d1.close()
    custom = load_runtime("systemone://127.0.0.1:9000/m", box=True, boxes=True)
    assert custom.caps.box and custom.caps.boxes
    custom.close()


def test_load_runtime_logprob_servers(no_network) -> None:
    from s1ght.runtimes.server.logprob import LogprobClient

    rt = load_runtime("vllm://localhost:8000/Qwen/Qwen3-VL-4B-Instruct")
    assert isinstance(rt, LogprobClient) and rt.server == "vllm" and rt.base_url == "http://localhost:8000"
    assert rt.model == "Qwen/Qwen3-VL-4B-Instruct"
    rt.close()
    rt = load_runtime("llama-server://127.0.0.1:8081", readout="letters")
    assert rt.server == "llama_server" and rt.base_url == "http://127.0.0.1:8081"  # type: ignore[attr-defined]
    rt.close()
    rt = load_runtime("llama-server://127.0.0.1:8081")  # a decision model: the server's own /v1/systemone
    assert isinstance(rt, SystemOneClient) and rt.provider == "llama_server" and not rt.caps.box
    rt.close()
    with pytest.raises(ValueError, match="readout"):
        load_runtime("llama-server://127.0.0.1:8081", readout="tokens")
    with pytest.raises(ValueError, match="llama-server://host:port"):
        resolve("llama-server:///model")
    with pytest.raises(NotImplementedError, match="llama_cpp"):
        load_runtime("d1-3b", runtime="llama_cpp")
