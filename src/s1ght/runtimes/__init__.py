"""Decision runtimes: native (in-process) and server (HTTP) back ends behind one `Runtime` protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.base import Runtime


def load_runtime(spec: str | RuntimeSpec, *, runtime: str | None = None, **kwargs: Any) -> Runtime:
    """Build a runtime from a catalog name, URI or resolved `RuntimeSpec`.

    Args:
        spec: E.g. "d1-3b", "ollama://clef-flash", "typesafe://jev-preview", or a `RuntimeSpec`.
        runtime: For catalog names, which runtime to use (default: the entry's first one).
        **kwargs: Passed to the runtime (e.g. `base_url`, `api_key`, `timeout` for servers).
    """
    from s1ght.hub import resolve

    if isinstance(spec, str):
        spec = resolve(spec, runtime)
    if spec.runtime in ("ollama", "ollaya", "typesafe"):
        from s1ght.runtimes.server.systemone import SystemOneClient

        if spec.base_url is not None:
            kwargs.setdefault("base_url", spec.base_url)
        return SystemOneClient(spec.runtime, spec.model, **kwargs)  # type: ignore[arg-type]
    if spec.runtime in ("vllm", "llama_server"):
        from s1ght.runtimes.server.logprob import LogprobClient

        if spec.base_url is not None:
            kwargs.setdefault("base_url", spec.base_url)
        return LogprobClient(spec.model, server=spec.runtime, **kwargs)
    if spec.runtime == "native":
        from s1ght.runtimes.native.engine import NativeRuntime

        return NativeRuntime(spec, **kwargs)
    raise NotImplementedError(f"the {spec.runtime!r} runtime is not implemented yet (a later step)")
