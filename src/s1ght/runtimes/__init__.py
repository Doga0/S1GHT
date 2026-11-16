"""Decision runtimes: server (HTTP) and in-process back ends behind one `Runtime` protocol."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from s1ght.hub import RuntimeSpec
    from s1ght.runtimes.base import Runtime

LETTERS = ("qwen_vl", "semif_generic")
"""Families read through option letters: chat VLMs with no decision head of their own."""


def load_runtime(spec: str | RuntimeSpec, *, runtime: str | None = None, **kwargs: Any) -> Runtime:
    """Build a runtime from a catalog name, URI or resolved `RuntimeSpec`.

    Args:
        spec: E.g. "d1-3b", "ollama://clef-flash", "typesafe://jev-preview", or a `RuntimeSpec`.
        runtime: For catalog names, which runtime to use (default: the entry's first one).
        **kwargs: Passed to the runtime (e.g. `base_url`, `api_key`, `timeout` for servers).
            `readout="letters"` reads a model on llama-server through option-letter logprobs
            instead of the server's own `/v1/systemone` (the default for models outside the
            letter families).
    """
    from s1ght.hub import resolve

    if isinstance(spec, str):
        spec = resolve(spec, runtime)
    readout = kwargs.pop("readout", None)
    if readout not in (None, "letters", "decisions"):
        raise ValueError(f"readout must be 'letters' or 'decisions', got {readout!r}")
    decisions = spec.runtime in ("ollama", "ollaya", "typesafe", "systemone") or (
        # llama-server reads decision models itself on /v1/systemone; chat VLMs through letters
        spec.runtime == "llama_server" and readout != "letters" and (readout or spec.family not in LETTERS)
    )
    if decisions:
        from s1ght.runtimes.server.systemone import SystemOneClient

        if spec.base_url is not None:
            kwargs.setdefault("base_url", spec.base_url)
        for key in ("box", "boxes"):
            if key in spec.options:
                kwargs.setdefault(key, bool(spec.options[key]))
        return SystemOneClient(spec.runtime, spec.model, **kwargs)  # type: ignore[arg-type]
    if spec.runtime in ("vllm", "llama_server"):
        from s1ght.runtimes.server.logprob import LogprobClient

        if spec.base_url is not None:
            kwargs.setdefault("base_url", spec.base_url)
        return LogprobClient(spec.model, server=spec.runtime, **kwargs)
    raise NotImplementedError(f"the {spec.runtime!r} runtime is not implemented yet (a later step)")
