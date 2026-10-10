"""Which server produced the tokens.

The gateway exists so a new inference method costs a config line plus an eval
run rather than a redesign. That claim has never been tested: mlx_lm.server is
the only engine this project has served, so nothing ever needed to say which
one a receipt came from.

It matters the moment a second one appears. `evals.core.comparable()` decides
whether two runs may share a table, and with no record of the engine two runs
across DIFFERENT servers look like the same exam -- the same failure as the
same card under two operating systems producing one accelerator string while
the instruments differed. Issue #190.

The HOST is deliberately not part of this: comparable() already argues that the
same aliases served from another machine answer the same questions. What
changes the exam is the implementation, not the address.
"""
from __future__ import annotations

import os
from typing import NamedTuple

#: Set this when serving the text lane through something else. A config line,
#: which is the whole claim under test.
ENV_VAR = "TEXT_ENGINE"

#: What scripts/serve-mlx.sh starts. Kept here rather than only in the shell so
#: the receipt and the launcher cannot disagree; a test asserts they match.
DEFAULT = "mlx_lm.server"


#: mlx_lm.server (scripts/serve-mlx.sh).
MLX_URL = "http://127.0.0.1:8081"
#: More queue in the gateway: at 32 a 4B model's footprint reached 49-57 GiB on the M5 Ultra. #542.
MLX_MAX_PARALLEL = 16


#: llama-server for structured output and GGUF candidates (scripts/serve-eval.sh).
LLAMACPP_URL = "http://127.0.0.1:8082"
LLAMACPP = "llama-server"
LLAMACPP_PREFIX = "llamacpp:"

#: vLLM on this Mac, an OpenAI-compatible server on its own port. #310.
VLLM_PREFIX = "vllm:"
VLLM_URL = "http://127.0.0.1:8086"
VLLM_PORT_VAR = "VLLM_PORT"
#: Which implementation answers `vllm:` specs; the receipt's engines map carries it.
VLLM_ENGINE_VAR = "VLLM_ENGINE"
VLLM_ENGINES = ("vllm-mlx", "vllm-metal")

#: antirez/ds4's ds4-server, one of its own GGUFs per launch. #611.
DS4_PREFIX = "ds4:"
DS4 = "ds4-server"


def vllm_url(environ=None) -> str:
    """The vLLM server, on VLLM_PORT when set."""
    environ = os.environ if environ is None else environ
    port = (environ.get(VLLM_PORT_VAR) or "").strip()
    return f"{VLLM_URL.rsplit(':', 1)[0]}:{port}" if port.isdigit() else VLLM_URL


def vllm_engine(environ=None) -> str:
    """The vLLM implementation serving `vllm:` specs here."""
    environ = os.environ if environ is None else environ
    got = (environ.get(VLLM_ENGINE_VAR) or "").strip() or VLLM_ENGINES[0]
    if got not in VLLM_ENGINES:
        raise ValueError(f"{VLLM_ENGINE_VAR}={got} is not one of {', '.join(VLLM_ENGINES)}")
    return got


def llamacpp_build(binary: str = "") -> str:
    """The installed llama-server's build number, or "" if it cannot say."""
    import re
    import subprocess
    try:
        out = subprocess.run([binary or os.environ.get("LLAMACPP_BIN")
                              or LLAMACPP, "--version"], capture_output=True,
                             text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return ""
    m = re.search(r"build (\d+)", out.stdout + out.stderr)
    return m.group(1) if m else ""


def engine_for(candidate: str, environ=None, config=None) -> str:
    """The server that answers this candidate, read from where it routes."""
    name = (candidate or "").partition(",")[0].strip()
    if name.startswith(LLAMACPP_PREFIX):
        return LLAMACPP
    if name.startswith(VLLM_PREFIX):
        return vllm_engine(environ)
    if name.startswith(DS4_PREFIX):
        return DS4
    from harness import ds4, gateway
    for entry in gateway.load(config).get("model_list") or []:
        if str(entry.get("model_name", "")).lower() == name.lower():
            base = str((entry.get("litellm_params") or {}).get("api_base", ""))
            base = base.rstrip("/").removesuffix("/v1")
            if base == LLAMACPP_URL:
                return LLAMACPP
            if base == vllm_url(environ):
                return vllm_engine(environ)
            if base == ds4.url(environ):
                return DS4
    return text_engine(environ)


def enforces_schema(spec: str, environ=None, config=None) -> bool:
    """Whether this spec is answered by llama-server, the one engine that enforces response_format. #572."""
    if engine_for(spec, environ, config) == LLAMACPP:
        return True
    try:
        where = route(spec, config=config)
    except ValueError:
        return False
    from harness import router
    return where.base.rstrip("/").removesuffix("/v1") in (LLAMACPP_URL, router.url())


def drops_schema(base: str, environ=None) -> bool:
    """Whether a request sent straight to `base` reaches mlx_lm.server, which ignores response_format."""
    return (base.rstrip("/").removesuffix("/v1") == MLX_URL
            and text_engine(environ) == DEFAULT)


def text_engine(environ=None) -> str:
    """The name of the server behind the text lane."""
    environ = os.environ if environ is None else environ
    return (environ.get(ENV_VAR) or "").strip() or DEFAULT


#: Sampling keys a spec may carry after a comma, e.g. `<model id>,temperature=0`. #90.
SAMPLING_KEYS = ("temperature", "top_p", "repetition_penalty")


class Route(NamedTuple):
    """Where a text spec is answered: the server's base URL and the model name it takes."""
    base: str
    model: str
    sampling: dict


def text_spec(spec: str) -> bool:
    """True when a text server can answer this spec: an alias, a repo id, a llamacpp: stem,
    a vllm: model or a ds4: file."""
    name = (spec or "").partition(",")[0].strip()
    return bool(name) and (":" not in name or name.startswith(
        (LLAMACPP_PREFIX, VLLM_PREFIX, DS4_PREFIX)))


def _sampling(optstr: str, spec: str) -> dict:
    out = {}
    for part in (p.strip() for p in optstr.split(",") if p.strip()):
        key, eq, value = part.partition("=")
        key = key.strip()
        if not eq or key not in SAMPLING_KEYS:
            raise ValueError(f"{spec}: unknown option(s) {key}")
        try:
            out[key] = float(value)
        except ValueError:
            raise ValueError(f"{spec}: {key} must be a number") from None
    return out


def _servable(stem: str) -> str:
    """The stem, unless its recorded context was refused (#498)."""
    from harness import context
    why = context.refusal(stem)
    if why:
        raise ValueError(f"llamacpp:{stem} is not served: {why}")
    return stem


def route(spec: str, gateway: str = "", config=None) -> Route:
    """The server that serves a text spec, shared by lane commands and evals.run. #297.

    `llamacpp:<stem>` and a GGUF-only repo id (fetched, or whole in the hub cache)
    go to llama-server's router by stem, whatever `gateway` says (#583). With an
    explicit `gateway` everything else goes there verbatim. Otherwise a gateway
    alias stays on the gateway and an mlx repo id goes to the upstream that
    hot-swaps to it (screen.routed_gateway).
    """
    name, _, optstr = (spec or "").partition(",")
    name = name.strip()
    from harness import ds4
    if name.startswith(DS4_PREFIX):
        got = ds4.parse(spec)
        return Route(ds4.url(), got.stem, got.sampling)
    sampling = _sampling(optstr, spec)
    if not text_spec(name):
        raise ValueError(f"{spec!r} is not served by a text server")
    from harness import router
    if name.startswith(LLAMACPP_PREFIX):
        return Route(router.url(), _servable(name[len(LLAMACPP_PREFIX):].strip()),
                     sampling)
    if name.startswith(VLLM_PREFIX):
        model = name[len(VLLM_PREFIX):].strip()
        if not model:
            raise ValueError(f"{spec!r} names no model; vllm:<repo id>")
        return Route(vllm_url(), model, sampling)
    from harness import screen
    from harness.completion import DEFAULT_GATEWAY
    alias = "/" not in name or name.lower() in screen.gateway_routes(config)[0]
    if not alias and ds4.recognised(name):
        # ds4's own GGUFs load nowhere else. #611.
        from harness import gguf
        stem = gguf.fetched(name)
        if stem and ds4.model_of(stem):
            return Route(ds4.url(), stem, sampling)
    stem = None if alias else gguf_stem(name)
    if stem:
        return Route(router.url(), _servable(stem), sampling)
    if gateway:
        return Route(gateway.rstrip("/"), name, sampling)
    if alias:
        return Route(DEFAULT_GATEWAY, name, sampling)
    return Route(screen.routed_gateway(name, config) or DEFAULT_GATEWAY, name, sampling)


def gguf_stem(repo: str) -> str | None:
    """The router stem a GGUF-only repo id is served under, linked or not yet. #583."""
    from harness import gguf
    try:
        return gguf.fetched(repo) or gguf.hub_stem(repo)
    except Exception:  # noqa: BLE001
        return None
