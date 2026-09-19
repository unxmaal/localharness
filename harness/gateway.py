"""Where the gateway config is, asked once.

Five modules resolved this independently -- coverage, discover, rank, and
screen twice -- with the same expression copied five times:

    Path(os.environ.get("GATEWAY_CONFIG", <repo> / "gateway" / "config.yaml"))

That is the "one question, two answers" class, and the fifth copy was added
while fixing a different instance of it. The copies agree today; nothing made
them, and a sixth would agree only by the author remembering.
"""
from __future__ import annotations

import os
from pathlib import Path

ENV_VAR = "GATEWAY_CONFIG"

#: The repo root, two parents up from this file.
REPO = Path(__file__).resolve().parent.parent

DEFAULT = REPO / "gateway" / "config.yaml"

#: LiteLLM prefixes a provider onto the upstream model id. Stripping is how an
#: alias resolves to the repo id mlx_lm.server can hot-swap to, so an
#: UNRECOGNISED prefix silently yields a model name nothing can serve. The
#: tracked configs are asserted against this list, so adding a provider to one
#: without adding it here fails rather than producing a wrong id.
PROVIDER_PREFIXES = ("openai/", "hosted_vllm/", "ollama/")


def config_path(config=None) -> Path:
    """The gateway config this machine uses."""
    return Path(config or os.environ.get(ENV_VAR) or DEFAULT)


def load(config=None) -> dict:
    """The config's parsed contents, or {} when it cannot be read.

    Fails OPEN by design: a missing or malformed config means "this machine
    serves nothing through a gateway", which is true of a fresh clone, and
    raising here would take down every tier that asks what is served.
    """
    try:
        import yaml
    except ImportError:      # pragma: no cover - yaml is a hard dependency
        return {}
    try:
        return yaml.safe_load(config_path(config).read_text(
            encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}


def strip_provider(model: str) -> str:
    """The upstream id without LiteLLM's provider prefix."""
    got = (model or "").strip()
    for prefix in PROVIDER_PREFIXES:
        if got.startswith(prefix):
            return got[len(prefix):]
    return got
