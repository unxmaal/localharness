"""A model's real id behind a gateway name, so nothing is reported only by an alias. #670.

A gateway entry is named by the upstream id it sends, except for three kinds of
alias: a lane alias (sohot-<lane>, repointed by adoption), a stable reference
alias (cloud-opus), and an old nickname kept one release with `deprecated_for`.
"""
from __future__ import annotations

import re

from harness import gateway

#: The key a deprecated nickname's config entry carries: the name that replaces it.
DEPRECATED_KEY = "deprecated_for"
#: The key an entry carries when it is this machine's build of another config's model.
IN_PLACE_KEY = "in_place_of"
#: Aliases that stay as stable endpoints; reports still show what they resolve to.
STABLE_ALIASES = ("cloud-opus",)
LANE_ALIAS_RE = re.compile("^" + re.escape(gateway.LANE_ALIAS).replace(r"\{\}", r"[a-z0-9_-]+") + "$")
UNKNOWN = "model unknown"
#: evals.run workflows over a text model that harness.methods does not parse.
WORKFLOW_PREFIXES = ("repair:",)


def _entries(config=None) -> list[dict]:
    """The served config's entries when serve-gateway.sh wrote one, else the base config's."""
    served = gateway.served_path(config)
    data = gateway.load(served) if served.exists() else gateway.load(config)
    return [e for e in data.get("model_list") or [] if isinstance(e, dict)]


def _name(spec: str) -> str:
    from harness import methods
    base = methods.base_of(spec or "") or spec or ""
    if base.startswith(WORKFLOW_PREFIXES):
        base = base.partition(":")[2]
    return base.partition(",")[0].strip()


def _entry(name: str, config=None) -> dict | None:
    want = name.lower()
    for e in _entries(config):
        if str(e.get("model_name", "")).strip().lower() == want:
            return e
    return None


def is_lane_alias(name: str) -> bool:
    return bool(LANE_ALIAS_RE.match(_name(name)))


def resolve(spec: str, config=None) -> str:
    """The upstream id a gateway name is sent as; the name itself when the gateway does not
    know it; "" for a lane alias nothing serves."""
    name = _name(spec)
    e = _entry(name, config)
    if e is not None:
        return gateway.strip_provider(str((e.get("litellm_params") or {}).get("model", "")))
    return "" if is_lane_alias(name) else name


def on_gateway(spec: str, config=None) -> bool:
    """True when the gateway config names `spec`, so the gateway, not a download, serves it."""
    return _entry(_name(spec), config) is not None


def is_alias(spec: str, config=None) -> bool:
    """True for a gateway name that is not the id it sends upstream."""
    name = _name(spec)
    if is_lane_alias(name):
        return True
    return _entry(name, config) is not None and resolve(name, config) != name


def deprecated(config=None) -> dict[str, str]:
    """Old nickname -> the real id it was renamed to."""
    return {str(e["model_name"]): resolve(str(e[DEPRECATED_KEY]), config)
            for e in _entries(config) if e.get(DEPRECATED_KEY)}


def old_names(spec: str, config=None) -> list[str]:
    """The deprecated nicknames that were renamed to `spec`."""
    want = resolve(spec, config)
    return [old for old, new in deprecated(config).items() if want and new == want]


def build_here(spec: str, config=None) -> str:
    """`spec`, or this machine's build of it when the config serves it under another id."""
    name = _name(spec)
    if not name or _entry(name, config) is not None:
        return spec
    for e in _entries(config):
        if str(e.get(IN_PLACE_KEY, "")).strip().lower() == name.lower():
            return spec.replace(name, str(e["model_name"]), 1)
    return spec


def deprecation(spec: str, config=None) -> str:
    """A warning when `spec` names a deprecated nickname, else ""."""
    name = _name(spec)
    new = deprecated(config).get(name)
    if not new:
        return ""
    return (f"{name} is a deprecated alias for {new} and goes in the next release; "
            f"use {new}")


def warn(spec: str, config=None) -> None:
    """Print deprecation() to stderr when there is one."""
    import sys
    why = deprecation(spec, config)
    if why:
        print(f"warning: {why}", file=sys.stderr)


def label(key: str, resolved: dict | None = None, config=None) -> dict:
    """{"model": real id or "" when unknown, "served_as": the alias or ""} for a result key.

    `resolved` is the receipt's record of what each key resolved to at run time. Without
    it a lane alias is unknown, since adoption repoints it; a fixed alias resolves now.
    """
    resolved = resolved or {}
    name = _name(key)
    alias = name if is_alias(name, config) else ""
    if key in resolved or name in resolved:
        model = resolved.get(key) or resolved.get(name) or ""
    elif is_lane_alias(name):
        model = ""
    else:
        model = resolve(name, config)
    return {"model": model, "served_as": alias if alias != model else ""}


def display(key: str, resolved: dict | None = None, config=None) -> str:
    """The model a result row names, with the alias it was served as beside it."""
    got = label(key, resolved, config)
    if not got["model"]:
        return f"{got['served_as'] or _name(key)} ({UNKNOWN})"
    if got["served_as"]:
        return f"{got['model']} (served as {got['served_as']})"
    return got["model"]


#: What a fact reads when the store holds nothing for it.
UNKNOWN_FACT = "unknown"
_PARAMS = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)([BbMm])(?![a-zA-Z])")
_ACTIVE = re.compile(r"(?<![\w.])A(\d+(?:\.\d+)?)B(?![a-zA-Z])")
_QUANT = re.compile(r"(?i)(?<![a-z0-9])(IQ\d_[A-Z]+|Q\d_K_[SML]|Q\d_K|Q\d_\d|\d+bit|bf16|fp16|f16|fp8|mxfp4)(?![a-z0-9])")
_ENGINE_PREFIXES = ("llamacpp:", "vllm:", "ds4:")
_REPO_ID = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _source(conn, model: str, config=None) -> tuple[str, str]:
    """(repo, weights file) behind a model id, from the config and the downloads table."""
    name = model
    for prefix in _ENGINE_PREFIXES:
        if name.startswith(prefix):
            name = name[len(prefix):]
    if not name or ":" in name:
        return "", ""
    e = _entry(name, config)
    if e is not None and e.get("source_repo"):
        return str(e["source_repo"]), str(e.get("source_file") or "")
    if "/" in name:
        return (name, "") if _REPO_ID.match(name) else ("", "")
    rows = conn.execute("SELECT DISTINCT repo, file FROM downloads WHERE repo != '' AND "
                        "(lower(file) = ? OR lower(file) = ?)",
                        (name.lower(), f"{name.lower()}.gguf")).fetchall()
    # Two repos shipping one file name is no answer; the row says unknown rather than pick one.
    repos = {r["repo"] for r in rows}
    return (rows[0]["repo"], rows[0]["file"]) if len(repos) == 1 else ("", name)


def about(conn, model: str, config=None) -> dict:
    """What a model is, from the store's proposals, lineage and downloads, and the config:
    base family, parameter count, quantisation, publisher and a source link. #670."""
    import json
    repo, file = _source(conn, model or "", config)
    prop = (conn.execute("SELECT id, registry, card_tags, model_type FROM proposals "
                         "WHERE lower(name) = lower(?)", (repo,)).fetchone() if repo else None)
    parent = ""
    if prop:
        try:
            tags = json.loads(prop["card_tags"] or "[]")
        except ValueError:
            tags = []
        parent = next((t.split(":")[-1] for t in tags if str(t).startswith("base_model:")), "")
        if not parent:
            got = conn.execute("SELECT parent FROM lineage WHERE proposal_id = ? ORDER BY id "
                               "LIMIT 1", (prop["id"],)).fetchone()
            parent = got["parent"] if got else ""
    family = parent or (prop["model_type"] if prop else "")
    known = repo or file
    text = " ".join(x for x in (repo, file, parent) if x)
    size = _PARAMS.search(text) if known else None
    active = _ACTIVE.search(text) if known else None
    params = (f"{size.group(1)}{size.group(2).upper()}"
              + (f" ({active.group(1)}B active)" if active else "")) if size else ""
    quant = _QUANT.search(" ".join((file, repo))) if known else None
    registry = prop["registry"] if prop else ""
    link = ""
    if _REPO_ID.match(repo):
        link = (f"https://github.com/{repo}" if registry == "github"
                else f"https://huggingface.co/{repo}")
    return {"family": family or UNKNOWN_FACT, "params": params or UNKNOWN_FACT,
            "quant": quant.group(1) if quant else UNKNOWN_FACT,
            "publisher": repo.split("/")[0] if _REPO_ID.match(repo) else UNKNOWN_FACT,
            "source": link}


def about_text(facts: dict | None) -> str:
    """One line for a row: "family X, 7B params, Q4_K_M, by org"; "unknown" when nothing is known."""
    facts = facts or {}
    keys = ("family", "params", "quant", "publisher")
    if all(facts.get(k, UNKNOWN_FACT) == UNKNOWN_FACT for k in keys) and not facts.get("source"):
        return UNKNOWN_FACT
    return (f"family {facts.get('family', UNKNOWN_FACT)}, {facts.get('params', UNKNOWN_FACT)} params, "
            f"{facts.get('quant', UNKNOWN_FACT)}, by {facts.get('publisher', UNKNOWN_FACT)}")
