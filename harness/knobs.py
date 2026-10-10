"""Every outcome-gating constant is a hypothesis: its range, the lanes it shapes, and the limit that shows it binding (#636).

This is the registry `soh sensitivity` sweeps (#98). A knob with a `probe` is
swept offline against cached data; a knob with a `limit` is one an eval can hit,
and the limit is the predicate name a result row or a verdict records when it
does (`timeout_s>180`, `max_tokens.code>32768`, `ceiling_gb:>66`).

`inventory()` finds the module-level numeric constants whose names say they gate
something. Every one is either registered here, named in NOT_GATING with a
reason, or in the shrinking backlog the tier-1 test holds.
"""
from __future__ import annotations

import ast
import importlib
import inspect as _inspect
import re
from dataclasses import dataclass
from pathlib import Path

from harness import completion

GIB = 1024 ** 3

#: Name words that mark a constant as one that gates an outcome.
GATING_WORDS = frozenset({"MAX", "MIN", "TIMEOUT", "CAP", "LIMIT", "FLOOR", "CEILING", "BUDGET",
                          "THRESHOLD", "TOLERANCE", "TEMPERATURE", "TOKENS", "RESERVE", "ALPHA",
                          "CUTOFF", "MARGIN", "STEPS", "FRACTION", "WER", "CER", "INK", "MOTION",
                          "STDDEV", "F1", "DAYS", "SHARED", "POPULATION", "VOTES"})
_UPPER = re.compile(r"_?[A-Z][A-Z0-9_]*")
_SITE = re.compile(r"(?P<rel>[\w/]+\.py):(?P<name>\w+)(?:\((?P<param>\w+)\))?")

#: The until-predicate family a knob's limit is written in: `limit:name>N` or a machine fact `name:>N`.
LIMIT, MACHINE = "limit", "machine"

#: The lanes an eval asks for a text reply, the ones a reply budget is set for.
TEXT_LANES = tuple(completion.BUDGET)
#: The lanes whose replies go through the text runner's request and load timeouts; agent steps have their own.
REQUEST_LANES = tuple(lane for lane in TEXT_LANES if lane != "agent")


@dataclass(frozen=True)
class Knob:
    name: str
    #: "rel:NAME" or "rel:func(param)"; the first is the value swept, a second is the fallback a table falls back to.
    sites: tuple[str, ...]
    values: tuple
    lanes: tuple[str, ...] = ()
    #: The predicate name that shows this knob binding; "{lane}" is replaced per lane.
    limit: str = ""
    predicate: str = LIMIT
    #: The recorded limit is the value divided by this.
    scale: float = 1.0
    probe: str = ""
    #: Where an eval's receipt records it: "max_tokens" (its own field), "knobs" (Receipt.knobs), or "" (not the exam).
    receipt: str = ""
    #: The runner attribute a run sets for an override; "" rebinds the first site for the run instead.
    runner_attr: str = ""
    note: str = ""
    #: lane -> the range swept there, where a lane's default sits outside `values`. #654.
    lane_values: tuple = ()

    def default(self, lane: str = ""):
        got = resolve(self.sites[0])
        if isinstance(got, dict):
            if lane in got:
                return got[lane]
            return resolve(self.sites[1]) if len(self.sites) > 1 else None
        return got

    def values_for(self, lane: str = "") -> tuple:
        return dict(self.lane_values).get(lane, self.values)

    def limit_name(self, lane: str = "") -> str:
        if "{lane}" not in self.limit:
            return self.limit
        return self.limit.format(lane=lane) if lane else self.limit.split(".{lane}")[0]

    def limit_value(self, lane: str = ""):
        got = self.default(lane)
        return got / self.scale if self.scale != 1.0 else got


def _k(name, sites, values, lane_values=None, **kw) -> tuple[str, Knob]:
    per = tuple((lane, tuple(v)) for lane, v in sorted((lane_values or {}).items()))
    return name, Knob(name, tuple(sites), tuple(values), lane_values=per, **kw)


KNOBS: dict[str, Knob] = dict([
    _k("reply_budget", ["harness/completion.py:BUDGET", "harness/completion.py:MAX_TOKENS"],
       [2000, 4000, 8000, 16384, 32768, 65536], lanes=TEXT_LANES,
       lane_values={"claims": (200, 400, 800, 1600)}, limit="max_tokens.{lane}",
       receipt="max_tokens", note="the reply budget an eval asks every text reply at (#628)"),
    _k("request_timeout", ["harness/completion.py:TIMEOUT_S", "harness/completion.py:MIN_DECODE_TOK_S"],
       [60.0, 120.0, 180.0, 300.0, 600.0], lanes=REQUEST_LANES, limit="timeout_s", receipt="knobs", runner_attr="timeout",
       note="seconds one request may take; a big budget stretches it at the slowest decode"),
    _k("load_timeout", ["harness/completion.py:LOAD_TIMEOUT_S"],
       [600.0, 1200.0, 1800.0, 3600.0], lanes=REQUEST_LANES, limit="load_timeout_s", receipt="knobs", runner_attr="load_timeout",
       note="seconds the untimed first request may spend loading weights"),
    _k("memory_ceiling", ["harness/inspect.py:MEMORY_CEILING", "harness/inspect.py:MEMORY_CEILING_RAM_GB"],
       [8 * GIB, 16 * GIB, 22 * GIB, 32 * GIB, 96 * GIB], limit="ceiling_gb", predicate=MACHINE,
       scale=GIB, probe="memory_ceiling",
       note="the largest weight this machine can hold; measured on 32 GB and scaled by RAM (#96)"),
    _k("runaway_rate", ["harness/audio.py:SECONDS_PER_WORD_CEILING", "harness/audio.py:RUNAWAY_MIN_WORDS"],
       [0.6, 0.8, 0.95, 1.2, 2.0], lanes=("tts",), limit="seconds_per_word", receipt="knobs",
       note="seconds of speech per word above which tts ran away (#91)"),
    _k("download_cap", ["harness/fetching.py:MAX_DOWNLOAD"],
       [20 * GIB, 40 * GIB, 60 * GIB, 120 * GIB], limit="download_gib", scale=GIB,
       note="the largest download one fetch may take"),
    _k("min_shared", ["harness/neighbors.py:MIN_SHARED"], [1, 2, 3, 5, 8], probe="min_shared",
       note="how many of the crowd must star a repo for it to rank"),
    _k("population", ["harness/neighbors.py:POPULATION"], [1e5, 1e6, 5e6, 2e7, 1e8], probe="population",
       note="the world's star count, the denominator of enrichment"),
    _k("half_life", ["harness/neighbors.py:HALF_LIFE_DAYS"], [90.0, 365.0, 1095.0, 1e9], probe="half_life",
       note="push-date decay; 1e9 is the decay switched off"),
    _k("min_degree", ["harness/neighbors.py:cohort(min_degree)"], [1, 2, 3, 5], probe="min_degree",
       note="how many of the crowd must follow someone for a second hop to reach them"),
    _k("per_repo", ["harness/neighbors.py:cohort(per_repo)"], [4, 8, 12, 24], probe="per_repo",
       note="contributors taken from each seed repo"),
    _k("hops", ["harness/neighbors.py:cohort(hops)"], [1, 2, 3], probe="hops",
       note="how far from the seeds the crowd reaches"),
    _k("crowd_limit", ["harness/neighbors.py:cohort(limit)"], [100, 250, 450], probe="crowd_limit",
       note="crowd size; read n= rather than the value, offline only cached people answer"),
    _k("size_limit", ["harness/inspect.py:SIZE_LIMIT"], [2, 4, 12, 24, 64], probe="size_limit",
       note="how many named weights get sized per repo"),
    _k("dead_days", ["harness/inspect.py:UPSTREAM_DEAD_DAYS"], [180, 365, 730, 1095, 3650], probe="dead_days",
       note="when a repo counts as abandoned"),
    _k("clone_kb_cap", ["harness/inspect.py:CLONE_KB_CAP"], [10_000, 50_000, 250_000, 1_000_000],
       probe="clone_kb_cap", note="source tree size above which a repo is weights in git"),
    _k("binding_window", ["harness/binding.py:BINDING_WINDOW_DAYS"], [7, 14, 30, 90],
       note="days of runs and verdicts a binding count reads"),
    _k("binding_fraction", ["harness/binding.py:BINDING_FRACTION"], [0.02, 0.05, 0.1, 0.25],
       note="share of a lane's rows hitting a limit at which its knob is reported binding"),
    _k("claims_match_threshold", ["harness/checks/claims.py:MATCH_THRESHOLD"], [0.3, 0.4, 0.5, 0.6, 0.7],
       lanes=("claims",), limit="claims_match_threshold", receipt="knobs", note="word overlap at which an emitted claim is a reviewed one (#654)"),
    _k("claims_min_recall", ["harness/checks/claims.py:MIN_RECALL"], [0.25, 0.5, 0.75, 1.0],
       lanes=("claims",), limit="claims_min_recall", receipt="knobs", note="share of a case's good claims a reply must recover (#654)"),
    _k("eval7b_slots", ["harness/context.py:EVAL_7B_SLOTS"], [1, 4, 8, 16, 32],
       note="llama-server slots Qwen2.5-7B-Instruct-Q4_K_M batches across; sets claims throughput, not a grade (#665)"),
    _k("eval7b_slot_ctx", ["harness/context.py:EVAL_7B_SLOT_CTX"], [3072, 4096, 8192],
       note="context each Qwen2.5-7B-Instruct-Q4_K_M slot is sized for; never under context.MIN_SLOT_CTX (#665)"),
    _k("binding_min_hits", ["harness/binding.py:BINDING_MIN_HITS"], [1, 3, 5, 10],
       note="fewer hits than this is never binding"),
])

#: Constants the inventory finds whose name says gate and whose value does not, each with why.
NOT_GATING: dict[str, str] = {
    "harness/mcp_server.py:MAX_INLINE_BYTES": "how a result is returned, inline or as a path; never what it is",
    "harness/context.py:MIN_SLOT_CTX": "the floor the eval7b_slot_ctx range is held to; a preset under it is refused (#665)",
    "evals/private.py:MIN_FRAGMENT": "which private text the leak scan looks for; never a lane's outcome (#654)",
}


def resolve(site: str):
    """The value a site names now: a module constant, or a function's argument default."""
    m = _SITE.fullmatch(site)
    if m is None:
        raise ValueError(f"not a site: {site!r}")
    module = importlib.import_module(m["rel"][:-3].replace("/", "."))
    got = getattr(module, m["name"])
    if m["param"]:
        return _inspect.signature(got).parameters[m["param"]].default
    return got


def _numeric(node) -> bool:
    if isinstance(node, ast.Constant):
        return isinstance(node.value, (int, float)) and not isinstance(node.value, bool)
    if isinstance(node, ast.UnaryOp):
        return _numeric(node.operand)
    if isinstance(node, ast.BinOp):
        return _num_or_ref(node.left) and _num_or_ref(node.right) and (_numeric(node.left) or _numeric(node.right))
    if isinstance(node, ast.Dict):
        return bool(node.values) and all(_num_or_ref(v) for v in node.values) and any(_numeric(v) for v in node.values)
    return False


def _num_or_ref(node) -> bool:
    return _numeric(node) or isinstance(node, (ast.Name, ast.Attribute))


def gating(text: str, rel: str) -> list[str]:
    """Sites 'rel:NAME' of module-level numeric constants whose name carries a gating word, in source order."""
    out = []
    for stmt in ast.parse(text).body:
        if isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name):
            name, value = stmt.targets[0].id, stmt.value
        elif isinstance(stmt, ast.AnnAssign) and isinstance(stmt.target, ast.Name) and stmt.value is not None:
            name, value = stmt.target.id, stmt.value
        else:
            continue
        if _UPPER.fullmatch(name) and GATING_WORDS & set(name.strip("_").split("_")) and _numeric(value):
            out.append(f"{rel}:{name}")
    return out


def _production(rel: str) -> bool:
    return rel.endswith(".py") and rel.startswith(("harness/", "evals/")) and not rel.startswith("evals/cases/")


def inventory(root) -> list[str]:
    """Every gating constant in production Python under `root`, by site."""
    from harness import repo
    root = Path(root)
    out = []
    for path in repo.publishable(root):
        rel = path.relative_to(root).as_posix()
        if _production(rel) and path.is_file():
            out += gating(path.read_text(encoding="utf-8", errors="replace"), rel)
    return out


def registered() -> set[str]:
    return {s for k in KNOBS.values() for s in k.sites}


def unregistered(found) -> list[str]:
    known = registered() | set(NOT_GATING)
    return [s for s in found if s not in known]


def census(root) -> dict:
    """How many gating constants the registry covers and how many it does not."""
    found = inventory(root)
    known = registered()
    return {"found": found, "registered": sum(1 for s in found if s in known),
            "not_gating": sum(1 for s in found if s in NOT_GATING),
            "unregistered": len(unregistered(found))}


def limits() -> dict:
    """The `limit:` values this harness currently chooses, by predicate name."""
    out = {}
    for k in KNOBS.values():
        if not k.limit or k.predicate != LIMIT:
            continue
        out[k.limit_name("")] = k.limit_value("")
        if "{lane}" in k.limit:
            for lane in k.lanes:
                out[k.limit_name(lane)] = k.limit_value(lane)
    return out


def settings(lane: str, overrides: dict | None = None) -> dict:
    """The exam knobs an eval of `lane` runs at, by knob name: their defaults, or the values a run names."""
    overrides = dict(overrides or {})
    unknown = sorted(n for n in overrides if n not in KNOBS or KNOBS[n].receipt != "knobs")
    if unknown:
        raise ValueError(f"not a knob a run can set: {', '.join(unknown)}")
    return {k.name: overrides.get(k.name, k.default(lane)) for k in KNOBS.values()
            if k.receipt == "knobs" and lane in k.lanes}


def rebind(overrides: dict):
    """Set each override that has no runner attribute on its module for the duration; restore after."""
    import contextlib

    @contextlib.contextmanager
    def held():
        saved = []
        try:
            for name, value in overrides.items():
                k = KNOBS[name]
                if k.runner_attr:
                    continue
                m = _SITE.fullmatch(k.sites[0])
                module = importlib.import_module(m["rel"][:-3].replace("/", "."))
                saved.append((module, m["name"], getattr(module, m["name"])))
                setattr(module, m["name"], value)
            yield
        finally:
            for module, attr, old in reversed(saved):
                setattr(module, attr, old)
    return held()


def apply(runner, overrides: dict) -> None:
    """Set the overrides a runner carries as attributes."""
    for name, value in overrides.items():
        attr = KNOBS[name].runner_attr
        if attr and hasattr(runner, attr):
            setattr(runner, attr, value)
