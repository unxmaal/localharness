"""What discovery never saw. Issue #99, assumption F1.

Extraction precision -- 0.45 from the recap, issue #49 -- measures the quality
of what gets CAUGHT. Nothing here has ever measured what is never seen at all,
and those are different numbers.

THE PROBE, and it needs no models and no network: take the things this project
actually adopted, and ask which configured source ever produced each. Anything
adopted that no source ever surfaced is a coverage hole with a name, and the
names are the useful part -- "we read three subreddits and a release feed" is a
list of sources, not a measurement of reach.

WHAT COUNTS AS ADOPTED is read from three places that cannot drift from the
truth, because each is what the thing itself says:

    served      the gateway config's upstream ids -- what this machine runs
    measured    candidate names in run receipts -- what was evaluated
    default     the typed lane defaults -- what a bare command reaches for

A THING FOUND ONLY AFTER IT WAS ADOPTED DID NOT LEAD US TO IT. The store's
first sighting is compared against the earliest receipt naming it, so a source
that turned it up a month later gets no credit for the find. That is the
difference between a source that works and a source that eventually agrees.
"""
from __future__ import annotations

import json

#: Sources that are not discovery: they record what a later tier did with a
#: name, so crediting them with finding it would be circular. `inspect` writes
#: what it read, `installed` is the seed list of what we already run, and
#: `lh discover --gap` is this project asking itself what it lacks.
NOT_DISCOVERY = {"inspect", "installed", "lh discover --gap"}


def aliases(config=None) -> dict[str, str]:
    """This machine's gateway alias -> the upstream id it actually serves.

    AN ALIAS IS OUR OWN NAMING and no source can propose one. Asking whether a
    feed ever surfaced `local-large` is asking whether strangers guessed a name
    we invented; the fair question is whether anything surfaced
    `mlx-community/Qwen2.5-7B-Instruct-4bit`, which is what that alias runs.
    """
    import os
    from pathlib import Path
    try:
        import yaml
    except ImportError:      # pragma: no cover - yaml is a hard dependency
        return {}
    root = Path(__file__).resolve().parent.parent
    config = config or Path(os.environ.get("GATEWAY_CONFIG",
                                           root / "gateway" / "config.yaml"))
    try:
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return {}
    out = {}
    for entry in data.get("model_list") or []:
        name = entry.get("model_name")
        upstream = (entry.get("litellm_params") or {}).get("model", "")
        parts = upstream.split("/")
        if name and len(parts) >= 3:
            out[name] = "/".join(parts[1:])
        elif name and upstream:
            out[name] = upstream
    return out


def model_of(candidate: str, speech: bool = False) -> str:
    """The MODEL inside a receipt key, which is the only part a source can
    propose.

    Two shapes share the same punctuation and want opposite halves:

        Kokoro-82M-bf16/af_sky        a model and the voice we chose
        mflux/flux2-klein-4b-q8       an engine and the model it ran

    The engines are named in evals.run, so that is what tells them apart rather
    than a guess about which side looks more like a model. A composition like
    `trace/mflux/...` reduces the same way, one layer at a time.
    """
    from evals.run import PROCESS_ENGINES, REPAIR_PREFIX, TRACE_PREFIXES
    wrappers = set(PROCESS_ENGINES) | set(TRACE_PREFIXES) | {REPAIR_PREFIX}
    parts = [p for p in candidate.replace(":", "/").split("/") if p]
    while len(parts) > 1 and parts[0].lower() in wrappers:
        parts = parts[1:]
    if speech and len(parts) == 2:
        # A SPEECH receipt key is `model/voice` -- or `model/reference-clip`
        # for a cloner. Either way the second half is ours to pick and no
        # source proposes it. Told by the MODALITY rather than by guessing
        # which half looks more like a model: `af_sky` and `fleurs-fr-male-1`
        # are both voices and share no shape, while `mlx-community/X` is a
        # registry id whose first half must stay.
        return parts[0]
    return "/".join(parts)


def adopted(runs=None, config=None, orgs=()) -> dict[str, set[str]]:
    """name -> how this project came to own it: served, measured, default.

    Aliases are resolved to what they serve, so the question asked of each
    source is one a source could possibly have answered.
    """
    from harness import rank, winners
    known = aliases(config)
    out: dict[str, set[str]] = {}

    def note(name: str, how: str):
        name = (name or "").strip()
        if not name:
            return
        name = known.get(name, name)
        # LOWERCASED, because the same model arrives spelled two ways: the
        # gateway config preserves `Qwen2.5-7B-Instruct-4bit` and the served
        # list lowercases it. Counted twice, one thing becomes two holes.
        out.setdefault(model_of(name).lower(), set()).add(how)

    for name in rank.serving():
        note(name, "served")
    for name in _first_receipt(runs):
        note(name, "measured")
    for name in winners.typed().values():
        note(name, "default")
    return _collapse_variants(out, {o.lower() for o in orgs})


def _collapse_variants(names: dict[str, set[str]],
                       orgs: set[str] = frozenset()) -> dict[str, set[str]]:
    """Fold `model/voice` into `model` where the data itself shows variants.

    A run from before receipts carried a modality cannot say which lane it was,
    so model_of() leaves its key whole. But a first segment carrying SEVERAL
    different second segments is a model measured across voices -- five for
    Kokoro, three reference clips for the cloner -- and counting those as five
    adopted things overstates the holes fivefold.

    Not applied to a segment that is an owner: `mlx-community` precedes a dozen
    models and folding on it would call them all one thing.
    """
    variants: dict[str, set[str]] = {}
    for name in names:
        head, _, tail = name.partition("/")
        if tail:
            variants.setdefault(head, set()).add(tail)
    # AN OWNER IS NOT A MODEL. `mlx-community` precedes a dozen different
    # models and folding on it turned eleven adopted things into one; the
    # publishers are the ones the store itself shows owning several names.
    fold = {head for head, tails in variants.items()
            if len(tails) > 1 and head not in orgs}
    out: dict[str, set[str]] = {}
    for name, how in names.items():
        head = name.partition("/")[0]
        out.setdefault(head if head in fold else name, set()).update(how)
    return out


#: The lanes whose receipt keys carry a voice or a reference clip.
SPEECH_LANES = {"tts", "stt"}


def _first_receipt(runs=None) -> dict[str, float]:
    """Earliest time each MODEL appears in a run receipt.

    Keyed by the model rather than the receipt key, so a tts model measured
    across five voices is one adopted thing rather than five.
    """
    from harness import paths
    seen: dict[str, float] = {}
    try:
        receipts = sorted((runs or paths.runs()).rglob("results.json"))
    except OSError:
        return seen
    for f in receipts:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
            when = f.stat().st_mtime
        except (OSError, ValueError):
            continue
        modality = ((data.get("receipt") or {}).get("modality") or "").lower()
        for candidate in (data.get("summary") or {}):
            name = model_of(candidate, speech=modality in SPEECH_LANES)
            if name not in seen or when < seen[name]:
                seen[name] = when
    return seen


def sightings(conn) -> list[dict]:
    """Every proposal with its first sighting and the sources that saw it."""
    q = """
        SELECT p.name AS name, MIN(s.seen_at) AS first_seen,
               GROUP_CONCAT(DISTINCT s.source) AS sources
        FROM proposals p JOIN sightings s ON s.proposal_id = p.id
        GROUP BY p.id
    """
    return [dict(r) for r in conn.execute(q)]


def _segments(name: str) -> set[str]:
    return {part for part in name.lower().replace(":", "/").split("/") if part}


def common_segments(names) -> set[str]:
    """Segments that identify a PUBLISHER rather than a thing.

    Derived by counting rather than listed, because a hand-kept list of orgs is
    one more thing to drift. An owner sits in FRONT: `mlx-community` precedes a
    dozen models, so a first segment shared by several names is a publisher and
    matching on it would call every MLX model the same model.

    Counted over ONE side only. The first cut counted across both, so a
    genuine match -- `parakeet-tdt-0.6b-v2` in a receipt and
    `mlx-community/parakeet-tdt-0.6b-v2` in the store -- made the model name
    itself look like an owner, and every single adopted thing came back a
    coverage hole. A 100% result is a broken matcher, not a finding.
    """
    seen: dict[str, set[str]] = {}
    for name in names:
        parts = [p for p in name.lower().replace(":", "/").split("/") if p]
        if len(parts) > 1:
            seen.setdefault(parts[0], set()).add(name)
    return {part for part, owners in seen.items() if len(owners) > 1}


def _known(name: str, rows: dict, orgs: set[str]) -> dict | None:
    """The store row naming the same thing, or None.

    THE TWO SIDES SPELL IT DIFFERENTLY and are trimmed from opposite ends: an
    adopted name may be a receipt key carrying a voice (`Kokoro-82M-bf16/
    af_sky`), while the store holds a registry id carrying an org
    (`mlx-community/parakeet-tdt-0.6b-v2`). So the comparison is on SEGMENTS,
    and a segment that names a publisher does not count -- otherwise every
    model under one org matches every other.

    Deliberately strict about artifacts: `Qwen/Qwen2.5-7B` in the store and
    `mlx-community/Qwen2.5-7B-Instruct-4bit` on the gateway are a base model
    and a requantisation of it, which is not the same thing to download or to
    run. Calling that a find would credit a source with surfacing something it
    did not.
    """
    if name in rows:
        return rows[name]
    mine = _segments(name) - orgs
    for candidate, row in rows.items():
        if mine & (_segments(candidate) - orgs):
            return row
    return None


def report(conn, runs=None) -> dict:
    """Which sources found what this project adopted, and what none of them did."""
    rows = {r["name"]: r for r in sightings(conn)}
    first_run = _first_receipt(runs)
    orgs = common_segments(rows)
    mine = adopted(runs, orgs=orgs)
    found, holes, late = [], [], []
    for name, how in sorted(mine.items()):
        row = _known(name, rows, orgs)
        if row is None:
            holes.append({"name": name, "how": sorted(how)})
            continue
        sources = sorted({s for s in (row["sources"] or "").split(",")
                          if s and s not in NOT_DISCOVERY})
        entry = {"name": name, "how": sorted(how), "sources": sources,
                 "first_seen": row["first_seen"]}
        if not sources:
            # In the store, but only because a later tier put it there. That
            # is the store remembering our own work, not a source finding it.
            holes.append({**entry, "why": "only recorded by a tier of ours"})
            continue
        measured_at = first_run.get(name)
        if measured_at and row["first_seen"] > measured_at:
            entry["days_late"] = (row["first_seen"] - measured_at) / 86400.0
            late.append(entry)
        else:
            found.append(entry)
    by_source: dict[str, int] = {}
    for entry in found:
        for source in entry["sources"]:
            by_source[source] = by_source.get(source, 0) + 1
    proposed = {}
    for row in sightings(conn):
        for source in (row["sources"] or "").split(","):
            if source and source not in NOT_DISCOVERY:
                proposed[source] = proposed.get(source, 0) + 1
    return {"adopted": len(mine), "found": found, "late": late,
            "holes": holes,
            "by_source": dict(sorted(by_source.items(),
                                     key=lambda kv: -kv[1])),
            # WHAT THE SOURCES DID PRODUCE, beside what was adopted. Without
            # this a reader cannot tell a source that finds nothing from one
            # that finds plenty of things nobody has run yet -- and those want
            # opposite fixes.
            "proposed": dict(sorted(proposed.items(),
                                    key=lambda kv: -kv[1]))}
