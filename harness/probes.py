"""The constants issue #98 asks about, wired to run against cached data.

Every probe here is offline and deterministic. The GitHub client is built with
a zero request budget and an effectively infinite TTL, so nothing is fetched
and a missing cache entry is an error rather than a silent network call; the
source probes read clones already on disk and size models from the size cache
only. That is what makes the sweep re-runnable in a second rather than a
morning, which is the difference between a check that gets run and one that
does not.

A cache miss skips a candidate, so the candidate set may be smaller than a live
sweep's. It is the SAME set at every value of a constant, so the comparison
stays fair; `coverage()` reports the size so a reader can see what it was.
"""
from __future__ import annotations

import json
from pathlib import Path

from harness import github, inspect as insp, neighbors as nb, paths
from harness.sensitivity import Finding, bound, measure

#: Big enough that every cached entry counts as fresh, so `get` never reaches
#: the fetch branch at all.
FOREVER_HOURS = 1e9


def offline(cache: Path | None = None) -> github.Client:
    return github.Client(budget=0, ttl_hours=FOREVER_HOURS, cache=cache)


def clones() -> Path:
    return paths.home() / "cache" / "clones"


def cached_facts(model_id: str, cache: dict | None = None) -> dict:
    """Sizes from the on-disk cache only. An unknown model is unsized, which is
    a verdict `decide` already understands, rather than a fetch."""
    hit = (cache or {}).get(model_id)
    if isinstance(hit, dict):
        return {"size": int(hit.get("size") or 0), "lane": hit.get("lane") or ""}
    if isinstance(hit, (int, float)):
        return {"size": int(hit), "lane": ""}
    return {"size": 0, "lane": ""}


# ---------------------------------------------------------------------------
# What the crowd ranking is made of
# ---------------------------------------------------------------------------

def _people(client) -> list[str]:
    return nb.cohort(client=client)


def ranking(client, people, **kw) -> list[str]:
    """The shipped ranking, in order. `lh discover --neighbors` calls this with
    these arguments, and a probe measuring anything else measures a toy."""
    return [n.repo for n in nb.neighbors(people, client, top=25,
                                         exclude=nb.DEFAULT_SEEDS, **kw)]


def verdict_of_control(client, people, **kw) -> str:
    """What the two-directional control says at these settings.

    A distance says the ranking moved; this says whether it still works. The
    positive half asks that the tools this project already runs still rank, the
    negative half that nothing over POPULAR_STARS leaked in. Both are counts
    over a set, which is fine HERE -- a count is blind to ordering, and quality
    is the one question that is not about ordering.
    """
    try:
        got = nb.control(list(nb.DEFAULT_SEEDS), people, client, **kw)
    except Exception as exc:  # noqa: BLE001
        return f"control failed: {type(exc).__name__}"
    # The ANSWERED count, not the cohort size. Offline, a person whose stars
    # are not cached is skipped silently, so a 450-person cohort can be a
    # 344-person crowd -- and crowd SIZE is the one thing this metric is known
    # not to be invariant in. Reporting the cohort size here would have read as
    # evidence about a crowd that was never asked.
    n = got["crowd"]
    if got["separates"]:
        return (f"separates {len(got['expected_found'])}/"
                f"{len(nb.DEFAULT_SEEDS)}, n={n}")
    if got["popularity_leaks"]:
        return f"LEAKS {len(got['popularity_leaks'])}, n={n}"
    return f"FAILS: none of the known-good ranked, n={n}"


def probe_min_shared() -> Finding:
    client = offline()
    people = _people(client)
    return measure("MIN_SHARED", nb.MIN_SHARED, [1, 2, 3, 5, 8],
                   lambda v: ranking(client, people, min_shared=v),
                   quality=lambda v: verdict_of_control(client, people,
                                                        min_shared=v),
                   note="how many of the crowd must star a repo for it to rank")


def probe_population() -> Finding:
    client = offline()
    people = _people(client)

    def run(v):
        # POPULATION reaches enrichment() as a DEFAULT ARGUMENT, so rebinding
        # neighbors.POPULATION would measure nothing. See sensitivity.bound.
        with bound(nb, "enrichment", population=v):
            return ranking(client, people)

    def check(v):
        with bound(nb, "enrichment", population=v):
            return verdict_of_control(client, people)

    return measure("POPULATION", nb.POPULATION,
                   [1e5, 1e6, 5e6, 2e7, 1e8], run, quality=check,
                   note="the world's star count, the denominator of enrichment")


def probe_half_life() -> Finding:
    client = offline()
    people = _people(client)

    def run(v):
        with bound(nb, "recency", half_life=v):
            return ranking(client, people)

    def check(v):
        with bound(nb, "recency", half_life=v):
            return verdict_of_control(client, people)

    return measure("HALF_LIFE_DAYS", nb.HALF_LIFE_DAYS,
                   [90.0, 365.0, 1095.0, 1e9], run, quality=check,
                   note="push-date decay; 1e9 is the decay switched off")


def probe_min_degree() -> Finding:
    client = offline()
    return measure("min_degree", 2, [1, 2, 3, 5],
                   lambda v: nb.cohort(client=client, min_degree=v),
                   quality=lambda v: verdict_of_control(
                       client, nb.cohort(client=client, min_degree=v)),
                   note="how many of the crowd must follow someone for a "
                        "second hop to reach them")


def probe_per_repo() -> Finding:
    client = offline()
    return measure("per_repo", 12, [4, 8, 12, 24],
                   lambda v: nb.cohort(client=client, per_repo=v),
                   note="contributors taken from each seed repo")


def probe_hops() -> Finding:
    client = offline()
    return measure("hops", 2, [1, 2, 3],
                   lambda v: nb.cohort(client=client, hops=v),
                   note="how far from the seeds the crowd reaches")


def probe_crowd_limit() -> Finding:
    client = offline()
    return measure("cohort limit", 250, [100, 250, 450],
                   lambda v: nb.cohort(client=client, limit=v),
                   quality=lambda v: verdict_of_control(
                       client, nb.cohort(client=client, limit=v)),
                   note="crowd size. The enrichment metric is NOT scale "
                        "invariant in this, so expect movement -- and read "
                        "n= rather than the value: offline, only people whose "
                        "stars are cached actually answer")


# ---------------------------------------------------------------------------
# What the source screen is made of
# ---------------------------------------------------------------------------

def _repos_on_disk() -> list[str]:
    root = clones()
    if not root.is_dir():
        return []
    return sorted(d.name.replace("__", "/", 1) for d in root.iterdir()
                  if d.is_dir() and "__" in d.name)


def _meta(client, repo: str) -> dict:
    """The repo's cached GitHub metadata, or {}.

    Not optional. `size` is what CLONE_KB_CAP compares against, and without it
    every source_kb is 0, the cap can never fire, and the constant reports
    INERT no matter what value it is given -- a false negative produced by the
    probe rather than by the code under test.
    """
    try:
        return client.repo(repo) or {}
    except github.GitHubError:
        return {}


def verdicts(**kw) -> dict[str, str]:
    """repo -> verdict, over every clone already on disk."""
    client = offline()
    out = {}
    for repo in _repos_on_disk():
        try:
            fit = insp.inspect(repo, clones(), meta=_meta(client, repo),
                               facts=cached_facts, **kw)
        except (insp.InspectError, OSError) as exc:
            out[repo] = f"error: {type(exc).__name__}"
            continue
        out[repo] = fit.verdict
    return out


def probe_memory_ceiling() -> Finding:
    GIB = insp.GIB
    return measure("MEMORY_CEILING", 22 * GIB,
                   [8 * GIB, 16 * GIB, 22 * GIB, 32 * GIB, 96 * GIB],
                   lambda v: verdicts(ceiling=v),
                   note="the largest weight this machine can hold")


def probe_size_limit() -> Finding:
    def run(v):
        # SIZE_LIMIT is read from the module INSIDE inspect(), not captured as
        # a default, so rebinding it is the honest way to vary it.
        old = insp.SIZE_LIMIT
        insp.SIZE_LIMIT = v
        try:
            return verdicts()
        finally:
            insp.SIZE_LIMIT = old

    return measure("SIZE_LIMIT", insp.SIZE_LIMIT, [2, 4, 12, 24, 64], run,
                   note="how many named weights get sized per repo")


def probe_dead_days() -> Finding:
    return measure("DEAD_DAYS", insp.DEAD_DAYS, [180, 365, 730, 1095, 3650],
                   lambda v: verdicts(dead_days=v),
                   note="when a repo counts as abandoned")


def probe_clone_kb_cap() -> Finding:
    return measure("CLONE_KB_CAP", insp.CLONE_KB_CAP,
                   [10_000, 50_000, 250_000, 1_000_000],
                   lambda v: verdicts(kb_cap=v),
                   note="source tree size above which a repo is weights in git")


PROBES = {
    "min_shared": probe_min_shared,
    "population": probe_population,
    "half_life": probe_half_life,
    "min_degree": probe_min_degree,
    "per_repo": probe_per_repo,
    "hops": probe_hops,
    "crowd_limit": probe_crowd_limit,
    "memory_ceiling": probe_memory_ceiling,
    "size_limit": probe_size_limit,
    "dead_days": probe_dead_days,
    "clone_kb_cap": probe_clone_kb_cap,
}

#: Constants #98 lists that no probe here covers, and the honest reason. A
#: sweep that quietly omitted them would read as coverage.
UNCOVERED = {
    "cache TTLs": "freshness, not output: nothing downstream reorders",
    "MAX_MENTIONS": "needs cached comment threads; feeds cache holds "
                    "posts, not per-thread comments",
    "mention_comments": "same",
    "DEFAULT_INTERVAL_DAYS": "decides WHEN a sweep runs, not what it returns",
    "DISK_FLOOR / MAX_DOWNLOAD": "needs a populated fetch queue",
    "H3_MIN_FRAMES / H3_FPS": "a product decision, and one generation is "
                              "40 minutes of GPU",
    "NOT_WEIGHTS / SKIP_DIRS / READ_SUFFIXES": "sets, not ordinals: a band "
                                               "has no meaning, so these need "
                                               "a different question",
}


def coverage() -> dict:
    client = offline()
    return {"crowd": len(_people(client)), "clones": len(_repos_on_disk())}


def run(names=None) -> list[Finding]:
    chosen = list(names) if names else list(PROBES)
    return [PROBES[n]() for n in chosen]
