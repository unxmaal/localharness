"""Did it run at all? The rung between a ranked queue and a measurement.

Issue #148's ladder is sweep, inspect, rank, SCREEN, measure. The screen was
specified in #53 and never connected: `evals.run --screen` exists and stamps a
receipt tier, `screened` has been in the store's verdict vocabulary since the
store was built, and nothing has ever written one. So the queue this project
now ranks is ranked for a tier that cannot see it -- the third time in this
ladder that two correct halves had no rung between them.

WHAT A SCREEN ASKS is binary and cheap: did it run, did it emit an artifact.
That is the right question because it is how things have actually failed here
-- seedvr2 crashed 0/3, local-small never closed a tag 0/9, q3-14b returned
null content behind a passing rate. Almost nothing has failed by running and
scoring slightly worse.

IT NEVER DOWNLOADS. Fetching is its own explicit step (`lh fetch --run`) with
its own disk budget, and a tier that quietly pulls gigabytes because something
ranked well is how a laptop fills up overnight. A candidate whose weights are
absent is reported as waiting on a fetch, not screened and not failed.
"""
from __future__ import annotations

#: lane -> how to spell a candidate of that lane for evals.run, given a model
#: id. Empty means this harness has no way to invoke an arbitrary model in that
#: lane, which is a fact about the harness rather than about the candidate.
#:
#: The specs are the ones evals.run already parses; nothing new is invented
#: here, because a second spelling of the candidate language is the defect that
#: this project has filed twice under other names.
LANE_CANDIDATE = {
    "image": "mflux:{model}",
    "stt": "stt:{model}",
    "tts": "tts:{model}",
    # mlx_lm.server treats the request's `model` as a live repo id and swaps to
    # it, so a text candidate needs no prefix and no config entry.
    "code": "{model}",
}

#: Why a row cannot be screened. Reported per row rather than filtered away: a
#: queue that silently drops what it cannot run looks like a queue that ran out.
WAITING, NO_RUNNER, READY = "waiting-on-fetch", "no-runner", "ready"


def candidate_for(lane: str, model: str) -> str:
    """The candidate spec for this lane, or "" when nothing here can run it."""
    spec = LANE_CANDIDATE.get((lane or "").strip().lower(), "")
    return spec.format(model=model) if spec else ""


def plan(rows, *, missing=None) -> list[dict]:
    """What a screen would do to each row, and what stands in the way.

    `missing` says what a candidate still needs that is not on disk -- ITSELF
    AND WHAT ITS CONFIG NAMES, because a complete repo is not a loadable model.
    Marvis-AI's 8-bit MLX repo is whole and names a tokenizer in a different
    repo; `ready` on that cost a real run to discover, and the screen tier
    exists to be cheap. Injected so this is testable without a cache and
    without a download. Issue #196.
    """
    if missing is None:
        from harness.fetching import missing as _missing
        missing = _missing
    out = []
    for row in rows:
        name = row["name"]
        lane = (row.get("lane") or "").strip().lower()
        spec = candidate_for(lane, name)
        absent = [] if not spec else missing(name)
        if not spec:
            state, why = NO_RUNNER, (
                f"no runner for the {lane} lane" if lane
                else "no lane, so no case and no metric")
        elif absent == [name]:
            state, why = WAITING, "weights are not on disk; lh fetch --run"
        elif absent:
            # NAME WHAT IS MISSING. "not ready" without the id sends whoever
            # reads it back to the server log to find out what to fetch.
            state, why = WAITING, (
                f"needs {', '.join(absent)}, which its config names and "
                f"nothing has fetched; lh fetch --run")
        else:
            state, why = READY, f"evals.run --modality {lane} --screen"
        out.append({**row, "state": state, "why_not": why, "candidate": spec,
                    "modality": lane})
    return out


def argv(row: dict, outdir=None) -> list[str]:
    """The exact command a screen runs. One case, one repeat, no quality
    metrics: the screen answers whether it runs, and a metric here would invite
    ranking a screen against a measurement."""
    out = ["python", "-m", "evals.run", "--modality", row["modality"],
           "--screen", "--repeat", "1", "--candidates", row["candidate"]]
    if outdir:
        out += ["--out", str(outdir)]
    return out


def outcome(returncode: int, summary: dict | None) -> tuple[str, str]:
    """A store verdict from one screen run.

    `broken` is TERMINAL and `screened` is not, which is the right way round: a
    thing that does not run is answered, and a thing that runs still has every
    measurement ahead of it.
    """
    if returncode != 0:
        return "broken", f"the screen exited {returncode}"
    rows = sum(int(v.get("pass", 0)) for v in (summary or {}).values()) \
        if summary else 0
    if not summary:
        return "broken", "the screen produced no rows"
    if rows == 0:
        return "broken", "it ran and passed nothing"
    return "screened", f"{rows} case(s) passed a screen"
