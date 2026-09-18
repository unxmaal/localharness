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

from harness import lanes

#: lane -> how to spell a candidate of that lane for evals.run, given a model
#: id. Empty means this harness has no way to invoke an arbitrary model in that
#: lane, which is a fact about the harness rather than about the candidate.
#:
#: The specs are the ones evals.run already parses; nothing new is invented
#: here, because a second spelling of the candidate language is the defect that
#: this project has filed twice under other names.
#: mlx_lm.server treats the request's `model` as a live repo id and swaps to
#: it, so a text candidate needs no prefix and no config entry. That is true of
#: every lane in lanes.TEXT_SERVED and only `code` was listed, so `web` and
#: `svg` -- the 3rd and 4th priorities -- reported `no-runner` for a candidate
#: the harness could in fact have screened. Issue #207.
LANE_CANDIDATE = {
    "image": "mflux:{model}",
    "stt": "stt:{model}",
    "tts": "tts:{model}",
    **{lane: "{model}" for lane in lanes.TEXT_SERVED},
}

#: Why a row cannot be screened. Reported per row rather than filtered away: a
#: queue that silently drops what it cannot run looks like a queue that ran out.
WAITING, NO_RUNNER, READY = "waiting-on-fetch", "no-runner", "ready"


#: Words in a card that mean "this attaches to a model" rather than "this is
#: one". An adapter, a node pack or a workflow cannot be handed to a runner as
#: a candidate: mflux loads a base model, and `mflux:org/some-style-lora`
#: downloads gigabytes and then fails, recording a verdict that says nothing
#: about the thing.
NOT_A_MODEL = ("lora", "comfyui", "workflow", "adapter", "controlnet",
               "textual_inversion", "embedding")

#: The one substring that is a model in its own right despite matching above.
#: Kept as an enumerated exception so the list can be read rather than guessed.
NOT_A_MODEL_EXCEPTIONS = ("lora-ready",)


def is_attachment(description: str) -> str:
    """The word that says this attaches to a model, or "" if none does.

    Read from the registry's own tags. A style LoRA and a ComfyUI node pack are
    both `text-to-image` and neither is something a lane can run alone.
    """
    text = (description or "").lower()
    for allowed in NOT_A_MODEL_EXCEPTIONS:
        text = text.replace(allowed, "")
    for word in NOT_A_MODEL:
        if word in text:
            return word
    return ""


def candidate_for(lane: str, model: str, description: str = "") -> str:
    """The candidate spec for this lane, or "" when nothing here can run it.

    ALREADY-SPELLED SPECS PASS THROUGH. A challenger arrives from the store as
    a bare repo id and is wrapped once; the INCUMBENT arrives from
    winners.typed() already spelled `mflux:flux2-klein-4b`, and wrapping it
    again produced `mflux:mflux:flux2-klein-4b`, which resolve() rejects. So
    the lane's own control contributed no cases, the challenger ran alone, and
    the run printed the pairing it intended above a receipt with one candidate
    in it. A candidate measured beside a control that did not run says nothing
    about the candidate. Issue #214.
    """
    if is_attachment(description):
        return ""
    spec = LANE_CANDIDATE.get(lanes.canonical(lane), "")
    if not spec:
        return ""
    prefix = spec.split("{model}", 1)[0]
    if prefix and model.startswith(prefix):
        return model
    return spec.format(model=model)


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
        spec = candidate_for(lane, name, row.get("description") or "")
        absent = [] if not spec else missing(name)
        attached = is_attachment(row.get("description") or "")
        if not spec:
            state, why = NO_RUNNER, (
                f"a {attached}: it attaches to a model rather than being one, "
                f"so no runner takes it as a candidate" if attached
                else f"no runner for the {lane} lane" if lane
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


def gateway_routes(config=None) -> tuple[set[str], str]:
    """The alias names the gateway will accept, and the upstream behind them.

    LiteLLM validates the request's `model` against its configured aliases and
    answers HTTP 400 for anything else. mlx_lm.server, which sits behind it,
    treats the name as a live repo id and swaps to it. So a DISCOVERED text
    candidate -- which is always a repo id and never an alias -- has to reach
    the upstream directly or it is refused before a token is generated.
    """
    import os
    from pathlib import Path
    try:
        import yaml
    except ImportError:      # pragma: no cover - yaml is a hard dependency
        return set(), ""
    root = Path(__file__).resolve().parent.parent
    config = config or Path(os.environ.get(
        "GATEWAY_CONFIG", root / "gateway" / "config.yaml"))
    try:
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return set(), ""
    names, base = set(), ""
    for entry in data.get("model_list") or []:
        if entry.get("model_name"):
            names.add(str(entry["model_name"]).strip().lower())
        base = base or (entry.get("litellm_params") or {}).get("api_base", "")
    return names, base


def routed_gateway(model: str, config=None) -> str:
    """Where to send this candidate, or "" to use the default gateway.

    An alias the gateway knows goes through the gateway. A repo id does not:
    it goes to the upstream that can hot-swap to it.
    """
    names, base = gateway_routes(config)
    if (model or "").strip().lower() in names:
        return ""
    return base if "/" in (model or "") else ""


def argv(row: dict, outdir=None) -> list[str]:
    """The exact command a screen runs. One case, one repeat, no quality
    metrics: the screen answers whether it runs, and a metric here would invite
    ranking a screen against a measurement."""
    out = ["python", "-m", "evals.run", "--modality", row["modality"],
           "--screen", "--repeat", "1", "--candidates", row["candidate"]]
    upstream = routed_gateway(row.get("name") or "")
    if upstream:
        out += ["--gateway", upstream.rsplit("/v1", 1)[0]]
    if outdir:
        out += ["--out", str(outdir)]
    return out


#: Things a run says when the HARNESS could not deliver the request, as opposed
#: to the candidate failing it. `broken` is terminal, so recording one of these
#: against a candidate declines it forever for something it never did.
NOT_THE_CANDIDATE = (
    "invalid model name",
    "gateway returned http 400",
    "is the gateway up",
    "connection refused",
    # THIRD OCCURRENCE OF THE CLASS. The runner could not build a spec for this
    # candidate, which is a gap in this harness, and it was recorded as BROKEN,
    # which is terminal. ERROR #22 and #60 are this same message from the stt
    # and tts lanes, so it has been settling real candidates falsely across
    # three lanes. #213, after #206 and #211.
    "no cases of a modality it can run",
    "no candidate matched any case",
)


def refused_by_harness(detail: str) -> str:
    """The phrase saying this never reached the candidate, or "".

    A gateway that declines to route a name is a fact about the gateway's alias
    table. Qwen3-8B-4bit was recorded `broken -- it ran and passed nothing`
    after an HTTP 400 stopped it before a single token.
    """
    text = (detail or "").lower()
    for phrase in NOT_THE_CANDIDATE:
        if phrase in text:
            return phrase
    return ""


def outcome(returncode: int, summary: dict | None,
            detail: str = "") -> tuple[str, str]:
    """A store verdict from one screen run.

    `broken` is TERMINAL and `screened` is not, which is the right way round: a
    thing that does not run is answered, and a thing that runs still has every
    measurement ahead of it.

    A request the harness could not deliver is NEITHER. It is recorded as
    `queued`, which is not terminal, so the candidate is asked again once the
    harness can route it.
    """
    refused = refused_by_harness(detail)
    if refused:
        return "queued", (f"not screened: {refused}. The harness could not "
                          f"deliver the request, which says nothing about the "
                          f"candidate")
    if returncode != 0:
        return "broken", f"the screen exited {returncode}"
    # THE SUMMARY SPELLS IT `passed`. Reading `pass` returned 0 for every run,
    # so a candidate that passed every case was recorded `broken`, which is
    # TERMINAL. The screen tier reported the opposite of what it measured.
    rows = sum(int(v.get("passed", 0)) for v in (summary or {}).values()) \
        if summary else 0
    if not summary:
        return "broken", "the screen produced no rows"
    if rows == 0:
        return "broken", "it ran and passed nothing"
    return "screened", f"{rows} case(s) passed a screen"
