"""Order a queue of candidates by what is worth LEARNING, not by who will win.

Issue #175. The judge tier scored twenty-five real candidates 3/10 each, behind
a control that separated at +7. Two further controls, shaped like the data the
tier actually meets, settled why:

    bare     a name and nothing else            0 of 3 runs separated
    carded   the registry's own description     0 of 3 runs separated

Six models with known opposite outcomes -- the STT engine this project measured
and adopted against the one it measured and dropped -- all scored 3. The reason
string says it plainly: "a new model with no concrete measured claim". The
rubric's top criterion is a measured claim, and A REGISTRY CARD NEVER CONTAINS
ONE. No wording fixes that, and no amount of extra metadata does either.

THE DEEPER ERROR WAS IN THE QUESTION. `parakeet` beat `whisper-large-v3` on WER
and latency, and nothing in either card could have told you that -- the fact was
produced by running them. A triage tier cannot order candidates by eventual
quality, because eventual quality is what the tiers below it exist to find out.

What triage CAN order is the value of the information a screen would buy:

    a lane nobody has measured yet     a screen there answers an open question
    a thing seen again and again       recurrence over time, this project's own
                                       answer to a feed measuring popularity
    something unlike what is served    a requantised copy of a model already
                                       running here teaches nothing
    cheap to screen                    small weights, a shorter round trip

Every one of those is a fact the store already holds, so this is arithmetic
rather than an opinion: deterministic, seeded by nothing, and testable without a
gateway. That matters as much as the ordering -- the instrument it replaces was
a sampling language model whose control had to be run three times to be believed.
"""
from __future__ import annotations

import re

GIB = 1024 ** 3

#: What each signal is worth. Weights are a policy, so they live in one place
#: and the test that pins the ordering names them.
RECURRENCE = 2.0
OPEN_LANE = 3.0
KNOWN_LINEAGE = -4.0
NO_LANE = -6.0
CHEAP = 1.0

#: A card says what it was built from as `base_model:...`, sometimes several
#: times and with a role in between (`base_model:quantized:org/name`).
_LINEAGE = re.compile(r"built from ([^;]+)")
_SIZE = re.compile(r"([\d.]+) GiB of weights")


def _parents(description: str) -> set[str]:
    m = _LINEAGE.search(description or "")
    if not m:
        return set()
    return {p.strip().lower() for p in m.group(1).split(",") if p.strip()}


def size_gib(description: str) -> float:
    m = _SIZE.search(description or "")
    return float(m.group(1)) if m else 0.0


def value(row: dict, *, serving: set[str] = frozenset(),
          measured_lanes: set[str] = frozenset(),
          ceiling_gib: float = 22.0) -> tuple[float, list[str]]:
    """What a screen would teach us about this candidate, and why.

    The reasons travel with the number. A ranking whose rows cannot say why
    they are where they are is a ranking nobody can argue with, and every
    metric in this project that went wrong went wrong quietly.
    """
    score, why = 0.0, []

    #: A feed source declares `all` to mean it covers every lane. That is a
    #: fact about the SOURCE, and 243 proposals carry it as though it were
    #: theirs.
    lane = (row.get("lane") or "").strip().lower()
    if lane in ("all", "text"):
        lane = ""
    if not lane:
        # NOTHING HERE CAN MEASURE IT. Not a judgement about the model: the
        # eval suite has no case, no runner and no metric for it, which is a
        # gap in this harness and sometimes the work worth doing.
        score += NO_LANE
        why.append("no lane can measure it")
    elif lane not in measured_lanes:
        score += OPEN_LANE
        why.append(f"no run receipt in the {lane} lane")
    if lane:
        bonus = priority_of(lane)
        if bonus:
            score += bonus
            why.append(f"{lane} is a wanted lane")

    times = int(row.get("times") or 0)
    if times > 1:
        # DIMINISHING, and the test pins that rather than the comment claiming
        # it: the first cut rose linearly to a cap, so the step from two
        # sightings to forty was larger than the step from one to two, which is
        # the opposite of what "the tenth sighting says less than the second"
        # means. Recurrence separates a lasting thing from one that trended.
        score += RECURRENCE * (1.0 - 1.0 / times)
        why.append(f"seen {times} times")

    parents = _parents(row.get("description") or "")
    known = {p for p in parents
             if any(p == s or p in s or s in p for s in serving)}
    if known:
        score += KNOWN_LINEAGE
        why.append(f"a requant of {', '.join(sorted(known))}, already served")

    gib = size_gib(row.get("description") or "")
    if 0 < gib <= ceiling_gib / 4:
        score += CHEAP
        why.append(f"{gib:.1f} GiB, cheap to screen")
    return score, why


#: What the lanes are worth, highest first, as stated by the person this
#: harness is for. A lane absent from this list is worth less than any lane in
#: it.
#:
#: WHY IT IS NEEDED: the value-of-information score rewards "no run receipt in
#: this lane" identically for every lane, so the LEAST wanted lane attracts the
#: most attention precisely because it has been ignored. Without this, the
#: queue tied seven candidates at +5.0 and broke the tie ALPHABETICALLY, which
#: put a tts model above an image one because S sorts after D.
LANE_PRIORITY = ("image", "code", "web", "svg", "video")

#: The most a lane's priority can add. Deliberately smaller than KNOWN_LINEAGE
#: and NO_LANE: a wanted lane breaks a tie, and never outranks "this teaches
#: nothing".
PRIORITY = 2.0


def priority_of(lane: str) -> float:
    """A lane's share of PRIORITY, 0.0 for one nobody asked for."""
    lane = (lane or "").strip().lower()
    if lane not in LANE_PRIORITY:
        return 0.0
    rank_from_top = LANE_PRIORITY.index(lane)
    return PRIORITY * (len(LANE_PRIORITY) - rank_from_top) / len(LANE_PRIORITY)


def lane_of(row) -> str:
    """The lane this row can be tested in, normalised. Empty means none."""
    lane = (row.get("lane") or "").strip().lower()
    return "" if lane in ("all", "text") else lane


def rank(rows, *, serving=(), measured_lanes=(), ceiling_gib: float = 22.0,
         keep_laneless: bool = False):
    """Best first. Ties break on name so the order is stable across runs.

    A candidate NO LANE CAN TEST IS DROPPED rather than ranked last. Penalising
    it left 66 of 94 queue slots holding things nothing could measure, which
    crowds out the candidates a screen could actually answer a question about.
    Most are tools rather than models -- mflux, mlx-vlm, nativ -- and no lane
    tests a library. See wanted() for what to do with them instead. Issue #201.
    """
    serving = {s.lower() for s in serving}
    measured = {m.lower() for m in measured_lanes}
    out = []
    for row in rows:
        if not keep_laneless and not lane_of(row):
            continue
        got, why = value(row, serving=serving, measured_lanes=measured,
                         ceiling_gib=ceiling_gib)
        out.append({**row, "value": got, "value_why": "; ".join(why)})
    return sorted(out, key=lambda r: (-r["value"], r["name"]))


def wanted(rows, minimum: int = 2) -> list[dict]:
    """Laneless candidates that keep coming back, most-seen first.

    A LANE IS A PERSON'S DECISION. Dropping these silently would throw away the
    signal that the harness is missing something; inventing a lane for them
    would be the harness deciding what it is for. So they are reported, and
    somebody chooses. Issue #201.
    """
    out = [r for r in rows
           if not lane_of(r) and int(r.get("times") or 0) >= minimum]
    return sorted(out, key=lambda r: (-int(r.get("times") or 0), r["name"]))


def serving(config=None) -> set[str]:
    """The upstream model ids this machine's gateway serves.

    Read from the gateway config rather than listed here, because a list of
    what we run, kept by hand beside the thing that runs it, is the same defect
    as an HF candidate list forked into two files.
    """
    import os
    from pathlib import Path
    try:
        import yaml
    except ImportError:      # pragma: no cover - yaml is a hard dependency
        return set()
    root = Path(__file__).resolve().parent.parent
    config = config or Path(os.environ.get("GATEWAY_CONFIG",
                                           root / "gateway" / "config.yaml"))
    try:
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return set()
    out = set()
    for entry in data.get("model_list") or []:
        upstream = (entry.get("litellm_params") or {}).get("model", "")
        # `openai/mlx-community/Qwen2.5-7B-Instruct-4bit` -- the first segment
        # is the provider LiteLLM routes through, not part of the id.
        parts = upstream.split("/")
        if len(parts) >= 3:
            out.add("/".join(parts[1:]).lower())
        elif upstream:
            out.add(upstream.lower())
    return out


def lanes_with_receipts() -> set[str]:
    """Lanes for which a run receipt exists on disk.

    NAMED FOR WHAT IT READS, not for what it means. Early measurements in this
    project were written to a gitignored directory before receipts existed, and
    the French TTS ranking is one of them -- so an absent lane means "no
    receipt here", which is weaker than "never measured" and must not be
    reported as the stronger claim.

    Read from what the runs wrote rather than from a list, for the same reason
    discover.measured() is: a hand-kept list of what has been measured, beside
    the thing that measures, drifts within a week.

    A lane with an incumbent is a lane where a screen answers "is this better
    than what we have"; a lane without one is where it answers "does anything
    here work at all", and the second is worth more.
    """
    import json
    from harness import paths
    lanes: set[str] = set()
    try:
        receipts = sorted(paths.runs().rglob("results.json"))
    except OSError:
        return lanes
    for f in receipts:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue        # a half-written run must not stop the ranking
        receipt = data.get("receipt") or {}
        lane = (receipt.get("modality") or "").strip().lower()
        # Only a run that actually produced rows counts. An empty receipt is a
        # run that was started, not a lane that was measured.
        if lane and (data.get("summary") or data.get("rows")):
            lanes.add(lane)
    return lanes
