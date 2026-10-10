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

from harness import lanes

GIB = 1024 ** 3

#: What each signal is worth. Weights are a policy, so they live in one place
#: and the test that pins the ordering names them.
RECURRENCE = 2.0
OPEN_LANE = 3.0
KNOWN_LINEAGE = -4.0
NO_LANE = -6.0
#: A lane somebody decided not to run here. BELOW no-lane on purpose: a
#: laneless candidate might get a lane tomorrow, while a parked one is waiting
#: on hardware. #244 taught the report and verify about parking and never
#: reached the ranking, so four of the top twelve were video candidates for a
#: lane nothing will run, and `--loop --run --top 4` would have downloaded
#: 16.2 GiB for it.
PARKED_LANE = -8.0
CHEAP = 1.0

def parents(row) -> set[str]:
    """The row's base_model parents, from the lineage rows inspect wrote. #414."""
    return {str(p[0] if isinstance(p, (tuple, list)) else p).strip().lower()
            for p in (row.get("parents") or []) if p}


def value(row: dict, *, serving: set[str] = frozenset(),
          measured_lanes: set[str] = frozenset(),
          ceiling_gib: float) -> tuple[float, list[str]]:
    """What a screen would teach us about this candidate, and why.

    The reasons travel with the number. A ranking whose rows cannot say why
    they are where they are is a ranking nobody can argue with, and every
    metric in this project that went wrong went wrong quietly.
    """
    score, why = 0.0, []

    #: A feed source declares `all` to mean it covers every lane. That is a
    #: fact about the SOURCE, and 243 proposals carry it as though it were
    #: theirs. lane_of() is the one place that knows, rather than the second
    #: copy of the rule that used to live here. #207.
    lane = lane_of(row)
    if not lane:
        # NOTHING HERE CAN MEASURE IT. Not a judgement about the model: the
        # eval suite has no case, no runner and no metric for it, which is a
        # gap in this harness and sometimes the work worth doing.
        score += NO_LANE
        why.append("no lane can measure it")
    elif lanes.parked(lane)[0]:
        # NOT DROPPED FROM THE QUEUE. When the Studio arrives the lane
        # un-parks and these candidates should still be here with their
        # recurrence intact, so this is a ranking answer and never a stored
        # verdict -- the same reasoning that keeps unrunnable() out of the
        # store, because the answer is a fact about this machine.
        score += PARKED_LANE
        why.append(f"the {lane} lane is parked: {lanes.parked(lane)[1]} "
                   f"would change that")
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

    known = {p for p in parents(row)
             if any(p == s or p in s or s in p for s in serving)}
    if known:
        score += KNOWN_LINEAGE
        why.append(f"a requant of {', '.join(sorted(known))}, already served")

    # The measured column, never the card prose. #413.
    gib = int(row.get("size_bytes") or 0) / GIB
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
LANE_PRIORITY = lanes.WANTED

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
    """The lane this row can be tested in, normalised. Empty means none.

    `text` used to be discarded here alongside `all`, which dropped 10
    proposals that ARE the code lane under its older name. Issue #207.
    """
    return lanes.canonical(row.get("lane"))


def rank(rows, *, serving=(), measured_lanes=(), ceiling_gib: float | None = None,
         keep_laneless: bool = False):
    """Best first. Ties break on name so the order is stable across runs.

    A candidate NO LANE CAN TEST IS DROPPED rather than ranked last. Penalising
    it left 66 of 94 queue slots holding things nothing could measure, which
    crowds out the candidates a screen could actually answer a question about.
    Most are tools rather than models -- mflux, mlx-vlm, nativ -- and no lane
    tests a library. See wanted() for what to do with them instead. Issue #201.

    AN ADAPTER IS DROPPED FOR THE SAME REASON. screen.is_attachment() already
    refuses a LoRA, a ComfyUI node pack or a workflow at screen time, because
    `mflux:org/some-style-lora` downloads gigabytes and then fails. Until #207
    it refused them only AFTER they had taken the top of the queue: the first
    six rows were video LoRAs and the next three ComfyUI packs, so the ranking
    spent its whole first page on things the next tier would not run.
    """
    from harness import screen
    if ceiling_gib is None:
        # This machine's ceiling, never a literal from one 32 GB box. #355, #415.
        from harness import memory_store as ms
        ceiling_gib = ms.this_machine()["ceiling_gb"]
    serving = {s.lower() for s in serving}
    measured = {m.lower() for m in measured_lanes}
    out = []
    for row in rows:
        if not keep_laneless and not lane_of(row):
            continue
        # The card's word, stored by inspect; and the NAME, which carries it
        # as often: `...-lora-I2V` and `...-ComfyUI` say what they are. #414.
        # Unless the lane has an engine that loads it onto its base. #423.
        kind = row.get("attaches_to") or screen.is_attachment(row.get("name"))
        if kind and not screen.takes_attachment(lane_of(row), kind):
            continue
        if unrunnable(row):
            continue
        if is_technique(row):
            continue
        got, why = value(row, serving=serving, measured_lanes=measured,
                         ceiling_gib=ceiling_gib)
        out.append({**row, "value": got, "value_why": "; ".join(why)})
    # THE TIEBREAK IS ABOUT THE CANDIDATE, NEVER ITS NAME. Ten code candidates
    # scored an identical +3.7, so the order was the alphabet: `AxiomicLabs`
    # won every run and `z-lab` never did. A STABLE arbitrary tiebreak is worse
    # than a random one, because it makes the tail of the queue unreachable
    # rather than merely last -- no amount of re-running ever gets there.
    #
    # Freshness breaks the tie instead: the loop works THROUGH a backlog over
    # successive runs rather than re-reading its first page. `name` stays as
    # the final term so one run's order is reproducible. #252.
    return sorted(out, key=lambda r: (-r["value"],
                                      -float(r.get("last_seen") or 0.0),
                                      r["name"]))


def unrunnable(row: dict, machine=None) -> str:
    """The runtime this row needs that this machine lacks, or "".

    DROPPED AT RANK TIME RATHER THAN SETTLED IN THE STORE, because the answer
    is a fact about the MACHINE and the store is shared. 28 GGUF proposals sit
    in the queue here, where the text lane is served by mlx_lm.server and
    nothing loads a GGUF; on a machine that builds llama.cpp the same rows are
    perfectly runnable, and writing `needs-llamacpp` into the store from here
    would answer for that machine too.

    The inspect tier records the requirement for rows it reads from now on
    (#228). This catches the ones inspected before the probe existed, and
    re-asks on every run rather than freezing the answer.
    """
    from harness import inspect as ins

    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()
    if ins.is_gguf(row.get("name") or "", {}):
        return machine.refuses("llamacpp") or ""
    # THE CARD SAYS WHAT IT NEEDS, and until #245 only GGUF was read. Four
    # candidates tagged cuda, gemlite, nvfp4 and modelopt ranked, and would
    # have been fetched and handed to a runner that cannot load them -- with
    # the screen recording a verdict about the CANDIDATE. The same class as
    # #228, whose fix was written for exactly one format.
    needs = row.get("runtime_needed") or ""
    if needs:
        return machine.refuses(needs) or ""
    return ""


def wanted(rows, minimum: int = 2) -> list[dict]:
    """Laneless candidates that keep coming back, most-seen first.

    A LANE IS A PERSON'S DECISION. Dropping these silently would throw away the
    signal that the harness is missing something; inventing a lane for them
    would be the harness deciding what it is for. So they are reported, and
    somebody chooses. Issue #201.
    """
    out = [r for r in _laneless(rows, minimum) if not is_tool(r) and not is_technique(r)]
    return sorted(out, key=lambda r: (-int(r.get("times") or 0), r["name"]))


def _laneless(rows, minimum: int) -> list[dict]:
    """Recurring laneless rows that are not attachments, as rank() drops them. #556."""
    from harness import screen
    return [r for r in rows
            if not lane_of(r) and int(r.get("times") or 0) >= minimum
            and not (r.get("attaches_to") or screen.is_attachment(r.get("name")))]


def is_tool(row) -> bool:
    """A GitHub repo with no model task: an engine or tool, not a model. #556.
    A repo that implements a paper is a technique instead. #576."""
    from harness.memory_store.schema import GITHUB
    return (row.get("registry") == GITHUB and not (row.get("hf_task") or "").strip()
            and not is_technique(row))


def is_technique(row) -> bool:
    """A method for a lane, from a paper or a repo implementing one: never a model to screen. #576."""
    from harness.memory_store.schema import TECHNIQUE
    return (row.get("category") or "") == TECHNIQUE


def technique_route(row) -> str:
    """A technique's lane or bucket; only code read from prose is re-read, its old evidence was any LLM word. #631."""
    lane = lane_of(row)
    if lane and not (lane == "code" and (row.get("lane_source") or "") == "prose"):
        return lane
    return lanes.technique_route(row.get("title") or "", row.get("description") or "")


def _by_seen(members) -> list[dict]:
    return sorted(members, key=lambda r: (-int(r.get("times") or 0),
                                          -float(r.get("last_seen") or 0.0), r["name"]))


def techniques_wanted(rows, want: str = "") -> list[dict]:
    """Laned techniques grouped by lane, most-seen first. Evidence only: a method is
    implemented by a person or an agent, never guessed. #576."""
    groups: dict[str, list[dict]] = {}
    for r in rows:
        lane = technique_route(r)
        if lane not in lanes.ALL or (want and not lanes.serves(lane, want)):
            continue
        groups.setdefault(lane, []).append(r)
    out = [{"lane": lane, "techniques": _by_seen(members),
            "sightings": sum(int(m.get("times") or 0) for m in members)}
           for lane, members in groups.items()]
    return sorted(out, key=lambda g: (-len(g["techniques"]), -g["sightings"], g["lane"]))


def techniques_unlaned(rows) -> dict[str, list[dict]]:
    """Techniques in no lane: {SERVING: [...], GENERAL: [...], "": unrouted}, most-seen first. #631."""
    out: dict[str, list[dict]] = {lanes.SERVING: [], lanes.GENERAL: [], "": []}
    for r in rows:
        got = technique_route(r)
        if got not in lanes.ALL:
            out[got].append(r)
    return {k: _by_seen(v) for k, v in out.items()}


def techniques_laneless(rows) -> int:
    """How many techniques route to no lane. #576."""
    return sum(len(v) for v in techniques_unlaned(rows).values())


def tools_wanted(rows, minimum: int = 2) -> list[dict]:
    """Recurring tooling repos: an engine entry is the decision, not a lane. #556."""
    out = [r for r in _laneless(rows, minimum) if is_tool(r)]
    return sorted(out, key=lambda r: (-int(r.get("times") or 0), r["name"]))


#: Tasks with an objective reference-based metric, so a negative control can
#: exist. Anything unlisted is unknown, not unmeasurable. #558.
MEASURABLE = {"translation": "chrF", "ocr": "CER", "image-to-text": "CER",
              "text-ranking": "recall@k", "text-retrieval": "recall@k",
              "retrieval": "recall@k", "token-classification": "F1"}

#: The group for a card that names no task anywhere. #558.
NO_TASK = "(no task on the card)"

_TASK_SUFFIXES = ("-classification", "-ranking", "-retrieval", "-detection",
                  "-segmentation", "-extraction", "-similarity", "-answering",
                  "-generation", "-estimation", "-forecasting")


def metric_for(task: str) -> str:
    return MEASURABLE.get(task, "")


def _tags(row) -> list[str]:
    import json
    got = row.get("card_tags") or []
    if isinstance(got, str):
        try:
            got = json.loads(got or "[]")
        except ValueError:
            got = []
    return [str(t).strip().lower() for t in got if str(t).strip()]


def task_of(row) -> str:
    """The task a laneless row is about: OCR, its hf_task, else a task tag. #558."""
    from harness import inspect as ins
    tags = _tags(row)
    if ins.is_ocr(tags):
        return "ocr"
    task = (row.get("hf_task") or "").strip().lower()
    if task:
        return task
    for t in tags:
        if t in MEASURABLE or "-to-" in t or t.endswith(_TASK_SUFFIXES) \
                or t in ("translation", "summarization", "fill-mask"):
            return t
    return NO_TASK


def wanted_groups(rows, *, ceiling_gib: float, minimum: int = 1) -> list[dict]:
    """Laneless models grouped by task with the evidence for a lane. Decides
    nothing: a lane is a person's decision. #558."""
    groups: dict[str, list[dict]] = {}
    for r in wanted(rows, minimum=minimum):
        groups.setdefault(task_of(r), []).append(r)
    out = []
    for task, members in groups.items():
        sizes = [int(m.get("size_bytes") or 0) / GIB for m in members]
        sized = [s for s in sizes if s > 0]
        fits = ("unknown" if not sized
                else "yes" if any(s <= ceiling_gib for s in sized) else "no")
        out.append({"task": task, "models": sorted(m["name"] for m in members),
                    "publishers": sorted({m["name"].split("/")[0].lower()
                                          for m in members}),
                    "sightings": sum(int(m.get("times") or 0) for m in members),
                    "fits": fits, "metric": metric_for(task)})
    return sorted(out, key=lambda g: (-len(g["models"]), -g["sightings"], g["task"]))


def runnerless(rows) -> list[dict]:
    """Candidates whose lane has a runner that cannot load them, with why.

    THE SAME ARGUMENT AS `wanted`, one rung along. A lane is a person's
    decision; so is an engine entry. These have a lane and a home, and the
    only thing between them and a screen is a line in engines.ENTRY_POINTS
    that nobody can write automatically -- guessing one yields a binary that
    rejects the model several seconds into loading, which is the reason that
    table is enumerated rather than inferred.
    """
    from harness import screen
    out = []
    for row in rows:
        lane = lane_of(row)
        if not lane:
            continue
        gap = screen.runner_gap(lane, row["name"], row.get("attaches_to") or "",
                                card=row)
        if gap:
            out.append({**row, "why_not": gap})
    return sorted(out, key=lambda r: (r.get("lane") or "", r["name"]))


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
        if entry.get("deprecated_for"):
            continue
        if entry.get("source_repo"):
            out.add(str(entry["source_repo"]).lower())
            continue
        upstream = (entry.get("litellm_params") or {}).get("model", "")
        # `openai/mlx-community/Qwen2.5-7B-Instruct-4bit` -- the first segment
        # is the provider LiteLLM routes through, not part of the id.
        parts = upstream.split("/")
        if len(parts) >= 3:
            out.add("/".join(parts[1:]).lower())
        elif upstream:
            out.add(upstream.lower())
    return out


def lanes_with_receipts(conn=None) -> set[str]:
    """Lanes with result rows in a stored run. #410.

    NAMED FOR WHAT IT READS, not for what it means. Early measurements in this
    project were written to a gitignored directory before receipts existed, and
    the French TTS ranking is one of them -- so an absent lane means "no
    receipt here", which is weaker than "never measured" and must not be
    reported as the stronger claim.

    Read from the stored runs rather than from a list, for the same reason
    discover.measured() is: a hand-kept list of what has been measured, beside
    the thing that measures, drifts within a week.

    A lane with an incumbent is a lane where a screen answers "is this better
    than what we have"; a lane without one is where it answers "does anything
    here work at all", and the second is worth more.
    """
    from harness import runs
    with runs.store(conn) as c:
        return runs.lanes(c)
