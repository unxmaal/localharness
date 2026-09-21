"""The whole circuit, in a fake world, in under a second. Issue #261.

The product is obsoletion-proofing: scheduled discovery finds something new,
evaluates it, sifts for gold, tests the gold, and replaces the incumbent when a
challenger genuinely wins. The loop has turned for days and has never completed
that circuit. Every defect in the way was found by running the real thing for
minutes to hours, and each was hidden behind the one before it.

This drives sweep -> inspect -> rank -> fetch -> screen -> measure -> adopt with
no weights, no gateway, no model server and no HuggingFace, and asserts the
thing that has never happened: THE LANE SERVES SOMETHING DIFFERENT AFTERWARDS.

Where the ladder is still broken these tests say so as xfail(strict=True), so
the day a fix lands the mark fails and gets deleted rather than rotting.
"""
import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakes  # noqa: E402

from harness import adopt, fetching, lanes, rank, screen  # noqa: E402
from harness import memory_store as ms  # noqa: E402


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "s.db")
    yield conn
    conn.close()


def summary_row(passed, total, metric, value):
    """A receipt summary row as evals.core.summarize writes one.

    `pass_rate` is the field winners.order_key ranks on, and it is easy to
    leave out of a hand-written fixture because `passed`/`total` look like
    enough. Without it every candidate ranks at 0.0 and no comparison can ever
    find a winner. Checked against a real receipt rather than assumed.
    """
    return {"passed": passed, "total": total,
            "pass_rate": round(passed / total, 3), "median_s": 1.0,
            "metrics": {metric: value}, "case_ids": [f"c{i}" for i in
                                                     range(total)]}


def rows(candidate, passes, total):
    return [{"case_id": f"c{i}", "candidate": candidate, "passed": i < passes}
            for i in range(total)]


# --- the closing step, which has never run outside a unit test ------------

def test_a_winner_becomes_what_the_lane_serves(store):
    """THE CIRCUIT. Four adopt verdicts exist in the real store and all four
    are declines, so nothing has ever been installed. A challenger that wins
    on the metric AND on the paired test must change what the lane serves,
    with no restart and no edit to a source file.
    """
    inc = {**summary_row(4, 27, "code_pass", 0.40), "candidate": "q3-4b"}
    ch = {**summary_row(24, 27, "code_pass", 0.91),
          "candidate": "org/challenger"}
    verdict = adopt.decide("code", inc, ch,
                           rows("q3-4b", 4, 27) + rows("org/challenger", 24, 27))
    assert verdict.adopt, verdict.why

    before = adopt.default_for("code", "q3-4b", conn=store)
    adopt.record(store, verdict)
    after = adopt.default_for("code", "q3-4b", conn=store)

    assert before == "q3-4b"
    assert after == "org/challenger", (
        f"the lane still serves {after!r} after adopting {verdict.challenger!r}; "
        f"the loop cannot change anything")


def test_a_loser_is_recorded_so_it_is_not_re_proposed(store):
    """Three of the four real adopt rows are the SAME model measured three
    times. A loss that is not remembered is a loop that re-answers itself."""
    inc = {**summary_row(20, 27, "code_pass", 0.89), "candidate": "q3-4b"}
    ch = {**summary_row(4, 27, "code_pass", 0.53), "candidate": "org/loser"}
    verdict = adopt.decide("code", inc, ch,
                           rows("q3-4b", 20, 27) + rows("org/loser", 4, 27))
    assert not verdict.adopt

    fakes.seeded_store(store, [("org/loser", "code", 0.7, 2)])
    adopt.record(store, verdict)
    last = store.execute(
        "SELECT v.outcome FROM verdicts v JOIN proposals p ON p.id = v.proposal_id "
        "WHERE p.name = ? ORDER BY v.id DESC LIMIT 1", ("org/loser",)).fetchone()
    assert last["outcome"] in ms.TERMINAL, (
        f"a measured loss recorded as {last['outcome']!r} is not terminal, so "
        f"the same candidate returns on the next sweep")


def test_an_adoption_survives_a_new_connection(tmp_path):
    """A STORE FACT, not an edit to a source file. If the answer lived in
    memory the lane would forget it the moment the loop exited."""
    path = tmp_path / "s.db"
    conn = ms.connect(path)
    # 27 cases, not 9. Zero lost against five gained is p=0.06 and the gate
    # correctly refuses it: a clean sweep of nine is not yet significant. The
    # fixture has to clear the bar the product enforces, not the other way
    # round.
    inc = {**summary_row(6, 27, "ink", 0.10), "candidate": "local-large"}
    ch = {**summary_row(27, 27, "ink", 0.30), "candidate": "org/better-svg"}
    adopt.record(conn, adopt.decide(
        "svg", inc, ch,
        rows("local-large", 6, 27) + rows("org/better-svg", 27, 27)))
    conn.close()

    again = ms.connect(path)
    try:
        assert adopt.default_for("svg", "local-large", conn=again) == \
            "org/better-svg"
    finally:
        again.close()


# --- the funnel: 367 in, 6 measured --------------------------------------

def test_a_candidate_travels_from_sweep_to_the_screen(store, tmp_path):
    """Each tier's output must be the next tier's input. The real loop lost
    candidates at every seam: inspect ordered by corroboration while fetch
    read by rank, and the screen was sliced before it was filtered."""
    fakes.seeded_store(store, [("org/real-model", "code", 2.0, 3)])
    store.execute("UPDATE proposals SET description = ? WHERE name = ?",
                  ("task text-generation; served by mlx; tagged qwen3, chat; "
                   "2.0 GiB of weights", "org/real-model"))

    queued = fetching.queued(store, needs_lane=False)
    assert [r["name"] for r in queued] == ["org/real-model"]

    ranked = fetching.in_rank_order(queued, store)
    assert ranked, "the candidate did not survive ranking"

    downloads = fakes.Downloads(tmp_path / "hub")
    got = fetching.run(store, {"org/real-model": 2 * 1024 ** 3}, limit=1,
                       snapshot=downloads)
    assert downloads.asked == ["org/real-model"], got

    plan = screen.plan(ranked, missing=lambda name: [])
    assert plan[0]["state"] == screen.READY, plan[0]["why_not"]


def test_an_answered_candidate_does_not_come_back(store):
    """#253. A candidate screened `broken` came straight back round, and three
    of the four adopt verdicts in the real store are one model measured,
    declined, and measured again."""
    from harness.cli import _queueable

    fakes.seeded_store(store, [("org/answered", "code", 0.5, 2),
                               ("org/fresh", "code", 0.5, 2)])
    ms.decide(store, "org/answered", "broken", tier="screen",
              detail="it ran and passed nothing")
    names = [r["name"] for r in _queueable(store, "code")[0]]
    assert "org/fresh" in names, "the unanswered candidate went missing"
    assert "org/answered" not in names, names


def test_a_retraction_puts_a_candidate_back(store):
    """THE OTHER HALF, and the reason this reads the LATEST verdict rather
    than any. A retraction is an appended `queued` row over a terminal one;
    treating "has ever been terminal" as final would make every retraction
    permanent, which is the defect schema 7 exists to undo."""
    from harness.cli import _queueable

    fakes.seeded_store(store, [("org/retracted", "code", 0.5, 2)])
    ms.decide(store, "org/retracted", "broken", tier="screen",
              detail="it ran and passed nothing")
    assert "org/retracted" not in [r["name"] for r in _queueable(store, "code")[0]]
    ms.decide(store, "org/retracted", "queued", tier="screen",
              detail="retracted: that was a fact about this harness")
    assert "org/retracted" in [r["name"] for r in _queueable(store, "code")[0]]


def test_a_candidate_this_machine_cannot_run_is_answered(store, tmp_path):
    """#254. Tagged gemlite and cuda. rank.unrunnable drops it from the
    ordering so it never reached a fetch and never got a verdict, and every
    sweep re-ranked it forever."""
    from harness import inspect as ins

    name = "prism-ml/bonsai-image-binary-4B-gemlite-1bit"
    fakes.seeded_store(store, [(name, "image", 3.8, 2)])
    store.execute("UPDATE proposals SET description = ? WHERE name = ?",
                  (ins.card_description(fakes.card(name)), name))

    downloads = fakes.Downloads(tmp_path / "hub")
    got = fetching.run(store, {name: int(3.8 * 1024 ** 3)}, limit=1,
                       snapshot=downloads)
    assert downloads.asked == [], "weights were fetched for an unrunnable model"
    assert got and "no runtime" in got[0]["why"], got

    last = store.execute(
        "SELECT v.outcome, v.detail FROM verdicts v "
        "JOIN proposals p ON p.id = v.proposal_id "
        "WHERE p.name = ? ORDER BY v.id DESC LIMIT 1", (name,)).fetchone()
    assert last["outcome"] in ms.TERMINAL, last["outcome"]
    assert "needs-cuda" in last["detail"], (
        f"say WHICH runtime is missing, and on which machine, or a reader of "
        f"the shared store cannot tell a model fault from ours: {last['detail']}")


def test_a_runnable_candidate_is_still_fetched(store, tmp_path):
    """THE NEGATIVE CONTROL. A refusal that fires on an ordinary model empties
    the queue and reads exactly like a queue that ran out."""
    fakes.seeded_store(store, [("org/ordinary", "code", 2.0, 2)])
    store.execute("UPDATE proposals SET description = ? WHERE name = ?",
                  ("task text-generation; served by mlx; tagged qwen3; "
                   "2.0 GiB of weights", "org/ordinary"))
    downloads = fakes.Downloads(tmp_path / "hub")
    fetching.run(store, {"org/ordinary": 2 * 1024 ** 3}, limit=1,
                 snapshot=downloads)
    assert downloads.asked == ["org/ordinary"]


def test_the_queue_does_not_always_offer_the_same_head(store):
    """#252. Ten code candidates tied at +3.7 and the tiebreak was the
    alphabet, so the loop screened the same six every run and everything below
    the first page was unreachable rather than merely last.

    A STABLE ARBITRARY TIEBREAK IS WORSE THAN A RANDOM ONE. Freshness breaks
    it, so the loop works through a backlog instead of re-reading page one.
    """
    import time as _t

    now = _t.time()
    tied = [(f"org/{c}-model", "code", 0.5, 2) for c in "abcdefghij"]
    fakes.seeded_store(store, tied)
    # Make the ALPHABETICALLY LAST candidate the freshest sighting.
    store.execute(
        "UPDATE sightings SET seen_at = ? WHERE proposal_id = "
        "(SELECT id FROM proposals WHERE name = ?)", (now + 500, "org/j-model"))

    ranked = fetching.in_rank_order(
        fetching.queued(store, needs_lane=False), store)
    assert ranked[0]["name"] == "org/j-model", (
        f"the freshest candidate did not reach the head; order is "
        f"{[r['name'] for r in ranked[:3]]}")


def test_one_run_orders_the_queue_reproducibly(store):
    """THE NEGATIVE CONTROL for the above. Freshness must break ties, not make
    the order arbitrary: two reads of an unchanged store agree, or nothing
    downstream can be debugged."""
    fakes.seeded_store(store, [(f"org/{c}-model", "code", 0.5, 2)
                               for c in "abcde"])
    first = [r["name"] for r in fetching.in_rank_order(
        fetching.queued(store, needs_lane=False), store)]
    second = [r["name"] for r in fetching.in_rank_order(
        fetching.queued(store, needs_lane=False), store)]
    assert first == second


# --- the schedule ---------------------------------------------------------

def test_a_periodic_job_is_scheduled_not_kept_alive():
    """#261. Obsoletion-proofing that has to be typed by a person is a script,
    not a property of the system: what it protects against is precisely the
    passage of unattended time.

    READ, NOT RUN. The first version executed launchd.sh, which is WinError
    193 on Windows -- and the plist SHAPE is already asserted by
    test_launchd.py, gated to macOS because launchd is macOS. What belongs
    here is the part that is true everywhere: the sweep is a service the
    generator knows about, and it is one of the ones that runs and exits.
    """
    gen = (Path(__file__).resolve().parents[1]
           / "scripts" / "launchd.sh").read_text(encoding="utf-8")
    services = re.search(r'^SERVICES="([^"]+)"', gen, re.M).group(1).split()
    periodic = re.search(r"^declare -a PERIODIC=\(([^)]*)\)", gen, re.M)
    assert "discover" in services, (
        f"no agent runs discovery, so the loop turns only when somebody types "
        f"the command. Generated: {services}")
    assert periodic and "discover" in periodic.group(1), (
        f"KeepAlive on a job that RUNS AND EXITS restarts a finished sweep at "
        f"once and the machine discovers in a tight loop: {gen[:0] or periodic}")


def test_the_scheduled_sweep_bounds_what_it_spends():
    """A scheduled job that downloads without a ceiling fills the disk
    overnight and is found the next morning."""
    body = (Path(__file__).resolve().parents[1] / "scripts"
            / "serve-discover.sh").read_text(encoding="utf-8")
    assert "--budget-gib" in body and "--top" in body, body
    assert "--run" in body, "a sweep that never spends discovers nothing"


# --- a lane's runner must take THIS model, asked before the disk is spent ---

def test_a_model_no_runner_can_load_never_reaches_a_download(
        store, tmp_path, monkeypatch):
    """FOUND IN THE FIRST SCHEDULED SWEEP, which is the point of scheduling it.

    Two image candidates were downloaded, handed to evals.run, refused with
    "no cases of a modality it can run", and re-queued -- correctly, since
    that refusal is about the harness and not about the candidate. So they
    came back on the next sweep, and would have forever.

    THE LANE IS PINNED TO ONE ENGINE HERE, which is the state it was actually
    in when this happened. Since then the image lane grew a second engine that
    takes any repo id, so today nothing in it can be refused before a download
    -- and no other lane has a strict engine either, acestep passing its model
    through by design. Left reading the live table, this test would assert
    nothing and say so by passing. The guard is still the right question the
    moment any lane's only engine enumerates what it accepts, which is the
    normal shape: mflux has to, because every mflux binary shares one argument
    parser.
    """
    from harness import fetching, screen

    monkeypatch.setitem(screen.LANE_CANDIDATES, "image", ("mflux:{model}",))
    # NOT A REAL REPO. The candidate this was found on is now in the
    # developer's HF cache, so `queued` skipped it as present and the test
    # passed for no reason. RULE #249: pin the ambient fact.
    name = "org/supra-a2a-nano-exp-notreal"
    fakes.seeded_store(store, [(name, "image", 1.2, 2)])
    downloads = fakes.Downloads(tmp_path / "hub")
    got = fetching.run(store, {name: int(1.2 * 1024 ** 3)}, limit=1,
                       snapshot=downloads)
    assert downloads.asked == [], "weights nothing can load were downloaded"
    assert got and "no runner" in got[0]["why"], got
    # NOT TERMINAL. An engine entry would make this candidate runnable, so
    # declining it would settle a real candidate for a gap that is ours.
    latest = store.execute(
        "SELECT outcome FROM verdicts v JOIN proposals p ON p.id = v.proposal_id"
        " WHERE p.name = ? ORDER BY v.id DESC LIMIT 1", (name,)).fetchone()
    assert latest["outcome"] == "queued", latest["outcome"]


def test_a_model_a_runner_can_load_is_still_fetched(store, tmp_path):
    """THE NEGATIVE CONTROL, and the half that decides whether the guard can
    ship. A check that refuses every image candidate empties the queue and
    reads exactly like a queue with nothing in it.

    THE NAME IS NOT A REAL REPO, and that is deliberate. Written first as
    `filipstrand/Z-Image-Turbo-mflux-4bit`, the control failed on this machine
    for the wrong reason: those weights are in the developer's HF cache, so
    the fetch skipped them as already present and `asked` was empty either
    way. It would have passed on a runner with a cold cache and failed here,
    which is RULE #249 a third time -- pin the ambient fact, do not read it.
    The tail still names a family mflux serves, which is what is under test.
    """
    from harness import fetching

    name = "org/z-image-turbo-4bit-notreal"
    fakes.seeded_store(store, [(name, "image", 5.5, 2)])
    downloads = fakes.Downloads(tmp_path / "hub")
    fetching.run(store, {name: int(5.5 * 1024 ** 3)}, limit=1,
                 snapshot=downloads)
    assert downloads.asked == [name], downloads.asked


def test_a_runner_gap_is_reported_to_a_person(monkeypatch):
    """An engine entry cannot be invented -- a guessed mflux binary rejects
    the model several seconds into loading, which is why that table is
    enumerated. So the gap is surfaced rather than worked around, the same as
    a laneless candidate. Pinned to one engine for the reason above."""
    from harness import rank, screen

    monkeypatch.setitem(screen.LANE_CANDIDATES, "image", ("mflux:{model}",))
    rows = [{"name": "SupraLabs/Supra-A2A-Nano-Exp", "lane": "image",
             "description": ""},
            {"name": "flux2-klein-4b", "lane": "image", "description": ""}]
    got = rank.runnerless(rows)
    assert [r["name"] for r in got] == ["SupraLabs/Supra-A2A-Nano-Exp"], got
    assert "unknown mflux model" in got[0]["why_not"]


def test_the_engine_that_forced_all_this_still_refuses_what_it_cannot_run():
    """The load-bearing fact under the second engine. If mflux ever started
    accepting arbitrary repo ids, the fallback would be unnecessary and the
    two tests above would be testing a fiction."""
    from harness import screen

    assert screen.no_runner("mflux:SupraLabs/Supra-A2A-Nano-Exp")
    assert not screen.no_runner("mflux:flux2-klein-4b")


def test_the_screen_never_hands_a_runnerless_model_to_a_run():
    """THE LIVE REPRODUCTION. Both these candidates were planned `ready`, run,
    refused with "no cases of a modality it can run" -- a refusal about the
    harness, correctly non-terminal -- and re-queued, forever.

    THEY ARE `ready` AGAIN NOW, AND THAT IS THE FIX RATHER THAN A REGRESSION.
    The image lane has a second engine that takes an arbitrary repo id, so
    these reach a runner instead of a refusal. What must not come back is the
    spec that NOTHING resolves: `mflux:` was chosen for them because it was
    the lane's only spelling, and engines.resolve raises on it.
    """
    from harness import engines, screen

    rows = [{"name": "SupraLabs/Supra-A2A-Nano-Exp", "lane": "image",
             "description": ""},
            {"name": "Danrisi/UltraReal_FineTune_Anima_base1_v3",
             "lane": "image", "description": ""}]
    for row in screen.plan(rows):
        assert row["state"] != screen.NO_RUNNER, row
        engines.resolve(row["candidate"])      # raises if it is unrunnable


def test_the_image_lane_can_spell_more_than_one_engine():
    """#263. LANE_CANDIDATES held one template per lane, so `mflux` was the
    image engine because it had been TYPED there rather than because it had
    won anything -- and since its entry points are an enumerated table of
    families, every challenger discovery found was refused before it could be
    measured. The incumbent had never faced one.

    An mflux family still goes to mflux: a second engine that swallowed the
    incumbent's own models would change the lane by accident.
    """
    from harness import screen

    assert screen.candidate_for("image", "flux2-klein-4b") == \
        "mflux:flux2-klein-4b"
    assert screen.candidate_for("image", "stabilityai/sdxl-turbo") == \
        "diffusers:stabilityai/sdxl-turbo"


def test_every_engine_reaches_a_process_runner():
    """`diffusers` was absent from evals.run.PROCESS_ENGINES, a hand-written
    tuple of three, so kind_of("diffusers:org/m") answered `gateway` and an
    image spec was routed to the TEXT gateway. It had been registered in
    harness/engines.py since the CUDA machine was brought up and had never
    once reached a ProcessRunner, on any machine."""
    from evals.run import kind_of
    from harness import engines

    for name in engines.names():
        assert kind_of(f"{name}:whatever") == "process", name
