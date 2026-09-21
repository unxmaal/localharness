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

    KeepAlive on a job that RUNS AND EXITS restarts a finished sweep at once
    and the machine discovers in a tight loop, so the two keys must not be
    confused. Read from the generator rather than from an installed plist,
    which would make this a test of the developer's machine.
    """
    import subprocess
    import tempfile

    repo = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run([str(repo / "scripts" / "launchd.sh"), "generate", tmp],
                       check=True, capture_output=True, text=True)
        plists = {p.stem.rsplit(".", 1)[-1]: p.read_text(encoding="utf-8")
                  for p in Path(tmp).glob("*.plist")}

    assert "discover" in plists, (
        f"no agent runs discovery, so the loop turns only when somebody types "
        f"the command. Generated: {sorted(plists)}")
    sweep = plists["discover"]
    assert "StartInterval" in sweep and "KeepAlive" not in sweep, sweep

    # THE NEGATIVE CONTROL: a server must still be kept alive, or this fix
    # trades a sweep that never runs for model servers that never restart.
    assert "KeepAlive" in plists["mlx"]
    assert "StartInterval" not in plists["mlx"]


def test_the_scheduled_sweep_bounds_what_it_spends():
    """A scheduled job that downloads without a ceiling fills the disk
    overnight and is found the next morning."""
    body = (Path(__file__).resolve().parents[1] / "scripts"
            / "serve-discover.sh").read_text(encoding="utf-8")
    assert "--budget-gib" in body and "--top" in body, body
    assert "--run" in body, "a sweep that never spends discovers nothing"
