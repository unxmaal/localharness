"""Issue #53/#148: the rung between a ranked queue and a measurement."""
import json

import pytest

from harness import screen
from harness import memory_store as ms
from harness.memory_store import Seen


def rows(*specs):
    return [{"name": n, "lane": lane} for n, lane in specs]


# --- what it would do, and what stands in the way ------------------------

def test_a_candidate_whose_weights_are_absent_is_waiting_not_failed():
    """A SCREEN NEVER DOWNLOADS. Fetching is its own step with its own disk
    budget, and a tier that quietly pulls gigabytes because something ranked
    well is how a laptop fills up overnight."""
    got = screen.plan(rows(("org/m", "image")), missing=lambda n: [n])[0]
    assert got["state"] == screen.WAITING
    assert "lh fetch" in got["why_not"]


def test_a_lane_with_no_runner_is_named_rather_than_dropped():
    """A queue that silently drops what it cannot run looks like a queue that
    ran out."""
    got = screen.plan(rows(("org/v", "video")), missing=lambda n: [])[0]
    assert got["state"] == screen.NO_RUNNER
    assert "video" in got["why_not"]


def test_a_candidate_with_no_lane_says_so_in_its_own_terms():
    got = screen.plan(rows(("org/x", "")), missing=lambda n: [])[0]
    assert got["state"] == screen.NO_RUNNER
    assert "no lane" in got["why_not"]


def test_a_ready_candidate_gets_the_spec_its_lane_actually_takes():
    """The specs are the ones evals.run already parses. A second spelling of
    the candidate language is a defect this project has filed twice."""
    from evals.run import kind_of
    for lane, expect in (("image", "process"), ("stt", "stt"), ("tts", "tts"),
                         ("code", "gateway")):
        spec = screen.candidate_for(lane, "org/m")
        assert spec, lane
        assert kind_of(spec) == expect, (lane, spec)


def test_the_command_is_one_case_one_repeat_and_no_metrics():
    """A screen asks whether it ran. A quality metric here would invite ranking
    a screen against a measurement, which is two different exams."""
    row = screen.plan(rows(("org/m", "image")), missing=lambda n: [])[0]
    argv = screen.argv(row)
    assert "--screen" in argv
    assert argv[argv.index("--repeat") + 1] == "1"
    assert "--adherence" not in argv


# --- turning a run into a verdict ----------------------------------------

def test_a_screen_that_did_not_run_is_broken_and_terminal():
    got, why = screen.outcome(1, None)
    assert got == "broken"
    assert got in ms.TERMINAL, "a thing that does not run is answered"


def test_a_screen_that_ran_and_passed_nothing_is_broken():
    """seedvr2 crashed 0/3 and local-small never closed a tag 0/9. Exiting zero
    is not the same as working."""
    assert screen.outcome(0, {"c": {"pass": 0}})[0] == "broken"


def test_a_screen_that_passed_is_screened_and_not_terminal():
    got, _ = screen.outcome(0, {"c": {"pass": 2}})
    assert got == "screened"
    assert got not in ms.TERMINAL, "it ran; every measurement is still ahead"


def test_screened_is_a_verdict_the_store_recognises(tmp_path):
    """It has been in the vocabulary since the store was built and nothing has
    ever written one."""
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, Seen(name="org/m", source="recap", resolved="org/m"))
    ms.decide(conn, "org/m", "screened", tier=ms.SCREEN, detail="1 passed")
    row = conn.execute("SELECT outcome, tier FROM verdicts").fetchone()
    conn.close()
    assert (row["outcome"], row["tier"]) == ("screened", "screen")


# ---- ready means loadable, not merely present (issue #196) -----------------

def test_a_model_whose_config_names_an_absent_repo_is_not_ready():
    """Marvis-AI's 8-bit MLX repo is complete -- every symlink resolving, no
    .incomplete files -- and names a tokenizer in a DIFFERENT repo that is not
    on disk. `ready` on that cost a real run to discover."""
    got = screen.plan(rows(("org/m", "tts")),
                      missing=lambda n: ["org/tokenizer"])[0]
    assert got["state"] == screen.WAITING
    assert "org/tokenizer" in got["why_not"], "say WHAT to fetch"


def test_a_model_missing_only_itself_reads_as_a_plain_fetch():
    """The ordinary case must keep its ordinary wording, or every
    waiting-on-fetch row starts shouting about dependencies."""
    got = screen.plan(rows(("org/m", "tts")), missing=lambda n: [n])[0]
    assert got["state"] == screen.WAITING
    assert got["why_not"] == "weights are not on disk; lh fetch --run"


def test_a_model_with_everything_present_is_ready():
    """The negative half: a closure check that never passes screens nothing."""
    got = screen.plan(rows(("org/m", "tts")), missing=lambda n: [])[0]
    assert got["state"] == screen.READY


def test_a_lane_with_no_runner_is_not_asked_about_weights():
    """Ordering matters: a missing runner is the answer regardless of disk, and
    asking the cache first would touch the filesystem for every unrunnable row."""
    asked = []

    def missing(n):
        asked.append(n)
        return []

    got = screen.plan(rows(("org/x", "")), missing=missing)[0]
    assert got["state"] == screen.NO_RUNNER
    assert asked == [], "no runner, so the cache was never consulted"
