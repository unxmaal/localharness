"""Verdicts a program cannot reach. Issue #273.

Deliberately NOT a research protocol: one person judges their own project, so
there is no panel, no inter-rater agreement and no self-consistency control.
What is tested is that an answer means what it says and survives a restart.
"""
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import human, lanes  # noqa: E402


@pytest.fixture(autouse=True)
def store(tmp_path, monkeypatch):
    """Never write the developer's real answers. RULE #249: pin the ambient."""
    monkeypatch.setattr(human.paths, "home", lambda: tmp_path)
    return tmp_path


RECEIPT = {"rows": [
    {"case_id": "fox#1", "candidate": "alpha", "passed": True,
     "artifact": "/runs/a-fox.png"},
    {"case_id": "fox#2", "candidate": "alpha", "passed": True,
     "artifact": "/runs/a-fox2.png"},
    {"case_id": "fox#1", "candidate": "beta", "passed": True,
     "artifact": "/runs/b-fox.png"},
    {"case_id": "sign#1", "candidate": "alpha", "passed": True,
     "artifact": "/runs/a-sign.png"},
    {"case_id": "sign#1", "candidate": "beta", "passed": True,
     "artifact": "/runs/b-sign.png"},
]}


# --- what a person is asked ------------------------------------------------

def test_candidates_are_compared_within_a_case_not_across():
    """Two different prompts are two different questions. Preferring a fox to
    a shop sign says nothing about either model."""
    got = human.pairings(RECEIPT)
    assert {p["case"] for p in got} == {"fox", "sign"}
    for p in got:
        assert p["a"] != p["b"]


def test_a_repeat_of_one_candidate_is_not_a_pairing_against_itself():
    """--repeat gives several rows per candidate per case. Asking someone to
    compare alpha with alpha is a question with no answer."""
    for p in human.pairings(RECEIPT):
        assert {p["a"], p["b"]} == {"alpha", "beta"}


def test_a_failed_run_is_not_offered_for_comparison():
    """A blank canvas beside a real image is not a preference, it is a bug
    already caught by the programmatic checks."""
    receipt = {"rows": [
        {"case_id": "fox#1", "candidate": "alpha", "passed": True,
         "artifact": "/a.png"},
        {"case_id": "fox#1", "candidate": "beta", "passed": False,
         "artifact": "/b.png"}]}
    assert human.pairings(receipt) == []


# --- what an answer means --------------------------------------------------

def test_asking_b_against_a_is_the_same_question_as_a_against_b():
    """Otherwise the sides randomising per showing would split one pairing's
    answers into two piles that never reach ENOUGH."""
    assert (human.key("music", "c", "alpha", "beta")
            == human.key("music", "c", "beta", "alpha"))


def test_a_verdict_needs_enough_answers():
    """One is a draw, not a measurement: every diffusion lane here varies run
    to run, and ACE-Step does so at a pinned seed (RULE #280)."""
    for _ in range(human.ENOUGH - 1):
        human.record("music", "c", "alpha", "beta", "a")
        assert human.decided("music", "c", "alpha", "beta") is None
    human.record("music", "c", "alpha", "beta", "a")
    assert human.decided("music", "c", "alpha", "beta") == "alpha"


def test_a_plurality_wins_rather_than_a_majority():
    """Two for alpha and one tie is a preference for alpha, not a stalemate."""
    human.record("music", "c", "alpha", "beta", "a")
    human.record("music", "c", "alpha", "beta", "tie")
    human.record("music", "c", "alpha", "beta", "a")
    assert human.decided("music", "c", "alpha", "beta") == "alpha"


def test_a_split_decision_is_decided_as_no_preference():
    """Three answers that disagree IS the finding. Asking a fourth time to
    break the tie is fishing for the answer you wanted."""
    human.record("music", "c", "alpha", "beta", "a")
    human.record("music", "c", "alpha", "beta", "b")
    human.record("music", "c", "alpha", "beta", "tie")
    assert human.decided("music", "c", "alpha", "beta") == ""


def test_cannot_tell_is_a_permitted_answer():
    """Not an evasion. The image lane has already recorded a statistical tie
    as a legitimate verdict, and forcing a choice manufactures a winner out
    of noise."""
    for _ in range(human.ENOUGH):
        human.record("music", "c", "alpha", "beta", "tie")
    assert human.decided("music", "c", "alpha", "beta") == ""


def test_an_unknown_answer_is_refused():
    with pytest.raises(ValueError, match="answer must be"):
        human.record("music", "c", "alpha", "beta", "maybe")


# --- it has to survive being closed ---------------------------------------

def test_answers_survive_a_restart(store):
    human.record("music", "c", "alpha", "beta", "a")
    assert json.loads((store / "human-verdicts.json")
                      .read_text(encoding="utf-8"))
    assert human.tally("music", "c", "alpha", "beta")["alpha"] == 1


def test_a_corrupt_store_does_not_take_the_command_down(store):
    """It is a JSON file on a laptop. Losing answers is bad; refusing to run
    because of a half-written file is worse."""
    (store / "human-verdicts.json").write_text("{not json",
                                                 encoding="utf-8")
    assert human.tally("music", "c", "alpha", "beta") == {}


# --- which lanes need a person --------------------------------------------

def test_a_human_judged_lane_still_runs_its_programmatic_checks():
    """music keeps its sung-lyric WER and duration adherence. What it gains
    is a verdict for the part those cannot see."""
    assert lanes.human_judged("music")
    assert not lanes.human_judged("code")
    assert set(lanes.HUMAN_JUDGED) <= set(lanes.ALL)


def test_pending_shuffles_which_side_is_shown_first(monkeypatch):
    """Knowing which one is the incumbent is the cheapest way to confirm what
    you already believed. Costs nothing to avoid."""
    pairs = human.pairings(RECEIPT)
    monkeypatch.setattr(human.random, "random", lambda: 0.9)
    right_first = human.pending("image", pairs)[0]["left"]
    monkeypatch.setattr(human.random, "random", lambda: 0.1)
    left_first = human.pending("image", pairs)[0]["left"]
    assert right_first != left_first


def test_a_settled_pairing_stops_being_offered():
    pairs = human.pairings(RECEIPT)
    p = pairs[0]
    for _ in range(human.ENOUGH):
        human.record("image", p["case"], p["a"], p["b"], "a")
    assert p["case"] not in {q["case"] for q in human.pending("image", pairs)}
