"""A verdict from a run where nothing ran, retracted from the receipts. #230.

The schema 8 retraction NAMED one row, because I had verified it by hand. It
missed the second member of the same class. Naming rows does not scale past
the ones you happened to look at, so this derives the set from the receipts.
"""
import json

import pytest

from harness import memory_store as ms

REFUSED = "gateway returned HTTP 400: Invalid model name passed in model=org/x"


def _receipt(runs, name, summary, rows):
    d = runs / name
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"summary": summary,
                                                "rows": rows}),
                                    encoding="utf-8")


@pytest.fixture
def home(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.paths.home", lambda: tmp_path)
    (tmp_path / "runs").mkdir()
    return tmp_path


def _seed(conn, name, outcome, detail):
    ms.record(conn, ms.Seen(name=name, source="t", url="", why="",
                            relevance=0, kind="candidate",
                            registry=ms.HUGGINGFACE, lane="code",
                            resolved=name))
    ms.decide(conn, name, "queued", tier=ms.INSPECT, detail="bytes=1")
    ms.decide(conn, name, outcome, tier="adopt", detail=detail)


def _migrate(path):
    conn = ms.connect(path)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', '9')")
    conn.commit()
    conn.close()
    return ms.connect(path)


def _latest(conn, name):
    return conn.execute(
        "SELECT v.outcome, v.detail FROM verdicts v JOIN proposals p "
        "ON p.id = v.proposal_id WHERE p.name = ? ORDER BY v.id DESC LIMIT 1",
        (name,)).fetchone()


def test_a_decline_from_a_run_that_never_ran_is_retracted(home):
    conn = ms.connect(home / "d.db")
    _seed(conn, "org/never-ran", "declined", "does not beat the incumbent")
    conn.close()
    _receipt(home / "runs", "r1",
             {"org/never-ran": {"passed": 0, "total": 27},
              "incumbent": {"passed": 20, "total": 27}},
             [{"candidate": "org/never-ran", "detail": REFUSED}])

    conn = _migrate(home / "d.db")
    try:
        last = _latest(conn, "org/never-ran")
        assert last["outcome"] == "queued"
        assert "nothing ran" in last["detail"]
    finally:
        conn.close()


def test_a_genuine_loss_is_left_alone(home):
    """THE NEGATIVE CONTROL, and the one that matters: a candidate that ran
    and lost must stay settled, or the ladder never declines anything."""
    conn = ms.connect(home / "d.db")
    _seed(conn, "org/lost", "declined", "does not beat the incumbent")
    conn.close()
    _receipt(home / "runs", "r1",
             {"org/lost": {"passed": 4, "total": 27}},
             [{"candidate": "org/lost", "detail": ""}])

    conn = _migrate(home / "d.db")
    try:
        assert _latest(conn, "org/lost")["outcome"] == "declined"
    finally:
        conn.close()


def test_one_row_that_reached_a_model_makes_the_zero_the_candidates_own(home):
    """0 of 27 is not automatically ours. A candidate that produced output on
    any case was measured, and a bad score is its own."""
    conn = ms.connect(home / "d.db")
    _seed(conn, "org/mixed", "declined", "does not beat the incumbent")
    conn.close()
    _receipt(home / "runs", "r1",
             {"org/mixed": {"passed": 0, "total": 2}},
             [{"candidate": "org/mixed", "detail": REFUSED},
              {"candidate": "org/mixed", "detail": "the output closed no tag"}])

    conn = _migrate(home / "d.db")
    try:
        assert _latest(conn, "org/mixed")["outcome"] == "declined"
    finally:
        conn.close()


def test_an_engine_receipt_key_still_finds_its_proposal(home):
    """The receipt key is `mflux/<id>-q8` and the proposal is the bare id.
    Matching on equality alone would retract nothing for the image lane."""
    conn = ms.connect(home / "d.db")
    _seed(conn, "org/pic", "declined", "does not beat the incumbent")
    conn.close()
    _receipt(home / "runs", "r1",
             {"mflux/org/pic-q8": {"passed": 0, "total": 9}},
             [{"candidate": "mflux/org/pic-q8", "detail": REFUSED}])

    conn = _migrate(home / "d.db")
    try:
        assert _latest(conn, "org/pic")["outcome"] == "queued"
    finally:
        conn.close()


def test_no_runs_directory_is_not_an_error(tmp_path, monkeypatch):
    monkeypatch.setattr("harness.paths.home", lambda: tmp_path / "absent")
    conn = ms.connect(tmp_path / "d.db")
    _seed(conn, "org/x", "declined", "does not beat the incumbent")
    conn.close()
    conn = _migrate(tmp_path / "d.db")
    try:
        assert _latest(conn, "org/x")["outcome"] == "declined"
    finally:
        conn.close()


def test_a_short_name_cannot_match_every_key_that_spells_it(home):
    """The negative control for containment. A proposal with no `/` must not
    be retracted because some unrelated receipt key contains its letters."""
    conn = ms.connect(home / "d.db")
    _seed(conn, "pic", "declined", "does not beat the incumbent")
    conn.close()
    _receipt(home / "runs", "r1",
             {"mflux/someone/picture-q8": {"passed": 0, "total": 9}},
             [{"candidate": "mflux/someone/picture-q8", "detail": REFUSED}])

    conn = _migrate(home / "d.db")
    try:
        assert _latest(conn, "pic")["outcome"] == "declined"
    finally:
        conn.close()
