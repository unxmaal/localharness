"""Schema 16 retracts screen verdicts that carry no evidence. Issue #281."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import memory_store as ms  # noqa: E402


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "s.db")
    yield conn
    conn.close()


def seed(conn, name, lane="image"):
    ms.record(conn, ms.Seen(name=name, source="test", kind="weights",
                            lane=lane, why="seeded"))


def latest(conn, name):
    row = conn.execute(
        "SELECT v.outcome, v.detail FROM verdicts v JOIN proposals p "
        "ON p.id = v.proposal_id WHERE p.name = ? ORDER BY v.id DESC LIMIT 1",
        (name,)).fetchone()
    return (row[0], row[1]) if row else (None, None)


def rewind(conn, version=15):
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                 (str(version),))
    conn.commit()


def test_an_evidence_free_broken_screen_is_reopened(store, tmp_path):
    seed(store, "org/answered")
    ms.decide(store, "org/answered", "broken", tier=ms.SCREEN,
              detail="it ran and passed nothing || WARNING: 2154 MB of swap")
    rewind(store)
    store.close()
    again = ms.connect(tmp_path / "s.db")
    try:
        got, why = latest(again, "org/answered")
        assert got == "queued", f"a verdict with no evidence stayed terminal: {why}"
        assert "#281" in why
    finally:
        again.close()


def test_a_screen_that_recorded_its_receipt_is_left_alone(store, tmp_path):
    """THE NEGATIVE CONTROL. Retracting every broken screen would empty the
    answered set and the loop would re-run work it had already settled."""
    seed(store, "org/with-evidence")
    ms.decide(store, "org/with-evidence", "broken", tier=ms.SCREEN,
              detail="it ran and passed nothing: fox-snow: exit 1",
              run_path=str(tmp_path / "runs" / "screen-1"))
    rewind(store)
    store.close()
    again = ms.connect(tmp_path / "s.db")
    try:
        assert latest(again, "org/with-evidence")[0] == "broken"
    finally:
        again.close()


def test_a_candidate_screened_since_is_not_reopened(store, tmp_path):
    """The bad verdict is not the latest word, so it is already settled."""
    seed(store, "org/moved-on")
    ms.decide(store, "org/moved-on", "broken", tier=ms.SCREEN,
              detail="it ran and passed nothing")
    ms.decide(store, "org/moved-on", "screened", tier=ms.SCREEN,
              detail="1 case(s) passed a screen")
    rewind(store)
    store.close()
    again = ms.connect(tmp_path / "s.db")
    try:
        assert latest(again, "org/moved-on")[0] == "screened"
    finally:
        again.close()


def test_a_broken_screen_for_another_reason_is_left_alone(store, tmp_path):
    """`the screen exited N` was retracted on 2026-09-20 and is not this
    migration's business."""
    seed(store, "org/exited")
    ms.decide(store, "org/exited", "broken", tier=ms.SCREEN,
              detail="the screen exited 1")
    rewind(store)
    store.close()
    again = ms.connect(tmp_path / "s.db")
    try:
        assert latest(again, "org/exited")[0] == "broken"
    finally:
        again.close()


def test_running_the_migration_twice_does_not_stack_retractions(store, tmp_path):
    seed(store, "org/answered")
    ms.decide(store, "org/answered", "broken", tier=ms.SCREEN,
              detail="it ran and passed nothing")
    rewind(store)
    store.close()
    for _ in range(2):
        again = ms.connect(tmp_path / "s.db")
        again.close()
    again = ms.connect(tmp_path / "s.db")
    try:
        n = again.execute(
            "SELECT COUNT(*) FROM verdicts v JOIN proposals p "
            "ON p.id = v.proposal_id WHERE p.name = ? AND v.outcome = 'queued'",
            ("org/answered",)).fetchone()[0]
        assert n == 1, f"{n} retractions stacked"
    finally:
        again.close()
