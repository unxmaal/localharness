"""A deterministic tier restating itself is one fact; a re-run after something
changed is a new one. Issues #184 and #225.

The skip used to match ANY earlier row of the tier, which made every retraction
permanent: a re-queued candidate would re-screen green, decide() would find the
old `screened` row three verdicts back, write nothing, and the retraction would
stay the latest verdict forever.
"""
import pytest

from harness import memory_store as ms


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, ms.Seen(name="org/x", source="t", url="", why="",
                            relevance=0, kind="candidate",
                            registry=ms.HUGGINGFACE, lane="code",
                            resolved="org/x"))
    yield conn
    conn.close()


def _rows(conn):
    return conn.execute(
        "SELECT tier, outcome, detail FROM verdicts ORDER BY id").fetchall()


def test_a_tier_restating_itself_back_to_back_writes_one_row(store):
    """#184: 128 of 671 rows were a deterministic tier repeating itself."""
    first = ms.decide(store, "org/x", "queued", tier="inspect", detail="fits")
    again = ms.decide(store, "org/x", "queued", tier="inspect", detail="fits")
    assert first == again
    assert len(_rows(store)) == 1


def test_a_repeat_after_an_intervening_verdict_is_a_new_fact(store):
    """The case that bit. Without this a retraction can never be undone."""
    ms.decide(store, "org/x", "screened", tier=ms.SCREEN, detail="1 passed")
    ms.decide(store, "org/x", "declined", tier="adopt", detail="lost")
    ms.decide(store, "org/x", "queued", tier=ms.SCREEN, detail="retracted")
    ms.decide(store, "org/x", "screened", tier=ms.SCREEN, detail="1 passed")
    assert [r["outcome"] for r in _rows(store)] == [
        "screened", "declined", "queued", "screened"]


def test_a_retracted_candidate_can_screen_green_again(store):
    """The behaviour the loop needs, stated as the loop sees it."""
    ms.decide(store, "org/x", "screened", tier=ms.SCREEN, detail="1 passed")
    ms.decide(store, "org/x", "queued", tier=ms.SCREEN, detail="retracted")
    assert not ms.survivors(store, limit=5)
    ms.decide(store, "org/x", "screened", tier=ms.SCREEN, detail="1 passed")
    assert [r["name"] for r in ms.survivors(store, limit=5)] == ["org/x"]


def test_a_differing_detail_is_always_a_new_row(store):
    ms.decide(store, "org/x", "queued", tier="inspect", detail="fits at 5 GiB")
    ms.decide(store, "org/x", "queued", tier="inspect", detail="fits at 6 GiB")
    assert len(_rows(store)) == 2


def test_a_run_is_always_a_new_row(store):
    """A run path means a real run happened, whatever it concluded."""
    ms.decide(store, "org/x", "measured", tier="adopt", detail="d",
              run_path="/runs/a")
    ms.decide(store, "org/x", "measured", tier="adopt", detail="d",
              run_path="/runs/b")
    assert len(_rows(store)) == 2
