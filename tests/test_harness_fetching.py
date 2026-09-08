"""Issue #62: the durable download queue."""
import pytest

from harness import fetching as f
from harness import memory_store as ms


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def seen(db, name):
    ms.record(db, ms.Seen(name=name, source="inspect", resolved=name))


# ---- what may be downloaded ------------------------------------------------

def test_an_unknown_size_is_refused_rather_than_attempted():
    """The point of inspecting before fetching is not to find out how big
    something is by downloading it."""
    assert not f.plan("a/b", 0).ok
    assert not f.plan("a/b", -1).ok


def test_a_download_that_would_fill_the_volume_is_refused():
    got = f.plan("a/b", 20 * f.GIB, free=60 * f.GIB)
    assert not got.ok and "floor" in got.why


def test_a_download_over_the_cap_is_refused_however_much_disk_there_is():
    assert not f.plan("a/b", 500 * f.GIB, free=5000 * f.GIB).ok


def test_a_download_that_fits_is_allowed():
    assert f.plan("a/b", 20 * f.GIB, free=900 * f.GIB).ok


# ---- the queue -------------------------------------------------------------

def test_only_the_inspect_tier_can_queue_a_download(db):
    """The JUDGE tier also writes `queued`, and it judges a DESCRIPTION: it
    queued a 122B model on a 32 GB machine. Nothing is downloaded on prose."""
    seen(db, "org/judged")
    seen(db, "org/inspected")
    ms.decide(db, "org/judged", "queued", tier="judge", score=9)
    ms.decide(db, "org/inspected", "queued", tier="inspect")
    assert [r["name"] for r in f.queued(db)] == ["org/inspected"]


def test_a_terminal_verdict_leaves_the_queue(db):
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")
    ms.decide(db, "org/a", "declined", tier="fetch", detail="too big")
    assert f.queued(db) == []


def test_the_queue_is_ordered_by_score(db):
    for name, score in [("org/low", 4), ("org/high", 9)]:
        seen(db, name)
        ms.decide(db, name, "queued", tier="inspect", score=score)
    assert [r["name"] for r in f.queued(db)] == ["org/high", "org/low"]


# ---- running it ------------------------------------------------------------

def test_one_at_a_time_by_default(db):
    calls = []
    for name in ["org/a", "org/b"]:
        seen(db, name)
        ms.decide(db, name, "queued", tier="inspect")
    f.run(db, {"org/a": 2 * f.GIB, "org/b": 2 * f.GIB},
          snapshot=lambda repo_id: calls.append(repo_id) or "/tmp/x",
          free=900 * f.GIB)
    assert len(calls) == 1


def test_something_too_big_is_declined_permanently(db):
    """Terminal, so the same oversized model is not re-queued every sweep."""
    seen(db, "org/huge")
    ms.decide(db, "org/huge", "queued", tier="inspect")
    f.run(db, {"org/huge": 500 * f.GIB}, snapshot=lambda **kw: "/x",
          free=900 * f.GIB)
    assert "org/huge" in ms.settled(db)


def test_a_failed_download_does_not_condemn_the_candidate(db):
    """A network error says nothing about the model."""
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")

    def boom(repo_id):
        raise RuntimeError("connection reset")

    got = f.run(db, {"org/a": 2 * f.GIB}, snapshot=boom, free=900 * f.GIB)
    assert got[0]["ok"] is False
    assert "org/a" not in ms.settled(db)


def test_a_finished_download_records_where_it_landed(db):
    seen(db, "org/a")
    ms.decide(db, "org/a", "queued", tier="inspect")
    f.run(db, {"org/a": 2 * f.GIB}, snapshot=lambda repo_id: "/Volumes/Models/hf/a",
          free=900 * f.GIB)
    row = db.execute("SELECT run_path FROM verdicts WHERE tier='fetch'").fetchone()
    assert row["run_path"] == "/Volumes/Models/hf/a"
