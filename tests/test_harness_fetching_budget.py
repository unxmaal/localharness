"""--budget-gib is the ceiling on one invocation, and it was never enforced.

`lh discover --loop --run --budget-gib 12` printed the budget in the fetch
header and downloaded 24.6 GiB, because _loop_spend never passed it to
fetching.run. Issue #215.

A REAL STORE, NOT A FAKE CONNECTION. These first used a stub whose execute()
answered one query shape, which broke the moment queued() asked the store a
second question (#224). A fake that models one call tests that call.
"""
import pytest

from harness import fetching
from harness import memory_store as ms

GIB = fetching.GIB


def _seed(conn, name, gib, lane="image"):
    ms.record(conn, ms.Seen(name=name, source="test", url="", why="",
                            relevance=0, kind="candidate",
                            registry=ms.HUGGINGFACE, lane=lane, resolved=name))
    ms.decide(conn, name, "queued", tier="inspect",
              detail=f"bytes={int(gib * GIB)}")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fetching, "have", lambda *a, **k: False)
    monkeypatch.setattr(fetching, "download",
                        lambda name, snapshot=None: f"/x/{name}")
    monkeypatch.setattr(fetching, "requires", lambda name: [])
    monkeypatch.setattr("harness.rank.serving", lambda *a, **k: set())
    monkeypatch.setattr("harness.rank.lanes_with_receipts", lambda *a, **k: set())
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def _sizes(conn):
    return {r["resolved"] or r["name"]: fetching.size_of(r)
            for r in fetching.queued(conn)}


def test_the_budget_stops_the_second_download(store):
    _seed(store, "org/a-small", 5.5)
    _seed(store, "org/b-big", 19.1)
    got = fetching.run(store, _sizes(store), limit=5, free=500 * GIB,
                       budget=int(12 * GIB))
    assert [g["repo"] for g in got if g["ok"]] == ["org/a-small"]
    refused = [g for g in got if not g["ok"]]
    assert refused and "budget" in refused[0]["why"]


def test_exceeding_the_budget_re_queues_rather_than_settling(store):
    """A budget is a fact about THIS INVOCATION, not about the candidate.
    Recording it as declined would suppress a real model forever (#211)."""
    _seed(store, "org/big", 19.1)
    fetching.run(store, _sizes(store), limit=5, free=500 * GIB,
                 budget=int(12 * GIB))
    rows = store.execute(
        "SELECT v.outcome, v.detail FROM verdicts v JOIN proposals p "
        "ON p.id = v.proposal_id WHERE p.name = 'org/big' AND v.tier = 'fetch' "
        "ORDER BY v.id DESC LIMIT 1").fetchall()
    assert rows and rows[0][0] == "queued"
    assert "budget" in rows[0][1]


def test_no_budget_means_no_ceiling(store):
    """The negative control: absent the flag, nothing new refuses anything."""
    _seed(store, "org/a-small", 5.5)
    _seed(store, "org/b-big", 19.1)
    got = fetching.run(store, _sizes(store), limit=5, free=500 * GIB,
                       budget=None)
    assert sorted(g["repo"] for g in got if g["ok"]) == ["org/a-small",
                                                         "org/b-big"]


def test_an_unsized_row_is_not_charged_to_the_budget(store):
    """A size of 0 means unknown, and plan() refuses it for that reason. It
    must not silently pass a budget check by counting as free."""
    _seed(store, "org/unsized", 0)
    got = fetching.run(store, {"org/unsized": 0}, limit=5, free=500 * GIB,
                       budget=int(12 * GIB))
    assert not got[0]["ok"]
    assert "no measured size" in got[0]["why"]
