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


# --- an adapter is not a candidate, and this tier is where that costs GiB ---

def test_an_adapter_is_declined_rather_than_downloaded(tmp_path):
    """rank drops attachments so they never reach the top of the queue, but
    rank is an ORDERING and fetch is a SPEND. A sized LoRA would be downloaded
    and handed to a runner that cannot load it.

    `declined` and not `queued`: unlike "no measured size", which says the
    store could not answer a question about itself, this is a fact about the
    candidate.
    """
    from harness import fetching
    from harness import memory_store as ms

    conn = ms.connect(tmp_path / "s.db")
    try:
        ms.record(conn, ms.Seen(
            name="org/style-lora", source="test", lane="image",
            registry="huggingface", resolved="org/style-lora",
            description="tagged lora, anime; adapter of org/base; "
                        "0.2 GiB of weights"))
        ms.decide(conn, "org/style-lora", "queued", tier="inspect",
                  detail="bytes=209715200 fits")
        got = fetching.run(conn, {"org/style-lora": 200 * 1024 ** 2}, limit=1,
                           snapshot=lambda *a, **k: pytest.fail(
                               "an adapter must never be downloaded"))
        assert got and not got[0]["ok"]
        assert "attaches to a model" in got[0]["why"]
        last = conn.execute(
            "SELECT v.outcome FROM verdicts v JOIN proposals p "
            "ON p.id = v.proposal_id WHERE p.name = ? "
            "ORDER BY v.id DESC LIMIT 1", ("org/style-lora",)).fetchone()
        assert last["outcome"] == "declined", (
            "an adapter is a fact about the candidate, not about this harness")
    finally:
        conn.close()


def test_an_ordinary_model_is_not_refused_as_an_attachment(tmp_path):
    """THE NEGATIVE CONTROL. A refusal that fires on everything empties the
    queue and looks exactly like a queue that ran out."""
    from harness import fetching
    from harness import memory_store as ms

    seen = []
    conn = ms.connect(tmp_path / "s.db")
    try:
        ms.record(conn, ms.Seen(
            name="org/real-model", source="test", lane="image",
            registry="huggingface", resolved="org/real-model",
            description="task text-to-image; served by mflux; "
                        "built from org/base; 4.0 GiB of weights"))
        ms.decide(conn, "org/real-model", "queued", tier="inspect",
                  detail="bytes=4294967296 fits")
        got = fetching.run(conn, {"org/real-model": 4 * 1024 ** 3}, limit=1,
                           snapshot=lambda *a, **k: (seen.append(1),
                                                     str(tmp_path))[1])
        assert seen, f"an ordinary model was not fetched: {got}"
        assert got and got[0]["ok"], got
    finally:
        conn.close()
