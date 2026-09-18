"""--budget-gib is the ceiling on one invocation, and it was never enforced.

`lh discover --loop --run --budget-gib 12` printed the budget in the fetch
header and downloaded 24.6 GiB, because _loop_spend never passed it to
fetching.run. Issue #215.
"""
import pytest

from harness import fetching

GIB = fetching.GIB


class _Conn:
    def __init__(self, rows):
        self.rows = rows
        self.decided = []

    def execute(self, *a, **k):
        conn = self

        class R:
            def fetchall(self_inner):
                return conn.rows
        return R()


def _row(name, lane="image"):
    return {"name": name, "resolved": name, "lane": lane, "kind": "weights",
            "registry": "huggingface", "outcome": "queued", "tier": "inspect",
            "detail": "", "sized": "", "score": 0}


@pytest.fixture
def wired(monkeypatch):
    monkeypatch.setattr(fetching, "have", lambda *a, **k: False)
    monkeypatch.setattr(fetching, "download", lambda name, snapshot=None: f"/x/{name}")
    monkeypatch.setattr(fetching, "requires", lambda name: [])
    decided = []
    monkeypatch.setattr("harness.memory_store.decide",
                        lambda conn, name, outcome, **kw: decided.append(
                            (name, outcome, kw.get("detail", ""))))
    return decided


def test_the_budget_stops_the_second_download(wired):
    conn = _Conn([_row("org/small"), _row("org/big")])
    sizes = {"org/small": int(5.5 * GIB), "org/big": int(19.1 * GIB)}
    got = fetching.run(conn, sizes, limit=5, free=500 * GIB,
                       budget=int(12 * GIB))
    assert [g["repo"] for g in got if g["ok"]] == ["org/small"]
    refused = [g for g in got if not g["ok"]]
    assert refused and "budget" in refused[0]["why"]


def test_exceeding_the_budget_re_queues_rather_than_settling(wired):
    """A budget is a fact about THIS INVOCATION, not about the candidate.
    Recording it as declined would suppress a real model forever (#211)."""
    conn = _Conn([_row("org/big")])
    fetching.run(conn, {"org/big": int(19.1 * GIB)}, limit=5, free=500 * GIB,
                 budget=int(12 * GIB))
    assert [(n, o) for n, o, _ in wired] == [("org/big", "queued")]


def test_no_budget_means_no_ceiling(wired):
    """The negative control: absent the flag, nothing new refuses anything."""
    conn = _Conn([_row("org/small"), _row("org/big")])
    sizes = {"org/small": int(5.5 * GIB), "org/big": int(19.1 * GIB)}
    got = fetching.run(conn, sizes, limit=5, free=500 * GIB, budget=None)
    assert sorted(g["repo"] for g in got if g["ok"]) == ["org/big", "org/small"]


def test_an_unsized_row_is_not_charged_to_the_budget(wired):
    """A size of 0 means unknown, and plan() refuses it for that reason. It
    must not silently pass a budget check by counting as free."""
    conn = _Conn([_row("org/unsized")])
    got = fetching.run(conn, {"org/unsized": 0}, limit=5, free=500 * GIB,
                       budget=int(12 * GIB))
    assert not got[0]["ok"]
    assert "no measured size" in got[0]["why"]
