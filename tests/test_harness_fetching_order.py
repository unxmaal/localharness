"""The fetch exists to feed the screen, so it must agree with it. Issue #224.

Measured on the code lane before the fix: the two top-six lists had ZERO
overlap, and three consecutive loop runs each downloaded something and each
ended with nothing to screen.
"""
import pytest

from harness import fetching, lanes, rank
from harness import memory_store as ms

GIB = fetching.GIB


def _seed(conn, rows):
    """Proposals with an inspect verdict queuing them, as a sweep leaves them."""
    for name, lane, kind, registry, times in rows:
        ms.record(conn, ms.Seen(name=name, source="test", url="", why="",
                                relevance=0, kind=kind, registry=registry,
                                lane=lane, resolved=name))
        for _ in range(times):
            ms.record(conn, ms.Seen(name=name, source=f"s{_}", url=f"u{_}",
                                    why="", relevance=0, kind=kind,
                                    registry=registry, lane=lane,
                                    resolved=name))
        ms.decide(conn, name, "queued", tier="inspect",
                  detail="bytes=1073741824 fits")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fetching, "have", lambda *a, **k: False)
    monkeypatch.setattr(rank, "serving", lambda *a, **k: set())
    monkeypatch.setattr(rank, "lanes_with_receipts", lambda *a, **k: set())
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def test_the_fetch_order_is_the_screen_order(store):
    """Both lists come from rank.rank, so the tier that spends the disk spends
    it on what the next tier will run."""
    _seed(store, [("org/zeta", "code", "candidate", ms.HUGGINGFACE, 4),
                  ("org/alpha", "code", "candidate", ms.HUGGINGFACE, 1)])
    fetch = [r["name"] for r in fetching.queued(store, lane="code")]
    rows = [r for r in ms.judgeable(store, limit=1_000_000)
            if lanes.serves(rank.lane_of(r), "code")]
    screen = [r["name"] for r in rank.rank(
        rows, serving=set(), measured_lanes=set())]
    assert fetch == screen, (
        "the fetch is downloading in an order the screen does not ask in")


def test_recurrence_beats_the_alphabet(store):
    """The old order sorted by a judged repo score that is 0 for every weight,
    so it fell back to the name. `zeta` is seen four times and must lead."""
    _seed(store, [("org/zeta", "code", "candidate", ms.HUGGINGFACE, 4),
                  ("org/alpha", "code", "candidate", ms.HUGGINGFACE, 1)])
    assert [r["name"] for r in fetching.queued(store, lane="code")][0] == "org/zeta"


# --- the registry is the authority, `kind` is the fallback -----------------

def test_a_huggingface_candidate_is_fetchable_whatever_its_kind(store):
    """The sweep records kind='candidate' for a name resolved out of prose.
    Filtering on kind='weights' excluded 25 of 28 ranked code candidates."""
    _seed(store, [("org/prose", "code", "candidate", ms.HUGGINGFACE, 1)])
    assert [r["name"] for r in fetching.queued(store)] == ["org/prose"]


def test_a_github_repo_is_still_refused(store):
    """The negative control, and the reason the guard exists: this queue once
    held GitHub names and snapshot_download 401'd on every one."""
    _seed(store, [("org/repo", "code", "repo", ms.GITHUB, 1)])
    assert fetching.queued(store) == []


def test_a_row_with_no_registry_falls_back_to_kind(store):
    """Rows written before the registry column. `kind` still answers for them."""
    _seed(store, [("org/old-weights", "code", "weights", "", 1),
                  ("org/old-other", "code", "candidate", "", 1)])
    assert [r["name"] for r in fetching.queued(store)] == ["org/old-weights"]
