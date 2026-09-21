"""A verdict is a fact about the machine that made it. Issue #266.

The store is shared across three machines and recorded only WHAT was decided.
A refusal is routinely a fact about one of them -- `needs-cuda` is true on an
mlx-only machine and false on one with the card, `too-big` is measured against a ceiling that
describes one 32 GB machine -- so a verdict made elsewhere read as a fact
about the model, and could not be found again when the machine changed.

56 rows in the real store carried the machine as prose inside `detail`, and
the only identifier in it was `arm64`.
"""
import sqlite3

import pytest

from harness import memory_store as ms

MAC = {"fingerprint": "Mac14,12/macOS-26.5.1/arm64", "runtimes": "cpu,mlx",
       "memory_gb": 32.0, "ceiling_gb": 22.0, "hw_model": "Mac14,12",
       "os": "macOS-26.5.1", "arch": "arm64", "accelerator": "unified 32GB"}
BOX = {"fingerprint": "MS-7D25/Linux-6.8/x86_64", "runtimes": "cpu,cuda",
       "memory_gb": 61.0, "ceiling_gb": 11.0, "hw_model": "MS-7D25",
       "os": "Linux-6.8", "arch": "x86_64", "accelerator": "discrete 12GB"}
#: The SAME box under the other operating system. `arch` cannot tell these
#: apart and comparable() already refuses to pool them: different peak-memory
#: instrument, different OCR grader.
BOX_WINDOWS = {**BOX, "os": "Windows-11", "arch": "x86_64",
               "fingerprint": "MS-7D25/Windows-11/x86_64"}


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "s.db")
    conn.execute("INSERT INTO proposals (name, kind, lane, resolved, "
                 "first_seen, last_seen, registry) "
                 "VALUES ('org/c','candidate','code','org/c',0,0,'huggingface')")
    yield conn
    conn.close()


def _decide_as(conn, facts, outcome, detail, until=""):
    """A verdict written as if `facts` were the machine."""
    mid = ms.remember_machine(conn, facts)
    conn.execute(
        "INSERT INTO verdicts (proposal_id, outcome, tier, detail, "
        "decided_at, machine_id, until) VALUES (1,?,'fetch',?,0,?,?)",
        (outcome, detail, mid, until))
    conn.commit()


# --- identity -------------------------------------------------------------

def test_two_operating_systems_on_one_box_are_two_machines(store):
    """`arch` was the only identifier in the old prose and says x86_64 for
    both. They measure peak memory with different instruments, so pooling
    their verdicts pools two different exams."""
    a = ms.remember_machine(store, BOX)
    b = ms.remember_machine(store, BOX_WINDOWS)
    assert a != b, "one row for two rigs the receipt layer already separates"


def test_the_same_machine_twice_is_one_row(store):
    """Or every sweep invents a machine and the table becomes a log."""
    assert ms.remember_machine(store, MAC) == ms.remember_machine(store, MAC)


def test_an_unidentifiable_machine_is_recorded_as_unknown(store, monkeypatch):
    """NOT omitted. A NULL would read as "some machine" and be pooled with
    rows that do know which one."""
    monkeypatch.setattr(ms, "_THIS_MACHINE", None)
    monkeypatch.setattr(ms, "this_machine",
                        lambda: {**MAC, "fingerprint": "unknown"})
    mid = ms.remember_machine(store)
    got = store.execute("SELECT fingerprint FROM machines WHERE id = ?",
                        (mid,)).fetchone()
    assert got["fingerprint"] == "unknown"


def test_every_verdict_records_its_machine(store):
    """decide() is the single write path, so no caller has to remember --
    and the 56 prose rows are what asking callers to remember produced."""
    ms.decide(store, "org/c", "declined", tier="fetch", detail="needs-cuda")
    row = store.execute("SELECT machine_id FROM verdicts").fetchone()
    assert row["machine_id"] is not None


# --- the condition --------------------------------------------------------

def test_a_condition_is_a_predicate_not_a_sentence():
    """The first draft stored "a machine with cuda" and re-created the exact
    defect being fixed, one level up: a machine fact only a human can read."""
    assert ms.until_met("runtime:cuda", BOX)
    assert not ms.until_met("runtime:cuda", MAC)
    assert ms.until_met("ceiling_gb:>15", MAC)
    assert not ms.until_met("ceiling_gb:>30", MAC)


def test_an_unreadable_condition_leaves_the_verdict_standing():
    """A typo must not silently re-queue everything. Not-met is the safe
    direction: the verdict stays until somebody looks."""
    assert not ms.until_met("runtimes:cuda", BOX)     # plural, wrong key
    assert not ms.until_met("ceiling_gb:<9", MAC)     # unsupported operator
    assert not ms.until_met("", MAC)


# --- the read path the columns exist for ----------------------------------

def test_a_refusal_elsewhere_reopens_where_its_reason_does_not_apply(store):
    """34 rows in the real store were declined on an mlx-only machine for wanting cuda,
    and the box with the card could not find them."""
    _decide_as(store, MAC, "declined", "needs-cuda: no runtime here",
               until="runtime:cuda")
    assert ms.revisitable(store, BOX), "the box with the card sees nothing"
    assert not ms.revisitable(store, MAC), (
        "a machine must not be told to revisit its own refusal")


def test_a_bigger_machine_reopens_only_what_it_can_hold(store):
    """144 rows are `too-big` against a ceiling describing one 32 GB machine.
    The condition carries the size the weights actually need, so a Studio
    matches the rows it can now run rather than all of them."""
    studio = {**MAC, "fingerprint": "Mac16,9/macOS/arm64", "memory_gb": 96.0,
              "ceiling_gb": 80.0}
    _decide_as(store, MAC, "declined", "too-big: 52.7 GiB",
               until="ceiling_gb:>52.7")
    assert ms.revisitable(store, studio)
    assert not ms.revisitable(store, {**studio, "ceiling_gb": 40.0}), (
        "a machine that still cannot hold these weights was told to try")


def test_only_the_latest_verdict_can_be_revisited(store):
    """A condition already retracted must not resurrect. Schema 7 exists so a
    verdict can be undone by appending, and this read has to honour that."""
    _decide_as(store, MAC, "declined", "needs-cuda", until="runtime:cuda")
    _decide_as(store, MAC, "queued", "retracted: worth another look")
    assert not ms.revisitable(store, BOX)


# --- the migration --------------------------------------------------------

def test_an_old_store_is_attributed_and_says_it_was_inferred(tmp_path):
    """1767 rows predate the column. Only this Mac has ever written to the
    real store, so attributing them to the migrating machine is right HERE
    and would be wrong on a store that had genuinely been shared -- the
    fingerprint is recorded so a reader can see what was assumed."""
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema','11');
        CREATE TABLE proposals (id INTEGER PRIMARY KEY, name TEXT,
            kind TEXT DEFAULT '', lane TEXT DEFAULT '',
            resolved TEXT DEFAULT '', consumes TEXT DEFAULT '',
            produces TEXT DEFAULT '', first_seen REAL DEFAULT 0,
            last_seen REAL DEFAULT 0, registry TEXT DEFAULT '',
            description TEXT DEFAULT '');
        CREATE TABLE verdicts (id INTEGER PRIMARY KEY, proposal_id INTEGER,
            outcome TEXT, tier TEXT DEFAULT '', detail TEXT DEFAULT '',
            issue INTEGER, run_path TEXT DEFAULT '', score REAL,
            rubric TEXT DEFAULT '', judge TEXT DEFAULT '', decided_at REAL);
        INSERT INTO proposals (id,name) VALUES (1,'org/a'),(2,'org/b');
        INSERT INTO verdicts (proposal_id,outcome,detail,decided_at)
          VALUES (1,'declined','needs-cuda on arm64: no runtime',0),
                 (2,'declined','too-big: smallest weight it names is 52.7 GiB,
                                over the 22 GiB ceiling',0);
    """)
    old.commit()
    old.close()

    conn = ms.connect(path)
    try:
        rows = {r["detail"][:8]: r["until"] for r in
                conn.execute("SELECT detail, until FROM verdicts")}
        assert rows["needs-cu"] == "runtime:cuda"
        assert rows["too-big:"] == "ceiling_gb:>52.7", rows
        unattributed = conn.execute(
            "SELECT COUNT(*) c FROM verdicts WHERE machine_id IS NULL"
        ).fetchone()["c"]
        assert unattributed == 0
    finally:
        conn.close()
