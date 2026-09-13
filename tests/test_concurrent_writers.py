"""A claimed failure mode must be the observed one.

THE VIOLATION: a claim that several processes writing one SQLite file is how
you corrupt it. It was wrong, and it was corrected on the spot. SQLite
SERIALISES writers; the failure is `database is locked`, and the difference
matters because the two have opposite fixes. Corruption says "use another
database". A lock timeout says "set one".

So this file does not reason about it. It runs two writers.

#148 is about to put several pods against this store, which is why the answer
had to stop being an opinion.
"""
import multiprocessing as mp
import sqlite3
import time
from pathlib import Path

import pytest

from harness import memory_store


def _write(path, key, rows, barrier=None):
    conn = memory_store.connect(Path(path))
    if barrier is not None:
        barrier.wait()
    for i in range(rows):
        conn.execute("INSERT OR REPLACE INTO meta VALUES (?, ?)",
                     (f"{key}-{i}", str(i)))
        conn.commit()
    conn.close()


def test_the_store_waits_for_the_lock_rather_than_failing():
    assert memory_store.BUSY_TIMEOUT_SECONDS > 0, (
        "without a busy timeout the second writer fails immediately, which is "
        "the behaviour that made 'concurrent writes do not work' look true")


def test_wal_and_a_timeout_are_actually_set(tmp_path):
    """Read back from the connection rather than trusting the source: a PRAGMA
    can be silently refused, and journal_mode in particular is per-database."""
    conn = memory_store.connect(tmp_path / "d.db")
    assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    assert conn.execute("PRAGMA busy_timeout").fetchone()[0] == \
        int(memory_store.BUSY_TIMEOUT_SECONDS * 1000)


def test_two_processes_writing_at_once_both_finish(tmp_path):
    """The actual claim, actually run. Not corruption and not failure:
    serialisation, and both writers get their rows in."""
    db = str(tmp_path / "d.db")
    memory_store.connect(Path(db)).close()          # create and migrate once
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(2)
    procs = [ctx.Process(target=_write, args=(db, key, 40, barrier))
             for key in ("a", "b")]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=120)
    assert [p.exitcode for p in procs] == [0, 0], \
        "a writer died rather than waiting its turn"

    conn = memory_store.connect(Path(db))
    for key in ("a", "b"):
        n = conn.execute("SELECT count(*) FROM meta WHERE key LIKE ?",
                         (f"{key}-%",)).fetchone()[0]
        assert n == 40, f"writer {key} lost rows"


def test_a_reader_is_not_blocked_by_a_writer(tmp_path):
    """Why WAL and not just a timeout. The sweep reads while the judge writes,
    and under the rollback journal the reader would wait for the writer."""
    db = tmp_path / "d.db"
    writer = memory_store.connect(db)
    reader = memory_store.connect(db)
    writer.execute("BEGIN IMMEDIATE")
    writer.execute("INSERT OR REPLACE INTO meta VALUES ('held', '1')")
    started = time.perf_counter()
    reader.execute("SELECT count(*) FROM meta").fetchone()
    assert time.perf_counter() - started < 1.0, \
        "the reader waited for an open write transaction, so WAL is not on"
    writer.rollback()


def test_the_failure_is_a_lock_not_corruption(tmp_path):
    """The negative control, with the timeout deliberately removed. Naming the
    real error keeps the retraction honest: `database is locked`, an
    OperationalError, not DatabaseError('malformed')."""
    db = tmp_path / "d.db"
    memory_store.connect(db).close()
    holder = sqlite3.connect(db, timeout=0)
    holder.execute("BEGIN IMMEDIATE")
    holder.execute("INSERT OR REPLACE INTO meta VALUES ('held', '1')")
    other = sqlite3.connect(db, timeout=0)
    with pytest.raises(sqlite3.OperationalError) as exc:
        other.execute("BEGIN IMMEDIATE")
    assert "locked" in str(exc.value)
    holder.rollback()
