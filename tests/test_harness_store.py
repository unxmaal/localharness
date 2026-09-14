"""The SQLite dialect, as Postgres would have it.

`harness/memory_store.py` holds 33 statements in one dialect and they stay
there. This translates. The alternative -- a second hand-written schema and a
second set of queries -- is the *one question, two answers* failure, and the
divergence always shows up as a column that exists on one backend only.

THE THREE DEFECTS THAT MATTERED WERE NOT FOUND BY READING. The incompatibilities
were counted by grep (33 `?`, 2 `.lastrowid`, 4 upserts, 5 `INTEGER PRIMARY
KEY`) and all three real failures were invisible to that count, because none is
a construct:

  * `HAVING times >= ?` referencing a select-list alias. SQLite allows it,
    standard SQL does not.
  * a `?` inside a SQL COMMENT becoming a placeholder.
  * a parameter in the non-recursive term of a recursive CTE, typed smallint by
    inference and then mismatching the integer column.

Running it found each one in minutes. That is the argument for the live test at
the bottom of this file.
"""
import pytest

from harness import memory_store, store


# ---- which backend -------------------------------------------------------

def test_sqlite_unless_told_otherwise(monkeypatch):
    """`lh` on a laptop must keep working with no cluster at all, which is a
    requirement of #148 rather than a nicety."""
    monkeypatch.delenv(store.BACKEND_ENV, raising=False)
    assert store.backend() == store.SQLITE


@pytest.mark.parametrize("value,want", [
    ("postgres", store.POSTGRES), ("POSTGRES", store.POSTGRES),
    (" postgres ", store.POSTGRES), ("sqlite", store.SQLITE),
    ("", store.SQLITE), ("mysql", store.SQLITE),
])
def test_the_backend_is_read_leniently_and_falls_back_safely(value, want, monkeypatch):
    """A typo must not silently reach for a database that is not there."""
    monkeypatch.setenv(store.BACKEND_ENV, value)
    assert store.backend() == want


# ---- placeholders, and the three places a `?` is not one ------------------

def test_a_parameter_becomes_a_placeholder():
    assert store.placeholders("WHERE a = ?") == "WHERE a = %s"


def test_a_question_mark_in_a_string_literal_is_left_alone():
    """A blanket replace yields a query that RUNS and returns the wrong rows,
    which is worse than one that raises."""
    assert store.placeholders("WHERE b = 'why?'") == "WHERE b = 'why?'"


def test_a_question_mark_in_a_line_comment_is_left_alone():
    """Found the hard way: the comment explaining the HAVING bug quoted
    `HAVING times >= ?` and grew a second placeholder for a one-parameter
    query. psycopg counts before binding, so it failed loudly."""
    got = store.placeholders("-- HAVING times >= ?\nWHERE a = ?")
    assert got.count("%s") == 1
    assert "times >= ?" in got


def test_a_question_mark_in_a_block_comment_is_left_alone():
    got = store.placeholders("/* what? */ WHERE a = ?")
    assert got.count("%s") == 1


def test_an_unterminated_comment_does_not_eat_the_query():
    assert store.placeholders("WHERE a = ? -- trailing?").count("%s") == 1


# ---- the upserts ---------------------------------------------------------

def test_or_ignore_becomes_a_targetless_do_nothing():
    """SQLite's OR IGNORE skips a row violating ANY unique constraint, so
    naming one column here would quietly stop ignoring the others."""
    got = store.translate("INSERT OR IGNORE INTO edges (src) VALUES (?)")
    assert got.endswith("ON CONFLICT DO NOTHING")
    assert "ON CONFLICT (" not in got


def test_or_replace_names_what_it_conflicts_on():
    got = store.translate("INSERT OR REPLACE INTO meta VALUES ('schema', ?)")
    assert "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value" in got


def test_an_unrecorded_upsert_is_refused_rather_than_guessed():
    """A guess here inserts duplicates instead of replacing, and nothing
    raises. Better to fail at translation than to be quietly wrong."""
    with pytest.raises(ValueError, match="no recorded conflict key"):
        store.translate("INSERT OR REPLACE INTO nowhere VALUES (?)")


# ---- the schema is derived, not written twice ----------------------------

def test_a_timestamp_does_not_lose_precision():
    """THE SILENT ONE. Postgres REAL is float4, which carries about seven
    significant digits; a unix timestamp needs ten. Left as REAL, every
    first_seen would round to the nearest couple of minutes and every
    recurrence window would be wrong by an amount nobody would look for."""
    ddl = store.postgres_ddl("first_seen REAL NOT NULL")
    assert "DOUBLE PRECISION" in ddl and "REAL" not in ddl


def test_an_autoincrementing_key_still_autoincrements():
    ddl = store.postgres_ddl("id INTEGER PRIMARY KEY")
    assert "IDENTITY" in ddl


def test_the_postgres_schema_comes_from_the_sqlite_one():
    """Derived rather than hand-written, so the two cannot drift into having
    different columns."""
    derived = store.postgres_ddl(memory_store._DDL)
    for table in ("proposals", "sightings", "verdicts", "edges", "extractions"):
        assert f"CREATE TABLE IF NOT EXISTS {table}" in derived
    assert derived.count("CREATE TABLE") == memory_store._DDL.count("CREATE TABLE")


# ---- the live half -------------------------------------------------------

@pytest.mark.postgres
def test_every_store_function_works_against_a_real_postgres():
    """The three defects above were all invisible to a reading of the code and
    took minutes to find by running. Needs a database:

        make test-postgres
    """
    import os
    if os.environ.get(store.BACKEND_ENV) != store.POSTGRES:
        pytest.skip("set LOCALHARNESS_STORE=postgres and point PG* at a database")
    conn = memory_store.connect()
    memory_store.record(conn, memory_store.Seen(
        name="t/live", kind="candidate", lane="image", source="feed",
        url="https://x", why="test", relevance=1, resolved=""))
    memory_store.decide(conn, "t/live", "queued", tier="inspect")
    conn.commit()
    assert memory_store.recurrence(conn, minimum=1)
    assert memory_store.precision(conn)["proposals"] >= 1
    assert memory_store.by_source(conn)
    memory_store.link(conn, "t/live", "t/live", "self", "")
    conn.commit()
    memory_store.traverse(conn, "t/live", depth=2)


@pytest.mark.postgres
def test_the_registry_column_and_its_migration_run_on_a_real_postgres():
    """Issue #167 added a column to a table the DDL only knows how to CREATE,
    so the migration has to ALTER -- and asking which columns exist is the one
    thing the two backends genuinely cannot share (PRAGMA vs information_schema).
    A migration exercised only on SQLite is a migration untested where it runs.

        make test-postgres
    """
    import os
    if os.environ.get(store.BACKEND_ENV) != store.POSTGRES:
        pytest.skip("set LOCALHARNESS_STORE=postgres and point PG* at a database")
    conn = memory_store.connect()
    # Verified by COLUMN EXISTENCE rather than by the version stamp: the stamp
    # says what the code believes, the column says what the database has.
    assert "registry" in memory_store._columns(conn, "proposals")
    memory_store.record(conn, memory_store.Seen(
        name="t/model", source="feed", resolved="t/model",
        url="https://huggingface.co/t/model",
        registry=memory_store.HUGGINGFACE))
    memory_store.record(conn, memory_store.Seen(
        name="t/tool", source="feed", resolved="t/tool",
        url="https://github.com/t/tool", registry=memory_store.GITHUB))
    conn.commit()
    assert "t/model" in memory_store.pending(conn, registry=memory_store.HUGGINGFACE)
    assert "t/model" not in memory_store.pending(conn, registry=memory_store.GITHUB)
    assert memory_store.by_registry(conn)[memory_store.HUGGINGFACE] >= 1
    # rowcount through the Postgres cursor wrapper, which SQLite gave for free.
    assert memory_store._backfill_registry(conn) >= 0

    # AND THE ALTER ITSELF, which a fresh database never reaches: the DDL
    # creates the column, so only a store that predates it takes this path.
    # Put the database back into that state and migrate it forward.
    conn.execute("ALTER TABLE proposals DROP COLUMN registry")
    conn.execute("UPDATE meta SET value = '2' WHERE key = 'schema'")
    conn.commit()
    assert "registry" not in memory_store._columns(conn, "proposals")
    memory_store._migrate(conn)
    assert "registry" in memory_store._columns(conn, "proposals")
    got = conn.execute("SELECT registry FROM proposals WHERE name = 't/tool'"
                       ).fetchone()["registry"]
    assert got == memory_store.GITHUB, "the sighting's URL says which registry"
