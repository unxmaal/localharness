"""Durable memory for the discovery loop. Issue #52.

Without this a sweep prints and forgets, so nothing compounds: every run
re-proposes what was already declined, and the precision of the extractor cannot
be measured because there is no record of what became of anything.

Three tables. `proposals` is identity, `sightings` is the time axis, `verdicts`
is what happened. The graph is the foreign keys; traversal is a join.
"""
from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from harness import paths

SCHEMA_VERSION = 1

#: Outcomes a proposal can reach. TERMINAL ones suppress re-proposal.
VERDICTS = ("measured", "declined", "broken", "queued", "ignored", "screened")
TERMINAL = ("measured", "declined", "broken", "ignored")

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS proposals (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'candidate',
    lane        TEXT NOT NULL DEFAULT '',
    resolved    TEXT NOT NULL DEFAULT '',
    -- What it consumes and produces, so valid compositions can be found
    -- without trying every pair. Empty means unknown.
    consumes    TEXT NOT NULL DEFAULT '',
    produces    TEXT NOT NULL DEFAULT '',
    first_seen  REAL NOT NULL,
    last_seen   REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS sightings (
    id          INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    source      TEXT NOT NULL,
    url         TEXT NOT NULL DEFAULT '',
    why         TEXT NOT NULL DEFAULT '',
    relevance   INTEGER NOT NULL DEFAULT 0,
    seen_at     REAL NOT NULL,
    UNIQUE (proposal_id, source, url)
);

CREATE TABLE IF NOT EXISTS verdicts (
    id          INTEGER PRIMARY KEY,
    proposal_id INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    outcome     TEXT NOT NULL,
    tier        TEXT NOT NULL DEFAULT '',
    detail      TEXT NOT NULL DEFAULT '',
    issue       INTEGER,
    run_path    TEXT NOT NULL DEFAULT '',
    score       REAL,
    rubric      TEXT NOT NULL DEFAULT '',
    judge       TEXT NOT NULL DEFAULT '',
    decided_at  REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS edges (
    id          INTEGER PRIMARY KEY,
    src         INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    dst         INTEGER NOT NULL REFERENCES proposals(id) ON DELETE CASCADE,
    relation    TEXT NOT NULL,
    note        TEXT NOT NULL DEFAULT '',
    UNIQUE (src, dst, relation)
);

CREATE INDEX IF NOT EXISTS ix_sight_prop ON sightings(proposal_id);
CREATE INDEX IF NOT EXISTS ix_verdict_prop ON verdicts(proposal_id);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst);
"""


def db_path() -> Path:
    return paths.home() / "discovery.db"


def connect(path: Path | None = None) -> sqlite3.Connection:
    """Open (creating if needed) and migrate to SCHEMA_VERSION."""
    path = Path(path) if path is not None else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_DDL)
    _migrate(conn)
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    row = conn.execute("SELECT value FROM meta WHERE key='schema'").fetchone()
    have = int(row["value"]) if row else 0
    if have == SCHEMA_VERSION:
        return
    if have > SCHEMA_VERSION:
        raise RuntimeError(
            f"discovery.db is schema {have}, this code speaks {SCHEMA_VERSION}. "
            f"Refusing to touch a newer store.")
    # Future migrations land here, keyed on `have`. The DDL above is
    # CREATE IF NOT EXISTS, so v0 -> v1 needs nothing beyond the stamp.
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                 (str(SCHEMA_VERSION),))
    conn.commit()


@dataclass
class Seen:
    """One proposal observed in one source at one time."""
    name: str
    source: str
    url: str = ""
    why: str = ""
    relevance: int = 0
    kind: str = "candidate"
    lane: str = ""
    resolved: str = ""


def record(conn: sqlite3.Connection, seen: Seen, at: float | None = None) -> int:
    """Upsert the proposal, add a sighting. Returns the proposal id.

    Seeing the same thing again is a SIGHTING, never a duplicate proposal:
    recurrence over time is the signal that separates a lasting thing from one
    that trended once.
    """
    now = time.time() if at is None else at
    cur = conn.execute("SELECT id FROM proposals WHERE name = ?", (seen.name,))
    row = cur.fetchone()
    if row:
        pid = row["id"]
        conn.execute(
            "UPDATE proposals SET last_seen = ?, "
            "  resolved = CASE WHEN ?<>'' THEN ? ELSE resolved END, "
            "  lane = CASE WHEN lane='' THEN ? ELSE lane END "
            "WHERE id = ?",
            (now, seen.resolved, seen.resolved, seen.lane, pid))
    else:
        pid = conn.execute(
            "INSERT INTO proposals (name, kind, lane, resolved, first_seen, "
            "last_seen) VALUES (?,?,?,?,?,?)",
            (seen.name, seen.kind, seen.lane, seen.resolved, now, now)).lastrowid
    conn.execute(
        "INSERT OR IGNORE INTO sightings (proposal_id, source, url, why, "
        "relevance, seen_at) VALUES (?,?,?,?,?,?)",
        (pid, seen.source, seen.url, seen.why, seen.relevance, now))
    conn.commit()
    return pid


def decide(conn: sqlite3.Connection, name: str, outcome: str, *, tier: str = "",
           detail: str = "", issue: int | None = None, run_path: str = "",
           score: float | None = None, rubric: str = "", judge: str = "",
           at: float | None = None) -> int:
    """Record what happened to a proposal.

    Verdicts accumulate rather than replace: a screen verdict and a later
    measurement are two facts, and which tier produced a row is part of whether
    two rows may be compared.
    """
    if outcome not in VERDICTS:
        raise ValueError(f"unknown outcome {outcome!r}; "
                         f"known: {', '.join(VERDICTS)}")
    row = conn.execute("SELECT id FROM proposals WHERE name = ?",
                       (name,)).fetchone()
    if not row:
        raise KeyError(f"no proposal named {name!r}")
    vid = conn.execute(
        "INSERT INTO verdicts (proposal_id, outcome, tier, detail, issue, "
        "run_path, score, rubric, judge, decided_at) VALUES (?,?,?,?,?,?,?,?,?,?)",
        (row["id"], outcome, tier, detail, issue, run_path, score, rubric,
         judge, time.time() if at is None else at)).lastrowid
    conn.commit()
    return vid


def link(conn: sqlite3.Connection, src: str, dst: str, relation: str,
         note: str = "") -> None:
    """An edge between two proposals. Both must already exist."""
    ids = {}
    for n in (src, dst):
        row = conn.execute("SELECT id FROM proposals WHERE name = ?",
                           (n,)).fetchone()
        if not row:
            raise KeyError(f"no proposal named {n!r}")
        ids[n] = row["id"]
    conn.execute("INSERT OR IGNORE INTO edges (src, dst, relation, note) "
                 "VALUES (?,?,?,?)", (ids[src], ids[dst], relation, note))
    conn.commit()


def settled(conn: sqlite3.Connection) -> set[str]:
    """Names with a terminal verdict, which must not be proposed again."""
    q = ("SELECT DISTINCT p.name FROM proposals p JOIN verdicts v "
         f"ON v.proposal_id = p.id WHERE v.outcome IN "
         f"({','.join('?' * len(TERMINAL))})")
    return {r["name"] for r in conn.execute(q, TERMINAL)}


def recurrence(conn: sqlite3.Connection, minimum: int = 2) -> list[dict]:
    """Proposals seen more than once, most-seen first.

    The answer to the README's own caveat that a feed measures popularity: a
    thing that keeps coming back over months is a different signal from a thing
    that trended once.
    """
    q = """
        SELECT p.name, p.kind, p.lane, p.resolved,
               COUNT(s.id) AS times, COUNT(DISTINCT s.source) AS sources,
               MIN(s.seen_at) AS first_seen, MAX(s.seen_at) AS last_seen
        FROM proposals p JOIN sightings s ON s.proposal_id = p.id
        GROUP BY p.id HAVING times >= ?
        ORDER BY times DESC, sources DESC, last_seen DESC
    """
    return [dict(r) for r in conn.execute(q, (minimum,))]


def precision(conn: sqlite3.Connection) -> dict:
    """How much of what the extractor proposes turns out to be worth anything.

    The number issue #49 asked for, as a query rather than a manual count.
    """
    total = conn.execute("SELECT COUNT(*) c FROM proposals").fetchone()["c"]
    resolved = conn.execute(
        "SELECT COUNT(*) c FROM proposals WHERE resolved <> ''").fetchone()["c"]
    counts = {o: 0 for o in VERDICTS}
    for r in conn.execute("SELECT outcome, COUNT(DISTINCT proposal_id) c "
                          "FROM verdicts GROUP BY outcome"):
        counts[r["outcome"]] = r["c"]
    judged = conn.execute("SELECT COUNT(DISTINCT proposal_id) c FROM verdicts "
                          "WHERE score IS NOT NULL").fetchone()["c"]
    return {"proposals": total, "resolved": resolved, "judged": judged,
            **{f"verdict_{k}": v for k, v in counts.items()}}


def traverse(conn: sqlite3.Connection, name: str, depth: int = 2,
             direction: str = "both") -> list[dict]:
    """Walk the edge graph from `name`, forwards, backwards, or both.

    Recursive CTE rather than a graph engine: at this scale the graph is joins,
    and a server would buy traversal performance nothing here needs.
    """
    if direction not in ("forward", "backward", "both"):
        raise ValueError("direction must be forward, backward or both")
    row = conn.execute("SELECT id FROM proposals WHERE name = ?",
                       (name,)).fetchone()
    if not row:
        raise KeyError(f"no proposal named {name!r}")
    step = {"forward": "e.src = w.id",
            "backward": "e.dst = w.id",
            "both": "(e.src = w.id OR e.dst = w.id)"}[direction]
    other = {"forward": "e.dst",
             "backward": "e.src",
             "both": "CASE WHEN e.src = w.id THEN e.dst ELSE e.src END"}[direction]
    q = f"""
        WITH RECURSIVE walk(id, depth, relation) AS (
            SELECT ?, 0, ''
            UNION
            SELECT {other}, w.depth + 1, e.relation
            FROM edges e JOIN walk w ON {step}
            WHERE w.depth < ?
        )
        SELECT p.name, p.kind, p.lane, w.depth, w.relation
        FROM walk w JOIN proposals p ON p.id = w.id
        WHERE w.depth > 0
        ORDER BY w.depth, p.name
    """
    return [dict(r) for r in conn.execute(q, (row["id"], depth))]


def composable(conn: sqlite3.Connection) -> list[tuple[str, str]]:
    """Pairs whose types line up: a produces what b consumes.

    This is the pruning that makes pair search affordable. 47 proposals is 1,081
    blind pairs; most are nonsense on their face, because speech does not compose
    with an image upscaler.
    """
    q = """
        SELECT a.name AS a, b.name AS b
        FROM proposals a JOIN proposals b
          ON a.produces <> '' AND b.consumes <> ''
         AND a.produces = b.consumes AND a.id <> b.id
        ORDER BY a.name, b.name
    """
    return [(r["a"], r["b"]) for r in conn.execute(q)]


def types(conn: sqlite3.Connection, name: str, consumes: str = "",
          produces: str = "") -> None:
    """Declare what a proposal consumes and produces."""
    conn.execute("UPDATE proposals SET "
                 "consumes = CASE WHEN ?<>'' THEN ? ELSE consumes END, "
                 "produces = CASE WHEN ?<>'' THEN ? ELSE produces END "
                 "WHERE name = ?",
                 (consumes, consumes, produces, produces, name))
    conn.commit()


def export(conn: sqlite3.Connection) -> str:
    """The whole store as JSON, for a human or another tool."""
    out = {t: [dict(r) for r in conn.execute(f"SELECT * FROM {t}")]
           for t in ("proposals", "sightings", "verdicts", "edges")}
    return json.dumps(out, indent=2)
