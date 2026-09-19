"""Durable memory for the discovery loop. Issue #52.

Without this a sweep prints and forgets, so nothing compounds: every run
re-proposes what was already declined, and the precision of the extractor cannot
be measured because there is no record of what became of anything.

Three tables. `proposals` is identity, `sightings` is the time axis, `verdicts`
is what happened. The graph is the foreign keys; traversal is a join.
"""
from __future__ import annotations

import json
import re
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path

from harness import paths, store

SCHEMA_VERSION = 9

#: Outcomes a proposal can reach. TERMINAL ones suppress re-proposal.
VERDICTS = ("measured", "declined", "broken", "queued", "ignored", "screened")
TERMINAL = ("measured", "declined", "broken", "ignored")

#: Where a name can be resolved. A proposal is `org/name` in both registries
#: and the two namespaces overlap, so the string alone cannot say which one
#: holds it: `openai/whisper-small` is a model, `openai/openai-python` a repo.
#: Issue #167. Empty means nothing recorded it, which is not the same as
#: neither -- an old row is unknown, and asking the wrong registry about it is
#: how 227 of 235 candidates 404ed.
GITHUB, HUGGINGFACE = "github", "huggingface"
REGISTRIES = (GITHUB, HUGGINGFACE)

#: WHICH TIER ANSWERED. Named here because these strings are a vocabulary
#: shared by writers and readers in different modules: the CLI writes
#: tier='inspect', judgeable() reads it back, and fetching.queued() filters on
#: it. Spelled as literals in three files, a rename in one would leave the
#: others silently returning nothing -- the same class as a list forked into
#: two configs. VERDICTS already had this treatment; the tiers did not.
INSPECT, JUDGE, SCREEN, MEASURE = "inspect", "judge", "screen", "measure"
#: The closing tier. A lane reads its adopted winner from here and falls
#: back to the typed constant when nothing has been adopted. See
#: harness/adopt.py.
ADOPT = "adopt"
TIERS = (INSPECT, JUDGE, SCREEN, MEASURE, ADOPT)

_DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT);

CREATE TABLE IF NOT EXISTS proposals (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL UNIQUE,
    kind        TEXT NOT NULL DEFAULT 'candidate',
    -- Which registry answers for this name. See REGISTRIES.
    registry    TEXT NOT NULL DEFAULT '',
    -- WHAT THE THING IS, as its registry describes it. A sighting's `why` is
    -- what one source said about it on one day; this is the card. The judge
    -- scores a description, and with only a name to read it floored six
    -- models with known opposite outcomes at the same number. Issue #175.
    description TEXT NOT NULL DEFAULT '',
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

-- Names pulled out of prose and REJECTED. Without these there is no
-- denominator: every proposal in the store resolved to something real, so
-- "100% resolved" was arithmetic, not precision. Issue #49.
CREATE TABLE IF NOT EXISTS extractions (
    id          INTEGER PRIMARY KEY,
    name        TEXT NOT NULL,
    source      TEXT NOT NULL,
    reason      TEXT NOT NULL,
    at          REAL NOT NULL,
    UNIQUE (name, source, reason)
);

CREATE INDEX IF NOT EXISTS ix_sight_prop ON sightings(proposal_id);
CREATE INDEX IF NOT EXISTS ix_verdict_prop ON verdicts(proposal_id);
CREATE INDEX IF NOT EXISTS ix_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS ix_edges_dst ON edges(dst);
"""


#: How long a writer waits for the lock before giving up. Generous because the
#: alternative is a failed sweep, and a discovery write is milliseconds: this
#: is a queue depth, not a latency budget.
BUSY_TIMEOUT_SECONDS = 30.0


def db_path() -> Path:
    return paths.home() / "discovery.db"


def connect(path: Path | None = None):
    """Open (creating if needed) and migrate to SCHEMA_VERSION.

    SQLite unless LOCALHARNESS_STORE says postgres, because `lh` on a laptop
    must keep working with no cluster at all -- a discovery engine that only
    runs in Kubernetes is a worse tool than the one that already exists.
    """
    if store.backend() == store.POSTGRES:
        conn = store.postgres_connect()
        conn.executescript(_DDL)
        _migrate(conn)
        return conn
    path = Path(path) if path is not None else db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=BUSY_TIMEOUT_SECONDS)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    # SQLite does not corrupt under concurrent writers -- it SERIALISES them,
    # and an unprepared second writer gets `database is locked` at once. These
    # two lines are the difference between "waits its turn" and "fails", and
    # the discovery store is about to have several writers (#148).
    #
    # WAL also lets readers proceed during a write, which matters because the
    # sweep reads while the judge writes. The caveat is the filesystem: WAL
    # needs real shared memory and is unsafe over NFS-style mounts, so a
    # network-mounted store wants a single writer rather than this.
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute(f"PRAGMA busy_timeout = {int(BUSY_TIMEOUT_SECONDS * 1000)}")
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
    # The DDL above is CREATE IF NOT EXISTS, so v0 -> v1 and v1 -> v2 (which
    # only adds the extractions table) need nothing beyond the stamp. v3 adds
    # a column to a table that already exists, which CREATE IF NOT EXISTS
    # cannot do.
    if have < 3:
        if "registry" not in _columns(conn, "proposals"):
            conn.execute("ALTER TABLE proposals "
                         "ADD COLUMN registry TEXT NOT NULL DEFAULT ''")
        _backfill_registry(conn)
    if have and have < 4 and "description" not in _columns(conn, "proposals"):
        conn.execute("ALTER TABLE proposals "
                     "ADD COLUMN description TEXT NOT NULL DEFAULT ''")
    if have and have < 5:
        _canonical_lanes(conn)
        _backfill_lanes(conn)
    if have and have < 7:
        _retract_harness_refusals(conn)
    if have and have < 8:
        _retract_verdicts_with_no_control(conn)
    if have and have < 9:
        _relane_from_the_card(conn)
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', ?)",
                 (str(SCHEMA_VERSION),))
    conn.commit()


def _canonical_lanes(conn) -> None:
    """One name per lane. `text` IS `code`, and `all` was never a lane.

    discover.py wrote `text` and inspect.PIPELINE_LANES wrote `code` for the
    same models; rank.lane_of then discarded `text` as laneless, so half the
    lane never reached the queue. `all` is a SOURCE's coverage claim. #207.
    """
    from harness import lanes
    for old, new in lanes.ALIASES.items():
        conn.execute("UPDATE proposals SET lane=? WHERE lower(lane)=?",
                     (new, old))


def _backfill_lanes(conn) -> None:
    """Route rows the registry never classified, using their own prose.

    A proposal arrives with a one-line description of what it does and the
    lane was only ever read from a HuggingFace `pipeline_tag`, so a GitHub
    repo could never have one and 287 of 381 rows sat laneless while carrying
    the answer. Only rows with NO lane are touched: the registry's own label
    is the publisher's answer and outranks a guess at it. #207.
    """
    from harness import lanes
    rows = conn.execute(
        "SELECT p.id, p.name, s.why FROM proposals p "
        "JOIN sightings s ON s.proposal_id = p.id WHERE p.lane = ''"
    ).fetchall()
    # A proposal seen twice gets both descriptions, and they can disagree.
    # Disagreement is exactly the case from_prose refuses to break, so it is
    # fed all of them at once and falls silent rather than picking one.
    prose: dict = {}
    for row in rows:
        pid, name = row[0], row[1]
        prose.setdefault(pid, [name]).append(row[2] or "")
    for pid, parts in prose.items():
        lane = lanes.from_prose(" ".join(parts))
        if lane:
            conn.execute("UPDATE proposals SET lane=? WHERE id=?", (lane, pid))


def _retract_harness_refusals(conn) -> None:
    """Re-queue anything settled for a reason that was about US, not about it.

    Every tier, not only the fetch: the screen recorded "no cases of a modality
    it can run" as `broken` for a candidate whose weights were on disk and
    whose lane has three cases.

    The fetch tier recorded "no measured size; inspect it first" as `declined`,
    which is TERMINAL, so 16 real candidates were suppressed from re-proposal
    forever because a size sat in a verdict row the reader did not look at.
    Among them was the only upgrade candidate the image lane had.

    A verdict is not deleted -- it is a record of what happened, and losing it
    would lose the evidence that this went wrong. A fresh `queued` row is
    appended saying why, and the latest verdict is what every tier reads.
    Issues #211, #206.
    """
    from harness import fetching, screen
    rows = conn.execute(
        "SELECT p.id, p.name, v.detail, v.tier FROM proposals p "
        "JOIN verdicts v ON v.proposal_id = p.id "
        "WHERE v.id = (SELECT v2.id FROM verdicts v2 "
        "               WHERE v2.proposal_id = p.id ORDER BY v2.id DESC LIMIT 1) "
        "  AND v.outcome IN ('declined', 'broken')").fetchall()
    now = time.time()
    for row in rows:
        pid, name, detail, tier = row[0], row[1], row[2] or "", row[3] or ""
        phrase = (fetching.refused_by_harness(detail)
                  or screen.refused_by_harness(detail))
        if not phrase:
            continue
        conn.execute(
            "INSERT INTO verdicts (proposal_id, tier, outcome, detail, "
            "decided_at) VALUES (?, ?, 'queued', ?, ?)",
            (pid, tier, f"retracted: {phrase!r} was a fact about this "
                        f"harness, not a verdict on {name}", now))


def _retract_verdicts_with_no_control(conn) -> None:
    """Undo an adoption verdict drawn from a run where the CONTROL scored zero.

    A doubled `/v1` in the gateway argument produced HTTP 404 for every
    request, so both the incumbent and the challenger passed 0 of 27, and the
    loop recorded `declined: does not beat the incumbent on the lane's metric`
    -- a statement about quality, from a run in which nothing ran.

    The receipts are on disk and say so, but re-deriving which runs were
    affected from them is guesswork after the fact. The one row this produced
    is named, because naming it is honest and a pattern match would catch
    legitimate declines too. Issue #223.
    """
    rows = conn.execute(
        "SELECT p.id, p.name FROM proposals p JOIN verdicts v "
        "ON v.proposal_id = p.id "
        "WHERE v.id = (SELECT v2.id FROM verdicts v2 "
        "               WHERE v2.proposal_id = p.id ORDER BY v2.id DESC LIMIT 1) "
        "  AND v.tier = 'adopt' AND v.outcome = 'declined' "
        "  AND v.detail LIKE '%does not beat the incumbent%' "
        "  AND p.name = 'LiquidAI/LFM2.5-350M'").fetchall()
    now = time.time()
    for row in rows:
        conn.execute(
            "INSERT INTO verdicts (proposal_id, tier, outcome, detail, "
            "decided_at) VALUES (?, ?, 'queued', ?, ?)",
            (row[0], SCREEN,
             "retracted: measured against a control that passed 0 of 27, so "
             "the verdict described a run in which nothing ran", now))


#: How the inspect tier spells the registry's own task inside a description.
_CARD_TASK = re.compile(r"task ([a-z0-9-]+)")


def _relane_from_the_card(conn) -> None:
    """Correct a lane the SOURCE supplied, using the candidate's own card.

    A feed declares the subject area it covers; that was recorded as every
    candidate's lane and never overwritten, so four video models sat in the
    image lane and a TTS model in the code lane. Each was screened against
    cases it could not pass and recorded `broken` -- terminal -- for a
    mismatch this harness created.

    Only rows whose card CONTRADICTS the recorded lane are touched. A lane
    with no card to check stays put: it may be right, and guessing again
    would be no better than the guess already there. #227.
    """
    from harness import inspect as ins
    from harness import lanes

    rows = conn.execute(
        "SELECT id, name, lane, description FROM proposals "
        "WHERE description <> ''").fetchall()
    moved = []
    for row in rows:
        m = _CARD_TASK.search((row["description"] or "").lower())
        card = ins.PIPELINE_LANES.get(m.group(1)) if m else None
        if not card or lanes.canonical(row["lane"]) == card:
            continue
        conn.execute("UPDATE proposals SET lane = ? WHERE id = ?",
                     (card, row["id"]))
        moved.append((row["id"], row["name"], row["lane"], card))

    # A terminal verdict reached in the WRONG LANE says nothing about the
    # candidate: the screen built its spec from the lane and handed it cases
    # from a modality it does not serve. Re-queued, not deleted.
    now = time.time()
    for pid, name, was, card in moved:
        last = conn.execute(
            "SELECT outcome FROM verdicts WHERE proposal_id = ? "
            "ORDER BY id DESC LIMIT 1", (pid,)).fetchone()
        if not last or last["outcome"] not in TERMINAL:
            continue
        conn.execute(
            "INSERT INTO verdicts (proposal_id, tier, outcome, detail, "
            "decided_at) VALUES (?, ?, 'queued', ?, ?)",
            (pid, SCREEN,
             f"retracted: settled in the {was or 'unknown'} lane, which came "
             f"from the source rather than from {name}'s own card ({card})",
             now))


def _columns(conn, table: str) -> set[str]:
    """Column names of one table, asked of whichever backend this is."""
    if store.backend() == store.POSTGRES:
        rows = conn.execute(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = ?", (table,))
        return {r["column_name"] for r in rows}
    return {r["name"] for r in conn.execute(f"PRAGMA table_info({table})")}


#: Evidence a v2 store already holds about where a name came from, strongest
#: first. Each rule only fills rows still empty, so a URL beats a kind: the
#: inspect tier wrote kind='repo' for a GitHub repo while from_feeds wrote the
#: same kind for a HuggingFace one, and the URL it recorded alongside says
#: which is which.
_BACKFILL = (
    ("url", "https://huggingface.co/%", HUGGINGFACE),
    ("url", "https://github.com/%", GITHUB),
    ("kind", "weights", HUGGINGFACE),
    ("kind", "tool", GITHUB),
    ("kind", "repo", HUGGINGFACE),
)


def _backfill_registry(conn) -> int:
    """Fill `registry` from what the store already recorded. Returns the count.

    A row with no evidence keeps the empty string. Guessing one for it would
    turn "nobody knows" into a wrong answer that nothing ever revisits, which
    is the failure this column exists to end.
    """
    filled = 0
    for field, value, registry in _BACKFILL:
        if field == "url":
            cur = conn.execute(
                "UPDATE proposals SET registry = ? WHERE registry = '' "
                "AND id IN (SELECT proposal_id FROM sightings WHERE url LIKE ?)",
                (registry, value))
        else:
            cur = conn.execute(
                "UPDATE proposals SET registry = ? WHERE registry = '' "
                "AND kind = ?", (registry, value))
        filled += cur.rowcount or 0
    conn.commit()
    return filled


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
    #: One of REGISTRIES, or empty when the caller genuinely cannot say.
    registry: str = ""
    #: What the registry says this IS, as opposed to what one source said
    #: about it. Only filled by a tier that read the registry.
    description: str = ""


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
            "  lane = CASE WHEN lane='' THEN ? ELSE lane END, "
            "  registry = CASE WHEN registry='' THEN ? ELSE registry END, "
            "  description = CASE WHEN ?<>'' THEN ? ELSE description END "
            "WHERE id = ?",
            (now, seen.resolved, seen.resolved, seen.lane, seen.registry,
             seen.description, seen.description, pid))
    else:
        pid = conn.execute(
            "INSERT INTO proposals (name, kind, registry, lane, resolved, "
            "description, first_seen, last_seen) VALUES (?,?,?,?,?,?,?,?)",
            (seen.name, seen.kind, seen.registry, seen.lane, seen.resolved,
             seen.description, now, now)).lastrowid
    conn.execute(
        "INSERT OR IGNORE INTO sightings (proposal_id, source, url, why, "
        "relevance, seen_at) VALUES (?,?,?,?,?,?)",
        (pid, seen.source, seen.url, seen.why, seen.relevance, now))
    conn.commit()
    return pid


def set_lane(conn, name: str, lane: str) -> bool:
    """Record a lane READ FROM THE REGISTRY, overwriting a guess.

    `record()` keeps the first non-empty lane, which is right for a value
    nothing can improve on. A lane is not that: the sweep can only guess from
    prose, and the inspect tier later reads the publisher's own task off the
    card. Without a way to correct it, the guess is permanent -- which is how
    four video models stayed in the image lane. #227.

    Returns whether anything changed, so a caller can say so.
    """
    from harness import lanes

    want = lanes.canonical(lane)
    if not want:
        return False
    row = conn.execute("SELECT id, lane FROM proposals WHERE name = ?",
                       (name,)).fetchone()
    if not row or lanes.canonical(row["lane"]) == want:
        return False
    conn.execute("UPDATE proposals SET lane = ? WHERE id = ?", (want, row["id"]))
    return True


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
    # A DETERMINISTIC TIER RESTATING ITSELF IS NOT A SECOND FACT. A judge
    # re-scoring is a new draw and a run is a new run, so the skip is narrow:
    # no score, no run path, and the LATEST row said exactly this. Issue #184.
    #
    # LATEST, not any earlier row of the tier. Matching any of them made every
    # retraction permanent: a re-queued candidate re-screened green, decide()
    # found the old `screened` row three verdicts back, wrote nothing, and the
    # retraction stayed the latest verdict forever. Issue #225.
    if score is None and not run_path:
        same = conn.execute(
            "SELECT id, tier, outcome, detail, score, run_path FROM verdicts "
            "WHERE proposal_id = ? ORDER BY id DESC LIMIT 1",
            (row["id"],)).fetchone()
        if (same and same["tier"] == tier and same["outcome"] == outcome
                and same["detail"] == detail and same["score"] is None
                and not same["run_path"]):
            return same["id"]
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


def pending(conn, limit: int = 50, registry: str | None = None) -> list[str]:
    """Proposals nothing has answered yet, most-corroborated first.

    THE MISSING RUNG. The sweep writes proposals and every later tier read a
    different source -- inspect went to the crowd, so 233 swept proposals sat in
    the store with nothing consuming them. The ladder in #148 is sweep ->
    inspect -> judge -> screen -> measure, and without this the first arrow
    does not exist.

    Ordered by how many independent sightings a name has, because that is the
    project's own answer to a feed measuring popularity: a thing that keeps
    coming back is a different signal from a thing that trended once. Ties
    break on recency so a fresh proposal is not stuck behind an old one.

    A terminal verdict removes a name for good; `queued` and `screened` do not,
    because those are waypoints rather than answers.

    `registry` narrows to names one registry can answer for, and a caller that
    resolves names SHOULD pass it: the tier that clones from GitHub asked
    GitHub about HuggingFace model ids and 227 of 235 came back 404 (#167).

    None means every registry INCLUDING the unknown ones, which is right for a
    report and wrong for resolving. The EMPTY STRING asks for the unknown ones
    on their own -- names the store cannot route, which is work waiting on one
    question rather than work nobody can do.
    """
    where = "" if registry is None else "AND p.registry = ?"
    args = (() if registry is None else (registry,)) + TERMINAL + (limit,)
    q = f"""
        SELECT p.name, COUNT(s.id) AS times, MAX(s.seen_at) AS last_seen
        FROM proposals p JOIN sightings s ON s.proposal_id = p.id
        WHERE p.resolved <> '' {where} AND p.name NOT IN (
            SELECT DISTINCT p2.name FROM proposals p2
            JOIN verdicts v ON v.proposal_id = p2.id
            WHERE v.outcome IN ({','.join('?' * len(TERMINAL))})
        )
        GROUP BY p.id
        ORDER BY times DESC, last_seen DESC
        LIMIT ?
    """
    return [r["name"] for r in conn.execute(q, args)]


def set_registry(conn, name: str, registry: str) -> None:
    """Record which registry answered for a name, once something has asked.

    The migration fills what the store already proves and stops there, so a
    name that arrived as prose keeps an empty registry: the sweep resolved it
    against a registry and did not write down which one. This is how that
    answer gets back in -- found by asking, not by guessing from the shape of
    the string.
    """
    if registry not in REGISTRIES:
        raise ValueError(f"{registry!r} is not one of {', '.join(REGISTRIES)}")
    conn.execute("UPDATE proposals SET registry = ? WHERE name = ?",
                 (registry, name))
    conn.commit()


def survivors(conn, limit: int = 10) -> list[dict]:
    """Candidates whose LATEST verdict is a passed screen, newest first.

    The latest verdict, not any verdict: a candidate that screened green and
    was later measured and declined must not be handed to the measure tier
    again every time the loop runs.
    """
    rows = conn.execute("""
        SELECT p.name, p.lane, v.detail
        FROM proposals p JOIN verdicts v ON v.proposal_id = p.id
        WHERE v.id = (SELECT v2.id FROM verdicts v2
                      WHERE v2.proposal_id = p.id ORDER BY v2.id DESC LIMIT 1)
          AND v.tier = ? AND v.outcome = 'screened'
        ORDER BY v.id DESC LIMIT ?
    """, (SCREEN, limit)).fetchall()
    return [dict(r) for r in rows]


def by_registry(conn) -> dict[str, int]:
    """How many unanswered proposals each registry owns, "" being the ones
    nothing can resolve. Reported rather than hidden: a name with no registry
    is work nobody can do, and it should be visible as that rather than as a
    404 from whichever tier guessed."""
    out = {}
    for r in conn.execute(
            "SELECT p.registry AS registry, COUNT(*) AS n FROM proposals p "
            "WHERE p.resolved <> '' AND p.id NOT IN "
            f"(SELECT proposal_id FROM verdicts WHERE outcome IN "
            f"({','.join('?' * len(TERMINAL))})) GROUP BY p.registry", TERMINAL):
        out[r["registry"]] = r["n"]
    return out


def judgeable(conn, limit: int = 50) -> list[dict]:
    """What the inspect tier queued and no judge has scored, best-corroborated
    first, with everything judge.describe() is shown.

    THE SAME MISSING RUNG ONE TIER ALONG. _judge_fits() scores the Fit objects
    sitting in memory from the inspect run that produced them, so a judge can
    only ever see candidates inspected in the same process. Everything the
    store already holds is unreachable, and after #167 that is a queue of real
    candidates with real verdicts that nothing ranks.

    A proposal already scored by a judge is not returned. Re-scoring it would
    cost a model call to learn what is already recorded, and the rubric and
    judge model are recorded beside the score, so a run under a NEW rubric is
    told apart by that rather than by scoring everything again.
    """
    q = """
        SELECT p.name, p.lane, p.registry, p.kind, p.description,
               COUNT(s.id) AS times, MAX(s.relevance) AS relevance,
               MAX(s.seen_at) AS last_seen,
               MIN(s.source) AS source, MIN(s.why) AS why,
               (SELECT v.detail FROM verdicts v
                 WHERE v.proposal_id = p.id AND v.tier = ?
                 ORDER BY v.id DESC LIMIT 1) AS inspected
        FROM proposals p JOIN sightings s ON s.proposal_id = p.id
        WHERE p.id IN (
            SELECT proposal_id FROM verdicts
            WHERE tier = ? AND outcome = 'queued'
        ) AND p.id NOT IN (
            SELECT proposal_id FROM verdicts WHERE tier = ?
        )
        GROUP BY p.id
        ORDER BY times DESC, last_seen DESC
        LIMIT ?
    """
    return [dict(r) for r in conn.execute(q, (INSPECT, INSPECT, JUDGE, limit))]


def judgeable_total(conn) -> int:
    """How many candidates are waiting, whatever one run's budget is. A tier
    that takes the top 25 of a backlog and says nothing about the rest reads as
    finished."""
    return len(judgeable(conn, limit=1_000_000))


def ranked(conn, limit: int = 50) -> list[dict]:
    """Everything a judge has scored, best first. What the tier is FOR."""
    q = """
        SELECT p.name, p.lane, v.score, v.rubric, v.judge, v.detail,
               v.decided_at
        FROM proposals p JOIN verdicts v ON v.proposal_id = p.id
        WHERE v.tier = ? AND v.score IS NOT NULL
        ORDER BY v.score DESC, v.id DESC
        LIMIT ?
    """
    return [dict(r) for r in conn.execute(q, (JUDGE, limit))]


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
        -- COUNT repeated rather than `HAVING times >= ?`. SQLite lets HAVING
        -- see a select-list alias and standard SQL does not, so the alias
        -- version runs here and raises UndefinedColumn on Postgres. ORDER BY
        -- may use the alias in both, which is why it still does.
        GROUP BY p.id HAVING COUNT(s.id) >= ?
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


#: Why a name pulled out of prose did not become a proposal.
REJECTIONS = ("unresolvable", "not-a-repo", "duplicate", "already-measured",
              "below-relevance", "settled")


def reject(conn: sqlite3.Connection, name: str, source: str, reason: str,
           at: float | None = None) -> None:
    """Record a name that was extracted and thrown away.

    The thrown-away ones are the whole measurement. A store holding only what
    survived can report that 100% of proposals resolved, which is true and
    means nothing.
    """
    if reason not in REJECTIONS:
        raise ValueError(f"unknown rejection {reason!r}, known: "
                         f"{', '.join(REJECTIONS)}")
    conn.execute("INSERT OR IGNORE INTO extractions (name, source, reason, at) "
                 "VALUES (?,?,?,?)",
                 (name[:200], source, reason,
                  time.time() if at is None else at))
    conn.commit()


def extraction(conn: sqlite3.Connection) -> list[dict]:
    """Per source: how many extracted names survived, and why the rest did not.

    Issue #49 asked for extraction precision. This is the number.
    """
    kept = {r["source"]: r["n"] for r in conn.execute(
        "SELECT source, COUNT(DISTINCT proposal_id) n FROM sightings "
        "GROUP BY source")}
    out = []
    for source in sorted(set(kept) | {r["source"] for r in conn.execute(
            "SELECT DISTINCT source FROM extractions")}):
        reasons = {r["reason"]: r["n"] for r in conn.execute(
            "SELECT reason, COUNT(*) n FROM extractions WHERE source=? "
            "GROUP BY reason", (source,))}
        dropped = sum(reasons.values())
        k = kept.get(source, 0)
        out.append({"source": source, "kept": k, "dropped": dropped,
                    "extracted": k + dropped,
                    "precision": (k / (k + dropped)) if (k + dropped) else 0.0,
                    "reasons": reasons})
    return sorted(out, key=lambda r: -r["extracted"])


def parents(conn: sqlite3.Connection, name: str,
            relation: str = "needs") -> list[str]:
    """Proposals with an edge INTO `name`. Which repos named this weight."""
    return [r["name"] for r in conn.execute(
        "SELECT src.name FROM edges e JOIN proposals src ON src.id = e.src "
        "JOIN proposals dst ON dst.id = e.dst "
        "WHERE dst.name = ? AND e.relation = ?", (name, relation))]


def retire_unlisted(conn: sqlite3.Connection, name: str, keep,
                    relation: str = "needs", reason: str = "",
                    outcome: str = "ignored") -> list[str]:
    """Retire things `name` queued that it no longer ranks.

    A queue entry is a decision a ranking made at a point in time, and when the
    ranking changes every decision it made is suspect. Re-inspecting a repo used
    to only ADD, so the queue mixed picks from rules that no longer exist.

    A weight named by TWO repos is not this one's to retire: if any other parent
    still ranks it, it stays. Returns what was retired.
    """
    keep = set(keep)
    retired = []
    rows = conn.execute(
        "SELECT dst.name AS name FROM edges e JOIN proposals src ON src.id = e.src "
        "JOIN proposals dst ON dst.id = e.dst "
        "WHERE src.name = ? AND e.relation = ?", (name, relation)).fetchall()
    for row in rows:
        other = row["name"]
        if other in keep:
            continue
        cur = conn.execute(
            "SELECT outcome FROM verdicts v JOIN proposals p ON p.id = v.proposal_id "
            "WHERE p.name = ? ORDER BY v.id DESC LIMIT 1", (other,)).fetchone()
        if not cur or cur["outcome"] != "queued":
            continue
        if any(p != name for p in parents(conn, other, relation)):
            continue      # another repo still names it; not ours to retire
        decide(conn, other, outcome, tier="inspect", detail=reason[:200])
        retired.append(other)
    return retired


def by_source(conn: sqlite3.Connection) -> list[dict]:
    """Per source: how many it proposed, how many resolved, how many settled.

    Issue #49 asked for extraction precision and it was impossible to answer
    without a store. It is a query now, and it compares sources against each
    other rather than reporting one number for all of them.
    """
    return [dict(r) for r in conn.execute("""
        SELECT s.source AS source,
               COUNT(DISTINCT s.proposal_id) AS proposals,
               COUNT(DISTINCT CASE WHEN p.resolved <> '' THEN p.id END) AS resolved,
               COUNT(DISTINCT CASE WHEN v.outcome IN ('measured','declined',
                     'broken','ignored') THEN p.id END) AS settled,
               COUNT(DISTINCT CASE WHEN v.outcome = 'measured' THEN p.id END)
                     AS measured
        FROM sightings s
        JOIN proposals p ON p.id = s.proposal_id
        LEFT JOIN verdicts v ON v.proposal_id = p.id
        GROUP BY s.source ORDER BY proposals DESC""")]


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
            -- CAST is load-bearing, not decoration. Postgres infers the
            -- parameter's type from the non-recursive term, decides smallint,
            -- and then refuses the recursive term where the column is integer:
            -- "column 1 has type smallint in the non-recursive term but type
            -- integer overall". SQLite accepts the cast and ignores it.
            SELECT CAST(? AS INTEGER), 0, ''
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
