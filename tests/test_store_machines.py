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


# --- the measured size, which code reads back -----------------------------

def test_a_measured_size_is_a_column_not_a_sentence(store):
    """It lived in `detail` as `bytes=N`, then as `weights from 5.5 to 8.9
    GiB` when the tier was reworded, and fetching.size_of parses BOTH with
    regexes because the rewording silently broke the numeric read: every
    candidate came back unsized and was declined TERMINALLY for a size sitting
    in the row above (#211)."""
    from harness import fetching

    ms.decide(store, "org/c", "queued", tier="inspect",
              detail="fits: MLX-native", size_bytes=5_500_000_000)
    row = store.execute(
        "SELECT size_bytes FROM verdicts ORDER BY id DESC LIMIT 1").fetchone()
    assert row["size_bytes"] == 5_500_000_000
    assert fetching.size_of(dict(row)) == 5_500_000_000


def test_the_prose_is_still_read_when_the_column_is_empty(store):
    """A row the migration could not parse still has its size in the
    sentence. Dropping the regexes would re-lose exactly the rows size_of was
    written for."""
    from harness import fetching

    assert fetching.size_of({"detail": "bytes=1234", "size_bytes": 0}) == 1234
    assert fetching.size_of(
        {"detail": "fits: weights from 0.1 to 8.9 GiB"}) == int(8.9 * 1024 ** 3)


def test_the_column_wins_over_the_prose(store):
    """Both present and disagreeing is the state every row is in right after
    the migration, since the prose is left alone. The column is the authority
    or there was no point moving it."""
    from harness import fetching

    assert fetching.size_of({"detail": "bytes=1", "size_bytes": 99}) == 99


# --- evidence -------------------------------------------------------------

def test_a_verdict_whose_run_is_gone_is_reported(store):
    ms.decide(store, "org/c", "measured", tier="measure",
              run_path="/nowhere/runs/gone")
    got = ms.dangling_receipts(store, exists=lambda p: False)
    assert [r["run_path"] for r in got] == ["/nowhere/runs/gone"]


def test_a_relative_run_path_is_resolved_before_it_is_called_missing(store):
    """THE FINDER'S OWN VERSION OF THE DEFECT IT FINDS. This first reported 7
    of 55 receipts gone; six were `runs/cycle-screen` and friends, relative to
    paths.home() and present. Reading them against the process cwd made a
    healthy store look half-rotten."""
    from harness import paths

    ms.decide(store, "org/c", "screened", tier="screen",
              run_path="runs/cycle-screen")
    home = str(paths.home() / "runs/cycle-screen")
    assert not ms.dangling_receipts(store, exists=lambda p: p == home), (
        "a receipt that exists under the project home was called missing")
    assert ms.dangling_receipts(store, exists=lambda p: False)


def test_a_run_path_that_is_not_a_path_is_refused(store):
    """One row in the real store holds `ok`, because a snapshot stand-in
    returned that string and the column took it. A verdict claiming evidence
    it cannot produce is worse than one claiming none."""
    with pytest.raises(ValueError, match="is not a path"):
        ms.decide(store, "org/c", "queued", tier="fetch", run_path="ok")
    ms.decide(store, "org/c", "queued", tier="fetch", run_path="runs/a")
    ms.decide(store, "org/c", "queued", tier="fetch", run_path="/abs/b")


# --- the attachment kind and the source's age -----------------------------

def test_the_attachment_kind_is_a_column(store):
    """It was `lora in its own card: this attaches to a model...`. The word
    that decided it is the fact; the sentence is the explanation."""
    ms.decide(store, "org/c", "declined", tier="fetch", attaches_to="lora",
              detail="lora in its own card: this attaches to a model rather "
                     "than being one, and no lane can run it alone")
    row = store.execute(
        "SELECT attaches_to FROM verdicts ORDER BY id DESC LIMIT 1").fetchone()
    assert row["attaches_to"] == "lora"


def test_not_an_attachment_is_distinguishable_from_nobody_asking(store):
    """"" is the answer for almost every row, and it has to mean "asked, and
    no" rather than being indistinguishable from an unfilled column."""
    ms.decide(store, "org/c", "queued", tier="inspect", detail="fits")
    row = store.execute(
        "SELECT attaches_to FROM verdicts ORDER BY id DESC LIMIT 1").fetchone()
    assert row["attaches_to"] == ""


def test_the_upstream_idle_time_is_a_number_not_one_decimal_of_years(store):
    """`last commit 2.9 years ago` is a number a reader acts on and no query
    can reach.

    THE NAME SAYS WHOSE. It was `stale_days` for a day and was read as the age
    of our own row -- this project is days old and the repos it refuses this
    way are years idle. A column named for a fact without its subject is the
    same defect as machine_id() answering `arm64`."""
    ms.decide(store, "org/c", "declined", tier="inspect",
              upstream_idle_days=1058.5,
              detail="dead: last commit 2.9 years ago")
    row = store.execute("SELECT upstream_idle_days FROM verdicts "
                        "ORDER BY id DESC LIMIT 1").fetchone()
    assert row["upstream_idle_days"] == pytest.approx(1058.5)


def test_a_judge_using_the_word_workflow_is_not_a_verdict_about_one(tmp_path):
    """THE MIGRATION'S OWN NEAR-MISS, and the reason it recovers rather than
    recomputes.

    The first cut ran screen.is_attachment over the verdict's DETAIL. The live
    code runs it over the candidate's DESCRIPTION, so 78 judge verdicts whose
    prose happens to contain "workflow", "gui" or "embedding" came back
    labelled attachments -- and "This is a composition of existing tools" is a
    judge explaining a score, not a declaration that a candidate is an adapter.
    Shipping it would have taught the fetch tier to refuse 57 real candidates.
    """
    path = tmp_path / "old.db"
    old = sqlite3.connect(path)
    old.executescript("""
        CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
        INSERT INTO meta VALUES ('schema','13');
        CREATE TABLE proposals (id INTEGER PRIMARY KEY, name TEXT,
            kind TEXT DEFAULT '', lane TEXT DEFAULT '',
            resolved TEXT DEFAULT '', consumes TEXT DEFAULT '',
            produces TEXT DEFAULT '', first_seen REAL DEFAULT 0,
            last_seen REAL DEFAULT 0, registry TEXT DEFAULT '',
            description TEXT DEFAULT '');
        CREATE TABLE verdicts (id INTEGER PRIMARY KEY, proposal_id INTEGER,
            outcome TEXT, tier TEXT DEFAULT '', detail TEXT DEFAULT '',
            issue INTEGER, run_path TEXT DEFAULT '', score REAL,
            rubric TEXT DEFAULT '', judge TEXT DEFAULT '', decided_at REAL,
            machine_id INTEGER, until TEXT DEFAULT '',
            size_bytes INTEGER DEFAULT 0);
        CREATE TABLE machines (id INTEGER PRIMARY KEY, fingerprint TEXT UNIQUE,
            hw_model TEXT DEFAULT '', os TEXT DEFAULT '', arch TEXT DEFAULT '',
            memory_gb REAL DEFAULT 0, accelerator TEXT DEFAULT '',
            runtimes TEXT DEFAULT '', ceiling_gb REAL DEFAULT 0,
            first_seen REAL DEFAULT 0, last_seen REAL DEFAULT 0);
        INSERT INTO proposals (id,name) VALUES (1,'org/judged'),(2,'org/lora'),
                                               (3,'org/old');
        INSERT INTO verdicts (proposal_id,outcome,tier,detail,decided_at)
          VALUES (1,'queued','judge',
                  'This is a composition of existing tools with a clean gui and a reusable workflow for embedding extraction',0),
                 (2,'declined','fetch',
                  'lora in its own card: this attaches to a model rather than being one, and no lane can run it alone',0),
                 (3,'declined','inspect','dead: last commit 2.9 years ago',0);
    """)
    old.commit()
    old.close()

    conn = ms.connect(path)
    try:
        got = {r["proposal_id"]: (r["attaches_to"], r["upstream_idle_days"])
               for r in conn.execute(
                   "SELECT proposal_id, attaches_to, upstream_idle_days "
                   "FROM verdicts")}
        assert got[1] == ("", 0.0), (
            f"a judge's prose was read as a verdict about an attachment: "
            f"{got[1]}")
        assert got[2][0] == "lora"
        assert got[3][1] == pytest.approx(2.9 * 365.0)
    finally:
        conn.close()


# --- a refusal that waits on somebody else's repository -------------------

def test_a_dead_upstream_can_be_reconsidered_when_it_commits(store):
    """A repo idle two years is refused, and that verdict is `declined`,
    which is TERMINAL. So the candidate stayed refused even after its upstream
    shipped -- and `mlx-community/Mistral-7B-Instruct-v0.3-4bit` is on that
    list, where a requantisation repo has no reason to receive commits at all.

    Every other machine-limited refusal got a condition in #266. This one was
    missed because its limit is not the machine."""
    ms.decide(store, "org/c", "declined", tier="inspect",
              upstream_idle_days=1058.5,
              until="commit_after:2023-10-22T03:10:14Z",
              detail="dead: last commit 2.9 years ago")
    fresh = {"fingerprint": "elsewhere", "last_commit": "2026-09-01T00:00:00Z"}
    assert ms.revisitable(store, fresh), "a revived upstream stays refused"
    stale = {"fingerprint": "elsewhere", "last_commit": "2023-01-01T00:00:00Z"}
    assert not ms.revisitable(store, stale), (
        "an upstream that has NOT moved was offered for reconsideration")


def test_an_unparseable_commit_date_does_not_reopen_everything():
    """Not-met is the safe direction: a registry field that is not ISO-8601
    must leave the verdict standing rather than re-queueing the corpus."""
    assert not ms.until_met("commit_after:2023-10-22T03:10:14Z",
                            {"last_commit": "last Tuesday"})
    assert not ms.until_met("commit_after:2023-10-22T03:10:14Z",
                            {"last_commit": ""})
    assert not ms.until_met("commit_after:2023-10-22T03:10:14Z", {})


def test_the_machine_conditions_still_ignore_an_upstream_fact():
    """The negative control for mixing two kinds of condition in one column.
    A machine with cuda must not satisfy a verdict waiting on a commit."""
    assert not ms.until_met("commit_after:2023-01-01T00:00:00Z",
                            {"runtimes": "cpu,cuda", "memory_gb": 61.0})
    assert not ms.until_met("runtime:cuda",
                            {"last_commit": "2026-09-01T00:00:00Z"})
