"""Issue #52: the discovery loop's memory."""
import pytest

from harness import memory_store as ms
from harness.memory_store import Seen


@pytest.fixture
def db(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def see(db, name, source="reddit", url="", why="", relevance=0, at=None, **kw):  # noqa: E501
    return ms.record(db, Seen(name=name, source=source, url=url, why=why,
                              relevance=relevance, **kw), at=at)


def test_seeing_the_same_thing_twice_is_two_sightings_one_proposal(db):
    """Recurrence is the signal. A second sighting must not become a second
    proposal or the count is meaningless."""
    a = see(db, "SDMLX", source="recap", url="https://x/1")
    b = see(db, "SDMLX", source="recap", url="https://x/2")
    assert a == b
    assert db.execute("SELECT COUNT(*) c FROM proposals").fetchone()["c"] == 1
    assert db.execute("SELECT COUNT(*) c FROM sightings").fetchone()["c"] == 2


def test_the_same_sighting_twice_is_recorded_once(db):
    """Re-reading a cached feed must not inflate recurrence."""
    see(db, "SDMLX", source="recap", url="https://x/1")
    see(db, "SDMLX", source="recap", url="https://x/1")
    assert db.execute("SELECT COUNT(*) c FROM sightings").fetchone()["c"] == 1


def test_first_and_last_seen_span_the_sightings(db):
    see(db, "SDMLX", url="a", at=1000.0)
    see(db, "SDMLX", url="b", at=5000.0)
    row = db.execute("SELECT first_seen, last_seen FROM proposals").fetchone()
    assert row["first_seen"] == 1000.0
    assert row["last_seen"] == 5000.0


def test_a_resolved_id_is_kept_once_learned(db):
    see(db, "Motif-3")
    see(db, "Motif-3", resolved="Motif-Technologies/Motif-3")
    see(db, "Motif-3")  # a later bare sighting must not erase it
    assert db.execute("SELECT resolved FROM proposals").fetchone()["resolved"] \
        == "Motif-Technologies/Motif-3"


# ---- verdicts --------------------------------------------------------------

def test_a_terminal_verdict_suppresses_re_proposal(db):
    """Every sweep re-proposing what was already declined is the failure that
    makes an operator stop reading the output."""
    see(db, "ComfyUI")
    see(db, "SDMLX")
    ms.decide(db, "ComfyUI", "declined", issue=20, detail="mflux ships them")
    assert ms.settled(db) == {"ComfyUI"}


def test_a_non_terminal_verdict_does_not_suppress(db):
    see(db, "SDMLX")
    ms.decide(db, "SDMLX", "queued")
    assert ms.settled(db) == set()


def test_verdicts_accumulate_rather_than_replace(db):
    """A screen result and a later measurement are two facts, and which tier
    produced a row decides whether it may be compared with another."""
    see(db, "LLaDA-Image")
    ms.decide(db, "LLaDA-Image", "screened", tier="screen", detail="ran")
    ms.decide(db, "LLaDA-Image", "measured", tier="measure", run_path="runs/x")
    rows = db.execute("SELECT outcome, tier FROM verdicts "
                      "ORDER BY id").fetchall()
    assert [r["tier"] for r in rows] == ["screen", "measure"]


def test_an_unknown_outcome_is_refused(db):
    see(db, "x")
    with pytest.raises(ValueError) as exc:
        ms.decide(db, "x", "vibes")
    assert "known:" in str(exc.value)


def test_a_verdict_on_an_unknown_proposal_is_refused(db):
    with pytest.raises(KeyError):
        ms.decide(db, "never-seen", "measured")


# ---- the questions that could not be answered before -----------------------

def test_recurrence_ranks_the_thing_that_keeps_coming_back(db):
    for i, u in enumerate(["a", "b", "c"]):
        see(db, "SDMLX", source="recap", url=u, at=1000.0 * (i + 1))
    see(db, "flash-in-the-pan", source="week", url="z")
    got = ms.recurrence(db, minimum=2)
    assert [r["name"] for r in got] == ["SDMLX"]
    assert got[0]["times"] == 3
    assert got[0]["last_seen"] - got[0]["first_seen"] == 2000.0


def test_recurrence_counts_distinct_sources(db):
    see(db, "SDMLX", source="recap", url="a")
    see(db, "SDMLX", source="week", url="b")
    assert ms.recurrence(db)[0]["sources"] == 2


def test_precision_is_a_query_not_a_manual_count(db):
    """Issue #49 asked for this number and it was impossible without a store."""
    for n in ["a", "b", "c", "d"]:
        see(db, n)
    see(db, "e", resolved="org/e")
    ms.decide(db, "a", "measured", run_path="runs/1")
    ms.decide(db, "b", "declined")
    p = ms.precision(db)
    assert p["proposals"] == 5
    assert p["resolved"] == 1
    assert p["verdict_measured"] == 1
    assert p["verdict_declined"] == 1


# ---- the graph -------------------------------------------------------------

def test_traversal_walks_forwards_and_backwards(db):
    for n in ["base", "finetune", "quant"]:
        see(db, n)
    ms.link(db, "base", "finetune", "base_model")
    ms.link(db, "finetune", "quant", "base_model")

    fwd = [r["name"] for r in ms.traverse(db, "base", direction="forward")]
    assert fwd == ["finetune", "quant"]

    back = [r["name"] for r in ms.traverse(db, "quant", direction="backward")]
    assert back == ["finetune", "base"]

    assert ms.traverse(db, "base", depth=1, direction="forward") == [
        {"name": "finetune", "kind": "candidate", "lane": "", "depth": 1,
         "relation": "base_model"}]


def test_traversal_terminates_on_a_cycle(db):
    for n in ["a", "b"]:
        see(db, n)
    ms.link(db, "a", "b", "r")
    ms.link(db, "b", "a", "r")
    got = ms.traverse(db, "a", depth=5)
    assert {r["name"] for r in got} <= {"a", "b"}


def test_composable_pairs_come_from_types_not_from_trying_everything(db):
    """47 proposals is 1,081 blind pairs, and most are nonsense: speech does not
    compose with an image upscaler."""
    for n in ["diffusion", "vectorizer", "transcriber"]:
        see(db, n)
    ms.types(db, "diffusion", produces="image")
    ms.types(db, "vectorizer", consumes="image", produces="svg")
    ms.types(db, "transcriber", consumes="audio", produces="text")
    assert ms.composable(db) == [("diffusion", "vectorizer")]


def test_an_edge_to_an_unknown_proposal_is_refused(db):
    see(db, "a")
    with pytest.raises(KeyError):
        ms.link(db, "a", "ghost", "base_model")


# ---- schema ----------------------------------------------------------------

def test_the_store_reopens_without_losing_anything(tmp_path):
    p = tmp_path / "d.db"
    c1 = ms.connect(p)
    see(c1, "SDMLX")
    ms.decide(c1, "SDMLX", "queued")
    c1.close()
    c2 = ms.connect(p)
    assert ms.recurrence(c2, minimum=1)[0]["name"] == "SDMLX"
    c2.close()


def test_a_newer_schema_is_refused_rather_than_migrated_backwards(tmp_path):
    """This is the one piece of state meant to outlive every other decision
    here, so a version it does not understand must not be written to."""
    p = tmp_path / "d.db"
    conn = ms.connect(p)
    conn.execute("UPDATE meta SET value = ? WHERE key='schema'",
                 (str(ms.SCHEMA_VERSION + 1),))
    conn.commit()
    conn.close()
    with pytest.raises(RuntimeError) as exc:
        ms.connect(p)
    assert "Refusing" in str(exc.value)


# ---- issue #49: extraction precision needs the DENOMINATOR ------------------

def test_a_store_of_survivors_alone_cannot_report_precision(db):
    """Every proposal in the store resolved to something real, so "100%
    resolved" is arithmetic, not precision. The names thrown away are the
    measurement."""
    see(db, "org/real", source="recap")
    got = ms.extraction(db)[0]
    assert got["precision"] == 1.0 and got["dropped"] == 0
    ms.reject(db, "RTX 3070", "recap", "unresolvable")
    ms.reject(db, "Diablo 4", "recap", "unresolvable")
    got = ms.extraction(db)[0]
    assert got["kept"] == 1 and got["dropped"] == 2
    assert got["extracted"] == 3
    assert got["precision"] == pytest.approx(1 / 3)


def test_rejections_are_broken_down_by_reason(db):
    ms.reject(db, "a", "recap", "unresolvable")
    ms.reject(db, "b", "recap", "not-a-repo")
    ms.reject(db, "c", "recap", "not-a-repo")
    assert ms.extraction(db)[0]["reasons"] == {"unresolvable": 1, "not-a-repo": 2}


def test_the_same_rejection_twice_is_recorded_once(db):
    """Re-reading a cached feed must not inflate the denominator, for the same
    reason a second sighting is not a second proposal."""
    for _ in range(3):
        ms.reject(db, "RTX 3070", "recap", "unresolvable")
    assert ms.extraction(db)[0]["dropped"] == 1


def test_an_unknown_rejection_reason_is_refused(db):
    with pytest.raises(ValueError) as exc:
        ms.reject(db, "x", "recap", "vibes")
    assert "known:" in str(exc.value)


def test_sources_are_compared_against_each_other(db):
    see(db, "org/a", source="crowd")
    ms.reject(db, "junk", "recap", "unresolvable")
    rows = {r["source"]: r for r in ms.extraction(db)}
    assert rows["crowd"]["precision"] == 1.0
    assert rows["recap"]["precision"] == 0.0
