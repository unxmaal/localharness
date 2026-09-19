"""A source's lane is its coverage; a candidate's lane is its own. Issue #227.

`DEFAULT_SOURCES` gives each feed the subject area it covers. That was recorded
as every candidate's lane, and a recorded lane was never overwritten, so the
real lane read off the model card could never land: four video models sat in
the image lane and a TTS model in the code lane, each screened against cases it
could not pass and recorded BROKEN for a mismatch the harness created.
"""
import pytest

from harness import feeds, lanes
from harness import memory_store as ms


def _latest(conn, name):
    """The verdict every tier actually reads."""
    return conn.execute(
        "SELECT v.outcome, v.detail FROM verdicts v JOIN proposals p "
        "ON p.id = v.proposal_id WHERE p.name = ? ORDER BY v.id DESC LIMIT 1",
        (name,)).fetchone()


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def _seed(conn, name, lane, description=""):
    ms.record(conn, ms.Seen(name=name, source="reddit-sd-week", url="",
                            why="", relevance=0, kind="candidate",
                            registry=ms.HUGGINGFACE, lane=lane,
                            resolved=name, description=description))


def test_every_default_source_declares_a_coverage_lane():
    """The premise. If a source stopped declaring one, this test is the wrong
    shape rather than passing vacuously."""
    declared = {s.lane for s in feeds.DEFAULT_SOURCES if s.lane}
    assert declared, "no source declares a lane; #227's premise is gone"
    assert declared & {"image", "tts", "all", "text"}


def test_a_candidates_lane_never_comes_from_its_source(store, monkeypatch):
    """A sweep from an image-lane feed that finds a speech model must record
    `tts`, from the candidate's own prose, not `image` from the feed."""
    from harness import discover

    src = feeds.Source("reddit-sd-week", "http://x", lane="image")
    prop = feeds.Proposal("org/a-speech-model",
                          "the smallest complete text-to-speech stack",
                          "reddit-sd-week", "http://x/p", "", "repo", 0)
    monkeypatch.setattr(feeds, "candidates", lambda batch, origin: [prop])
    monkeypatch.setattr(discover, "measured", lambda: set())
    monkeypatch.setattr(discover, "_was_measured", lambda *a, **k: False)

    discover.from_feeds(sources=[src], reader=lambda s: [object()],
                        verify=False, store=store)

    row = store.execute("SELECT lane FROM proposals WHERE name = ?",
                        ("org/a-speech-model",)).fetchone()
    assert row, "the proposal was not recorded at all"
    assert row["lane"] == "tts", (
        f"recorded {row['lane']!r}; the feed's `image` is its coverage, "
        f"not this candidate's lane")
    assert src.lane == "image", "the source still declares what it covers"


def test_the_card_overwrites_a_guessed_lane(store):
    """record() keeps the first non-empty lane, which is right for a value
    nothing can improve on. A lane is not that."""
    _seed(store, "org/x", "image")
    assert ms.set_lane(store, "org/x", "video") is True
    assert store.execute("SELECT lane FROM proposals WHERE name='org/x'"
                         ).fetchone()["lane"] == "video"


def test_setting_the_same_lane_changes_nothing(store):
    """The negative control: no write, no spurious 'corrected' line."""
    _seed(store, "org/x", "video")
    assert ms.set_lane(store, "org/x", "video") is False
    assert ms.set_lane(store, "org/x", "") is False


def test_an_alias_is_canonicalised_before_it_is_compared(store):
    """`text` IS `code`, so this is not a correction and must not report one."""
    _seed(store, "org/x", "code")
    assert ms.set_lane(store, "org/x", "text") is False


def test_a_missing_proposal_is_not_an_error(store):
    assert ms.set_lane(store, "org/never-seen", "video") is False


# --- the migration --------------------------------------------------------

def test_the_migration_moves_a_row_its_card_contradicts(tmp_path):
    conn = ms.connect(tmp_path / "m.db")
    _seed(conn, "org/vid", "image", "task text-to-video; tagged video")
    # As a real row arrives: inspect queues it, then the screen settles it.
    ms.decide(conn, "org/vid", "queued", tier=ms.INSPECT, detail="bytes=1")
    ms.decide(conn, "org/vid", "broken", tier=ms.SCREEN,
              detail="it ran and passed nothing")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', '8')")
    conn.commit()
    conn.close()

    conn = ms.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT lane FROM proposals WHERE name='org/vid'"
                            ).fetchone()["lane"] == "video"
        last = _latest(conn, "org/vid")
        assert last["outcome"] == "queued", (
            "a terminal verdict reached in the wrong lane must be retracted")
        assert "came from the source" in last["detail"]
    finally:
        conn.close()


def test_the_migration_leaves_a_row_with_no_card_alone(tmp_path):
    """The negative control. A lane with nothing to check against may be
    right, and guessing again is no better than the guess already there."""
    conn = ms.connect(tmp_path / "m.db")
    _seed(conn, "org/bare", "image", "")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', '8')")
    conn.commit()
    conn.close()

    conn = ms.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT lane FROM proposals WHERE name='org/bare'"
                            ).fetchone()["lane"] == "image"
    finally:
        conn.close()


def test_the_migration_leaves_an_agreeing_row_alone(tmp_path):
    conn = ms.connect(tmp_path / "m.db")
    _seed(conn, "org/img", "image", "task text-to-image; tagged diffusion")
    ms.decide(conn, "org/img", "queued", tier=ms.INSPECT, detail="bytes=1")
    ms.decide(conn, "org/img", "declined", tier=ms.SCREEN, detail="too big")
    conn.execute("INSERT OR REPLACE INTO meta VALUES ('schema', '8')")
    conn.commit()
    conn.close()

    conn = ms.connect(tmp_path / "m.db")
    try:
        assert conn.execute("SELECT lane FROM proposals WHERE name='org/img'"
                            ).fetchone()["lane"] == "image"
        last = _latest(conn, "org/img")
        assert last["outcome"] == "declined", (
            "a verdict reached in the RIGHT lane stays settled")
    finally:
        conn.close()
