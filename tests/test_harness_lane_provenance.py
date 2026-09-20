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


# --- a card's decisive tag must survive the summary (#201) ----------------

def test_a_lora_tag_survives_the_six_tag_cap():
    """Xanthius/Ace-Step-1.5-XL-Concept-Sliders ranked TOP OF THE QUEUE at
    +5.3 and is an adapter. HF tagged it `lora` plainly -- and `lora` was the
    seventh tag, so `plain[:6]` dropped the one word that decides whether a
    runner can load it at all.

    A summary may drop what a reader would merely like to know. It may not
    drop what the next decision is made on.
    """
    from harness import inspect as ins, screen
    data = {"tags": ["music", "audio", "sound", "singing", "concepts",
                     "slider", "lora"],
            "siblings": [{"size": 700 * 1024 ** 2}]}
    assert screen.is_attachment(ins.card_description(data)) == "lora"


def test_the_base_model_relation_type_travels_with_the_id():
    """HF types the lineage -- base_model:adapter:X against
    base_model:finetune:X -- and describe() kept only the id. That threw away
    the field separating a thing that runs from a thing that attaches to
    something that runs."""
    from harness import inspect as ins, screen
    adapter = {"tags": ["base_model:adapter:ACE-Step/acestep-v15-xl-base"],
               "siblings": [{"size": 700 * 1024 ** 2}]}
    got = ins.card_description(adapter)
    assert "adapter of ACE-Step/acestep-v15-xl-base" in got
    assert screen.is_attachment(got)


def test_a_genuine_finetune_is_not_called_an_attachment():
    """THE NEGATIVE CONTROL, and the half that matters most. A finetune
    declares a base model too and IS runnable; refusing those would empty the
    queue of exactly the candidates worth screening."""
    from harness import inspect as ins, screen
    finetune = {"tags": ["text-generation", "mlx",
                         "base_model:finetune:Qwen/Qwen3-4B"],
                "pipeline_tag": "text-generation",
                "siblings": [{"size": 4 * 1024 ** 3}]}
    got = ins.card_description(finetune)
    assert "built from Qwen/Qwen3-4B" in got
    assert not screen.is_attachment(got)


def test_an_ordinary_model_card_is_untouched():
    from harness import inspect as ins, screen
    plain = {"tags": ["text-generation", "mlx", "qwen", "chat", "instruct"],
             "pipeline_tag": "text-generation", "library_name": "mlx",
             "siblings": [{"size": 4 * 1024 ** 3}]}
    assert not screen.is_attachment(ins.card_description(plain))


# --- registry over registry, before registry over prose (#246) ------------

REAL_OMNISVG = {"pipeline_tag": "text-generation",
                "tags": ["pytorch", "qwen2_5_vl", "SVG", "Image-to-SVG",
                         "Text-to-SVG", "text-generation", "en", "zh"]}


def test_a_supertype_pipeline_tag_yields_to_a_specific_one():
    """OmniSVG1.1_8B declares `text-generation` and is tagged SVG three times.
    It was filed under `code`, where the lane hands a candidate to
    mlx_lm.server -- a path that cannot run it, so the screen would have
    recorded a verdict about the candidate.

    The publisher is not wrong: a text-to-SVG model IS a text-generation
    model. `text-generation` is simply the supertype of four lanes at once.
    """
    from harness import inspect as ins
    assert ins.lane_for(REAL_OMNISVG) == "svg"


def test_an_ordinary_text_model_still_lands_in_code():
    """THE NEGATIVE CONTROL. This is a routing change, and routing changes are
    how lanes get poisoned: if incidental tags could move a candidate, the
    code lane would empty into whatever its cards happen to mention."""
    from harness import inspect as ins
    assert ins.lane_for({"pipeline_tag": "text-generation",
                         "tags": ["qwen3", "text-generation", "chat",
                                  "conversational", "reasoning"]}) == "code"


def test_tags_naming_two_lanes_leave_the_publishers_answer_standing():
    """Ambiguity is not a tiebreak -- the rule lanes.from_prose already
    follows. With the tags disagreeing among themselves, pipeline_tag is a
    better fallback than a coin flip."""
    from harness import inspect as ins
    assert ins.lane_for({"pipeline_tag": "text-generation",
                         "tags": ["text-to-svg", "tts"]}) == "code"


def test_a_specific_pipeline_tag_is_never_overruled():
    """Only `text-generation` is a supertype. Every other PIPELINE_LANES entry
    already names exactly one lane."""
    from harness import inspect as ins
    assert ins.lane_for({"pipeline_tag": "text-to-speech",
                         "tags": ["tts", "text-to-image"]}) == "tts"
    assert ins.lane_for({"pipeline_tag": "text-to-image",
                         "tags": ["diffusion", "svg"]}) == "image"
