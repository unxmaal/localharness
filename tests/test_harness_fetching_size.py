"""The fetch tier must find a size the inspect tier already measured. #211.

`queued()` returns the LATEST verdict, and the latest verdict format dropped
the `bytes=` field that `size_of` read, so every candidate came back unsized
and was declined -- terminally -- for a number sitting in the row above.
"""
import pytest

from harness import fetching

GIB = fetching.GIB


def test_the_exact_byte_count_is_read_where_it_exists():
    row = {"detail": "bytes=5911138787 lane=image named by mflux"}
    assert fetching.size_of(row) == 5911138787


def test_the_later_human_range_is_read_when_no_byte_count_survives():
    """`fits: weights from 5.5 to 5.5 GiB` replaced `bytes=` and nothing read
    it. The upper bound is taken: over-estimating refuses a fetch, under-
    estimating fills the disk."""
    row = {"detail": "fits: weights from 5.5 to 8.9 GiB"}
    assert fetching.size_of(row) == pytest.approx(8.9 * GIB, rel=1e-6)


def test_a_size_in_an_older_row_beats_a_newer_row_with_none():
    """The case that bit: the newest verdict is the refusal, which has no size
    in it at all, so reading only `detail` loses a fact the store holds."""
    row = {"detail": "no measured size; inspect it first",
           "sized": "bytes=5911138787 named by mflux"}
    assert fetching.size_of(row) == 5911138787


def test_the_exact_count_is_preferred_over_the_range():
    """bytes= is the SUM of the weights, which is what a download costs; the
    range is smallest-to-largest of individual files."""
    row = {"detail": "fits: weights from 5.5 to 5.5 GiB",
           "sized": "bytes=5911138787 named by mflux"}
    assert fetching.size_of(row) == 5911138787


def test_a_row_with_no_size_anywhere_is_still_zero():
    """The negative control. An unsized repo must not acquire a size."""
    assert fetching.size_of({"detail": "named by mflux-community/mflux"}) == 0
    assert fetching.size_of({}) == 0
    assert fetching.size_of({"detail": "", "sized": ""}) == 0


# --- a refusal about this harness is not a verdict about the candidate -----

def test_our_own_gap_is_named_as_ours():
    assert fetching.refused_by_harness("no measured size; inspect it first")


def test_a_real_refusal_is_the_candidates():
    """The negative control, and the one that matters: a genuinely oversized
    model must still be settled, or every sweep re-offers it."""
    for why in ("64.6 GiB is over the 60 GiB cap",
                "49.3 GiB would leave under the 50 GiB floor (60 GiB free)"):
        assert not fetching.refused_by_harness(why)


def test_the_plan_for_an_unsized_repo_still_refuses():
    """The refusal is right. Only the VERDICT it produced was wrong."""
    p = fetching.plan("org/x", 0)
    assert not p.ok and fetching.refused_by_harness(p.why)


# --- the scoped loop must scope the step that spends the disk ---------------
#
# A REAL STORE, NOT A FAKE CONNECTION. These used a stub whose execute()
# returned an object with only fetchall(), which stopped working the moment
# queued() asked the store a second question. A fake that models one call is a
# test of that call, not of the function.

from harness import memory_store as ms       # noqa: E402


def _seed(conn, name, lane, registry=ms.HUGGINGFACE, kind="candidate"):
    ms.record(conn, ms.Seen(name=name, source="test", url="", why="",
                            relevance=0, kind=kind, registry=registry,
                            lane=lane, resolved=name))
    ms.decide(conn, name, "queued", tier="inspect", detail="bytes=104857600")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setattr(fetching, "have", lambda *a, **k: False)
    monkeypatch.setattr("harness.rank.serving", lambda *a, **k: set())
    monkeypatch.setattr("harness.rank.lanes_with_receipts", lambda *a, **k: set())
    conn = ms.connect(tmp_path / "d.db")
    yield conn
    conn.close()


def test_a_lane_scoped_fetch_leaves_the_other_lanes_alone(store):
    """The loop printed "(spending only on the image lane)" and then considered
    whisper-tiny, wav2vec2, gpt2 and vicuna. Issue #211."""
    _seed(store, "org/img", "image")
    _seed(store, "org/asr", "stt")
    assert [r["name"] for r in fetching.queued(store, lane="image")] == ["org/img"]
    assert len(fetching.queued(store)) == 2, "unscoped must keep both"


def test_a_text_candidate_is_fetchable_for_the_web_lane(store):
    """lanes.serves, not equality, so #208 survives here too."""
    _seed(store, "org/txt", "code")
    assert len(fetching.queued(store, lane="web")) == 1
    assert len(fetching.queued(store, lane="image")) == 0
