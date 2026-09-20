"""Issue #175: ordering a queue by what a screen would teach.

The judge could not rank these candidates, and two controls shaped like the
real data proved it rather than assuming it. This instrument is arithmetic over
what the store already holds, so its own control is deterministic: no gateway,
no sampling, and no need to run it three times to believe it.
"""
import pytest

from harness import rank


def row(name, **kw):
    base = {"name": name, "lane": "", "times": 1, "description": ""}
    base.update(kw)
    return base


SERVING = {"qwen/qwen2.5-7b", "mlx-community/kokoro-82m-bf16"}
MEASURED = {"code", "image", "svg"}


def value(r):
    return rank.value(r, serving=SERVING, measured_lanes=MEASURED)[0]


# --- the control: known cases, in a known order ---------------------------

def test_a_lane_with_no_receipt_outranks_one_already_covered():
    """Where a screen answers an open question it is worth more than where it
    re-answers a settled one."""
    open_lane = row("a/new", lane="tts")
    covered = row("b/another", lane="image")
    assert value(open_lane) > value(covered)


def test_a_requant_of_something_already_served_ranks_below_it():
    """Screening a 4-bit copy of the model already running here teaches
    nothing that is not already known."""
    copy = row("someone/qwen-copy", lane="code",
               description="built from Qwen/Qwen2.5-7B; 4.0 GiB of weights")
    fresh = row("someone/new-thing", lane="code",
                description="built from nobody/unheard-of; 4.0 GiB of weights")
    assert value(copy) < value(fresh)


def test_a_candidate_no_lane_can_measure_ranks_last():
    """Not a judgement about the model: the suite has no case, no runner and no
    metric for it, so a screen cannot say anything at all."""
    unmeasurable = row("x/no-lane")
    anything = row("y/has-lane", lane="code")
    assert value(unmeasurable) < value(anything)
    assert "no lane can measure it" in rank.value(unmeasurable)[1][0]


def test_recurrence_counts_and_stops_counting():
    """A thing seen again and again is a different signal from one that trended
    once -- and the tenth sighting says much less than the second, so the bump
    saturates rather than letting a popular row outrank everything."""
    once, twice, many = (row("a", times=1), row("a", times=2), row("a", times=40))
    assert value(once) < value(twice) < value(many)
    assert value(many) - value(twice) < value(twice) - value(once)
    # and it can never outweigh being unmeasurable
    assert value(row("a", times=999)) < value(row("b", lane="code"))


def test_a_source_covering_every_lane_is_not_a_lane():
    """`all` is a feed source's COVERAGE. Recorded as a candidate's lane it made
    243 of 323 proposals claim membership of a lane that does not exist, and
    since a recorded lane is never overwritten, the real one read off a model
    card could never land."""
    assert value(row("a", lane="all")) == value(row("a", lane=""))
    assert "no lane can measure it" in rank.value(row("a", lane="all"))[1][0]


def test_every_row_says_why_it_is_where_it_is():
    """A ranking whose rows cannot say why is one nobody can argue with, and
    every metric in this project that went wrong went wrong quietly."""
    got = rank.rank([row("a/thing", lane="tts", times=3,
                         description="0.5 GiB of weights")],
                    serving=SERVING, measured_lanes=MEASURED)
    assert got[0]["value_why"]
    assert "tts" in got[0]["value_why"]


def test_the_order_is_stable_when_two_rows_tie():
    """Two runs of a deterministic instrument must not disagree. The judge
    needed three runs to be believed; this one should need none."""
    rows = [row("b/second", lane="tts"), row("a/first", lane="tts")]
    assert [r["name"] for r in rank.rank(rows, measured_lanes=MEASURED)] == \
           ["a/first", "b/second"]
    assert rank.rank(rows) == rank.rank(rows)


# --- reading the facts out of a description ------------------------------

def test_a_size_is_read_from_the_card_text():
    assert rank.size_gib("task x; 2.3 GiB of weights") == pytest.approx(2.3)
    assert rank.size_gib("nothing here") == 0.0


def test_serving_is_read_from_the_gateway_config_not_a_list():
    """A hand-kept list of what we run, beside the thing that runs it, drifts
    within a week."""
    got = rank.serving()
    assert got, "no upstream ids read from the gateway config"
    assert all("/" in name for name in got)
    # The provider prefix LiteLLM routes through is not part of the id.
    assert not any(name.startswith("openai/") for name in got)


# ---- the lanes are not equally wanted (2026-09-18) ------------------------

def test_the_priority_order_is_the_one_that_was_asked_for():
    """image, code, web, svg, video, then music. Pinned because a reordering
    here silently changes what the loop spends its download budget on.

    `music` is last and appended, never inserted: it was asked for after the
    other five were ranked and has never been ranked against them (#237).
    """
    assert rank.LANE_PRIORITY == ("image", "code", "web", "svg", "video",
                                  "music")
    got = [rank.priority_of(l) for l in rank.LANE_PRIORITY]
    assert got == sorted(got, reverse=True), "priority must fall down the list"


def test_a_lane_nobody_asked_for_gets_nothing():
    assert rank.priority_of("tts") == 0.0
    assert rank.priority_of("stt") == 0.0
    assert rank.priority_of("") == 0.0


def test_a_wanted_lane_outranks_an_unwanted_one_on_equal_information():
    """The defect this fixes: every candidate tied at +5.0 and the tiebreak was
    ALPHABETICAL, so a tts model sat above an image one because of its name."""
    rows = [{"name": "zzz/image-model", "lane": "image", "times": 1, "bytes": 0},
            {"name": "aaa/tts-model", "lane": "tts", "times": 1, "bytes": 0}]
    got = rank.rank(rows, serving=(), measured_lanes=())
    assert [r["name"] for r in got] == ["zzz/image-model", "aaa/tts-model"]


def test_priority_never_outranks_teaching_nothing():
    """A requant of something already served teaches nothing whatever lane it
    is in. If priority could overcome that, the loop would spend its budget
    re-measuring what it already runs."""
    rows = [{"name": "org/requant", "lane": "image", "times": 1, "bytes": 0,
             "description": "built from org/served"},
            {"name": "org/fresh", "lane": "video", "times": 1, "bytes": 0,
             "description": ""}]
    got = rank.rank(rows, serving={"org/served"}, measured_lanes=())
    assert [r["name"] for r in got] == ["org/fresh", "org/requant"], (
        "image is the top lane, and a requant of something already served "
        "still teaches nothing")


def test_the_reason_names_the_lane():
    rows = [{"name": "org/m", "lane": "image", "times": 1, "bytes": 0}]
    got = rank.rank(rows, serving=(), measured_lanes=())
    assert "wanted lane" in got[0]["value_why"]
