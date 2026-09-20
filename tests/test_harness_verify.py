"""Does each lane work? Run its own default and find out. Issue #234.

Whether any of this works is not known, because most lanes have never been
run here: three appear in no receipt on this machine and four more were last
measured twelve days ago.
"""
import pytest

from harness import verify

NEVER = {"lane": "extract", "serves": "local-large", "unverified": True,
         "stale": False, "age_days": None}
STALE = {"lane": "svg", "serves": "local-large", "unverified": False,
         "stale": True, "age_days": 12.0}
FRESH = {"lane": "code", "serves": "q3-4b", "unverified": False,
         "stale": False, "age_days": 0.2}
VIDEO = {"lane": "video", "serves": "h3", "unverified": True,
         "stale": False, "age_days": None}


def _lanes(*rows):
    return [dict(r) for r in rows]


def test_a_lane_with_no_receipt_is_planned():
    got = verify.plan(_lanes(NEVER))
    assert [t.lane for t in got] == ["extract"]
    assert got[0].why == "no receipt on this machine"


def test_a_stale_lane_is_planned_and_says_how_old():
    assert verify.plan(_lanes(STALE))[0].why == "last measured 12 days ago"


def test_a_fresh_lane_is_left_alone():
    """The negative control. A pass that re-runs everything every time stops
    being run."""
    assert verify.plan(_lanes(FRESH)) == []


def test_every_lane_can_be_forced():
    assert [t.lane for t in verify.plan(_lanes(FRESH), force=True)] == ["code"]


def test_cheapest_first():
    """A pass that opens with forty minutes of video learns nothing for forty
    minutes."""
    got = verify.plan(_lanes(STALE, NEVER))
    assert [t.lane for t in got] == ["extract", "svg"]


def test_video_is_never_started_without_being_asked_for_by_name():
    """~45 minutes for one case, behind a six-hour engine timeout. This is how
    a machine ends up unusable overnight."""
    got = verify.plan(_lanes(VIDEO))
    assert got[0].skip and "--lane video" in got[0].skip
    assert not got[0].argv, "a skipped task must carry no command"


def test_naming_video_opts_into_it():
    got = verify.plan(_lanes(VIDEO), only="video")
    assert not got[0].skip
    assert got[0].argv


def test_a_lane_naming_no_default_is_skipped_rather_than_run():
    bare = dict(NEVER, serves="")
    got = verify.plan(_lanes(bare))
    assert got[0].skip == "the lane names no default to run"
    assert not got[0].argv


def test_the_command_runs_the_lanes_own_default():
    """NOT the screen and NOT the measure tier: both of those judge a
    challenger. This runs the incumbent alone."""
    argv = verify.plan(_lanes(NEVER))[0].argv
    assert "--modality" in argv and argv[argv.index("--modality") + 1] == "extract"
    assert "local-large" in argv


def test_an_alias_lane_is_not_routed_upstream():
    """`local-large` is a gateway alias, so it goes through the gateway."""
    assert "--gateway" not in verify.plan(_lanes(NEVER))[0].argv


# --- the verdict -----------------------------------------------------------

def _receipt(passed, total):
    return {"summary": {"c": {"passed": passed, "total": total}}}


def test_a_default_that_passes_everything_works():
    assert verify.verdict(_receipt(9, 9)) == "works"


def test_a_default_that_passes_nothing_is_broken():
    """Whatever the reason. That is the finding this pass exists to produce."""
    assert verify.verdict(_receipt(0, 9)) == "broken"


def test_a_default_that_passes_some_is_partial_not_broken():
    """A lane can legitimately fail a hard case: web has cases no small model
    passes, and calling that broken would be wrong."""
    assert verify.verdict(_receipt(6, 9)) == "partial"


def test_no_receipt_and_no_cases_are_both_broken():
    assert verify.verdict({}) == "broken"
    assert verify.verdict(_receipt(0, 0)) == "broken"


def test_the_summary_line_names_the_candidate_and_its_metric():
    data = {"summary": {"local-large": {"passed": 8, "total": 9,
                                        "median_s": 5.1,
                                        "metrics": {"ink": 0.29}}}}
    line = verify.summarise("svg", data)
    assert "local-large 8/9" in line and "ink 0.290" in line
