"""The status page. Issue #231.

Answering "how have our lanes changed recently" took six commands and ad-hoc
SQL, and the answer landed in a chat message that was stale immediately.
"""
import json
import re

import pytest

from harness import report

#: Assert on the MARKUP, not the phrase: the page's own explanatory note
#: contains the words "no receipt here", so a bare substring check passes for
#: the wrong reason and fails for the right one.
NO_RECEIPT = '<span class="tag bad">no receipt here</span>'
STALE = '<span class="tag warn">last run'

LANE = {"lane": "code", "wanted": True, "serves": "q3-4b", "adopted": False,
        "measured": "q3-4b", "match": "exact", "pass_rate": 0.78,
        "median_s": 1.4, "metrics": {"code_pass": 0.9}, "run": "r1",
        "age_days": 0.2, "unverified": False, "stale": False}


def _state(**over):
    base = {"generated": 0.0,
            "machine": {"runtimes": ["mlx"], "accelerator": "x", "kind":
                        "unified", "total_gb": 32.0, "available_gb": 8.0},
            "funnel": [{"tier": "inspect", "candidates": 9, "verdicts": 9}],
            "lanes": [dict(LANE)],
            "queue": {"waiting": 3, "rankable": 2, "by_lane": {"code": 2},
                      "top": [], "wanted_with_none": ["web"]},
            "sources": [{"name": "s", "last_fetched": 1.0, "age_days": 0.5,
                         "stale": False}]}
    base.update(over)
    return base


# --- "changed" needs a referent -------------------------------------------

def test_a_first_report_marks_nothing_changed():
    """Nothing to compare against is not the same as nothing having changed,
    and the page has to say which one it means."""
    page = report.render(_state(), before={})
    assert 'class="changed"' not in page
    assert "No previous report" in page


def test_a_lane_that_changed_what_it_serves_is_highlighted():
    before = _state()
    now = _state(lanes=[dict(LANE, serves="something-new")])
    got = report.changes(now, before)
    assert got["code"] == "serves something-new, was q3-4b"
    assert 'class="changed"' in report.render(now, before)


def test_a_lane_re_measured_in_a_new_run_is_highlighted():
    before = _state()
    now = _state(lanes=[dict(LANE, run="r2")])
    assert report.changes(now, before)["code"] == "re-measured in r2"


def test_an_unchanged_lane_is_not_highlighted():
    """The negative control. A page that highlights every row highlights
    nothing."""
    assert report.changes(_state(), _state()) == {}
    assert 'class="changed"' not in report.render(_state(), _state())


def test_a_lane_that_did_not_exist_before_is_new_rather_than_changed():
    before = _state(lanes=[])
    assert report.changes(_state(), before)["code"] == "new lane"


# --- staleness is three different things ----------------------------------

def test_a_lane_with_no_receipt_is_unverified_not_stale():
    """Quoting an age for a lane that never ran would be a fabrication."""
    page = report.render(_state(lanes=[
        dict(LANE, run="", age_days=None, unverified=True, stale=False)]), {})
    assert NO_RECEIPT in page
    assert STALE not in page


def test_a_lane_measured_long_ago_reports_its_age():
    page = report.render(_state(lanes=[
        dict(LANE, age_days=42.0, stale=True)]), {})
    assert f'{STALE} 42d ago' in page
    assert NO_RECEIPT not in page


def test_only_a_quantisation_having_run_is_called_out():
    """`~` in --winners. A finding, not a mismatch to smooth over."""
    page = report.render(_state(lanes=[dict(LANE, match="quantised")]), {})
    assert "only a quantisation ran" in page


def test_a_wanted_lane_with_an_empty_queue_is_named():
    assert "nothing queued for web" in report.render(_state(), {})


# --- self-contained -------------------------------------------------------

def test_the_page_fetches_nothing(tmp_path):
    """It has to open on a machine with no network, which is the machine this
    runs on. No CDN, no fonts, no <script src>."""
    page = report.render(_state(), {})
    assert not re.search(r'<(?:script|link|img)[^>]*\bsrc=|<link[^>]*\bhref=',
                         page), "the page pulls an external resource"
    assert "<style>" in page, "styling must be inline"


def test_the_page_escapes_what_the_store_holds():
    """Proposal names come from feeds written by strangers. Untrusted text."""
    page = report.render(_state(lanes=[
        dict(LANE, serves="<script>alert(1)</script>")]), {})
    assert "<script>alert(1)</script>" not in page
    assert "&lt;script&gt;" in page


def test_writing_leaves_the_state_beside_the_page(tmp_path):
    """The NEXT run needs it; without it, 'changed' has no referent."""
    out = tmp_path / "r.html"
    monkey = report.state
    report.state = lambda conn=None: _state()
    try:
        report.write(out)
    finally:
        report.state = monkey
    assert out.exists()
    assert json.loads(out.with_suffix(".json").read_text(encoding="utf-8"))["lanes"]


# --- the ladder -----------------------------------------------------------

def test_the_ladder_is_shown_in_the_order_work_flows_through_it():
    """memory_store.TIERS enumerates valid tier NAMES and omits `fetch`.
    A different question, and using it here put fetch last."""
    assert report.LADDER.index("fetch") < report.LADDER.index("screen")
    assert report.LADDER.index("screen") < report.LADDER.index("adopt")


def test_the_funnel_orders_by_the_ladder_not_by_count(tmp_path):
    from harness import memory_store as ms

    conn = ms.connect(tmp_path / "d.db")
    try:
        for name, tier in (("a", "adopt"), ("b", "inspect"), ("c", "screen")):
            ms.record(conn, ms.Seen(name=name, source="t", url="", why="",
                                    relevance=0, kind="candidate",
                                    registry=ms.HUGGINGFACE, lane="code",
                                    resolved=name))
            ms.decide(conn, name, "queued", tier=tier, detail="d")
        got = [r["tier"] for r in report.funnel(conn)]
        assert got == ["inspect", "screen", "adopt"]
    finally:
        conn.close()
