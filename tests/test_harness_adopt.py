"""Issue #201: the loop's closing step."""
import pytest

from harness import adopt, rank
from harness import memory_store as ms


def row(candidate, wer, rate=1.0, median=0.5):
    return {"candidate": candidate, "pass_rate": rate, "median_s": median,
            "metrics": {"wer": wer}}


def cells(candidate, before, after):
    b = [{"candidate": candidate, "case_id": f"c{i}", "passed": v}
         for i, v in enumerate(before)]
    a = [{"candidate": candidate, "case_id": f"c{i}", "passed": v}
         for i, v in enumerate(after)]
    return b, a


def test_a_faster_candidate_that_loses_on_the_metric_is_not_adopted():
    """RULE #254's case. parakeet-ctc passes as often and is faster, and loses
    on WER, which is what the stt lane is about."""
    inc, ch = row("parakeet-tdt-v2", 0.0162, median=0.135), row("parakeet-ctc", 0.0225, median=0.116)
    assert not adopt.better(inc, ch)
    assert not adopt.decide("stt", inc, ch).adopt


def test_a_better_metric_alone_is_not_enough():
    """A run is a sample. Adopting on the point estimate is how a 12/27 to
    9/27 trend at p=0.45 becomes a decision."""
    inc, ch = row("inc", 0.05), row("ch", 0.01)
    b, a = cells("ch", [False] * 8, [True] * 5 + [False] * 3)
    got = adopt.decide("tts", inc, ch, b, a)
    assert not got.adopt
    assert "not established" in got.why


def test_a_significant_win_on_the_metric_is_adopted():
    inc, ch = row("inc", 0.05), row("ch", 0.01)
    b, a = cells("ch", [False] * 8, [True] * 8)
    got = adopt.decide("tts", inc, ch, b, a)
    assert got.adopt
    assert "p=0.01" in got.why or "p=0.00" in got.why


def test_without_paired_rows_nothing_is_adopted():
    """Silence is not evidence. A challenger with no paired comparison has an
    unmeasured difference, not a proven one."""
    inc, ch = row("inc", 0.05), row("ch", 0.01)
    assert not adopt.decide("tts", inc, ch).adopt


def test_a_lane_serves_its_typed_constant_until_something_is_adopted(tmp_path):
    conn = ms.connect(tmp_path / "d.db")
    assert adopt.default_for("tts", "CONST", conn) == "CONST"


def test_an_adoption_changes_what_the_lane_serves(tmp_path):
    """The whole point. Before this the loop could measure a winner and the
    lane went on serving the constant somebody typed."""
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, ms.Seen(name="org/new", source="feeds", lane="tts",
                            kind="weights", resolved="org/new"))
    adopt.record(conn, adopt.Verdict("tts", "old", "org/new", True, "won"))
    assert adopt.adopted(conn) == {"tts": "org/new"}
    assert adopt.default_for("tts", "CONST", conn) == "org/new"


def test_a_loss_is_recorded_too(tmp_path):
    """So the same candidate is not re-proposed every sweep."""
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, ms.Seen(name="org/new", source="feeds", lane="tts",
                            kind="weights", resolved="org/new"))
    adopt.record(conn, adopt.Verdict("tts", "old", "org/new", False, "lost"))
    assert adopt.adopted(conn) == {}
    got = conn.execute("SELECT outcome, tier FROM verdicts").fetchall()
    assert [tuple(r) for r in got] == [("declined", "adopt")]


def test_an_unreachable_store_leaves_the_lane_on_its_constant():
    """A generation must never fail because the discovery store is missing,
    locked, or newer than this code."""
    class Broken:
        def execute(self, *a, **k):
            raise RuntimeError("no store")
    assert adopt.default_for("tts", "CONST", Broken()) == "CONST"


# ---- the queue drops what no lane can test --------------------------------

def proposal(name, lane, times=1):
    return {"name": name, "lane": lane, "times": times, "bytes": 0}


def test_a_candidate_with_no_lane_is_not_ranked():
    """Penalising it left 66 of 94 queue slots holding things nothing could
    measure."""
    got = rank.rank([proposal("org/tool", ""), proposal("org/model", "tts")],
                    serving=(), measured_lanes=())
    assert [r["name"] for r in got] == ["org/model"]


def test_a_lane_of_all_or_text_counts_as_none():
    """A feed source declares lane 'all' to mean it covers everything, and that
    was recorded as the candidate's own lane on hundreds of proposals."""
    got = rank.rank([proposal("org/a", "all"), proposal("org/b", "text")],
                    serving=(), measured_lanes=())
    assert got == []


def test_the_laneless_are_reported_rather_than_discarded():
    """A lane is a person's decision. Dropping them silently throws away the
    signal that the harness is missing something."""
    rows = [proposal("org/tool", "", times=3), proposal("org/once", "", times=1),
            proposal("org/model", "tts", times=5)]
    got = rank.wanted(rows)
    assert [r["name"] for r in got] == ["org/tool"]


def test_keeping_them_is_possible_for_a_report_that_wants_them():
    got = rank.rank([proposal("org/tool", "")], serving=(), measured_lanes=(),
                    keep_laneless=True)
    assert [r["name"] for r in got] == ["org/tool"]
