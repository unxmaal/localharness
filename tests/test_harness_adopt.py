"""Issue #201: the loop's closing step."""
import pytest

from harness import adopt, rank
from harness import memory_store as ms


def row(candidate, wer, rate=1.0, median=0.5):
    return {"candidate": candidate, "pass_rate": rate, "median_s": median,
            "metrics": {"wer": wer}}


def head_to_head_rows(incumbent, challenger):
    """One run's rows: the incumbent and the challenger over the same cases."""
    return ([{"candidate": "inc", "case_id": f"c{i}", "passed": v}
             for i, v in enumerate(incumbent)]
            + [{"candidate": "ch", "case_id": f"c{i}", "passed": v}
               for i, v in enumerate(challenger)])


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
    got = adopt.decide("tts", inc, ch, head_to_head_rows(
        [False] * 8, [True] * 5 + [False] * 3))
    assert not got.adopt
    assert "not established" in got.why


def test_a_significant_win_on_the_metric_is_adopted():
    inc, ch = row("inc", 0.05), row("ch", 0.01)
    got = adopt.decide("tts", inc, ch, head_to_head_rows(
        [False] * 8, [True] * 8))
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


def test_a_lane_of_all_counts_as_none_and_text_is_the_code_lane():
    """A feed source declares lane 'all' to mean it covers everything, and that
    was recorded as the candidate's own lane on hundreds of proposals.

    `text` is a different case and used to be dropped beside it: it is the
    code lane under discover.py's older name, and discarding it hid 10
    proposals from the queue entirely. Issue #207.
    """
    got = rank.rank([proposal("org/a", "all"), proposal("org/b", "text")],
                    serving=(), measured_lanes=())
    assert [r["name"] for r in got] == ["org/b"]
    assert got[0]["lane"] == "text", "the stored row is not rewritten in place"
    assert rank.lane_of(got[0]) == "code"


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


# ---- the chain's measure-and-adopt step (issue #201) ----------------------

def test_the_incumbent_and_challenger_run_in_one_paired_invocation(monkeypatch, tmp_path):
    """Two separate runs would be two receipts that comparable() refuses, and
    rightly: the machine, the cases and the repeat count must all be held."""
    import argparse
    import subprocess

    from harness import cli

    seen = {}

    class Done:
        returncode = 0
        stderr = ""

    def fake_run(argv, **kw):
        seen["argv"] = argv
        return Done()

    monkeypatch.setattr(subprocess, "run", fake_run)
    # PIN THE INCUMBENT. Speech defaults are platform-branched: this resolves
    # to kokoro-onnx/Kokoro-82M on Windows and the MLX name elsewhere, so a
    # fixture naming one of them tests the runner's machine. RULE #249, third
    # occurrence.
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "org/Kokoro-82M-bf16")
    monkeypatch.setattr(cli, "_latest_receipt", lambda m: {
        "summary": {"Kokoro-82M-bf16": {"pass_rate": 1.0, "median_s": 1.0,
                                        "metrics": {"wer": 0.05}},
                    "better-tts": {"pass_rate": 1.0, "median_s": 1.0,
                                   "metrics": {"wer": 0.01}}},
        "rows": ([{"candidate": "Kokoro-82M-bf16", "case_id": f"c{i}",
                   "passed": False} for i in range(8)]
                 + [{"candidate": "better-tts", "case_id": f"c{i}",
                     "passed": True} for i in range(8)]),
    })
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    rc = cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                {"name": "org/better-tts", "lane": "tts"})
    assert rc == 0
    argv = seen["argv"]
    assert "--modality" in argv and "tts" in argv
    cands = argv[argv.index("--candidates") + 1]
    assert "better-tts" in cands, "the challenger must be in the run"
    assert cands.count(",") >= 1, "the incumbent must be in the SAME run"


def test_a_lane_with_no_incumbent_measures_nothing(monkeypatch, capsys):
    """Nothing to beat is not a win. Adopting against an empty incumbent would
    make the first candidate through the door the lane's default."""
    import argparse

    from harness import adopt as adopt_mod
    from harness import cli, winners

    monkeypatch.setattr(winners, "typed", lambda: {})
    monkeypatch.setattr(adopt_mod, "default_for", lambda lane, fb, conn=None: "")
    rc = cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                {"name": "org/x", "lane": "tts"})
    assert rc == 0
    assert "no incumbent" in capsys.readouterr().out


def test_a_summary_key_is_matched_on_its_stem():
    """A run reports `Kokoro-82M-bf16/bm_george` for a candidate named
    `mlx-community/Kokoro-82M-bf16`. Demanding the caller's exact string would
    make every tts comparison unmatchable."""
    from harness import cli

    got = cli._summary_row({"Kokoro-82M-bf16/bm_george": {"pass_rate": 1.0}},
                           "mlx-community/Kokoro-82M-bf16")
    assert got and got["candidate"] == "Kokoro-82M-bf16/bm_george"


def test_an_unmatched_candidate_is_an_error_not_a_silent_skip():
    from harness import cli

    assert cli._summary_row({"something-else": {}}, "org/wanted") is None


def test_only_the_latest_verdict_counts_as_a_survivor(tmp_path):
    """A candidate that screened green and was later declined must not be
    handed to the measure tier again on every loop."""
    conn = ms.connect(tmp_path / "d.db")
    ms.record(conn, ms.Seen(name="org/a", source="feeds", lane="tts",
                            kind="weights", resolved="org/a"))
    ms.decide(conn, "org/a", "screened", tier=ms.SCREEN, detail="ran")
    assert [r["name"] for r in ms.survivors(conn)] == ["org/a"]
    ms.decide(conn, "org/a", "declined", tier=adopt.TIER, detail="tts: lost")
    assert ms.survivors(conn) == []


def test_two_candidates_sharing_no_case_is_refused_not_called_a_tie():
    """Using cells() here returned nothing, and an empty result reads as "no
    difference" rather than "wrong comparison"."""
    inc, ch = row("inc", 0.05), row("ch", 0.01)
    rows = ([{"candidate": "inc", "case_id": "a", "passed": True}]
            + [{"candidate": "ch", "case_id": "z", "passed": True}])
    got = adopt.decide("tts", inc, ch, rows)
    assert not got.adopt
    assert "share no case" in got.why


def test_head_to_head_pairs_on_the_case_not_the_candidate():
    from harness import paired

    rows = head_to_head_rows([False, False, True], [True, True, True])
    cell = paired.head_to_head(rows, "inc", "ch")
    assert (cell.lost, cell.gained, cell.unchanged) == (0, 2, 1)
