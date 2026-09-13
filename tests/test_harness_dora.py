"""The four DORA metrics, computed from git and gh.

Red-proofed against fixtures rather than the live repo: a test that asserts
"median lead time is 0.45h" is asserting this week, and would go red on a quiet
one. What is pinned here is the ARITHMETIC and the degradations.
"""
import json
from pathlib import Path

import pytest

from harness import dora


def fake_git(log_line, branch_commits=("c1",), started="2026-09-01T09:00:00+00:00"):
    """A git that answers the three questions dora asks, and nothing else."""
    def git(*args):
        if args[1] == "--merges" or "--merges" in args:
            return log_line
        if args[0] == "rev-list":
            return "\n".join(branch_commits)
        if args[0] == "show":
            return started
        raise AssertionError(f"unexpected git call: {args}")
    return git


MERGE = "\x1f".join(["abc123def4567", "2026-09-01T12:00:00+00:00",
                     "p1 p2", "Merge pull request #1"])


# ---- lead time is the branch's own history, not the window ----------------

def test_lead_time_measures_the_branch_not_the_window():
    """An earlier cut asked for commits `--not main~50` and reported six days
    for a branch created that morning: it measured how far back the window
    reached. The start is the oldest commit on the branch's own side."""
    got = dora.changes(14, git=fake_git(MERGE))[0]
    assert got.first_commit_at == "2026-09-01T09:00:00+00:00"
    assert got.lead_hours == 3.0


def test_a_fast_forward_merge_is_skipped():
    """One parent means no branch to measure, and subtracting a commit from
    itself would report a lead time of zero, dragging the median down."""
    ff = "\x1f".join(["abc", "2026-09-01T12:00:00+00:00", "p1", "ff"])
    assert dora.changes(14, git=fake_git(ff)) == []


def test_a_merge_whose_branch_has_no_commits_is_skipped():
    assert dora.changes(14, git=fake_git(MERGE, branch_commits=())) == []


# ---- restore time runs forward ------------------------------------------

def run(conclusion, created, updated, branch="main", event="push"):
    return {"conclusion": conclusion, "createdAt": created,
            "updatedAt": updated, "headBranch": branch, "event": event}


def test_restore_is_measured_forward_in_time():
    """gh returns runs newest first. Reading them as given measures a green run
    followed by an earlier red one, which is not a restore at all."""
    runs = [run("success", "2026-09-01T12:00:00Z", "2026-09-01T12:30:00Z"),
            run("failure", "2026-09-01T10:00:00Z", "2026-09-01T10:05:00Z")]
    assert dora.restore_times(runs) == [2.5]


def test_a_still_broken_branch_contributes_no_restore():
    """An open incident has no duration yet. Counting it as zero would make an
    unfixed break look like the fastest recovery on record."""
    runs = [run("failure", "2026-09-01T10:00:00Z", "2026-09-01T10:05:00Z")]
    assert dora.restore_times(runs) == []


def test_only_the_first_failure_of_a_streak_starts_the_clock():
    runs = [run("success", "2026-09-01T14:00:00Z", "2026-09-01T14:30:00Z"),
            run("failure", "2026-09-01T11:00:00Z", "2026-09-01T11:05:00Z"),
            run("failure", "2026-09-01T10:00:00Z", "2026-09-01T10:05:00Z")]
    assert dora.restore_times(runs) == [4.5]


def test_another_branch_is_not_this_branch():
    runs = [run("failure", "2026-09-01T10:00:00Z", "2026-09-01T10:05:00Z",
                branch="feature"),
            run("success", "2026-09-01T12:00:00Z", "2026-09-01T12:30:00Z")]
    assert dora.restore_times(runs) == []


# ---- missing is not zero -------------------------------------------------

def test_no_ci_history_reports_missing_rather_than_zero():
    """gh absent, unauthenticated or offline all look the same from here, and
    all three mean the number was not gathered. Reporting 0.0 would say the
    branch never broke."""
    got = dora.report(14, git=fake_git(MERGE), runs=[])
    assert got["ci_history"] is False
    assert got["change_failure_rate"]["value"] is None
    assert got["time_to_restore_hours"]["median"] is None


def test_gh_failing_does_not_raise():
    def explode(argv):
        raise FileNotFoundError("gh")
    assert dora._gh_runs(10, run=explode) == []


def test_no_merges_does_not_divide_by_zero():
    got = dora.report(14, git=fake_git(""), runs=[])
    assert got["deployment_frequency"]["merges"] == 0
    assert got["lead_time_hours"]["median"] is None


# ---- the caveats travel with the numbers ---------------------------------

def test_change_failure_rate_carries_what_it_actually_measures():
    """It is CI health, not change quality. Quoted bare beside the other three
    it reads as the same grade of number, which is how a floor becomes a rate."""
    got = dora.report(14, git=fake_git(MERGE), runs=[run("success", "2026-09-01T12:00:00Z", "2026-09-01T12:30:00Z")])
    assert "CI health" in got["change_failure_rate"]["caveat"]


def test_a_median_of_one_sample_is_marked_thin():
    runs = [run("success", "2026-09-01T12:00:00Z", "2026-09-01T12:30:00Z"),
            run("failure", "2026-09-01T10:00:00Z", "2026-09-01T10:05:00Z")]
    got = dora.report(14, git=fake_git(MERGE), runs=runs)
    assert got["time_to_restore_hours"]["n"] == 1
    assert got["time_to_restore_hours"]["thin"] is True


def test_a_healthy_sample_is_not_marked_thin():
    line = "\n".join([MERGE] * 5)
    got = dora.report(14, git=fake_git(line), runs=[])
    assert got["lead_time_hours"]["n"] == 5
    assert got["lead_time_hours"]["thin"] is False


def test_the_json_shape_survives_a_round_trip():
    got = dora.report(14, git=fake_git(MERGE), runs=[])
    assert json.loads(json.dumps(got))["branch"] == "main"


# ---- the entry point, because that is what anyone actually runs -----------

def test_main_prints_the_four_and_their_caveat(capsys, monkeypatch):
    monkeypatch.setattr(dora, "_git", fake_git(MERGE))
    monkeypatch.setattr(dora, "_gh_runs", lambda *a, **k: [])
    assert dora.main([]) == 0
    out = capsys.readouterr().out
    for want in ("deployment frequency", "lead time", "time to restore",
                 "change failure rate", "CI health"):
        assert want in out


def test_main_says_when_the_ci_half_is_missing(capsys, monkeypatch):
    """Missing and zero look identical in a table, and one of them is a lie."""
    monkeypatch.setattr(dora, "_git", fake_git(MERGE))
    monkeypatch.setattr(dora, "_gh_runs", lambda *a, **k: [])
    dora.main([])
    assert "MISSING, not zero" in capsys.readouterr().out


def test_main_emits_parseable_json(capsys, monkeypatch):
    monkeypatch.setattr(dora, "_git", fake_git(MERGE))
    monkeypatch.setattr(dora, "_gh_runs", lambda *a, **k: [])
    dora.main(["--json"])
    assert json.loads(capsys.readouterr().out)["window_days"] == 14
