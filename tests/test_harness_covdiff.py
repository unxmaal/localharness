"""Coverage on the diff, not on the repo.

A repo-wide percentage is the whole reported as if it described the part: it
barely moves when a change adds fifty unexercised lines. The question a reviewer
has is narrow -- of the lines THIS change adds, which does nothing run?

The negative cases carry the weight, as ever: a differential that reports a
config file or a shell script as an untested gap is one nobody reads twice.
"""
import json
import subprocess
from pathlib import Path

import pytest

from harness import covdiff


@pytest.fixture
def repo(tmp_path):
    def git(*a):
        return subprocess.run(("git", "-C", str(tmp_path)) + a,
                              capture_output=True, text=True, check=True)
    git("init", "-q")
    (tmp_path / "m.py").write_text("a = 1\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "base")
    git("branch", "-M", "base")
    # A branch to change things ON. Without it `base` tracks HEAD, every
    # `base...HEAD` diff is empty, and every test passes by measuring nothing.
    git("checkout", "-q", "-b", "work")
    return tmp_path, git


def test_only_added_lines_are_considered(repo):
    root, git = repo
    (root / "m.py").write_text("a = 1\nb = 2\nc = 3\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "more")
    got = covdiff.added_lines("base", root)
    assert got == {"m.py": {2, 3}}, "the unchanged first line is not this change"


def test_a_deleted_line_adds_nothing(repo):
    root, git = repo
    (root / "m.py").write_text("", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "cut")
    assert covdiff.added_lines("base", root) == {}


def test_an_added_line_nothing_runs_is_reported(repo):
    root, git = repo
    (root / "m.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "more")
    report = {"files": {"m.py": {"missing_lines": [2]}}}
    assert covdiff.uncovered("base", root, report) == {"m.py": [2]}


def test_an_added_line_that_runs_is_not(repo):
    root, git = repo
    (root / "m.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "more")
    report = {"files": {"m.py": {"missing_lines": []}}}
    assert covdiff.uncovered("base", root, report) == {}


def test_a_file_coverage_never_measured_is_not_called_a_gap(repo):
    """A shell script, a YAML config, or anything outside --cov. Reporting it
    as uncovered is a false positive, and a differential that cries wolf on
    every config edit is one that gets switched off."""
    root, git = repo
    (root / "s.sh").write_text("echo hi\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "script")
    assert covdiff.uncovered("base", root, {"files": {}}) == {}


def test_a_missing_line_outside_the_diff_is_not_this_change_s_problem(repo):
    """Pre-existing gaps belong to the repo figure, not to this review."""
    root, git = repo
    (root / "m.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    git("add", "-A")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "more")
    report = {"files": {"m.py": {"missing_lines": [1]}}}
    assert covdiff.uncovered("base", root, report) == {}


def test_a_missing_report_says_what_to_run(tmp_path):
    with pytest.raises(FileNotFoundError) as exc:
        covdiff.uncovered("base", tmp_path)
    assert "make coverage" in str(exc.value)


# ---- the entry point ------------------------------------------------------

def test_main_reports_the_gap_and_its_count(capsys, repo, monkeypatch):
    root, git = repo
    (root / "m.py").write_text("a = 1\nb = 2\n", encoding="utf-8")
    monkeypatch.chdir(root)
    (root / covdiff.COVERAGE_JSON).write_text(
        json.dumps({"files": {"m.py": {"missing_lines": [2]}}}), encoding="utf-8")
    assert covdiff.main(["--base", "base"]) == 0
    out = capsys.readouterr().out
    assert "m.py: 2" in out and "1 added line(s)" in out


def test_main_without_a_coverage_run_says_what_to_do(capsys, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert covdiff.main(["--base", "base"]) == 1
    assert "make coverage" in capsys.readouterr().out


def test_an_untracked_file_is_entirely_new(repo):
    """The case the first cut missed: nothing committed, so `base...HEAD` was
    empty and the differential reported a clean bill on two whole modules."""
    root, git = repo
    (root / "fresh.py").write_text("x = 1\ny = 2\n", encoding="utf-8")
    assert covdiff.added_lines("base", root) == {"fresh.py": {1, 2}}
    assert covdiff.added_lines("base", root, committed_only=True) == {}
