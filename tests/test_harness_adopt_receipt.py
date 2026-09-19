"""The loop must read the receipt it asked for, not the one that sorts best.

`_latest_receipt` called itself "the newest run receipt" and implemented newest
as "sorts highest by directory name". Names are timestamps by CONVENTION, and
`legacy-ev-small-code` does not follow it: `l` sorts above `2`, so it has been
the code lane's answer since it was created. Issue #222.
"""
import argparse
import json
import subprocess

import pytest

from harness import adopt, cli
from harness import memory_store as ms


class _Done:
    returncode = 0
    stderr = ""


def _write(d, summary):
    d.mkdir(parents=True, exist_ok=True)
    (d / "results.json").write_text(json.dumps({"summary": summary, "rows": []}),
                                    encoding="utf-8")


def test_a_directory_outside_the_timestamp_convention_sorts_to_the_top(tmp_path,
                                                                       monkeypatch):
    """The defect, pinned. This is why sorting by name cannot be the answer."""
    runs = tmp_path / "runs"
    _write(runs / "20260919-165029-311-0000-code", {"right": {}})
    _write(runs / "legacy-ev-small-code", {"wrong": {}})
    monkeypatch.setattr("harness.paths.home", lambda: tmp_path)
    got = cli._latest_receipt("code")
    assert sorted(got["summary"]) == ["wrong"], (
        "if this passes, a legacy name still outranks every timestamp, which "
        "is exactly why nothing that decides may use _latest_receipt")


def test_the_receipt_is_read_from_the_path_that_was_named(tmp_path):
    _write(tmp_path / "mine", {"a": {}, "b": {}})
    assert sorted(cli._receipt_at(tmp_path / "mine")["summary"]) == ["a", "b"]


def test_a_path_with_no_receipt_is_none_rather_than_someone_elses(tmp_path):
    _write(tmp_path / "other", {"a": {}})
    assert cli._receipt_at(tmp_path / "mine") is None


def test_the_measure_names_its_own_output_directory(monkeypatch, tmp_path):
    """The subprocess must be told where to write, or the caller is guessing."""
    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: (seen.update(argv=argv), _Done())[1])
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: None)
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    cli._measure_and_adopt(argparse.Namespace(repeat=3),
                           {"name": "org/challenger", "lane": "code"})
    argv = seen["argv"]
    assert "--out" in argv, argv
    out = argv[argv.index("--out") + 1]
    assert out.endswith("-adopt-code"), out


def test_a_receipt_naming_neither_candidate_is_refused(monkeypatch, tmp_path,
                                                       capsys):
    """Naming the directory stops the loop reading a stranger's receipt. This
    stops it reading one that is its own and describes a different exam."""
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Done())
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: {
        "summary": {"local-small": {}, "q3-1.7b": {}}, "rows": []})
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    rc = cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                {"name": "org/challenger", "lane": "code"})
    assert rc == 1
    assert "does not describe the run that was just made" in capsys.readouterr().err


# --- a candidate that never ran is not a candidate that lost ---------------

REFUSED = "gateway returned HTTP 400: Invalid model name passed in model=org/x"


def test_the_measure_routes_a_repo_id_the_way_the_screen_does(monkeypatch,
                                                              tmp_path):
    """LiteLLM validates the model name against its alias table and a
    discovered candidate is always a repo id. The screen passes --gateway and
    the measure did not. Issue #223, which is #206 one tier along."""
    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: (seen.update(argv=argv), _Done())[1])
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "http://127.0.0.1:8081")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: None)
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    cli._measure_and_adopt(argparse.Namespace(repeat=3),
                           {"name": "org/challenger", "lane": "code"})
    argv = seen["argv"]
    assert "--gateway" in argv, argv
    assert argv[argv.index("--gateway") + 1] == "http://127.0.0.1:8081"


def test_rows_that_are_all_harness_refusals_are_named_as_such():
    rows = [{"candidate": "org/x", "detail": REFUSED} for _ in range(9)]
    assert cli._all_refused(rows, "org/x")


def test_one_row_that_reached_a_model_makes_it_the_candidates_result():
    """The negative control. A candidate that genuinely scores badly must
    still be measured, or nothing is ever declined."""
    rows = ([{"candidate": "org/x", "detail": REFUSED}]
            + [{"candidate": "org/x", "detail": "the output closed no tag"}])
    assert not cli._all_refused(rows, "org/x")


def test_a_real_low_score_is_not_a_refusal():
    rows = [{"candidate": "org/x", "detail": ""} for _ in range(9)]
    assert not cli._all_refused(rows, "org/x")


def test_a_wholly_refused_challenger_is_requeued_not_declined(monkeypatch,
                                                              tmp_path, capsys):
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Done())
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: {
        "summary": {"q3-4b": {"passed": 21, "total": 27, "pass_rate": 0.78,
                              "median_s": 1.4, "metrics": {"code_pass": 0.91}},
                    "org/challenger": {"passed": 0, "total": 27,
                                       "pass_rate": 0.0, "median_s": 0.011,
                                       "metrics": {}}},
        "rows": [{"candidate": "org/challenger", "detail": REFUSED}
                 for _ in range(27)]})
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    rc = cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                {"name": "org/challenger", "lane": "code"})
    assert rc == 1
    err_text = capsys.readouterr().err
    assert "refused before it reached a model" in err_text, err_text
    assert "says nothing about the candidate" in err_text


# --- a run with no working control settles nothing -------------------------

def test_a_control_that_passed_nothing_blocks_any_verdict(monkeypatch,
                                                          tmp_path, capsys):
    """The general form of the refusal check. A doubled /v1 produced HTTP 404,
    both candidates scored 0/27, and the loop reported "does not beat the
    incumbent on the lane's metric". Issue #223."""
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Done())
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: {
        "summary": {"q3-4b": {"passed": 0, "total": 27, "pass_rate": 0.0,
                              "median_s": 0.004, "metrics": {}},
                    "org/challenger": {"passed": 0, "total": 27,
                                       "pass_rate": 0.0, "median_s": 0.004,
                                       "metrics": {}}},
        "rows": [{"candidate": "org/challenger",
                  "detail": "gateway returned HTTP 404: Not Found"}]})
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    rc = cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                {"name": "org/challenger", "lane": "code"})
    assert rc == 1
    assert "no working control" in capsys.readouterr().err


def test_a_working_control_still_lets_a_loss_be_recorded(monkeypatch, tmp_path):
    """The negative control for the guard: a real contest must still resolve,
    or nothing is ever declined and the ladder stops closing."""
    monkeypatch.setattr(subprocess, "run", lambda argv, **kw: _Done())
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: {
        "summary": {"q3-4b": {"passed": 21, "total": 27, "pass_rate": 0.78,
                              "median_s": 1.4, "metrics": {"code_pass": 0.91}},
                    "org/challenger": {"passed": 9, "total": 27,
                                       "pass_rate": 0.33, "median_s": 2.0,
                                       "metrics": {"code_pass": 0.40}}},
        "rows": [{"candidate": "org/challenger", "case_id": f"c{i}",
                  "passed": False, "detail": ""} for i in range(9)]
                + [{"candidate": "q3-4b", "case_id": f"c{i}",
                    "passed": True, "detail": ""} for i in range(9)]})
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    assert cli._measure_and_adopt(argparse.Namespace(repeat=3),
                                  {"name": "org/challenger", "lane": "code"}) == 0


def test_the_gateway_argument_needs_no_stripping_by_its_caller(monkeypatch):
    """routed_gateway returns the form --gateway wants. argv() used to strip
    /v1 and _measure_and_adopt did not, so one of the two 404'd."""
    from harness import screen

    monkeypatch.setattr(screen, "gateway_routes",
                        lambda config=None: ({"q3-4b"}, "http://h:8081/v1"))
    assert screen.routed_gateway("org/repo-id") == "http://h:8081"
    assert screen.routed_gateway("q3-4b") == ""


# --- one gateway, two candidates -------------------------------------------

CFG = """model_list:
  - model_name: q3-4b
    litellm_params:
      model: openai/mlx-community/Qwen3-4B-Instruct-2507-4bit
      api_base: http://up:8081/v1
"""


def test_an_alias_resolves_to_the_upstream_behind_it(tmp_path):
    from harness import screen

    cfg = tmp_path / "g.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    assert screen.upstream_of("q3-4b", cfg) == \
        "mlx-community/Qwen3-4B-Instruct-2507-4bit"


def test_a_repo_id_is_not_an_alias_and_resolves_to_nothing(tmp_path):
    """The negative control: translating a repo id would rename a candidate."""
    from harness import screen

    cfg = tmp_path / "g.yaml"
    cfg.write_text(CFG, encoding="utf-8")
    assert screen.upstream_of("org/some-repo-id", cfg) == ""
    assert screen.upstream_of("", cfg) == ""


def test_the_incumbent_travels_as_a_repo_id_when_the_run_goes_upstream(
        monkeypatch, tmp_path):
    """A paired run has ONE gateway. When the challenger sends it to
    mlx_lm.server the incumbent cannot go as a LiteLLM alias: :8081 has never
    heard of `q3-4b`, so the control scored 0 of 27. Issue #223."""
    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: (seen.update(argv=argv), _Done())[1])
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "http://up:8081")
    monkeypatch.setattr("harness.screen.upstream_of",
                        lambda alias, config=None: "mlx-community/Q3-4B")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: None)
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    cli._measure_and_adopt(argparse.Namespace(repeat=3),
                           {"name": "org/challenger", "lane": "code"})
    cands = seen["argv"][seen["argv"].index("--candidates") + 1]
    assert cands == "mlx-community/Q3-4B,org/challenger", cands


def test_the_incumbent_keeps_its_alias_when_the_run_stays_on_the_gateway(
        monkeypatch, tmp_path):
    """The negative control. A challenger that IS an alias needs no
    translation, and renaming the incumbent would change the exam."""
    seen = {}
    monkeypatch.setattr(subprocess, "run",
                        lambda argv, **kw: (seen.update(argv=argv), _Done())[1])
    monkeypatch.setattr(adopt, "default_for",
                        lambda lane, fallback, conn=None: "q3-4b")
    monkeypatch.setattr("harness.screen.routed_gateway",
                        lambda model, config=None: "")
    monkeypatch.setattr(cli, "_receipt_at", lambda out: None)
    real = ms.connect
    monkeypatch.setattr(ms, "connect", lambda *a, **k: real(tmp_path / "d.db"))

    cli._measure_and_adopt(argparse.Namespace(repeat=3),
                           {"name": "q3-8b", "lane": "code"})
    cands = seen["argv"][seen["argv"].index("--candidates") + 1]
    assert cands == "q3-4b,q3-8b", cands


# --- gauntlet #9: a lookup miss must not look like a pass ------------------

def test_a_key_that_matches_no_row_is_reported_not_silently_ok():
    """The receipt key for an engine lane is `mflux/<id>-q8`, not the bare
    repo id. Asking with the wrong one used to return "" -- the same answer as
    "every case reached a model" -- so the guard would be skipped without a
    word. A safety check whose lookup miss looks like a pass is worse than no
    check at all."""
    rows = [{"candidate": "mflux/org/x-q8", "detail": REFUSED}
            for _ in range(9)]
    assert cli._all_refused(rows, "org/x") == cli.NO_ROWS_FOR_CANDIDATE


def test_the_right_key_still_answers_the_real_question():
    rows = [{"candidate": "mflux/org/x-q8", "detail": REFUSED}
            for _ in range(9)]
    assert cli._all_refused(rows, "mflux/org/x-q8") == "invalid model name"


def test_an_empty_receipt_is_not_a_key_mismatch():
    """The negative control. No rows AT ALL is a different fact, handled by
    the caller, and must not be reported as a naming problem."""
    assert cli._all_refused([], "org/x") == ""
    assert cli._all_refused(None, "org/x") == ""
