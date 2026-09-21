"""`lh compare`: two models, one command, a verdict. Issue #259.

Every lane's path is exercised here with no weights, no gateway and no model
server. A day of this week went into discovering, by running real models for
hours, faults that these catch in milliseconds: a candidate spelled for the
wrong server, an alias sent to a server that had never heard of it, a control
that contributed no rows, a service that held a port and answered nothing.

The rule these are built around: A CANDIDATE MEASURED BESIDE A CONTROL THAT DID
NOT RUN SAYS NOTHING. That is the single most common way this project produced
a confident wrong answer, and several tests below exist only to keep it true.
"""
import json
from pathlib import Path

import pytest

from harness import compare, lanes

UP = lambda port: True          # noqa: E731 - every service answering
DOWN = lambda port: False       # noqa: E731 - none of them answering


def receipt(tmp_path, summary, rows=None) -> Path:
    """A receipt shaped like the one evals.run writes."""
    out = tmp_path / "run"
    out.mkdir(parents=True, exist_ok=True)
    (out / "results.json").write_text(
        json.dumps({"summary": summary, "rows": rows or []}), encoding="utf-8")
    return out


def rows_for(candidate, passes, total=3):
    """Per-case rows, so head_to_head has something to pair on."""
    return [{"case_id": f"c{i}", "candidate": candidate,
             "passed": i < passes} for i in range(total)]


# --- resolution: the caller should not need to know the spelling ----------

@pytest.mark.parametrize("lane,a,b", [
    ("code", "q3-4b", "local-large"),
    ("web", "q3-4b", "local-large"),
    ("svg", "q3-4b", "local-large"),
    ("extract", "q3-4b", "local-large"),
    ("image", "mflux:z-image-turbo", "mflux:flux2-klein-4b"),
    ("music", "acestep:acestep-v15-turbo", "acestep:acestep-v15-base"),
    ("tts", "mlx-community/Kokoro-82M-bf16", "mlx-community/other-tts"),
    ("stt", "mlx-community/parakeet-tdt-0.6b-v2", "mlx-community/whisper"),
])
def test_every_lane_can_plan_a_comparison(lane, a, b):
    """ONE CODE PATH FOR EVERY LANE. A branch per modality is how `music`
    arrived with --repeat silently doing nothing."""
    p = compare.plan(lane, a, b, probe=UP)
    assert p.ok, p.blocked
    assert all(p.specs), p.specs


def test_an_alias_and_a_repo_id_are_made_to_meet_at_one_server():
    """#223. One run reaches one server. An alias sent to the server that
    hot-swaps repo ids scored the control 0/27, so the run had nothing to
    compare against and said so only after spending the time."""
    p = compare.plan("code", "q3-4b",
                     "mlx-community/Qwen3-4B-Instruct-2507-4bit", probe=UP)
    assert p.ok, p.blocked
    assert p.gateway.endswith(":8081"), p.gateway
    assert "/" in p.incumbent, (
        f"the alias should have been resolved to its upstream, got "
        f"{p.incumbent!r}")


def test_two_aliases_stay_on_the_gateway():
    """THE NEGATIVE CONTROL for the translation above. Translating when it is
    not needed would send both to a server that knows neither name."""
    p = compare.plan("code", "q3-4b", "local-large", probe=UP)
    assert p.gateway == "", p.gateway
    assert p.incumbent == "q3-4b"


def test_a_process_engine_needs_no_server_at_all():
    """mflux runs a binary. Requiring a gateway for it would refuse a
    comparison that needs nothing running."""
    p = compare.plan("image", "mflux:z-image-turbo", "mflux:flux2-klein-4b",
                     probe=DOWN)
    assert p.ok, p.blocked
    assert p.needs == []


# --- refuse early, and say which service and how --------------------------

def test_a_missing_service_is_named_before_anything_is_spent():
    """Hanging for 180 seconds against a dead server is how a morning went.
    The answer is available in milliseconds and should be given then."""
    p = compare.plan("code", "q3-4b", "local-large", probe=DOWN)
    assert not p.ok
    assert "gateway" in p.blocked
    assert "services.sh" in p.blocked, (
        f"say HOW to fix it, not just that it is broken: {p.blocked}")


def test_a_wedged_service_counts_as_absent():
    """A BOUND PORT IS NOT A WORKING SERVICE (#255). The probe asks for an
    answer, so a server holding a port and serving nothing is refused."""
    assert compare.answers(4000, probe=lambda port: False) is False


def test_an_unknown_lane_lists_the_real_ones():
    p = compare.plan("pictures", "a", "b", probe=UP)
    assert not p.ok
    assert all(name in p.blocked for name in ("image", "code", "music"))


def test_a_candidate_cannot_be_compared_with_itself():
    p = compare.plan("code", "q3-4b", "q3-4b", probe=UP)
    assert not p.ok


def test_a_lane_that_cannot_run_the_candidate_says_so():
    """video has no per-model runner. Better to say that than to fetch
    weights and discover it at the screen."""
    p = compare.plan("video", "org/some-video-model", "h3", probe=UP)
    assert not p.ok
    assert "no way to run" in p.blocked


# --- the control is the whole argument ------------------------------------

def test_a_missing_control_blocks_the_verdict(tmp_path):
    """THE DEFECT THIS PROJECT KEEPS MAKING. The challenger measured beside an
    incumbent that contributed nothing is not a result, and recording one
    settles a real model on our own gap."""
    out = receipt(tmp_path, {"local-large": {"passed": 3, "total": 3}})
    got = compare.run("code", "q3-4b", "local-large", out=out,
                      runner=lambda argv: 0, probe=UP)
    assert not got.ok
    assert "no control" in got.blocked
    assert got.verdict is None


def test_a_missing_challenger_blocks_the_verdict(tmp_path):
    out = receipt(tmp_path, {"q3-4b": {"passed": 3, "total": 3}})
    got = compare.run("code", "q3-4b", "local-large", out=out,
                      runner=lambda argv: 0, probe=UP)
    assert not got.ok
    assert "never measured" in got.blocked


def test_an_eval_that_exits_non_zero_records_nothing(tmp_path):
    got = compare.run("code", "q3-4b", "local-large", out=tmp_path / "run",
                      runner=lambda argv: 2, probe=UP)
    assert not got.ok
    assert "exited 2" in got.blocked


def test_a_receipt_key_that_differs_from_the_spec_is_still_matched(tmp_path):
    """#219. `mflux:flux2-klein-4b` is written `mflux/flux2-klein-4b-q8` in a
    receipt: different separator, resolved quantisation. Matching by equality
    reported the incumbent absent while its rows sat in the summary."""
    out = receipt(
        tmp_path,
        {"mflux/flux2-klein-4b-q8": {"passed": 3, "total": 3,
                                     "metrics": {"cer": 0.0}},
         "mflux/z-image-turbo-q8": {"passed": 1, "total": 3,
                                    "metrics": {"cer": 0.5}}},
        rows_for("mflux/flux2-klein-4b-q8", 3) +
        rows_for("mflux/z-image-turbo-q8", 1))
    got = compare.run("image", "mflux:flux2-klein-4b", "mflux:z-image-turbo",
                      out=out, runner=lambda argv: 0, probe=UP)
    assert got.ok, got.blocked
    assert got.verdict is not None


# --- the verdict ----------------------------------------------------------

def test_a_challenger_that_wins_is_adopted(tmp_path):
    """THE PATH THAT HAS NEVER RUN IN PRODUCTION. The adopt tier has reached a
    verdict four times and declined every one, so the half that installs a
    winner exists only here. A fake that simply wins proves it."""
    out = receipt(
        tmp_path,
        {"q3-4b": {"passed": 1, "total": 9, "metrics": {"code_pass": 0.2}},
         "local-large": {"passed": 9, "total": 9,
                         "metrics": {"code_pass": 0.95}}},
        rows_for("q3-4b", 1, 9) + rows_for("local-large", 9, 9))
    got = compare.run("code", "q3-4b", "local-large", out=out,
                      runner=lambda argv: 0, probe=UP)
    assert got.ok, got.blocked
    assert got.verdict.adopt, got.verdict.why
    assert "local-large" in compare.report(got)


def test_a_challenger_that_ties_keeps_the_incumbent(tmp_path):
    """A tie is not a win. The image lane produced exactly this: 9/9 against
    9/9, and the incumbent stayed."""
    same = {"passed": 9, "total": 9, "metrics": {"code_pass": 0.9}}
    out = receipt(tmp_path, {"q3-4b": dict(same), "local-large": dict(same)},
                  rows_for("q3-4b", 9, 9) + rows_for("local-large", 9, 9))
    got = compare.run("code", "q3-4b", "local-large", out=out,
                      runner=lambda argv: 0, probe=UP)
    assert got.ok, got.blocked
    assert not got.verdict.adopt


def test_a_win_too_small_to_establish_keeps_the_incumbent(tmp_path):
    """Both gates, and this is the one that fired on Z-Image: better on the
    metric, and 0 lost against 1 gained is p=1.00."""
    out = receipt(
        tmp_path,
        {"q3-4b": {"passed": 8, "total": 9, "metrics": {"code_pass": 0.80}},
         "local-large": {"passed": 9, "total": 9,
                         "metrics": {"code_pass": 0.81}}},
        rows_for("q3-4b", 8, 9) + rows_for("local-large", 9, 9))
    got = compare.run("code", "q3-4b", "local-large", out=out,
                      runner=lambda argv: 0, probe=UP)
    assert got.ok, got.blocked
    assert not got.verdict.adopt
    assert "not established" in got.verdict.why


def test_the_report_names_the_numbers_behind_the_verdict(tmp_path):
    """A verdict without its counts is an opinion."""
    out = receipt(
        tmp_path,
        {"q3-4b": {"passed": 1, "total": 9, "metrics": {"code_pass": 0.2}},
         "local-large": {"passed": 9, "total": 9,
                         "metrics": {"code_pass": 0.95}}},
        rows_for("q3-4b", 1, 9) + rows_for("local-large", 9, 9))
    text = compare.report(compare.run("code", "q3-4b", "local-large", out=out,
                                      runner=lambda argv: 0, probe=UP))
    assert "1/9" in text and "9/9" in text
    assert "code_pass" in text
    assert str(out) in text, "say where the receipt is"


# --- the command line -----------------------------------------------------

def test_the_eval_runs_from_the_checkout(tmp_path):
    """`uv run` walks UP to the nearest pyproject. A caller standing in a
    parent directory got that project's virtualenv and a ModuleNotFoundError
    for our own eval package, which cost two runs today."""
    assert (compare.REPO / "pyproject.toml").is_file()


def test_the_command_exits_non_zero_when_it_cannot_answer(monkeypatch, capsys):
    """EXIT STATUS IS PART OF THE ANSWER, so a script can tell "the challenger
    lost" from "nothing ran"."""
    from harness import cli

    monkeypatch.setattr(compare, "answers", lambda port, probe=None: False)
    assert cli.main(["compare", "code", "q3-4b", "local-large"]) != 0


def test_a_dry_run_spends_nothing(monkeypatch, capsys):
    from harness import cli

    monkeypatch.setattr(compare, "answers", lambda port, probe=None: True)
    monkeypatch.setattr(compare, "run", lambda *a, **k: pytest.fail(
        "a dry run must not measure anything"))
    assert cli.main(["compare", "code", "q3-4b", "local-large",
                     "--dry-run"]) == 0
    assert "against" in capsys.readouterr().out


def test_every_wanted_lane_is_reachable_from_the_command():
    """A lane the product cannot compare in is a lane the product does not
    have."""
    for lane in lanes.WANTED:
        assert lane in lanes.ALL
