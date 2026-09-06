"""The eval suite's own entry point: candidate strings to runners, and the
modality selection that decides what actually gets measured.
"""
import pytest

from evals.run import ALL_MODALITIES, build_runner, select_cases
from evals.core import Case
from evals.runners.process import ProcessRunner
from evals.runners.text import CompletionRunner


def test_a_gateway_alias_becomes_a_completion_runner():
    r = build_runner("local-mid", "http://gw", None)
    assert isinstance(r, CompletionRunner)
    assert r.candidate == "local-mid"


def test_an_engine_spec_becomes_a_process_runner(tmp_path):
    r = build_runner("mflux:z-image-turbo", "http://gw", tmp_path)
    assert isinstance(r, ProcessRunner)
    assert r.candidate == "mflux/z-image-turbo-q8"


def test_a_candidate_is_an_engine_when_it_names_a_known_engine(tmp_path):
    assert isinstance(build_runner("h3", "http://gw", tmp_path), ProcessRunner)


def test_a_bad_engine_spec_is_reported_before_anything_runs(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("mflux:x,quantise=4", "http://gw", tmp_path)
    assert "quantise" in str(e.value)


def test_process_candidates_need_an_output_directory():
    """Images have to land somewhere; silently writing to the cwd is worse."""
    with pytest.raises(SystemExit):
        build_runner("mflux:z-image-turbo", "http://gw", None)


# ---- modality selection ----------------------------------------------------

CASES = [Case(id="a", modality="svg", prompt="x"),
         Case(id="b", modality="web", prompt="x"),
         Case(id="c", modality="image", prompt="x")]


def test_selecting_one_modality():
    assert [c.id for c in select_cases(CASES, "svg")] == ["a"]


def test_all_means_all_of_them_not_just_the_text_ones():
    """`--modality all` used to mean text only, so mixing in an image candidate
    raised SystemExit and the flag quietly measured half the suite."""
    assert {c.id for c in select_cases(CASES, "all")} == {"a", "b", "c"}
    assert "image" in ALL_MODALITIES


def test_a_modality_with_no_cases_is_an_error_not_an_empty_run():
    with pytest.raises(SystemExit):
        select_cases(CASES, "video")


def test_an_unknown_modality_is_rejected():
    with pytest.raises(SystemExit):
        select_cases(CASES, "telepathy")


# ---- pairing candidates with the cases they can actually run ---------------

def test_a_text_candidate_is_only_given_text_cases():
    """Handing mflux an SVG case, or the gateway an image case, produces a
    failure row that says nothing about the candidate."""
    assert [c.id for c in runnable("local-mid", CASES)] == ["a", "b"]


def test_an_image_engine_is_only_given_image_cases():
    assert [c.id for c in runnable("mflux:z-image-turbo", CASES)] == ["c"]


def runnable(candidate, cases):
    from evals.run import cases_for
    return cases_for(candidate, cases)
