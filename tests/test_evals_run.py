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


# ---- speech candidates -----------------------------------------------------

def test_a_tts_candidate_becomes_a_speech_runner(tmp_path):
    from evals.runners.speech import SpeechRunner
    r = build_runner("tts:mlx-community/Kokoro-82M-bf16,voice=am_adam",
                     "http://gw", tmp_path)
    assert isinstance(r, SpeechRunner)
    assert r.voice == "am_adam"
    assert r.model == "mlx-community/Kokoro-82M-bf16"
    assert r.candidate == "Kokoro-82M-bf16/am_adam"


def test_a_tts_candidate_without_a_voice_uses_a_cached_default(tmp_path):
    from harness import audio
    r = build_runner("tts:mlx-community/Kokoro-82M-bf16", "http://gw", tmp_path)
    assert r.voice == audio.DEFAULT_VOICE


def test_a_tts_candidate_needs_somewhere_to_put_the_audio():
    with pytest.raises(SystemExit):
        build_runner("tts:mlx-community/Kokoro-82M-bf16", "http://gw", None)


def test_an_unknown_tts_option_is_rejected(tmp_path):
    with pytest.raises(SystemExit, match="viice"):
        build_runner("tts:mlx-community/Kokoro-82M-bf16,viice=am_adam",
                     "http://gw", tmp_path)


def test_a_tts_candidate_only_gets_tts_cases():
    cases = CASES + [Case(id="d", modality="tts", prompt="hello")]
    got = runnable("tts:mlx-community/Kokoro-82M-bf16", cases)
    assert [c.id for c in got] == ["d"]


def test_a_gateway_alias_never_gets_tts_cases():
    """The gateway serves text; handing it a sentence to speak produces a
    failure row that says nothing about the candidate."""
    cases = CASES + [Case(id="d", modality="tts", prompt="hello")]
    assert "d" not in [c.id for c in runnable("local-mid", cases)]


# ---- the report ------------------------------------------------------------

from evals.core import Result, summarize  # noqa: E402
from evals.run import report  # noqa: E402


def two_candidates(wer_a, wer_b):
    return summarize(
        [Result("c", "a", True, 1.0, 0, "", metrics={"wer": wer_a}),
         Result("c", "b", True, 2.0, 0, "", metrics={"wer": wer_b})])


def test_the_report_shows_the_metric_it_ranks_on(capsys):
    report(two_candidates(0.05, 0.30))
    out = capsys.readouterr().out
    assert "wer" in out
    assert "0.05" in out and "0.3" in out


def test_candidates_are_ordered_by_the_metric_when_both_pass(capsys):
    """Both at 100% is the situation the suite kept landing in. Sorting on
    latency alone made the faster of two put the worse one on top."""
    report(two_candidates(0.30, 0.05))
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(("a ", "b "))]
    assert lines[0].startswith("b")


def test_the_competence_gate_warning_only_fires_when_nothing_discriminates(capsys):
    report(two_candidates(0.05, 0.30))
    assert "competence gate" not in capsys.readouterr().out


def test_the_warning_does_fire_when_every_candidate_looks_identical(capsys):
    s = summarize([Result("c", "a", True, 1.0, 0, ""),
                   Result("c", "b", True, 1.0, 0, "")])
    report(s)
    assert "competence gate" in capsys.readouterr().out


def test_the_worst_case_is_shown_next_to_the_average(capsys):
    s = summarize([Result("a", "k", True, 1.0, 0, "", metrics={"wer": 0.0}),
                   Result("b", "k", True, 1.0, 0, "", metrics={"wer": 0.4})])
    report(s)
    out = capsys.readouterr().out
    assert "0.2" in out and "0.4" in out
