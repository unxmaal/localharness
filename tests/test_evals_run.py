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


def test_a_tts_candidate_can_name_its_voice(tmp_path):
    r = build_runner("tts:mlx-community/Kokoro-82M-bf16,voice=bm_george",
                     "http://gw", tmp_path)
    assert r.voice == "bm_george"


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


# ---- repeats ---------------------------------------------------------------
#
# One sample per prompt ranks noise. Diffusion output varies enormously with
# the seed, and a language model at temperature 0.2 is not deterministic
# either, so a single run decides a comparison on luck.

from evals.run import expand_cases  # noqa: E402


def test_without_repeats_the_cases_are_untouched():
    got = expand_cases(CASES, 1)
    assert [c.id for c in got] == ["a", "b", "c"]
    assert got[0] is CASES[0]


def test_repeats_produce_distinctly_named_rows():
    got = expand_cases([CASES[2]], 3)
    assert [c.id for c in got] == ["c#1", "c#2", "c#3"]


def test_each_repeat_gets_a_different_seed():
    got = expand_cases([Case(id="c", modality="image", prompt="x",
                             params={"seed": 42})], 3)
    seeds = [c.params["seed"] for c in got]
    assert len(set(seeds)) == 3
    assert seeds[0] == 42, "the case's own seed should still be one of them"


def test_a_case_with_no_seed_still_gets_distinct_ones():
    got = expand_cases([Case(id="c", modality="image", prompt="x")], 3)
    assert len({c.params["seed"] for c in got}) == 3


def test_repeating_does_not_mutate_the_original_case():
    original = Case(id="c", modality="image", prompt="x", params={"seed": 42})
    expand_cases([original], 3)
    assert original.params == {"seed": 42}


def test_deterministic_modalities_are_not_repeated():
    """Kokoro at a fixed voice and speed produces the same bytes every time.
    Repeating it burns time to average three identical numbers."""
    got = expand_cases([Case(id="t", modality="tts", prompt="hello")], 3)
    assert [c.id for c in got] == ["t"]


def test_repeats_and_singletons_coexist_in_one_run():
    cases = [Case(id="img", modality="image", prompt="x"),
             Case(id="say", modality="tts", prompt="x")]
    got = expand_cases(cases, 2)
    assert [c.id for c in got] == ["img#1", "img#2", "say"]


def test_a_repeat_count_below_one_is_rejected():
    with pytest.raises(SystemExit):
        expand_cases(CASES, 0)


def test_cases_no_candidate_can_run_are_reported(capsys):
    """A case that silently never runs is invisible, which is the same class of
    problem as a silent pass: the summary looks complete and one lane was
    never measured. A full-suite run with no video candidate did exactly this.
    """
    cases = CASES + [Case(id="clip", modality="video", prompt="x")]
    unrun = report_unrun(cases, ["local-mid", "mflux:z-image-turbo"])
    assert "clip" in unrun and "video" in unrun


def test_nothing_is_reported_when_every_case_has_a_candidate():
    assert report_unrun(CASES, ["local-mid", "mflux:z-image-turbo"]) == ""


def report_unrun(cases, candidates):
    from evals.run import unrun_summary
    return unrun_summary(cases, candidates)


def test_a_higher_is_better_metric_ranks_the_larger_value_first(capsys):
    """`ink` and `motion` were ranked ascending like an error rate, so the
    candidate that drew least came top."""
    s = summarize([Result("c", "sparse", True, 1.0, 0, "", metrics={"ink": 0.02}),
                   Result("c", "rich", True, 1.0, 0, "", metrics={"ink": 0.40})])
    report(s)
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(("sparse", "rich"))]
    assert lines[0].startswith("rich")


def test_a_lower_is_better_metric_still_ranks_the_smaller_value_first(capsys):
    report(two_candidates(0.30, 0.05))
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(("a ", "b "))]
    assert lines[0].startswith("b")


def test_the_report_marks_which_way_each_metric_runs(capsys):
    """A column of numbers with no direction is a column you cannot read."""
    report(two_candidates(0.05, 0.30))
    out = capsys.readouterr().out
    assert "wer" in out
    assert "lower is better" in out or "↓" in out


def test_the_adherence_flag_reaches_the_process_runner(tmp_path):
    r = build_runner("mflux:z-image-turbo", "http://gw", tmp_path,
                     adherence="hpsv2")
    assert r.score_kwargs() == {"adherence": "hpsv2"}


def test_adherence_is_off_by_default(tmp_path):
    assert build_runner("mflux:z-image-turbo", "http://gw",
                        tmp_path).score_kwargs() == {}


def test_a_disagreement_between_pass_rate_and_a_metric_is_called_out(capsys):
    """Seen live on the code lane: local-mid passed 50% of cases to
    local-large's 33%, while local-large wrote 75.6% correct code to
    local-mid's 52.8%. A case with six strict checks fails outright on one
    miss, so the binary view punishes the better model. Both numbers are true
    and the reader has to be told they point different ways."""
    s = summarize([
        # Passes 2 of 3, but the one it fails it fails badly.
        Result("a", "narrow", True, 1.0, 0, "", metrics={"code_pass": 1.0}),
        Result("b", "narrow", True, 1.0, 0, "", metrics={"code_pass": 1.0}),
        Result("c", "narrow", False, 1.0, 0, "x", metrics={"code_pass": 0.4}),
        # Passes 1 of 3, but misses only one check in each of the others.
        Result("a", "strong", True, 1.0, 0, "", metrics={"code_pass": 1.0}),
        Result("b", "strong", False, 1.0, 0, "x", metrics={"code_pass": 0.9}),
        Result("c", "strong", False, 1.0, 0, "x", metrics={"code_pass": 0.9}),
    ])
    report(s)
    out = capsys.readouterr().out
    assert "disagree" in out.lower()
    assert "code_pass" in out


def test_no_disagreement_note_when_the_orders_agree(capsys):
    s = summarize([
        Result("a", "best", True, 1.0, 0, "", metrics={"code_pass": 1.0}),
        Result("a", "worst", False, 1.0, 0, "x", metrics={"code_pass": 0.2}),
    ])
    report(s)
    assert "disagree" not in capsys.readouterr().out.lower()


# ---- stt candidates --------------------------------------------------------

def test_an_stt_candidate_becomes_a_transcription_runner(tmp_path):
    from evals.runners.transcription import TranscriptionRunner
    r = build_runner("stt:mlx-community/parakeet-tdt-0.6b-v2", "http://gw", tmp_path)
    assert isinstance(r, TranscriptionRunner)
    assert r.model == "mlx-community/parakeet-tdt-0.6b-v2"
    assert r.candidate == "parakeet-tdt-0.6b-v2"


def test_an_stt_candidate_needs_no_output_directory():
    """It produces text, not files."""
    from evals.runners.transcription import TranscriptionRunner
    assert isinstance(build_runner("stt:some/model", "http://gw", None),
                      TranscriptionRunner)


def test_an_stt_candidate_only_gets_stt_cases():
    cases = CASES + [Case(id="u", modality="stt", prompt="hello")]
    assert [c.id for c in runnable("stt:some/model", cases)] == ["u"]


def test_tts_and_stt_candidates_are_told_apart():
    """`tts:` and `stt:` differ by one letter and route to different runners."""
    from evals.run import modality_of
    assert modality_of("tts:m") == "tts"
    assert modality_of("stt:m") == "stt"


def test_a_candidate_with_no_metric_is_ranked_last_not_first(capsys):
    """Seen live: whisper failed all 40 stt cases, reported no wer at all, and
    a missing lower-is-better metric defaulting to 0.0 read as a PERFECT error
    rate. The summary table got it right; the ranking and the disagreement note
    did not."""
    s = summarize([
        Result("a", "working", True, 1.0, 0, "",
               metrics={"wer": 0.02, "wer_errors": 1, "wer_words": 50}),
        Result("a", "broken", False, 0.1, 0, "server error"),
    ])
    report(s)
    lines = [ln for ln in capsys.readouterr().out.splitlines()
             if ln.startswith(("working", "broken"))]
    assert lines[0].startswith("working")


def test_a_candidate_with_no_metric_is_left_out_of_the_disagreement_note(capsys):
    s = summarize([
        Result("a", "working", True, 1.0, 0, "",
               metrics={"wer": 0.02, "wer_errors": 1, "wer_words": 50}),
        Result("a", "broken", False, 0.1, 0, "server error"),
    ])
    report(s)
    assert "disagree" not in capsys.readouterr().out.lower()


def test_a_tts_candidate_without_a_voice_asks_for_none(tmp_path):
    """Defaulting to a Kokoro voice name would send bm_george to Qwen3-TTS."""
    assert build_runner("tts:mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16",
                        "http://gw", tmp_path).voice == ""


# ---- stt backends and options ---------------------------------------------

def test_an_stt_candidate_takes_a_backend_and_a_language():
    r = build_runner(
        "stt:mlx-community/whisper-large-v3-mlx,backend=whisper,language=fr",
        "http://gw", None)
    assert r.backend == "whisper"
    assert r.language == "fr"
    assert r.model == "mlx-community/whisper-large-v3-mlx"
    assert r.candidate == "whisper-large-v3-mlx/fr"


def test_an_stt_candidate_still_defaults_to_the_server_backend():
    assert build_runner("stt:some/model", "http://gw", None).backend == "server"


def test_an_unknown_stt_option_is_named_before_the_run(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("stt:some/model,lang=fr", "http://gw", None)
    assert "lang" in str(e.value)


def test_an_unknown_stt_backend_is_reported_as_a_bad_spec():
    """A ValueError from deep in audio.transcriber would print a traceback; a
    bad candidate string deserves the same treatment as a bad engine spec."""
    with pytest.raises(SystemExit) as e:
        build_runner("stt:some/model,backend=wisper", "http://gw", None)
    assert "wisper" in str(e.value)
