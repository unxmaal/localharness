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
    candidate that moved least came top. `ink` has since become neutral --
    see test_ink_is_a_floor_and_is_not_ranked_on -- so `motion` carries this."""
    s = summarize([Result("c", "sparse", True, 1.0, 0, "", metrics={"motion": 0.02}),
                   Result("c", "rich", True, 1.0, 0, "", metrics={"motion": 0.40})])
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


# ---- tts cloning options ---------------------------------------------------

def test_a_tts_candidate_takes_a_reference_clip_and_a_lang_code(tmp_path):
    ref = tmp_path / "fleurs-fr-male-1.wav"
    ref.write_bytes(b"x")
    r = build_runner(
        f"tts:litmudoc/Chatterbox-Multilingual-MLX-v2-Q8,ref_audio={ref},"
        "lang_code=fr,ear=whisper:fr", "http://gw", tmp_path)
    assert r.ref_audio == ref
    assert r.lang_code == "fr"
    assert r.ear == "whisper:fr"
    assert r.candidate.endswith("/fleurs-fr-male-1")


def test_a_tts_candidate_still_defaults_to_the_server_ear(tmp_path):
    r = build_runner("tts:mlx-community/Kokoro-82M-bf16,voice=am_adam",
                     "http://gw", tmp_path)
    assert r.ear == "server"
    assert r.ref_audio is None


def test_an_unknown_tts_option_is_named_before_the_run(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("tts:some/model,language=fr", "http://gw", tmp_path)
    assert "language" in str(e.value)
    assert "lang_code" in str(e.value)


def test_an_unknown_ear_is_reported_as_a_bad_spec(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("tts:some/model,ear=wisper:fr", "http://gw", tmp_path)
    assert "wisper" in str(e.value)


def test_a_reference_clip_that_is_not_there_is_caught_before_the_run(tmp_path):
    """A cloning run is the slow lane; finding the typo forty cases in is the
    expensive way."""
    with pytest.raises(SystemExit) as e:
        build_runner(f"tts:some/model,ref_audio={tmp_path}/gone.wav",
                     "http://gw", tmp_path)
    assert "gone.wav" in str(e.value)


# ---- language selection ----------------------------------------------------
# Handing a French case to an English-only candidate produces a word error rate
# near 1.0, which is a row that says nothing about the candidate -- the same
# mistake as handing an SVG case to mflux.

def test_an_english_candidate_does_not_get_french_cases(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="en1", modality="tts", prompt="hello"),
             Case(id="fr1", modality="tts", prompt="bonjour", language="fr")]
    got = cases_for("tts:mlx-community/Kokoro-82M-bf16,voice=am_adam", cases)
    assert [c.id for c in got] == ["en1"]


def test_a_french_eared_candidate_gets_the_french_cases(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="en1", modality="tts", prompt="hello"),
             Case(id="fr1", modality="tts", prompt="bonjour", language="fr")]
    got = cases_for("tts:some/model,ear=whisper:fr", cases)
    assert [c.id for c in got] == ["fr1"]


def test_an_stt_candidate_selects_by_its_own_language(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="en1", modality="stt", prompt="hello"),
             Case(id="fr1", modality="stt", prompt="bonjour", language="fr")]
    fr = "stt:mlx-community/whisper-large-v3-mlx,backend=whisper,language=fr"
    assert [c.id for c in cases_for(fr, cases)] == ["fr1"]
    assert [c.id for c in cases_for("stt:some/model", cases)] == ["en1"]


def test_language_selection_leaves_the_other_lanes_alone(tmp_path):
    """Only speech candidates have an ear. An image model has no language and
    must not lose its cases to this filter."""
    from evals.run import cases_for
    cases = [Case(id="fox", modality="image", prompt="a fox",
                  params={"width": 64, "height": 64})]
    assert [c.id for c in cases_for("mflux:z-image-turbo", cases)] == ["fox"]


def test_the_disagreement_note_is_silent_when_a_metric_is_partial(capsys):
    """Caught by the PARTIAL line on its first live run: the note announced
    that local-mid 'scores better on ink' at 0.239 against 0.195, while its
    0.239 came from the ONE case it passed and the other from two. Ranking on a
    metric computed over different samples is the mistake this whole item is
    about, and the note was making it."""
    s = summarize([
        Result("a", "thorough", True, 1.0, 0, "", metrics={"ink": 0.19}),
        Result("b", "thorough", True, 1.0, 0, "", metrics={"ink": 0.20}),
        Result("c", "thorough", False, 1.0, 0, "drew 1 shape"),
        Result("a", "lucky", False, 1.0, 0, "drew 1 shape"),
        Result("b", "lucky", False, 1.0, 0, "drew 1 shape"),
        Result("c", "lucky", True, 1.0, 0, "", metrics={"ink": 0.24}),
    ])
    report(s)
    out = capsys.readouterr().out
    assert "PARTIAL" in out
    assert "disagree" not in out.lower(), out


# ---- the svg lane's second method ------------------------------------------

def test_a_trace_candidate_becomes_a_trace_runner(tmp_path):
    from evals.runners.trace import TraceRunner
    r = build_runner("trace:mflux:flux2-klein-4b", "http://gw", tmp_path)
    assert isinstance(r, TraceRunner)
    assert r.candidate.startswith("trace/")


def test_an_omnisvg_candidate_becomes_an_omnisvg_runner(tmp_path):
    from evals.runners.omnisvg import OmniSVGRunner
    r = build_runner("omnisvg:4B", "http://gw", tmp_path)
    assert isinstance(r, OmniSVGRunner)
    assert r.candidate == "omnisvg:4B"


def test_an_omnisvg_candidate_needs_no_output_directory(tmp_path):
    # It returns SVG text, not a file this suite has to place.
    assert build_runner("omnisvg:4B", "http://gw", None).size == "4B"


def test_an_omnisvg_candidate_only_gets_svg_cases(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="s", modality="svg", prompt="a gear"),
             Case(id="i", modality="image", prompt="a fox",
                  params={"width": 64, "height": 64})]
    assert [c.id for c in cases_for("omnisvg:4B", cases)] == ["s"]


def test_a_bad_omnisvg_size_is_caught_before_anything_runs(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("omnisvg:2B", "http://gw", tmp_path)
    assert "2B" in str(e.value)


def test_an_unknown_omnisvg_option_is_named(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("omnisvg:4B,temperature=0.5", "http://gw", tmp_path)
    assert "temperature" in str(e.value)
    assert "candidates" in str(e.value)


def test_a_trace_candidate_only_gets_svg_cases(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="s", modality="svg", prompt="a gear"),
             Case(id="i", modality="image", prompt="a fox",
                  params={"width": 64, "height": 64})]
    got = cases_for("trace:mflux:flux2-klein-4b", cases)
    assert [c.id for c in got] == ["s"]


def test_a_trace_candidate_needs_an_output_directory(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("trace:mflux:flux2-klein-4b", "http://gw", None)
    assert "--out" in str(e.value)


def test_a_bad_engine_inside_a_trace_spec_is_caught_early(tmp_path):
    with pytest.raises(SystemExit):
        build_runner("trace:mflux:no-such-model-xyz", "http://gw", tmp_path)


def test_a_case_that_is_unfair_to_tracing_is_not_given_to_it():
    """A traced SVG has no <text> element by construction, so a case asserting
    one measures the method rather than the candidate."""
    from evals.run import cases_for
    cases = [Case(id="bars", modality="svg", prompt="bars", methods=("llm",)),
             Case(id="gear", modality="svg", prompt="a gear")]
    assert [c.id for c in cases_for("trace:mflux:flux2-klein-4b", cases)] == ["gear"]
    assert [c.id for c in cases_for("local-large", cases)] == ["bars", "gear"]


def test_the_report_says_when_candidates_sat_different_exams(capsys):
    """Caught live: `trace` ran 4 rows and `local-large` 6, because chart-bars
    is llm-only. 4/4 against 2/6 reads as a pass-rate comparison and is not
    one. Whether two runs are comparable was item 2; this is the same question
    one level down, INSIDE a single run."""
    s = summarize([
        Result("gear", "trace", True, 1.0, 0, ""),
        Result("mark", "trace", True, 1.0, 0, ""),
        Result("gear", "llm", False, 1.0, 0, "drew 1"),
        Result("mark", "llm", False, 1.0, 0, "drew 1"),
        Result("bars", "llm", True, 1.0, 0, ""),
    ])
    report(s)
    out = capsys.readouterr().out
    assert "different cases" in out.lower()
    assert "bars" in out


def test_no_such_note_when_everyone_sat_the_same_exam(capsys):
    s = summarize([
        Result("gear", "a", True, 1.0, 0, ""),
        Result("gear", "b", False, 1.0, 0, "drew 1"),
    ])
    report(s)
    assert "different cases" not in capsys.readouterr().out.lower()


def test_the_disagreement_note_needs_an_actual_disagreement(capsys):
    """Caught live on the parakeet run: all three candidates passed 40/40 and
    the note still announced that one 'passes more cases'. A tie on pass rate
    cannot disagree with anything."""
    s = summarize([
        Result("a", "x", True, 1.0, 0, "", metrics={"wer": 0.01,
                                                    "wer_errors": 1, "wer_words": 100}),
        Result("a", "y", True, 1.0, 0, "", metrics={"wer": 0.05,
                                                    "wer_errors": 5, "wer_words": 100}),
    ])
    report(s)
    assert "disagree" not in capsys.readouterr().out.lower()


def test_a_run_gets_its_own_directory_without_being_told(monkeypatch, tmp_path):
    """--out was required for every lane that writes anything, so every
    invocation in this repo's history picked a directory by hand and they all
    picked differently: .logs/img, .logs/voices, .logs/ev-svg-fair."""
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    from evals.run import resolve_outdir
    d = resolve_outdir(None, "svg")
    assert d.is_dir()
    assert d.parent.name == "runs"
    assert "svg" in d.name


def test_an_explicit_out_is_still_honoured(monkeypatch, tmp_path):
    from evals.run import resolve_outdir
    assert resolve_outdir(str(tmp_path / "here"), "svg") == tmp_path / "here"


def test_a_repair_candidate_becomes_a_repair_runner(tmp_path):
    from evals.runners.repair import RepairRunner
    r = build_runner("repair:local-large", "http://gw", tmp_path)
    assert isinstance(r, RepairRunner)
    assert r.candidate == "repair/local-large"


def test_the_repair_budget_is_settable(tmp_path):
    r = build_runner("repair:local-large,attempts=5", "http://gw", tmp_path)
    assert r.attempts == 5


def test_a_repair_candidate_gets_the_text_lanes(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="s", modality="svg", prompt="a gear"),
             Case(id="i", modality="image", prompt="a fox",
                  params={"width": 64, "height": 64})]
    assert [c.id for c in cases_for("repair:local-large", cases)] == ["s"]


def test_an_unknown_repair_option_is_named(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("repair:local-large,tries=5", "http://gw", tmp_path)
    assert "tries" in str(e.value)


# ---- --compare -------------------------------------------------------------
# comparable() has existed, tested, and been called by nothing. A guard that
# nothing invokes is a function.

def _run_dir(tmp_path, name, receipt, summary):
    import json
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps(
        {"receipt": receipt, "summary": summary}))
    return d / "results.json"


RECEIPT = {"modality": "svg", "case_ids": ["a", "b"], "repeat": 1,
           "sampling": {"svg": {"temperature": 0.4}}, "gateway": "http://gw",
           "adherence": ""}


def test_two_runs_of_the_same_shape_compare(tmp_path, capsys):
    from evals.run import compare_runs
    a = _run_dir(tmp_path, "a", RECEIPT, {"local-large": {"pass_rate": 0.5,
                                                          "total": 2, "passed": 1,
                                                          "median_s": 1.0,
                                                          "metrics": {"ink": 0.1}}})
    b = _run_dir(tmp_path, "b", RECEIPT, {"q3-4b": {"pass_rate": 1.0,
                                                    "total": 2, "passed": 2,
                                                    "median_s": 2.0,
                                                    "metrics": {"ink": 0.3}}})
    assert compare_runs([str(a), str(b)]) == 0
    out = capsys.readouterr().out
    assert "local-large" in out and "q3-4b" in out


def test_incomparable_runs_are_refused_with_the_axis_that_differs(tmp_path, capsys):
    from evals.run import compare_runs
    a = _run_dir(tmp_path, "a", RECEIPT, {"x": {"pass_rate": 1.0, "total": 2,
                                                "passed": 2, "median_s": 1.0,
                                                "metrics": {}}})
    other = {**RECEIPT, "sampling": {"svg": {"temperature": 0.9}}}
    b = _run_dir(tmp_path, "b", other, {"y": {"pass_rate": 1.0, "total": 2,
                                              "passed": 2, "median_s": 1.0,
                                              "metrics": {}}})
    assert compare_runs([str(a), str(b)]) == 1
    err = capsys.readouterr().out + capsys.readouterr().err
    assert "sampling" in err.lower() or "refus" in err.lower()


def test_a_run_with_no_receipt_cannot_be_compared(tmp_path, capsys):
    """Runs written before receipts existed. Silently comparing them is
    exactly the thing this guard is for."""
    import json
    from evals.run import compare_runs
    d = tmp_path / "old"
    d.mkdir()
    (d / "results.json").write_text(json.dumps({"summary": {}}))
    b = _run_dir(tmp_path, "new", RECEIPT, {})
    assert compare_runs([str(d / "results.json"), str(b)]) == 1
    assert "receipt" in (capsys.readouterr().out).lower()


def test_compare_works_through_the_command_line(tmp_path, capsys):
    """The unit tests called compare_runs() directly and passed while the CLI
    path was broken: --candidates was still required, so `--compare` alone
    died in argparse. Test the entry point, not just the function."""
    from evals.run import main
    a = _run_dir(tmp_path, "a", RECEIPT, {"x": {"pass_rate": 1.0, "total": 2,
                                                "passed": 2, "median_s": 1.0,
                                                "metrics": {}}})
    assert main(["--compare", str(a), str(a)]) == 0


def test_a_normal_run_still_requires_its_arguments(tmp_path):
    from evals.run import main
    with pytest.raises(SystemExit):
        main(["--modality", "svg"])          # no --candidates
    with pytest.raises(SystemExit):
        main(["--candidates", "local-mid"])  # no --modality


def test_an_upscale_candidate_becomes_a_chain_runner(tmp_path):
    from evals.runners.chain import ChainRunner
    r = build_runner("upscale-seedvr2:mflux:flux2-klein-4b", "http://gw", tmp_path)
    assert isinstance(r, ChainRunner)
    assert r.candidate.startswith("upscale-seedvr2/")


def test_a_chain_candidate_only_gets_image_cases(tmp_path):
    from evals.run import cases_for
    cases = [Case(id="i", modality="image", prompt="a fox",
                  params={"width": 64, "height": 64}),
             Case(id="s", modality="svg", prompt="a gear")]
    got = cases_for("upscale-seedvr2:mflux:flux2-klein-4b", cases)
    assert [c.id for c in got] == ["i"]


def test_a_chain_candidate_needs_an_output_directory(tmp_path):
    with pytest.raises(SystemExit) as e:
        build_runner("upscale-seedvr2:mflux:flux2-klein-4b", "http://gw", None)
    assert "--out" in str(e.value)


def test_screen_picks_one_case_per_modality_and_shrinks_it():
    """1,081 pairs at full settings does not run. The screen is what makes
    traversal affordable."""
    from evals.core import Case
    from evals.run import screen_cases
    cases = [Case(id="b", modality="image", prompt="p",
                  params={"width": 1024, "height": 1024, "steps": 20}),
             Case(id="a", modality="image", prompt="p",
                  params={"width": 512, "height": 512, "steps": 8}),
             Case(id="c", modality="svg", prompt="p")]
    got = screen_cases(cases)
    assert {c.modality for c in got} == {"image", "svg"}
    assert len(got) == 2
    img = next(c for c in got if c.modality == "image")
    assert img.id == "a", "picks deterministically, by id"
    assert img.params["width"] == 256 and img.params["steps"] == 2


def test_screen_never_enlarges_a_case_that_is_already_smaller():
    from evals.core import Case
    from evals.run import screen_cases
    got = screen_cases([Case(id="a", modality="image", prompt="p",
                             params={"width": 64, "height": 64, "steps": 1})])
    assert got[0].params["width"] == 64
    assert got[0].params["steps"] == 1


def test_screen_does_not_mutate_the_case_it_was_given():
    from evals.core import Case
    from evals.run import screen_cases
    c = Case(id="a", modality="image", prompt="p", params={"width": 1024})
    screen_cases([c])
    assert c.params["width"] == 1024


def test_omnisvg_is_not_an_llm_for_the_purpose_of_case_methods():
    # It emits move/line/curve/arc/close and nothing else, so a case that needs
    # a <text> element is measuring the method, not the candidate. Caught live:
    # chart-bars failed omnisvg for "missing required content: <text".
    from evals.run import method_of
    assert method_of("omnisvg:4B") == "omnisvg"
    assert method_of("local-large") == "llm"
    assert method_of("trace:mflux:flux2-klein-4b") == "trace"


def test_a_text_only_case_is_withheld_from_omnisvg():
    from evals.run import cases_for
    cases = [Case(id="chart", modality="svg", prompt="a bar chart",
                  methods=("llm",)),
             Case(id="gear", modality="svg", prompt="a gear")]
    assert [c.id for c in cases_for("omnisvg:4B", cases)] == ["gear"]
    assert [c.id for c in cases_for("local-large", cases)] == ["chart", "gear"]
