"""The music lane: sung-lyric WER, its controls, and the acestep engine.

Issue #237. The lane exists because the round-trip was measured to SEPARATE
(#236): a sung vocal scored 0.091 against a 0.000 spoken floor and a 1.000
instrumental ceiling. These tests pin that shape so it cannot quietly stop
being true, and they pin the negative half as hard as the positive one -- a
checker that passes everything is the failure mode, not a checker that is
occasionally strict.
"""
import wave
from pathlib import Path

import pytest

from evals.core import Case, CHECKERS
from harness import engines
from harness.audio import AudioError
from harness.checks import music

#: The lyric line the spike used, and what parakeet actually heard back off the
#: generated vocal. Three errors in 33 words. Kept verbatim rather than
#: invented so the fixture is a real measurement and not a guess at one.
LYRICS = """[Verse]
The morning light is on the open road
I hold the wheel and let the river flow
We drive all day and watch the city glow
The radio is playing soft and low
"""
HEARD_SUNG = ("The morning light is on the open road I hold the wheel and let "
              "the river flow We drive all day and watch the city go The "
              "radius playing soft and low")


def wav(path: Path, seconds: float = 30.0) -> Path:
    """A real wav of a given length, big enough to clear MIN_AUDIO_BYTES."""
    with wave.open(str(path), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(b"\0\0" * int(8000 * seconds))
    return path


def ear(text):
    """A transcriber that hears exactly what it is told to hear."""
    return lambda path: text


# --- the metric, and the controls that make it mean something -------------

def test_a_sung_vocal_scores_a_low_rate(tmp_path):
    """The positive case: the words are there and the ear finds them."""
    r = music.check(wav(tmp_path / "a.wav"), LYRICS, transcriber=ear(HEARD_SUNG))
    assert r.ok
    assert r.wer == pytest.approx(3 / 33, abs=0.01)
    assert r.metrics["wer_words"] == 33


def test_an_instrumental_scores_the_ceiling_against_the_same_lyrics(tmp_path):
    """THE NEGATIVE CONTROL, and the reason the number above means anything.

    Scored as a vocal case, silence is a total miss. A sung 0.091 beside a
    ceiling of 1.000 is a measurement; a sung 0.091 beside nothing is a number.
    """
    r = music.check(wav(tmp_path / "a.wav"), LYRICS, transcriber=ear(""))
    assert r.wer == 1.0
    assert r.errors == r.words


def test_the_gap_between_the_classes_beats_the_spread_within_them(tmp_path):
    """Gauntlet class 1, stated as an assertion rather than as a hope.

    Two known-SAME (a clean read and a sung read of the same words) and one
    known-DIFFERENT (no words at all). The metric is only usable if the
    between-class gap dwarfs the within-class spread.
    """
    clip = wav(tmp_path / "a.wav")
    spoken = music.check(clip, LYRICS,
                         transcriber=ear(music.lyric_words(LYRICS))).wer
    sung = music.check(clip, LYRICS, transcriber=ear(HEARD_SUNG)).wer
    silent = music.check(clip, LYRICS, transcriber=ear("")).wer

    within = abs(sung - spoken)
    between = silent - sung
    assert between > 5 * within, (
        f"spoken {spoken} sung {sung} silent {silent}: the ear cannot tell "
        f"singing from silence well enough to rank anything")


def test_a_bad_vocal_fails_its_limit(tmp_path):
    r = music.check(wav(tmp_path / "a.wav"), LYRICS, max_wer=0.35,
                    transcriber=ear("something else entirely"))
    assert not r.ok
    assert "word error rate" in r.reason


# --- asking for no vocal is a requirement, not only a control -------------

def test_an_instrumental_that_stays_silent_passes(tmp_path):
    r = music.check(wav(tmp_path / "a.wav"), "[Instrumental]",
                    expect_vocals=False, transcriber=ear(""))
    assert r.ok


def test_an_instrumental_that_sings_anyway_fails(tmp_path):
    """The capability this case exists to check. No duration or tempo check
    can see a model ignoring the one instruction it was given."""
    r = music.check(wav(tmp_path / "a.wav"), "[Instrumental]",
                    expect_vocals=False,
                    transcriber=ear("we drive all day and watch the city"))
    assert not r.ok
    assert "asked for no vocal" in r.reason


def test_a_stray_syllable_does_not_fail_an_instrumental(tmp_path):
    """The negative half of the check above. A transcriber will occasionally
    find a word in a guitar, and failing a model for that measures the ear."""
    r = music.check(wav(tmp_path / "a.wav"), "[Instrumental]",
                    expect_vocals=False, transcriber=ear("oh"))
    assert r.ok


# --- structure tags are directions, not words anybody sings ---------------

def test_structure_tags_are_not_scored_as_missed_words():
    """Charging a candidate for obeying the lyric format would measure the
    prompt language rather than the singing."""
    assert "[Verse]" not in music.lyric_words(LYRICS)
    assert music.lyric_words("[Chorus]\nhold the line\n") == "hold the line"


def test_a_line_that_merely_contains_a_bracket_survives():
    """The negative half: only a line that is ENTIRELY a tag is a direction."""
    assert music.lyric_words("she said [nothing] at all") == (
        "she said [nothing] at all")


# --- duration, the axis with no prior to be dragged by --------------------

def test_duration_adherence_passes_when_it_is_honoured(tmp_path):
    r = music.check(wav(tmp_path / "a.wav", 30.0), LYRICS, duration_s=30,
                    transcriber=ear(HEARD_SUNG))
    assert r.ok
    assert r.metrics["seconds"] == pytest.approx(30.0, abs=0.1)
    assert r.metrics["duration_error_s"] == pytest.approx(0.0, abs=0.1)


def test_a_track_that_ignores_the_requested_length_fails(tmp_path):
    r = music.check(wav(tmp_path / "a.wav", 12.0), LYRICS, duration_s=30,
                    transcriber=ear(HEARD_SUNG))
    assert not r.ok
    assert "asked for 30s" in r.reason


# --- a broken instrument is not a bad candidate ---------------------------

def test_a_transcriber_outage_is_not_recorded_as_a_perfect_failure(tmp_path):
    """Scoring an STT outage as 100% error records a broken instrument as a
    bad model, which is the defect class this project has filed six times."""
    def dead(path):
        raise AudioError("the audio server is not running")

    r = music.check(wav(tmp_path / "a.wav"), LYRICS, transcriber=dead)
    assert not r.ok
    assert "could not transcribe" in r.reason
    assert r.wer is None
    assert "wer" not in r.metrics, "an unscored run must not report a rate"


def test_a_file_too_small_to_hold_music_is_refused(tmp_path):
    path = tmp_path / "a.wav"
    path.write_bytes(b"RIFF")
    r = music.check(path, LYRICS, transcriber=ear(HEARD_SUNG))
    assert not r.ok
    assert "too small" in r.reason


def test_metrics_omit_the_rate_rather_than_reporting_zero():
    """A zero error rate is the BEST possible score. Reporting it for a run
    that scored nothing puts a candidate that never ran at the top."""
    assert music.MusicResult(True).metrics == {}


# --- the case files the lane ships ----------------------------------------

def test_the_lane_ships_its_own_negative_control():
    """If the instrumental case is ever deleted, every sung number in this
    lane loses the thing that brackets it."""
    cases = Path(__file__).resolve().parents[1] / "evals" / "cases" / "music"
    import yaml
    loaded = [yaml.safe_load(p.read_text(encoding="utf-8"))
              for p in cases.glob("*.yaml")]
    assert any(c["assert"].get("expect_vocals") is False for c in loaded), (
        "no music case asks for an instrumental, so nothing establishes what "
        "'no words present' scores on this ear")
    assert any(c["assert"].get("max_wer") for c in loaded)


def test_every_music_case_routes_through_the_music_checker(tmp_path):
    case = Case(id="x", modality="music", prompt="a song",
                params={"lyrics": LYRICS, "duration": 30},
                assertions={"max_wer": 0.35})
    out = CHECKERS["music"](wav(tmp_path / "a.wav"), case,
                            transcriber=ear(HEARD_SUNG))
    assert out.ok
    assert out.metrics["wer"] == pytest.approx(0.0909, abs=0.001)


# --- the engine -----------------------------------------------------------

@pytest.fixture
def checkout(tmp_path, monkeypatch):
    """An ACE-Step checkout with a virtualenv in it, named by $ACESTEP_ROOT."""
    python = tmp_path / ".venv" / "bin" / "python"
    python.parent.mkdir(parents=True)
    python.write_text("", encoding="utf-8")
    monkeypatch.setenv(engines.ACESTEP_ROOT_ENV, str(tmp_path))
    return tmp_path


def test_the_engine_runs_the_checkouts_own_interpreter(checkout):
    """NOT `uv run`, and the reason is a measurement rather than taste.

    proc.run measures peak on macOS with `/usr/bin/time -l`, which reports the
    phys_footprint of the process it wraps. uv FORKS python rather than
    exec'ing it, so the footprint recorded is uv's: the lane's first receipt
    said 0.0 GiB for a process independently measured at 13.82 GiB. One
    process for the instrument to measure. RULE #225 is the same defect on
    Linux.
    """
    argv = engines.resolve("acestep:acestep-v15-turbo").argv(
        "a song", Path("/tmp/o.wav"), {})
    assert argv[0] == str(checkout / ".venv" / "bin" / "python")
    assert "uv" not in argv


def test_a_checkout_with_no_virtualenv_says_what_to_run(tmp_path, monkeypatch):
    monkeypatch.setenv(engines.ACESTEP_ROOT_ENV, str(tmp_path))
    with pytest.raises(ValueError, match="uv sync"):
        engines.resolve("acestep:acestep-v15-turbo").argv(
            "a song", Path("/tmp/o.wav"), {})


def test_no_checkout_at_all_is_refused_rather_than_defaulted(monkeypatch):
    """Falling back to "." would pick up whichever virtualenv the caller was
    standing in -- this repo's own -- and run a generator against an
    environment with no acestep in it."""
    monkeypatch.delenv(engines.ACESTEP_ROOT_ENV, raising=False)
    with pytest.raises(ValueError, match="ACESTEP_ROOT"):
        engines.resolve("acestep:acestep-v15-turbo").argv(
            "a song", Path("/tmp/o.wav"), {})


def test_the_engine_passes_a_repo_id_through_untouched(checkout):
    """RULE #271: a discovered candidate is ALWAYS a repo id, and a validator
    that enumerates known names refuses everything discovery finds."""
    e = engines.resolve("acestep:ACE-Step/acestep-v15-xl-turbo")
    argv = e.argv("a song", Path("/tmp/o.wav"), {})
    assert "ACE-Step/acestep-v15-xl-turbo" in argv
    assert e.modality == "music"
    assert e.output_suffix == ".wav"


def test_the_engine_needs_a_model():
    with pytest.raises(ValueError, match="acestep needs a model"):
        engines.resolve("acestep")


def test_an_unknown_engine_option_is_refused_before_anything_runs():
    with pytest.raises(ValueError, match="unknown option"):
        engines.resolve("acestep:acestep-v15-turbo,tempo=fast")


def test_case_params_reach_the_command_line(checkout):
    e = engines.resolve("acestep:acestep-v15-turbo")
    argv = e.argv("a song", Path("/tmp/o.wav"),
                  {"lyrics": "hold the line", "bpm": 90, "duration": 30,
                   "keyscale": "C Major", "seed": 42})
    for flag, value in (("--lyrics", "hold the line"), ("--bpm", "90"),
                        ("--duration", "30"), ("--keyscale", "C Major"),
                        ("--seed", "42")):
        assert argv[argv.index(flag) + 1] == value


def test_an_unset_param_is_omitted_rather_than_passed_as_none(checkout):
    """`--bpm None` is accepted by argparse as a string and fails deep inside
    generation, which reads as a model fault."""
    argv = engines.resolve("acestep:acestep-v15-turbo").argv(
        "a song", Path("/tmp/o.wav"), {"bpm": None})
    assert "--bpm" not in argv
    assert "None" not in argv


def test_the_instrumental_flag_is_a_flag_not_a_value(checkout):
    argv = engines.resolve("acestep:acestep-v15-turbo").argv(
        "a song", Path("/tmp/o.wav"), {"instrumental": True})
    assert "--instrumental" in argv
    argv = engines.resolve("acestep:acestep-v15-turbo").argv(
        "a song", Path("/tmp/o.wav"), {"instrumental": False})
    assert "--instrumental" not in argv


def test_the_checkout_is_read_at_call_time(checkout):
    """Read from the environment per call, not captured at import, so a second
    checkout does not need a module reload. Same shape as H3_BIN."""
    argv = engines.resolve("acestep:acestep-v15-turbo").argv(
        "a song", Path("/tmp/o.wav"), {})
    assert argv[argv.index("--root") + 1] == str(checkout)
