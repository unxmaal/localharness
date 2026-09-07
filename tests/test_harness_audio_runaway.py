"""Issue #6: Chatterbox continuing past the end of the input.

The failure is not an error. The server returns 200 with a minute of fluent
invented speech, so nothing downstream notices until a human plays the file.

MEASURED 2026-09-07 through mlx_audio 0.5.1 on port 8890, and the numbers below
are those measurements rather than invented thresholds:

  * on lang_code="fr", three of five cases ran to EXACTLY 48.00 seconds. An
    identical duration across different sentences is a token budget being
    exhausted, not sentences ending. Rates: 1.37, 3.43 and 5.33 s/word.
  * the same three at repetition_penalty=1.2: 9.92s, 5.40s, 4.28s.
  * on lang_code="en" through the accent voice it does NOT reproduce: short,
    medium and long all came back at 0.25-0.36 s/word with and without the
    penalty.

So the guard is against a failure mode that has been observed, on a path that
is currently healthy. That is worth stating plainly rather than implying
English was ever broken.
"""
import struct
import wave

import pytest

from harness import audio
from harness.audio import (RUNAWAY_MIN_WORDS, SECONDS_PER_WORD_CEILING,
                           audio_seconds, runaway_reason, token_budget)


def wav(path, seconds, rate=24000):
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(struct.pack("<h", 0) * int(rate * seconds))
    return path


def words(n):
    return " ".join(["mot"] * n)


def test_the_measured_runaway_is_caught(tmp_path):
    """48.00s for 9 words = 5.33 s/word, the worst case actually observed."""
    p = wav(tmp_path / "runaway.wav", 48.0)
    why = runaway_reason(p, words(9))
    assert why
    assert "48.00s" in why and "9 words" in why


def test_the_penalised_rerun_of_that_same_case_passes(tmp_path):
    """4.28s for 9 words, which is what repetition_penalty=1.2 produced. A
    guard that also rejects the fixed output would be useless."""
    assert runaway_reason(wav(tmp_path / "ok.wav", 4.28), words(9)) == ""


@pytest.mark.parametrize("seconds,n", [(2.66, 8), (3.98, 13), (10.26, 33),
                                       (9.92, 35), (5.40, 14)])
def test_real_measured_clips_are_not_false_positives(tmp_path, seconds, n):
    """Every one of these is a real duration recorded on 2026-09-07. A guard
    that fires on healthy audio gets switched off, and then it guards nothing."""
    assert runaway_reason(wav(tmp_path / "m.wav", seconds), words(n)) == ""


def test_a_short_phrase_is_never_judged(tmp_path):
    """Onset and trailing silence dominate a three-word clip, so its per-word
    rate is noise. `lh say` output is often this short."""
    p = wav(tmp_path / "short.wav", 3.0)
    assert runaway_reason(p, words(RUNAWAY_MIN_WORDS - 1)) == ""
    assert runaway_reason(p, words(2)) == ""


def test_unreadable_audio_is_not_reported_as_a_runaway(tmp_path):
    """A truncated or non-WAV file is a different failure, already caught by
    the byte-count check. Claiming it ran away sends the reader elsewhere."""
    p = tmp_path / "junk.wav"
    p.write_bytes(b"not a wav")
    assert audio_seconds(p) == 0.0
    assert runaway_reason(p, words(20)) == ""


def test_the_budget_scales_with_the_text_and_has_a_floor():
    assert token_budget("one two three") == 200        # floor
    assert token_budget(words(35)) == 420
    assert token_budget(words(100)) > token_budget(words(35))
    # The whole point: below mlx_audio's flat 1200 for ordinary input.
    assert token_budget(words(35)) < 1200


def test_cloning_sends_the_penalty_the_server_would_otherwise_clobber(
        tmp_path, monkeypatch):
    """mlx_audio's SpeechRequest defaults repetition_penalty to 1.0 and forwards
    it, overriding the 1.2 Chatterbox itself defaults to. Sending nothing is
    therefore NOT the same as letting the model decide."""
    sent = {}

    class R:
        content = b"\x00" * (audio.MIN_AUDIO_BYTES + 1)

        def raise_for_status(self):
            pass

    def fake_post(url, json=None, timeout=None):
        sent.update(json)
        return R()

    monkeypatch.setattr(audio.httpx, "post", fake_post)
    monkeypatch.setattr(audio, "runaway_reason", lambda *a, **k: "")
    ref = tmp_path / "ref.wav"
    wav(ref, 1.0)

    audio.speak("one two three four five six", tmp_path / "a.wav", ref_audio=ref)
    assert sent["repetition_penalty"] == 1.2
    assert sent["max_tokens"] == token_budget("one two three four five six")

    sent.clear()
    audio.speak("one two three four five six", tmp_path / "b.wav")
    assert "repetition_penalty" not in sent, "Kokoro takes neither field"
    assert "max_tokens" not in sent


def test_a_runaway_raises_rather_than_returning_the_audio(tmp_path, monkeypatch):
    """Returning the path lets 48 seconds of invented speech flow onward as a
    success. It must be loud, and it must keep the file so it can be heard."""
    made = tmp_path / "out.wav"

    class R:
        content = wav(tmp_path / "src.wav", 48.0).read_bytes()

        def raise_for_status(self):
            pass

    monkeypatch.setattr(audio.httpx, "post",
                        lambda url, json=None, timeout=None: R())
    ref = tmp_path / "ref.wav"
    wav(ref, 1.0)
    with pytest.raises(audio.AudioError) as exc:
        audio.speak(words(9), made, ref_audio=ref)
    assert "#6" in str(exc.value)
    assert made.exists(), "the audio must survive so it can be listened to"


def test_the_ceiling_is_well_clear_of_the_measured_range():
    """Fastest healthy 0.25, slowest healthy 0.56, slowest runaway 5.33."""
    assert 0.56 < SECONDS_PER_WORD_CEILING < 1.37
