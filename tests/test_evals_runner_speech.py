"""The speech runner: half the goal, and until now zero eval coverage.

A tts case is a sentence. The runner speaks it, the checker reads it back, and
the row carries the word error rate. Both sides run on this machine, so the
whole thing is offline apart from the two local servers.
"""
import httpx
import pytest
import respx

from evals.core import Case
from evals.runners.speech import SpeechRunner

BASE = "http://127.0.0.1:8890/v1"


def wav(samples=16000):
    import struct
    data = b"\x00\x01" * samples
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


def case(**over):
    base = dict(id="pangram", modality="tts",
                prompt="the quick brown fox jumps over the lazy dog",
                assertions={"max_wer": 0.2})
    base.update(over)
    return Case(**base)


def runner(tmp_path, **over):
    kwargs = dict(model="mlx-community/Kokoro-82M-bf16", voice="am_adam",
                  outdir=tmp_path, base_url=BASE)
    kwargs.update(over)
    return SpeechRunner(**kwargs)


@respx.mock
def test_a_clean_round_trip_passes(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    respx.post(f"{BASE}/audio/transcriptions").mock(return_value=httpx.Response(
        200, json={"text": "The quick brown fox jumps over the lazy dog."}))
    r = runner(tmp_path).run(case())
    assert r.passed, r.detail


@respx.mock
def test_the_candidate_name_carries_the_voice(tmp_path):
    """Kokoro at am_adam and at ff_siwis are different candidates; sharing a
    row is the same mistake as sharing one between two quantizations."""
    assert "am_adam" in runner(tmp_path).candidate
    assert "Kokoro" in runner(tmp_path).candidate


@respx.mock
def test_a_garbled_round_trip_fails_with_what_was_heard(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "a slow green ox"}))
    r = runner(tmp_path).run(case())
    assert not r.passed
    assert "slow green ox" in r.detail


@respx.mock
def test_the_audio_is_kept_so_you_can_listen_to_a_failure(tmp_path):
    from pathlib import Path
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "nonsense"}))
    r = runner(tmp_path).run(case())
    assert r.artifact and Path(r.artifact).exists()


@respx.mock
def test_a_tts_outage_is_a_failed_row_not_an_exception(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(side_effect=httpx.ConnectError("x"))
    r = runner(tmp_path).run(case())
    assert not r.passed
    assert "unreachable" in r.detail.lower()


@respx.mock
def test_voice_and_speed_come_from_the_case_params(tmp_path):
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "x"}))
    runner(tmp_path).run(case(params={"speed": 1.4}))
    assert json.loads(route.calls[0].request.read())["speed"] == 1.4


@respx.mock
def test_generation_time_is_measured_not_transcription_time(tmp_path):
    """The row's seconds column has to mean the candidate's latency. Scoring
    happens after generate() returns, so the checker's own request must not be
    inside the timer."""
    import time
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))

    def slow_transcribe(request):
        time.sleep(0.4)
        return httpx.Response(200, json={"text": case().prompt})

    respx.post(f"{BASE}/audio/transcriptions").mock(side_effect=slow_transcribe)
    r = runner(tmp_path).run(case())
    assert r.passed
    assert r.seconds < 0.4, (
        f"{r.seconds}s includes the transcription, so the column does not "
        "mean what the header says")


@respx.mock
def test_a_candidate_with_no_voice_is_named_for_the_model_alone(tmp_path):
    """Only Kokoro has a voice table. `Qwen3-TTS-.../` with a trailing slash
    would be a lie about what varies."""
    r = runner(tmp_path, voice="")
    assert r.candidate == "Kokoro-82M-bf16"
    assert not r.candidate.endswith("/")


# ---- speaking a language the resident ear cannot hear ----------------------
# The joint measurement has a hidden requirement: the transcriber has to speak
# the same language as the candidate. Parakeet is English-only, so scoring
# French through it reports a word error rate near 1.0 for every candidate and
# ranks them all as equally broken.

class FakeWhisper:
    def __init__(self, text="Bonjour, la passerelle est en marche."):
        self.text = text
        self.calls = []

    def transcribe(self, path, **kw):
        self.calls.append((path, kw))
        return {"text": self.text}


@pytest.fixture
def fake_whisper(monkeypatch):
    from harness import audio
    fake = FakeWhisper()
    monkeypatch.setattr(audio, "_whisper_module", lambda: fake)
    return fake


def french_case():
    return case(id="fr-greeting",
                prompt="Bonjour, la passerelle est en marche.",
                assertions={"max_wer": 0.2})


@respx.mock
def test_the_ear_can_be_whisper_pinned_to_a_language(tmp_path, fake_whisper):
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    r = runner(tmp_path, ear="whisper:fr").run(french_case())
    assert r.passed, r.detail
    assert r.metrics["wer"] == 0.0
    assert fake_whisper.calls[0][1]["language"] == "fr"


@respx.mock
def test_the_default_ear_is_still_the_server(tmp_path):
    assert runner(tmp_path).ear == "server"


def test_an_unknown_ear_is_refused_when_the_runner_is_built(tmp_path):
    with pytest.raises(ValueError):
        runner(tmp_path, ear="wisper:fr")


@respx.mock
def test_a_reference_clip_is_sent_for_cloning(tmp_path):
    import json
    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav())
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": french_case().prompt}))
    runner(tmp_path, voice="", ref_audio=ref, lang_code="fr").run(french_case())
    sent = json.loads(route.calls[0].request.read())
    assert sent["ref_audio"] == str(ref)
    assert sent["lang_code"] == "fr"


@respx.mock
def test_the_reference_clip_is_part_of_the_candidate_name(tmp_path):
    """The same weights cloning two different speakers are two products, the
    same way Kokoro at two voices is."""
    ref = tmp_path / "fleurs-fr-male-1.wav"
    ref.write_bytes(wav())
    name = runner(tmp_path, voice="", model="litmudoc/Chatterbox-x",
                  ref_audio=ref).candidate
    assert "fleurs-fr-male-1" in name
    assert "Chatterbox-x" in name


@respx.mock
def test_a_missing_reference_clip_is_a_failed_row_not_an_exception(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav()))
    r = runner(tmp_path, voice="",
               ref_audio=tmp_path / "gone.wav").run(french_case())
    assert not r.passed
    assert "gone.wav" in r.detail


def test_a_cloning_candidate_asks_for_speaker_similarity(tmp_path):
    """WER scores an intelligible clone in the wrong voice at a perfect 0.000.
    A candidate given a reference clip must also be asked whether it sounds
    like it."""
    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav())
    assert "ref_audio" in runner(tmp_path, voice="", ref_audio=ref).score_kwargs()


def test_a_non_cloning_candidate_does_not(tmp_path):
    """Kokoro has no reference to be similar to."""
    assert "ref_audio" not in runner(tmp_path).score_kwargs()
