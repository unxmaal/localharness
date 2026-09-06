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
