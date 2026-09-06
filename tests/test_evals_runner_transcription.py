"""The transcription runner: STT measured on its own, not jointly with TTS."""
import httpx
import pytest
import respx

from evals.core import Case
from evals.runners.transcription import TranscriptionRunner

BASE = "http://127.0.0.1:8890/v1"


def case(tmp_path, **over):
    clip = tmp_path / "a.wav"
    clip.write_bytes(b"x" * 9000)
    base = dict(id="u1", modality="stt",
                prompt="the quick brown fox jumps over the lazy dog",
                audio=clip, assertions={"max_wer": 0.2})
    base.update(over)
    return Case(**base)


def runner(**over):
    kwargs = dict(model="mlx-community/parakeet-tdt-0.6b-v2", base_url=BASE)
    kwargs.update(over)
    return TranscriptionRunner(**kwargs)


@respx.mock
def test_a_clean_transcription_passes(tmp_path):
    respx.post(f"{BASE}/audio/transcriptions").mock(return_value=httpx.Response(
        200, json={"text": "The quick brown fox jumps over the lazy dog."}))
    r = runner().run(case(tmp_path))
    assert r.passed, r.detail
    assert r.metrics["wer"] == 0.0


@respx.mock
def test_a_bad_transcription_fails_with_what_was_heard(tmp_path):
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "a slow green ox"}))
    r = runner().run(case(tmp_path))
    assert not r.passed
    assert "slow green ox" in r.detail


def test_the_candidate_is_named_for_the_model(tmp_path):
    assert "parakeet" in runner().candidate.lower()


@respx.mock
def test_a_server_outage_is_a_failed_row_not_an_exception(tmp_path):
    respx.post(f"{BASE}/audio/transcriptions").mock(
        side_effect=httpx.ConnectError("x"))
    r = runner().run(case(tmp_path))
    assert not r.passed
    assert "unreachable" in r.detail.lower()


@respx.mock
def test_the_model_is_sent_so_two_candidates_are_actually_different(tmp_path):
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "x"}))
    runner(model="mlx-community/canary-qwen-2.5b").run(case(tmp_path))
    body = route.calls[0].request.read().decode("utf-8", "replace")
    assert "canary-qwen" in body


@respx.mock
def test_timing_measures_transcription_not_scoring(tmp_path):
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": case(tmp_path).prompt}))
    r = runner().run(case(tmp_path))
    assert r.seconds >= 0


def test_a_case_with_no_audio_is_a_failed_row(tmp_path):
    """Only reachable if a Case is built by hand; load_cases rejects it."""
    r = runner().run(Case(id="u", modality="stt", prompt="x"))
    assert not r.passed
    assert "audio" in r.detail.lower()
