"""Behaviour of the STT model-rewriting proxy.

The shim exists for one reason: voicemode hardcodes the STT model as "whisper-1"
and mlx_audio 404s on it. Everything here pins that contract, plus the failure
modes a proxy sitting in a live audio path has to survive.
"""
import httpx
import pytest
import respx
from fastapi.testclient import TestClient

import stt_shim

UPSTREAM = stt_shim.UPSTREAM
TRANSCRIBE = f"{UPSTREAM}/v1/audio/transcriptions"
SPEECH = f"{UPSTREAM}/v1/audio/speech"

WAV = b"RIFF$\x00\x00\x00WAVEfmt " + b"\x00" * 32


@pytest.fixture
def client():
    # As a context manager, so lifespan runs and app.state.client exists.
    # A bare TestClient(app) skips lifespan and every handler fails on
    # AttributeError, which looks like a code bug and is a test bug.
    with TestClient(stt_shim.app) as c:
        yield c


# ---- the one job -----------------------------------------------------------

@respx.mock
def test_rewrites_model_to_parakeet(client):
    """whisper-1 in, the configured Parakeet repo id out."""
    route = respx.post(TRANSCRIBE).mock(
        return_value=httpx.Response(200, json={"text": "hello"}))

    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")},
                    data={"model": "whisper-1"})

    assert r.status_code == 200
    assert r.json()["text"] == "hello"
    sent = route.calls.last.request.content
    assert stt_shim.MODEL.encode() in sent
    assert b"whisper-1" not in sent


@respx.mock
def test_rewrites_even_when_model_absent(client):
    """A client that omits `model` still gets a working transcription."""
    route = respx.post(TRANSCRIBE).mock(
        return_value=httpx.Response(200, json={"text": "hi"}))
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 200
    assert stt_shim.MODEL.encode() in route.calls.last.request.content


@respx.mock
def test_passes_other_fields_through(client):
    """Only `model` is rewritten; language, prompt etc. survive."""
    route = respx.post(TRANSCRIBE).mock(
        return_value=httpx.Response(200, json={"text": "x"}))
    client.post("/v1/audio/transcriptions",
                files={"file": ("a.wav", WAV, "audio/wav")},
                data={"model": "whisper-1", "language": "en",
                      "response_format": "json"})
    sent = route.calls.last.request.content
    assert b"language" in sent and b"en" in sent
    assert b"response_format" in sent


@respx.mock
def test_preserves_upstream_status_and_body(client):
    """A proxy must not invent success. Upstream 422 stays a 422."""
    respx.post(TRANSCRIBE).mock(
        return_value=httpx.Response(422, json={"detail": "bad audio"}))
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 422
    assert r.json()["detail"] == "bad audio"


# ---- failure modes ---------------------------------------------------------

def test_missing_file_is_400(client):
    r = client.post("/v1/audio/transcriptions", data={"model": "whisper-1"})
    assert r.status_code == 400


def test_file_field_that_is_not_a_file_is_400_not_500(client):
    """A string in the `file` field must not raise AttributeError.

    Anything can POST to this port. A 500 with a stack trace is not an
    acceptable answer to a malformed request.
    """
    r = client.post("/v1/audio/transcriptions",
                    data={"file": "not-an-upload", "model": "whisper-1"})
    assert r.status_code == 400


@respx.mock
def test_upstream_down_is_502_not_500(client):
    """mlx_audio being down is an upstream failure, and must read as one."""
    respx.post(TRANSCRIBE).mock(side_effect=httpx.ConnectError("refused"))
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 502
    assert "upstream" in r.text.lower()


@respx.mock
def test_upstream_timeout_is_504(client):
    respx.post(TRANSCRIBE).mock(side_effect=httpx.ReadTimeout("slow"))
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("a.wav", WAV, "audio/wav")})
    assert r.status_code == 504


# ---- the other routes the port has to carry --------------------------------

def test_models_advertises_whisper_one(client):
    """voicemode probes this to build its provider registry."""
    r = client.get("/v1/models")
    assert r.status_code == 200
    assert [m["id"] for m in r.json()["data"]] == ["whisper-1"]


@respx.mock
def test_speech_proxies_untouched(client):
    """TTS passes through byte for byte; the shim must not rewrite it."""
    route = respx.post(SPEECH).mock(
        return_value=httpx.Response(200, content=b"WAVDATA",
                                    headers={"content-type": "audio/wav"}))
    body = {"model": "mlx-community/Kokoro-82M-bf16", "voice": "am_adam",
            "input": "hi"}
    r = client.post("/v1/audio/speech", json=body)
    assert r.status_code == 200
    assert r.content == b"WAVDATA"
    import json
    assert json.loads(route.calls.last.request.content) == body


@respx.mock
def test_speech_upstream_down_is_502(client):
    respx.post(SPEECH).mock(side_effect=httpx.ConnectError("refused"))
    r = client.post("/v1/audio/speech", json={"input": "hi"})
    assert r.status_code == 502


def test_root_reports_upstream(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.json()["upstream"] == UPSTREAM
