"""Speaking and hearing, against the local mlx-audio server.

Every test here is offline: the point is the contract with the server and the
handling of its two real failure modes, not the model.
"""
from pathlib import Path

import httpx
import pytest
import respx

from harness import audio

BASE = "http://127.0.0.1:8890/v1"

# A minimal but structurally real 16-bit mono WAV: 44-byte header plus data.
def wav_bytes(samples: int = 16000) -> bytes:
    import struct
    data = b"\x00\x01" * samples
    return (b"RIFF" + struct.pack("<I", 36 + len(data)) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, 24000, 48000, 2, 16)
            + b"data" + struct.pack("<I", len(data)) + data)


@respx.mock
def test_speak_writes_the_audio_it_was_given(tmp_path):
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    out = audio.speak("bonjour", out=tmp_path / "a.wav", base_url=BASE)
    assert out.exists() and out.stat().st_size > 8000
    body = route.calls[0].request.read().decode()
    assert "bonjour" in body


@respx.mock
def test_speak_sends_voice_speed_and_model(tmp_path):
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE,
                voice="ff_siwis", speed=1.3, model="mlx-community/Kokoro-82M-bf16")
    sent = json.loads(route.calls[0].request.read())
    assert sent["voice"] == "ff_siwis"
    assert sent["speed"] == 1.3
    assert sent["model"] == "mlx-community/Kokoro-82M-bf16"
    assert sent["input"] == "hi"


@respx.mock
def test_an_empty_200_is_a_failure_not_a_silent_success(tmp_path):
    """mlx_audio returns HTTP 200 with an EMPTY BODY when misaki is missing and
    logs the ImportError server-side. Trusting the status code writes a 0-byte
    file and reports success."""
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=b""))
    with pytest.raises(audio.AudioError) as e:
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)
    assert "empty" in str(e.value).lower()
    assert not (tmp_path / "a.wav").exists()


@respx.mock
def test_a_bare_wav_header_is_also_empty(tmp_path):
    """44 bytes is a header with no samples, and it passes `test -s`."""
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes(0)))
    with pytest.raises(audio.AudioError):
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)


@respx.mock
def test_server_down_says_so_and_names_the_script_that_starts_it(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(
        side_effect=httpx.ConnectError("nope"))
    with pytest.raises(audio.AudioError) as e:
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)
    assert "serve-tts.sh" in str(e.value)


@respx.mock
def test_http_error_carries_the_server_message(tmp_path):
    respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(500, text="model not found"))
    with pytest.raises(audio.AudioError) as e:
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)
    assert "model not found" in str(e.value)


# ---- transcription --------------------------------------------------------

@respx.mock
def test_transcribe_returns_the_text(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "  hello there  "}))
    assert audio.transcribe(clip, base_url=BASE) == "hello there"


@respx.mock
def test_transcribe_sends_a_repo_id_never_whisper_1(tmp_path):
    """mlx_audio rejects 'whisper-1'; it wants a HuggingFace repo id."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    route = respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "x"}))
    audio.transcribe(clip, base_url=BASE)
    body = route.calls[0].request.read().decode("utf-8", "replace")
    assert "whisper-1" not in body
    assert "/" in audio.DEFAULT_STT_MODEL


def test_transcribe_on_a_missing_file_fails_before_the_request(tmp_path):
    with pytest.raises(audio.AudioError) as e:
        audio.transcribe(tmp_path / "nope.wav", base_url=BASE)
    assert "nope.wav" in str(e.value)


@respx.mock
def test_transcribe_handles_a_response_with_no_text_key(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"error": "boom"}))
    with pytest.raises(audio.AudioError):
        audio.transcribe(clip, base_url=BASE)


# ---- recording ------------------------------------------------------------

def test_record_builds_a_sox_command_with_an_explicit_format():
    """`rec` guesses rate and channels from the file suffix otherwise, and the
    STT model wants 16k mono."""
    cmd = audio.record_argv(Path("/tmp/x.wav"), seconds=5)
    assert cmd[0].endswith("rec")
    assert "16000" in cmd and "/tmp/x.wav" in cmd
    assert cmd[cmd.index("trim") + 2] == "5"


def test_record_argv_uses_an_absolute_path_for_gui_spawned_shells():
    """wezterm hands children PATH=/usr/bin:/bin:/usr/sbin:/sbin, so a bare
    `rec` is not found when Claude Code is launched from the GUI."""
    assert audio.record_argv(Path("/tmp/x.wav"), seconds=1)[0].startswith("/")


@respx.mock
def test_a_midstream_close_is_not_reported_as_a_server_that_is_down(tmp_path):
    """mlx_audio streams the response. When generation fails part way -- a voice
    that is not in the cache, say -- the connection closes mid-body and httpx
    raises RemoteProtocolError. Telling the user to start a server that is
    already running sends them to the wrong place; the answer is in its log."""
    respx.post(f"{BASE}/audio/speech").mock(
        side_effect=httpx.RemoteProtocolError("incomplete chunked read"))
    with pytest.raises(audio.AudioError) as e:
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)
    msg = str(e.value)
    assert "serve-tts.sh" not in msg
    assert "log" in msg.lower()


def test_the_default_voice_is_one_that_is_actually_cached():
    """af_heart is Kokoro's own default and is NOT in this machine's cache, so
    every `lh say` failed with an opaque mid-stream close."""
    assert audio.DEFAULT_VOICE in audio.KNOWN_VOICES


def test_known_voices_are_offered_so_a_typo_is_recoverable():
    assert "am_adam" in audio.KNOWN_VOICES and "ff_siwis" in audio.KNOWN_VOICES
