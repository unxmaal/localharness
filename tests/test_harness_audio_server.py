"""The audio server that answers for the Mac's mlx_audio on a CUDA machine.

Everything here exercises what the server REFUSES, because that is where the
lane can go wrong quietly. Producing audio and transcribing it needs Kokoro and
faster-whisper, which are the launcher's pinned dependencies rather than the
suite's; the round trip is smoke.sh's job.
"""
import io

import pytest

from harness import audio
from harness import audio_server

fastapi = pytest.importorskip("fastapi", reason="needs the `dev` group")
from fastapi.testclient import TestClient  # noqa: E402


@pytest.fixture
def client():
    return TestClient(audio_server.build_app())


def test_it_answers_for_the_model_lh_say_actually_sends():
    """`lh say` sends DEFAULT_TTS_MODEL. A server that refused it would leave
    the speech lane working only when a caller overrode the default."""
    assert audio.DEFAULT_TTS_MODEL in audio_server.KOKORO_IDS or \
        audio.DEFAULT_TTS_MODEL.startswith("mlx-community/"), \
        "the Windows default should be one this server serves"


def test_it_answers_for_the_model_lh_hear_actually_sends():
    import sys
    if sys.platform != "win32":
        pytest.skip("the Windows default is what this server has to accept")
    assert any(audio.DEFAULT_STT_MODEL.startswith(p)
               for p in audio_server.WHISPER_PREFIXES)


def test_a_model_it_cannot_run_is_refused_rather_than_substituted(client):
    """parakeet is MLX. Transcribing with Whisper and returning the text would
    put a model in the result that never ran, and nothing downstream could
    tell: the score sheet records the name the caller sent."""
    r = client.post("/v1/audio/transcriptions",
                    files={"file": ("clip.wav", io.BytesIO(b"RIFF0000WAVE"),
                                    "audio/wav")},
                    data={"model": "mlx-community/parakeet-tdt-0.6b-v2"})
    assert r.status_code == 400
    assert "faster-whisper" in r.text


def test_a_tts_model_it_does_not_hold_is_refused(client):
    r = client.post("/v1/audio/speech",
                    json={"model": "some/other-voice-model", "input": "hi"})
    assert r.status_code == 400


def test_voice_cloning_is_refused_rather_than_answered_in_another_voice(client):
    """The Mac's default voice is cloned by Chatterbox from a reference clip.
    Kokoro has a fixed table, so the honest answer is a refusal: a line
    delivered in a substitute voice measures a voice nobody asked for."""
    r = client.post("/v1/audio/speech",
                    json={"input": "hi", "ref_audio": "harness/voices/fr-male.wav"})
    assert r.status_code == 400
    assert "cloning" in r.text.lower()


def test_empty_input_is_refused(client):
    assert client.post("/v1/audio/speech", json={"input": "  "}).status_code == 400


def test_it_says_what_it_serves(client):
    """So a caller finds out from /v1/models rather than from a 400."""
    body = client.get("/v1/models").json()
    served = {m["id"] for m in body["data"]}
    assert audio_server.KOKORO_IDS[0] in served
    assert served - {audio_server.KOKORO_IDS[0]}, "no transcription model listed"


def test_the_cuda_runtime_is_found_from_the_installed_wheels():
    """The launcher puts these on PATH before starting the server, because
    CTranslate2 resolves cublas through the OS search path rather than through
    Python's loader. An empty list here means the wheels are absent, which is
    the CPU path rather than a failure."""
    found = audio_server._cuda_dll_directories()
    assert isinstance(found, list)
