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


# ---- backends -------------------------------------------------------------
# The server backend cannot run whisper at all (mlx_audio wants a HuggingFace
# processor the mlx repos do not ship), so a multilingual candidate has to be
# read in-process. The runner picks the ear; the case does not care.

class FakeWhisper:
    def __init__(self, text="Le renard brun rapide saute par-dessus le chien."):
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


def french_case(tmp_path):
    clip = tmp_path / "fr.wav"
    clip.write_bytes(b"x" * 9000)
    return Case(id="fr1", modality="stt",
                prompt="Le renard brun rapide saute par-dessus le chien.",
                audio=clip, assertions={"max_wer": 0.2})


def test_the_whisper_backend_scores_french(tmp_path, fake_whisper):
    r = runner(model="mlx-community/whisper-large-v3-mlx",
               backend="whisper", language="fr").run(french_case(tmp_path))
    assert r.passed, r.detail
    assert r.metrics["wer"] == 0.0
    assert fake_whisper.calls[0][1]["language"] == "fr"


def test_the_whisper_backend_never_touches_the_server(tmp_path, fake_whisper):
    """respx is not mocking anything here: a real HTTP call would raise."""
    with respx.mock:
        r = runner(backend="whisper").run(french_case(tmp_path))
    assert r.passed, r.detail


def test_the_language_is_part_of_the_candidate_name(tmp_path):
    """The same weights decoding French and English are two measurements, and
    sharing a row would average them."""
    name = runner(model="mlx-community/whisper-large-v3-mlx",
                  backend="whisper", language="fr").candidate
    assert "fr" in name
    assert "whisper-large-v3-mlx" in name


def test_the_default_backend_is_still_the_server(tmp_path):
    assert runner().backend == "server"


def test_an_unknown_backend_is_refused_when_the_runner_is_built():
    """Before a corpus of forty clips runs, not on the first one."""
    with pytest.raises(ValueError):
        runner(backend="wisper")


def test_a_whisper_failure_is_a_failed_row_not_an_exception(tmp_path,
                                                            monkeypatch):
    from harness import audio

    class Boom:
        def transcribe(self, *a, **kw):
            raise RuntimeError("weights not found")
    monkeypatch.setattr(audio, "_whisper_module", lambda: Boom())
    r = runner(backend="whisper").run(french_case(tmp_path))
    assert not r.passed
    assert "weights not found" in r.detail
