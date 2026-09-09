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
    out = Path("/tmp/x.wav")
    cmd = audio.record_argv(out, seconds=5)
    assert cmd[0].endswith("rec")
    # str(out), not the literal: a Path renders with backslashes on
    # Windows, and the assertion is about what was passed, not about
    # which separator this machine spells it with.
    assert "16000" in cmd and str(out) in cmd
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


def test_the_default_voice_resolves():
    """It is a preset now, not a Kokoro table entry, so the check that it is
    real is that it resolves rather than that it is in KNOWN_VOICES."""
    v = audio.resolve_voice(audio.DEFAULT_VOICE)
    assert v.model and Path(v.ref_audio or __file__).exists()


def test_the_default_voice_is_a_cloned_french_accent():
    """A stated preference: Eric auditioned three and kept this one."""
    assert audio.DEFAULT_VOICE in audio.VOICE_PRESETS
    assert audio.resolve_voice(audio.DEFAULT_VOICE).lang_code == "en"


def test_the_raw_kokoro_default_is_still_cached_and_male():
    """speak() talks to the server directly and its default has to be a name
    the server's voice table holds. af_heart is Kokoro's own default and is NOT
    in this machine's cache, so every `lh say` failed with an opaque mid-stream
    close until this was pinned."""
    assert audio.DEFAULT_KOKORO_VOICE in audio.KNOWN_VOICES
    assert audio.DEFAULT_KOKORO_VOICE.startswith(("am_", "bm_"))


def test_known_voices_are_offered_so_a_typo_is_recoverable():
    assert "am_adam" in audio.KNOWN_VOICES and "ff_siwis" in audio.KNOWN_VOICES


@respx.mock
def test_an_empty_voice_is_omitted_rather_than_sent_blank(tmp_path):
    """Kokoro has named voices; Qwen3-TTS and Chatterbox do not. Sending
    voice="" to a model with no voice table is a request for a voice called
    empty string."""
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE, voice="")
    assert "voice" not in json.loads(route.calls[0].request.read())


# ---- the local whisper backend --------------------------------------------
# Parakeet is English-only: it hears cloned French as "Monjour passed in March",
# which measures nothing. Both multilingual candidates fail THROUGH mlx_audio's
# server -- whisper repos ship no preprocessor_config.json, and Voxtral rejects
# the word_timestamps argument the server passes unconditionally -- so this
# backend bypasses the server and calls mlx_whisper in-process.

class FakeWhisper:
    """Stands in for the mlx_whisper module. Records what it was asked."""

    def __init__(self, text=" bonjour le monde "):
        self.text = text
        self.calls = []

    def transcribe(self, path, **kw):
        self.calls.append((path, kw))
        return {"text": self.text, "language": kw.get("language") or "en"}


@pytest.fixture
def fake_whisper(monkeypatch):
    fake = FakeWhisper()
    monkeypatch.setattr(audio, "_whisper_module", lambda: fake)
    return fake


def test_whisper_transcribe_returns_the_stripped_text(tmp_path, fake_whisper):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    assert audio.transcribe_whisper(clip) == "bonjour le monde"


def test_whisper_passes_the_repo_as_a_path_or_hf_repo(tmp_path, fake_whisper):
    """mlx_whisper takes the repo under its own keyword, not `model`. Getting
    this wrong silently transcribes with tiny, whose French is unusable."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    audio.transcribe_whisper(clip, model="mlx-community/whisper-large-v3-mlx")
    _, kw = fake_whisper.calls[0]
    assert kw["path_or_hf_repo"] == "mlx-community/whisper-large-v3-mlx"


def test_whisper_pins_the_language_when_one_is_given(tmp_path, fake_whisper):
    """Auto-detection on a short clip picks the wrong language often enough to
    poison a corpus rate; a French eval knows it is French."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    audio.transcribe_whisper(clip, language="fr")
    assert fake_whisper.calls[0][1]["language"] == "fr"


def test_whisper_omits_the_language_when_none_is_given(tmp_path, fake_whisper):
    """Passing language="" is not the same as not passing one: mlx_whisper
    would take the empty string as a language code."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    audio.transcribe_whisper(clip, language="")
    assert "language" not in fake_whisper.calls[0][1]


def test_whisper_on_a_missing_file_fails_before_loading_the_model(tmp_path):
    """A 3GB load is a slow way to find out the path was wrong."""
    with pytest.raises(audio.AudioError) as e:
        audio.transcribe_whisper(tmp_path / "nope.wav")
    assert "nope.wav" in str(e.value)


def test_whisper_names_the_package_when_it_is_not_installed(tmp_path,
                                                            monkeypatch):
    def boom():
        raise ImportError("No module named 'mlx_whisper'")
    monkeypatch.setattr(audio, "_whisper_module", boom)
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    with pytest.raises(audio.AudioError) as e:
        audio.transcribe_whisper(clip)
    assert "mlx-whisper" in str(e.value)


def test_a_whisper_failure_is_an_audio_error_not_a_raw_traceback(tmp_path,
                                                                 monkeypatch):
    """Every other transcription failure arrives as AudioError, and the checker
    catches exactly that to record 'broken instrument' rather than 100% error."""
    class Boom:
        def transcribe(self, *a, **kw):
            raise RuntimeError("weights not found")
    monkeypatch.setattr(audio, "_whisper_module", lambda: Boom())
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    with pytest.raises(audio.AudioError) as e:
        audio.transcribe_whisper(clip)
    assert "weights not found" in str(e.value)


def test_the_default_whisper_model_is_multilingual():
    """The whole reason this backend exists. tiny.en or any .en repo cannot
    score French however fast it is."""
    assert not audio.DEFAULT_WHISPER_MODEL.endswith(".en")
    assert "/" in audio.DEFAULT_WHISPER_MODEL


# ---- picking a backend ----------------------------------------------------

def test_transcriber_defaults_to_the_server_backend(tmp_path):
    fn = audio.transcriber()
    assert fn.backend == "server"


def test_transcriber_whisper_backend_calls_whisper(tmp_path, fake_whisper):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    fn = audio.transcriber(backend="whisper", language="fr")
    assert fn(clip) == "bonjour le monde"
    assert fake_whisper.calls[0][1]["language"] == "fr"


@respx.mock
def test_transcriber_server_backend_calls_the_server(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    respx.post(f"{BASE}/audio/transcriptions").mock(
        return_value=httpx.Response(200, json={"text": "hello"}))
    fn = audio.transcriber(backend="server", base_url=BASE)
    assert fn(clip) == "hello"


def test_an_unknown_backend_is_rejected_by_name():
    with pytest.raises(ValueError) as e:
        audio.transcriber(backend="wisper")
    assert "wisper" in str(e.value)
    assert "whisper" in str(e.value)


def test_the_transcriber_carries_a_label_for_the_results_table(tmp_path):
    """Two rows scored by different ears are not comparable, so the report has
    to be able to say which one read them back."""
    fn = audio.transcriber(backend="whisper",
                           model="mlx-community/whisper-large-v3-mlx",
                           language="fr")
    assert "whisper-large-v3-mlx" in fn.label
    assert "fr" in fn.label


# ---- voice cloning and non-English speech ---------------------------------
# Chatterbox clones a voice from a reference clip. Three things have to be
# right and each fails differently; the one this code can prevent is the field
# name -- mlx_audio calls it lang_code, defaults it to "a" (Kokoro American
# English), and silently ignores a field called `language` before failing as
# "Unsupported language code 'a'", naming a code the caller never sent.

@respx.mock
def test_speak_sends_a_reference_clip_for_cloning(tmp_path):
    import json
    ref = tmp_path / "ref.wav"
    ref.write_bytes(wav_bytes())
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak("bonjour", out=tmp_path / "a.wav", base_url=BASE, voice="",
                ref_audio=ref)
    sent = json.loads(route.calls[0].request.read())
    assert sent["ref_audio"] == str(ref)


@respx.mock
def test_speak_sends_lang_code_never_language(tmp_path):
    """The field is lang_code. `language` is accepted by the HTTP layer and
    then ignored, so the request fails claiming an unsupported code 'a'."""
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak("bonjour", out=tmp_path / "a.wav", base_url=BASE, voice="",
                lang_code="fr")
    sent = json.loads(route.calls[0].request.read())
    assert sent["lang_code"] == "fr"
    assert "language" not in sent


@respx.mock
def test_no_lang_code_is_omitted_rather_than_sent_blank(tmp_path):
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE)
    assert "lang_code" not in json.loads(route.calls[0].request.read())


def test_a_missing_reference_clip_fails_before_the_request(tmp_path):
    """Otherwise the server reports it as a generation failure, which sends you
    to the wrong log."""
    with pytest.raises(audio.AudioError) as e:
        audio.speak("hi", out=tmp_path / "a.wav", base_url=BASE,
                    ref_audio=tmp_path / "nope.wav")
    assert "nope.wav" in str(e.value)


# ---- cloned voice presets --------------------------------------------------
# A cloned voice needs a model, a reference clip and a language code that all
# have to agree. Asking a user to remember three coupled settings to hear a
# French accent is how a working feature goes unused.

def test_a_preset_names_a_cloned_voice():
    v = audio.resolve_voice("fr-male")
    assert v.model != audio.DEFAULT_TTS_MODEL
    assert v.ref_audio and Path(v.ref_audio).exists()
    assert v.voice == ""


def test_the_preset_speaks_english_in_a_french_voice():
    """The reference clip is French and the output is English: Chatterbox
    clones across languages and the accent comes with the voice. Setting
    lang_code to fr here would produce French, which is a different product."""
    assert audio.resolve_voice("fr-male").lang_code == "en"


def test_a_kokoro_voice_name_still_resolves_to_kokoro():
    v = audio.resolve_voice("bm_george")
    assert v.model == audio.DEFAULT_TTS_MODEL
    assert v.voice == "bm_george"
    assert v.ref_audio is None
    assert v.lang_code == ""


def test_the_reference_clips_ship_with_the_package():
    """Otherwise the preset works only on the machine with the volume mounted."""
    for name in audio.VOICE_PRESETS:
        assert Path(audio.resolve_voice(name).ref_audio).exists()


def test_an_unknown_voice_is_rejected_and_lists_what_exists():
    with pytest.raises(ValueError) as e:
        audio.resolve_voice("fr_male")
    msg = str(e.value)
    assert "fr_male" in msg
    assert "fr-male" in msg and "bm_george" in msg


@respx.mock
def test_speaking_through_a_preset_sends_all_three_settings(tmp_path):
    import json
    route = respx.post(f"{BASE}/audio/speech").mock(
        return_value=httpx.Response(200, content=wav_bytes()))
    audio.speak_as("fr-male", "the tests all passed", out=tmp_path / "a.wav",
                   base_url=BASE)
    sent = json.loads(route.calls[0].request.read())
    assert sent["lang_code"] == "en"
    assert sent["ref_audio"].endswith("fr-male.wav")
    assert "voice" not in sent
    assert "Chatterbox" in sent["model"]


# ---- WhisperKit: a third backend, and the first non-MLX one ----------------
# The stt lane had measured three models and never a RUNTIME: parakeet 0.6b,
# parakeet 1.1b and whisper all run through MLX. WhisperKit is CoreML, shipped
# as `whisperkit-cli` (brew), and is what EnviousWispr uses in production.

def test_the_whisperkit_command_names_the_audio_and_the_model(tmp_path):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    argv = audio.whisperkit_argv(clip, model="large-v3", language="fr")
    assert argv[0].endswith("whisperkit-cli")
    assert argv[1] == "transcribe"
    assert str(clip) in argv
    assert "large-v3" in argv
    assert "fr" in argv


def test_no_language_means_no_language_flag(tmp_path):
    """Passing an empty string would pin the decode to a language called ''."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    assert "--language" not in audio.whisperkit_argv(clip, language="")


def test_whisperkit_is_a_named_backend():
    assert "whisperkit" in audio.STT_BACKENDS


def test_the_transcriber_factory_builds_a_whisperkit_reader(tmp_path):
    fn = audio.transcriber(backend="whisperkit", model="large-v3", language="fr")
    assert fn.backend == "whisperkit"
    assert "large-v3" in fn.label and "fr" in fn.label


def test_whisperkit_output_is_stripped_of_the_cli_furniture(tmp_path, monkeypatch):
    """The CLI prints timing and a banner around the transcript. Handing that
    to a word error rate would score the tool's logging as speech."""
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    noisy = ("Loading models...\n"
             "[00:00.000 --> 00:03.120] Bonjour, la passerelle est en marche.\n"
             "Transcription time: 1.2s\n")
    monkeypatch.setattr(audio, "_run_whisperkit", lambda argv, timeout: noisy)
    assert audio.transcribe_whisperkit(clip) == "Bonjour, la passerelle est en marche."


def test_a_missing_whisperkit_binary_says_how_to_get_it(tmp_path, monkeypatch):
    clip = tmp_path / "c.wav"
    clip.write_bytes(wav_bytes())
    def boom(argv, timeout):
        raise FileNotFoundError("whisperkit-cli")
    monkeypatch.setattr(audio, "_run_whisperkit", boom)
    with pytest.raises(audio.AudioError) as e:
        audio.transcribe_whisperkit(clip)
    assert "brew install whisperkit-cli" in str(e.value)


# ---- issue #9: FluidAudio, a Swift/CoreML Parakeet -------------------------

def test_fluidaudio_is_a_known_backend():
    assert "fluidaudio" in audio.STT_BACKENDS


def test_the_version_is_pinned_to_the_one_mlx_runs():
    """FluidAudio carries its own CoreML conversions of v2, v3 and 110m, and
    defaults to v3. v3 is already measured as WORSE than v2, so comparing the
    default would confound the runtime with a known model regression."""
    assert audio.DEFAULT_FLUIDAUDIO_MODEL == "v2"
    assert "--model-version" in audio.fluidaudio_argv("/tmp/a.flac")
    assert "v2" in audio.fluidaudio_argv("/tmp/a.flac")


def test_an_empty_language_is_not_passed(tmp_path):
    """Empty is not "no language": it would pin the decode to a language named
    the empty string, the same trap lang_code has."""
    assert "--language" not in audio.fluidaudio_argv("/tmp/a.flac")
    assert "--language" in audio.fluidaudio_argv("/tmp/a.flac", language="en")


def test_the_transcript_is_read_from_json_never_stdout(tmp_path, monkeypatch):
    """fluidaudiocli writes CoreML runtime errors to STDOUT, unprefixed and on
    the SAME LINE as the transcript. Reading stdout measured a corpus WER of
    1.016 with a worst case of 10.5, which reads as a broken model rather than
    a broken reader."""
    import json as _json

    def fake_run(argv, timeout):
        out = Path(argv[argv.index("--output-json") + 1])
        out.write_text(_json.dumps({"text": "the real transcript"}), encoding="utf-8")
        return ("E5RT encountered an STL exception... zero shape error."
                "the real transcript")

    monkeypatch.setattr(audio, "_run_whisperkit", fake_run)
    clip = tmp_path / "a.flac"
    clip.write_bytes(b"x")
    assert audio.transcribe_fluidaudio(clip) == "the real transcript"


def test_json_that_never_appeared_is_an_error_not_an_empty_transcript(tmp_path,
                                                                      monkeypatch):
    """An empty transcript scores as a total miss and would be recorded as a
    measurement of the model."""
    monkeypatch.setattr(audio, "_run_whisperkit", lambda argv, timeout: "")
    clip = tmp_path / "a.flac"
    clip.write_bytes(b"x")
    with pytest.raises(audio.AudioError):
        audio.transcribe_fluidaudio(clip)


def test_the_binary_is_found_by_env_then_path_then_our_own_bin(monkeypatch):
    monkeypatch.setenv(audio.FLUIDAUDIO_CLI_ENV, "/somewhere/fluidaudiocli")
    assert audio.fluidaudio_cli() == "/somewhere/fluidaudiocli"
    monkeypatch.delenv(audio.FLUIDAUDIO_CLI_ENV)
    assert audio.fluidaudio_cli().endswith("fluidaudiocli")


def test_the_ear_says_which_runtime_produced_the_number():
    """A WER from Parakeet-on-MLX and one from Parakeet-on-CoreML are not the
    same number even at the same model version."""
    assert audio.transcriber(backend="fluidaudio", model="v2").label == \
        "fluidaudio:v2"
