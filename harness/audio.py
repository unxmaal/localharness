"""Speaking and hearing through the local mlx-audio server.

Kokoro for TTS and Parakeet for STT, both on 8890, both OpenAI-compatible.
Measured through this stack: stt 0.1s, tts generation 0.4s, 12-13x realtime.

Two failure modes are handled explicitly because both look like success:

  * mlx_audio answers HTTP 200 with an EMPTY BODY when misaki (Kokoro's
    grapheme-to-phoneme frontend) is missing, logging the ImportError
    server-side. A caller that trusts the status code writes a 0-byte file.
  * A bare 44-byte WAV header passes `test -s` and plays as silence.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

import httpx

DEFAULT_BASE_URL = "http://127.0.0.1:8890/v1"
DEFAULT_TTS_MODEL = "mlx-community/Kokoro-82M-bf16"
# A HuggingFace repo id, never "whisper-1": mlx_audio rejects the OpenAI model
# name outright. This is why port 8890 matters to voicemode's provider probe.
DEFAULT_STT_MODEL = "mlx-community/parakeet-tdt-0.6b-v2"
# Parakeet is English-only and there is no way around that: it hears French as
# English words that rhyme. Whisper is the multilingual ear, and it has to be
# called in-process because mlx_audio's server cannot load it -- the mlx repos
# ship weights.npz and config.json but no preprocessor_config.json, which the
# server demands. mlx_whisper wants exactly what those repos hold.
DEFAULT_WHISPER_MODEL = "mlx-community/whisper-large-v3-mlx"
DEFAULT_WHISPERKIT_MODEL = "large-v3"
#: server   -- Parakeet on mlx-audio :8890, English, fastest.
#: whisper  -- mlx-whisper in this process, multilingual.
#: whisperkit -- CoreML via `whisperkit-cli` (brew). The first NON-MLX runtime
#:   here, and what EnviousWispr ships in production. The lane had measured
#:   three models and never a runtime.
STT_BACKENDS = ("server", "whisper", "whisperkit")
# Kokoro's own default is af_heart, which is NOT in this machine's cache: only
# these five voice packs were pulled, and HF_HUB_OFFLINE=1 stops the server
# fetching a sixth. Asking for an absent voice fails as a mid-stream close with
# nothing useful in it, so the default has to be one that exists here.
KNOWN_VOICES = ("am_adam", "am_onyx", "bm_george", "af_sky", "ff_siwis")
# bm_george: male, and the best of the male voices on the tts eval -- 0.000
# mean word error rate over the five cases against am_adam's 0.031 and
# am_onyx's 0.013. Re-derive with:
#   uv run python -m evals.run --modality tts --out .logs/voices --candidates \
#     'tts:mlx-community/Kokoro-82M-bf16,voice=bm_george,...'
DEFAULT_KOKORO_VOICE = "bm_george"

# Cloned voices. Kokoro has a fixed table and no French-accented English in it;
# Chatterbox clones from a reference clip, and it clones ACROSS LANGUAGES -- the
# reference speaks French, the output speaks English, and the accent comes with
# the voice. That is what makes a French-accented male voice reachable at all
# here, and it is why no accented-English corpus had to be sourced.
#
# The clips ship in harness/voices/ so a preset works without a mounted volume.
# See that directory's README for provenance (google/fleurs, CC-BY-4.0) and for
# why the reference clip is a first-class variable rather than a detail.
CHATTERBOX_MULTILINGUAL = "litmudoc/Chatterbox-Multilingual-MLX-v2-Q8"
VOICES_DIR = Path(__file__).resolve().parent / "voices"
VOICE_PRESETS = {
    "fr-male": {"model": CHATTERBOX_MULTILINGUAL, "clip": "fr-male.wav",
                "lang_code": "en"},
    "fr-male-2": {"model": CHATTERBOX_MULTILINGUAL, "clip": "fr-male-2.wav",
                  "lang_code": "en"},
}


# What `lh say` uses when nobody says otherwise. A cloned preset rather than a
# Kokoro voice: Eric auditioned three French-accented candidates and kept this
# one. It costs a Chatterbox load -- seconds rather than the sub-second Kokoro
# reply -- which is the price of the voice being the one that was wanted.
DEFAULT_VOICE = "fr-male"


@dataclass
class Voice:
    """Everything the server needs to produce one voice.

    A cloned voice is three coupled settings -- model, reference clip and
    language code -- and getting any of them wrong fails in a way that names
    one of the others. Resolving them together is the point.
    """
    model: str
    voice: str = ""
    ref_audio: str | None = None
    lang_code: str = ""


def resolve_voice(name: str) -> Voice:
    """A preset name or a Kokoro voice name, resolved to a full Voice."""
    preset = VOICE_PRESETS.get(name)
    if preset:
        return Voice(model=preset["model"],
                     ref_audio=str(VOICES_DIR / preset["clip"]),
                     lang_code=preset["lang_code"])
    if name in KNOWN_VOICES:
        return Voice(model=DEFAULT_TTS_MODEL, voice=name)
    raise ValueError(
        f"unknown voice {name!r}; cloned: {', '.join(sorted(VOICE_PRESETS))}; "
        f"kokoro: {', '.join(KNOWN_VOICES)}")

# A WAV header is 44 bytes and 8000 bytes is a fifth of a second at 16k mono:
# below that there is no speech in the file whatever the status code said.
MIN_AUDIO_BYTES = 8000

_START_HINT = "is the server up? ./scripts/serve-tts.sh"


class AudioError(RuntimeError):
    """Anything that stops audio from being produced or understood."""


def speak(text: str, out: str | Path, voice: str = DEFAULT_KOKORO_VOICE,
          speed: float = 1.0, model: str = DEFAULT_TTS_MODEL,
          base_url: str = DEFAULT_BASE_URL,
          timeout: float = 120.0,
          ref_audio: str | Path | None = None,
          lang_code: str = "") -> Path:
    """Synthesize `text` to a WAV at `out`. Returns the path.

    `ref_audio` clones a voice from a reference clip, which is how Chatterbox
    works and what Kokoro's fixed voice table cannot do.

    `lang_code` is spelled exactly that way on purpose. mlx_audio names the
    field lang_code and defaults it to "a", Kokoro's American English. A field
    called `language` is accepted by the HTTP layer, ignored, and the request
    then fails as "Unsupported language code 'a'" -- naming a code the caller
    never sent, which sends you looking in the wrong place.
    """
    out = Path(out)
    payload = {"model": model, "input": text,
               "speed": speed, "response_format": "wav"}
    # Only some models have a voice table. Sending voice="" to one that does
    # not is a request for a voice named empty string.
    if voice:
        payload["voice"] = voice
    if lang_code:
        payload["lang_code"] = lang_code
    if ref_audio is not None:
        ref_audio = Path(ref_audio)
        if not ref_audio.exists():
            # The server reports a missing reference as a generation failure,
            # which sends you to its log rather than to the typo in the path.
            raise AudioError(f"no reference audio at {ref_audio}")
        payload["ref_audio"] = str(ref_audio)
    try:
        r = httpx.post(f"{base_url.rstrip('/')}/audio/speech", json=payload,
                       timeout=timeout)
        r.raise_for_status()
    except httpx.HTTPStatusError as exc:
        raise AudioError(
            f"tts failed: HTTP {exc.response.status_code}: "
            f"{exc.response.text[:400]}") from exc
    except httpx.RemoteProtocolError as exc:
        # The server accepted the request and then died part way through
        # streaming. Sending the user to start a server that is already running
        # is the wrong place: the traceback is in its log. The usual cause is a
        # voice that is not in the local cache.
        raise AudioError(
            f"tts failed while generating: {exc}. The server is up but the "
            f"request failed; the traceback is in its log (.logs/tts.log). "
            f"If you passed --voice, check it is one of: "
            f"{', '.join(KNOWN_VOICES)}") from exc
    except httpx.HTTPError as exc:
        raise AudioError(f"tts unreachable at {base_url}: {exc} ({_START_HINT})") from exc

    if len(r.content) < MIN_AUDIO_BYTES:
        raise AudioError(
            f"tts returned {len(r.content)} bytes, which is empty audio. "
            "mlx_audio does this on a 200 when misaki is not installed; check "
            "the server log for an ImportError.")

    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(r.content)
    return out


def speak_as(voice_name: str, text: str, out: str | Path, speed: float = 1.0,
             base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0) -> Path:
    """speak(), but the voice name carries its model and reference clip."""
    v = resolve_voice(voice_name)
    return speak(text, out=out, voice=v.voice, speed=speed, model=v.model,
                 base_url=base_url, timeout=timeout,
                 ref_audio=v.ref_audio, lang_code=v.lang_code)


def transcribe(path: str | Path, model: str = DEFAULT_STT_MODEL,
               base_url: str = DEFAULT_BASE_URL,
               timeout: float = 120.0) -> str:
    """Transcribe an audio file. Returns the text, stripped."""
    path = Path(path)
    if not path.exists():
        raise AudioError(f"no audio file at {path}")

    try:
        with path.open("rb") as fh:
            r = httpx.post(f"{base_url.rstrip('/')}/audio/transcriptions",
                           files={"file": (path.name, fh, "audio/wav")},
                           data={"model": model}, timeout=timeout)
        r.raise_for_status()
        body = r.json()
    except httpx.HTTPStatusError as exc:
        raise AudioError(
            f"stt failed: HTTP {exc.response.status_code}: "
            f"{exc.response.text[:400]}") from exc
    except httpx.HTTPError as exc:
        raise AudioError(f"stt unreachable at {base_url}: {exc} ({_START_HINT})") from exc
    except ValueError as exc:
        raise AudioError(f"stt returned non-JSON: {exc}") from exc

    text = body.get("text")
    if text is None:
        raise AudioError(f"stt response had no 'text': {body}")
    return text.strip()


def record_argv(out: str | Path, seconds: float) -> list[str]:
    """The sox command that captures a clip from the default input.

    Rate and channels are explicit because `rec` otherwise infers them from the
    file suffix, and Parakeet wants 16k mono.

    The binary is resolved to an absolute path: a GUI-spawned wezterm hands its
    children PATH=/usr/bin:/bin:/usr/sbin:/sbin, so a bare `rec` is not found
    when Claude Code was not started from a login shell.
    """
    rec = shutil.which("rec") or "/opt/homebrew/bin/rec"
    return [rec, "-q", "-r", "16000", "-c", "1", "-b", "16",
            str(out), "trim", "0", str(seconds)]


def play_argv(path: str | Path) -> list[str]:
    """afplay ships with macOS, so this one needs no PATH hedging."""
    return ["/usr/bin/afplay", str(path)]


# ---------------------------------------------------------------------------
# The local whisper backend
# ---------------------------------------------------------------------------

def _whisper_module():
    """Imported here, not at module scope, for two reasons: `lh say` should not
    pay numba's import cost to speak a sentence, and a test needs somewhere to
    hang a stub."""
    import mlx_whisper
    return mlx_whisper


def transcribe_whisper(path: str | Path,
                       model: str = DEFAULT_WHISPER_MODEL,
                       language: str = "") -> str:
    """Transcribe in-process with mlx_whisper. Returns the text, stripped.

    Bypasses the audio server entirely, so it costs a model load per process
    rather than per request. That is the right trade for an eval, which runs
    one process over a whole corpus, and the wrong one for a chat loop.

    `language` pins the decode. Whisper auto-detects when it is empty, which is
    fine on a paragraph and unreliable on a five-second clip -- and a French
    corpus scored as English produces a WER near 1.0 that looks like a bad TTS
    model rather than a misconfigured ear.
    """
    path = Path(path)
    if not path.exists():
        # A three gigabyte load is a slow way to learn the path was wrong.
        raise AudioError(f"no audio file at {path}")

    try:
        whisper = _whisper_module()
    except ImportError as exc:
        raise AudioError(
            f"the whisper backend needs the mlx-whisper package: {exc} "
            "(uv add mlx-whisper)") from exc

    kwargs = {"path_or_hf_repo": model}
    # Not the same as omitting it: mlx_whisper would take "" as a language code.
    if language:
        kwargs["language"] = language
    try:
        result = whisper.transcribe(str(path), **kwargs)
    except Exception as exc:
        # Everything downstream catches AudioError to tell a broken instrument
        # apart from a candidate that scored 100% error. A raw traceback here
        # would abort the run instead of recording a row.
        raise AudioError(f"whisper failed on {path.name}: {exc}") from exc

    text = result.get("text") if isinstance(result, dict) else None
    if text is None:
        raise AudioError(f"whisper returned no text: {result!r}")
    return text.strip()


_WK_TIMESTAMP = re.compile(r"^\[[\d:.]+\s*-+>\s*[\d:.]+\]\s*")
#: Lines the CLI prints around the transcript. Scoring these as speech would
#: measure the tool's logging.
_WK_NOISE = ("loading", "transcription time", "model", "downloading",
             "progress", "warning", "argmax", "compiling", "%")


def whisperkit_argv(path, model: str = DEFAULT_WHISPERKIT_MODEL,
                    language: str = "") -> list:
    """The `whisperkit-cli transcribe` command line."""
    argv = [shutil.which("whisperkit-cli") or "/opt/homebrew/bin/whisperkit-cli",
            "transcribe", "--audio-path", str(path), "--model", model]
    # Empty is not "no language": it would pin the decode to a language named
    # the empty string, the same trap lang_code has.
    if language:
        argv += ["--language", language]
    return argv


def _run_whisperkit(argv: list, timeout: float) -> str:
    import subprocess
    r = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    if r.returncode != 0:
        raise AudioError(
            f"whisperkit-cli exited {r.returncode}: "
            f"{(r.stderr or r.stdout).strip()[-300:]}")
    return r.stdout


def transcribe_whisperkit(path, model: str = DEFAULT_WHISPERKIT_MODEL,
                          language: str = "", timeout: float = 300.0) -> str:
    """Transcribe with CoreML WhisperKit, shelling out to `whisperkit-cli`.

    A THIRD RUNTIME, which is the point. parakeet-on-mlx_audio,
    whisper-on-mlx and this are three implementations of two model families;
    comparing only the first two measures models and never the runtime they
    sit on.
    """
    path = Path(path)
    if not path.exists():
        raise AudioError(f"no audio file at {path}")
    try:
        raw = _run_whisperkit(whisperkit_argv(path, model, language), timeout)
    except FileNotFoundError as exc:
        raise AudioError(
            f"whisperkit-cli is not installed: {exc} "
            f"(brew install whisperkit-cli)") from exc
    except OSError as exc:
        raise AudioError(f"could not run whisperkit-cli: {exc}") from exc

    lines = []
    for line in raw.splitlines():
        line = _WK_TIMESTAMP.sub("", line.strip())
        if not line or any(m in line.lower() for m in _WK_NOISE):
            continue
        lines.append(line)
    text = " ".join(lines).strip()
    if not text:
        raise AudioError(f"whisperkit-cli produced no transcript for {path.name}")
    return text


def transcriber(backend: str = "server", model: str = "", language: str = "",
                base_url: str = DEFAULT_BASE_URL, timeout: float = 120.0):
    """Build the callable that reads audio back, as `check()` wants it.

    The backend is part of the measurement, not an implementation detail: a WER
    from Parakeet and a WER from Whisper are not the same number, so the
    returned function carries a `label` saying which ear produced it.
    """
    if backend not in STT_BACKENDS:
        raise ValueError(
            f"unknown stt backend {backend!r}; "
            f"expected one of {', '.join(STT_BACKENDS)}")

    if backend == "whisperkit":
        model = model or DEFAULT_WHISPERKIT_MODEL

        def _run(path):
            return transcribe_whisperkit(path, model=model, language=language,
                                         timeout=timeout)
    elif backend == "whisper":
        model = model or DEFAULT_WHISPER_MODEL

        def _run(path):
            return transcribe_whisper(path, model=model, language=language)
    else:
        model = model or DEFAULT_STT_MODEL

        def _run(path):
            return transcribe(path, model=model, base_url=base_url,
                              timeout=timeout)

    _run.backend = backend
    _run.model = model
    label = f"{backend}:{model.split('/')[-1]}"
    _run.label = f"{label}/{language}" if language else label
    return _run
