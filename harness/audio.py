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

import shutil
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
STT_BACKENDS = ("server", "whisper")
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
DEFAULT_VOICE = "bm_george"

# A WAV header is 44 bytes and 8000 bytes is a fifth of a second at 16k mono:
# below that there is no speech in the file whatever the status code said.
MIN_AUDIO_BYTES = 8000

_START_HINT = "is the server up? ./scripts/serve-tts.sh"


class AudioError(RuntimeError):
    """Anything that stops audio from being produced or understood."""


def speak(text: str, out: str | Path, voice: str = DEFAULT_VOICE,
          speed: float = 1.0, model: str = DEFAULT_TTS_MODEL,
          base_url: str = DEFAULT_BASE_URL,
          timeout: float = 120.0) -> Path:
    """Synthesize `text` to a WAV at `out`. Returns the path."""
    out = Path(out)
    payload = {"model": model, "input": text,
               "speed": speed, "response_format": "wav"}
    # Only some models have a voice table. Sending voice="" to one that does
    # not is a request for a voice named empty string.
    if voice:
        payload["voice"] = voice
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

    if backend == "whisper":
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
