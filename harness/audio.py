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
