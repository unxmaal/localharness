"""The audio lanes on a machine with an NVIDIA card.

mlx_audio answers both endpoints on the Mac. Nothing on PyPI answers both here,
so this is the smallest server that takes the same two requests: Kokoro through
onnxruntime for speech, faster-whisper through CTranslate2 for transcription.
harness/audio.py is unchanged apart from which model it names by default, and
does not know which of the two it is talking to.

IT REFUSES A MODEL IT CANNOT SERVE. Asked for parakeet it answers 400 rather
than transcribing with Whisper and returning the text, because a result that
names one model and was produced by another is worse than no result: the score
sheet cannot tell, and neither can the person reading it. This is the same rule
llama-server applies to a GGUF name it does not hold.

VOICE CLONING IS NOT IMPLEMENTED. The Mac's default voice is cloned from a
reference clip by Chatterbox, and Kokoro has a fixed table of 54. A request
carrying ref_audio is refused for the reason above: a line delivered in some
other voice measures something the caller did not ask for.
"""
from __future__ import annotations

import glob
import io
import os
import sys
import tempfile
from pathlib import Path

# Imported here rather than inside build_app(): `from __future__ import
# annotations` makes every annotation a string, and pydantic resolves the ones
# on a route against this module's globals. Bound inside the factory they
# resolve to nothing and every multipart request answers 500.
try:
    from fastapi import FastAPI, File, Form, HTTPException, UploadFile
    from fastapi.responses import Response
except ImportError:  # the constants above are importable without the server
    FastAPI = File = Form = HTTPException = UploadFile = Response = None

#: Kokoro-82M in ONNX form. The Mac names the same weights as an MLX repo, and
#: the runtime differs, so an eval comparing the two machines should say which
#: it ran rather than letting one id stand for both.
KOKORO_IDS = (
    "kokoro-onnx/Kokoro-82M",
    "mlx-community/Kokoro-82M-bf16",   # what `lh say` sends; same weights
)
#: What faster-whisper will load by name. Anything else is refused.
WHISPER_PREFIXES = ("Systran/faster-whisper-", "whisper-", "tiny", "base",
                    "small", "medium", "large")

DEFAULT_WHISPER = os.environ.get("WHISPER_MODEL", "base.en")
DEFAULT_KOKORO_LANG = "en-gb"
#: Below this a WAV is silence or a truncated stream. harness/audio.py applies
#: the same floor to what it receives.
MIN_AUDIO_BYTES = 8000


def _cuda_dll_directories() -> list[str]:
    """Where the nvidia pip wheels keep the CUDA runtime.

    CTranslate2 resolves cublas64_12.dll through the OS search path rather than
    through Python's loader, so os.add_dll_directory does not reach it and the
    directories have to be on PATH before this process starts. The launcher
    does that; this reports them so it can.
    """
    try:
        import nvidia
    except ImportError:
        return []
    found = []
    for root in list(getattr(nvidia, "__path__", [])):
        found += [d for d in glob.glob(os.path.join(root, "*", "bin"))
                  if os.path.isdir(d)]
    return found


def _load_whisper(name: str):
    """Load on the card, and fall back to the CPU saying why.

    Counting CUDA devices is not the test: ctranslate2 reports the card through
    the driver and then fails to load cublas64_12.dll if the runtime is not on
    PATH, so the only honest check is to build the model and see. Falling back
    without a word would make a slow run look like a slow model.
    """
    from faster_whisper import WhisperModel
    try:
        return WhisperModel(name, device="cuda", compute_type="float16")
    except Exception as exc:  # noqa: BLE001 - any failure means the CPU
        print(f"audio: CUDA unavailable for transcription ({exc}); "
              f"using the CPU", file=sys.stderr)
        return WhisperModel(name, device="cpu", compute_type="int8")


def build_app():
    """The FastAPI app. Built in a function so importing this module for its
    constants does not need fastapi installed."""
    if FastAPI is None:
        raise RuntimeError("fastapi is not installed; scripts/serve-audio-cuda.sh "
                           "supplies it with pinned versions")

    app = FastAPI()
    state: dict = {}

    def kokoro():
        if "kokoro" not in state:
            from kokoro_onnx import Kokoro
            model = os.environ.get("KOKORO_MODEL", "")
            voices = os.environ.get("KOKORO_VOICES", "")
            if not (model and voices):
                raise HTTPException(
                    500, "set KOKORO_MODEL and KOKORO_VOICES to the "
                         "kokoro-v1.0.onnx and voices-v1.0.bin paths")
            state["kokoro"] = Kokoro(model, voices)
        return state["kokoro"]

    def whisper(name: str):
        key = f"whisper:{name}"
        if key not in state:
            state[key] = _load_whisper(name)
        return state[key]

    @app.get("/v1/models")
    def models():
        """What this server can actually answer for, so a caller can find out
        without guessing from a 400."""
        return {"object": "list", "data": [
            {"id": KOKORO_IDS[0], "object": "model", "owned_by": "kokoro-onnx"},
            {"id": DEFAULT_WHISPER, "object": "model",
             "owned_by": "faster-whisper"},
        ]}

    @app.post("/v1/audio/speech")
    async def speech(payload: dict):
        if payload.get("ref_audio"):
            raise HTTPException(
                400, "voice cloning is not implemented here: Kokoro has a "
                     "fixed voice table and Chatterbox is the Mac's cloner. "
                     "Ask for one of the 54 Kokoro voices instead.")
        model = payload.get("model") or KOKORO_IDS[0]
        if model not in KOKORO_IDS:
            raise HTTPException(
                400, f"model {model!r} is not served here; this server holds "
                     f"Kokoro ({', '.join(KOKORO_IDS)})")
        text = (payload.get("input") or "").strip()
        if not text:
            raise HTTPException(400, "input is empty")

        engine = kokoro()
        voice = payload.get("voice") or "bm_george"
        if voice not in engine.get_voices():
            raise HTTPException(
                400, f"no voice {voice!r} in this Kokoro build; a substituted "
                     f"voice would measure a different one than was asked for")
        samples, rate = engine.create(
            text, voice=voice, speed=float(payload.get("speed") or 1.0),
            lang=payload.get("lang_code") or DEFAULT_KOKORO_LANG)

        import soundfile as sf
        buf = io.BytesIO()
        sf.write(buf, samples, rate, format="WAV")
        body = buf.getvalue()
        if len(body) < MIN_AUDIO_BYTES:
            raise HTTPException(
                500, f"synthesis produced {len(body)} bytes, which is silence")
        return Response(content=body, media_type="audio/wav")

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(file: UploadFile = File(...),
                             model: str = Form(DEFAULT_WHISPER)):
        if not any(model.startswith(p) for p in WHISPER_PREFIXES):
            raise HTTPException(
                400, f"model {model!r} is not served here; this server "
                     f"transcribes with faster-whisper. Naming a model it does "
                     f"not run and returning text anyway would put the wrong "
                     f"model in the result.")
        body = await file.read()
        suffix = Path(file.filename or "clip.wav").suffix or ".wav"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as fh:
            fh.write(body)
            clip = fh.name
        try:
            segments, _ = whisper(model).transcribe(clip)
            return {"text": " ".join(s.text for s in segments).strip()}
        finally:
            Path(clip).unlink(missing_ok=True)

    return app


def main() -> None:
    import uvicorn
    uvicorn.run(build_app(),
                host=os.environ.get("AUDIO_HOST", "0.0.0.0"),
                port=int(os.environ.get("AUDIO_PORT", "8890")))


if __name__ == "__main__":
    main()
