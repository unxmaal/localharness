"""Runner for text-to-speech candidates.

A tts case is a sentence. The runner speaks it; the checker reads it back and
reports the word error rate. Both ends run on this machine, so a full
comparison costs nothing but time.

The measurement is JOINT: it scores the TTS model and the STT model together
and cannot separate them. Holding the STT side fixed while varying TTS gives a
real ordering of TTS candidates, which is what the suite is for, but a number
from here is not an absolute score for either model alone.

That joint-ness has a requirement the English lane never exposed: the ear has
to speak the language. Parakeet is English-only, so scoring French through it
returns a word error rate near 1.0 for every candidate and ranks them all as
equally broken. `ear` picks the transcriber, and it is part of the candidate's
identity for the same reason the voice is.
"""
from __future__ import annotations

from pathlib import Path

from harness import audio

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


def parse_ear(ear: str) -> tuple[str, str]:
    """"whisper:fr" -> ("whisper", "fr"). "server" -> ("server", "")."""
    backend, _, language = ear.partition(":")
    return backend.strip(), language.strip()


class SpeechRunner(BaseRunner):
    def __init__(self, model: str, outdir: str | Path,
                 voice: str = audio.DEFAULT_VOICE,
                 base_url: str = audio.DEFAULT_BASE_URL,
                 timeout: float = 120.0,
                 ref_audio: str | Path | None = None,
                 lang_code: str = "",
                 ear: str = "server"):
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.timeout = timeout
        self.ref_audio = Path(ref_audio) if ref_audio else None
        self.lang_code = lang_code
        self.ear = ear
        backend, language = parse_ear(ear)
        # Raises on an unknown backend here, before a whole lane runs.
        self._transcribe = audio.transcriber(
            backend=backend, language=language,
            base_url=base_url, timeout=timeout)
        # What varies is part of the candidate: Kokoro at am_adam and at
        # ff_siwis are different products, and so are the same cloning weights
        # given two different speakers to imitate. Sharing a row is the same
        # mistake as sharing one between two quantizations.
        name = model.split("/")[-1]
        if voice:
            name = f"{name}/{voice}"
        if self.ref_audio is not None:
            name = f"{name}/{self.ref_audio.stem}"
        self.candidate = name
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def generate(self, case: Case):
        out = self.outdir / f"{self.candidate.replace('/', '_')}--{case.id}.wav"
        try:
            audio.speak(case.prompt, out=out, voice=self.voice,
                        speed=float(case.params.get("speed", 1.0)),
                        model=self.model, base_url=self.base_url,
                        timeout=self.timeout,
                        ref_audio=self.ref_audio, lang_code=self.lang_code)
        except audio.AudioError as exc:
            raise RunnerError(str(exc)) from exc
        # Peak memory is not observable across HTTP; the server holds the model.
        return out, 0

    def score_kwargs(self) -> dict:
        return {"transcriber": self._transcribe}
