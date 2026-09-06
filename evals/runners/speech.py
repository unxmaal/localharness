"""Runner for text-to-speech candidates.

A tts case is a sentence. The runner speaks it; the checker reads it back with
the resident STT model and reports the word error rate. Both ends run on this
machine, so a full comparison costs nothing but time.

The measurement is JOINT: it scores the TTS model and the STT model together
and cannot separate them. Holding the STT side fixed while varying TTS gives a
real ordering of TTS candidates, which is what the suite is for, but a number
from here is not an absolute score for either model alone.
"""
from __future__ import annotations

from pathlib import Path

from harness import audio

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


class SpeechRunner(BaseRunner):
    def __init__(self, model: str, outdir: str | Path,
                 voice: str = audio.DEFAULT_VOICE,
                 base_url: str = audio.DEFAULT_BASE_URL,
                 timeout: float = 120.0):
        self.model = model
        self.voice = voice
        self.base_url = base_url
        self.timeout = timeout
        # The voice is part of the candidate: Kokoro at am_adam and at ff_siwis
        # are different products, and sharing a row is the same mistake as
        # sharing one between two quantizations.
        self.candidate = f"{model.split('/')[-1]}/{voice}"
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)

    def generate(self, case: Case):
        out = self.outdir / f"{self.candidate.replace('/', '_')}--{case.id}.wav"
        try:
            audio.speak(case.prompt, out=out, voice=self.voice,
                        speed=float(case.params.get("speed", 1.0)),
                        model=self.model, base_url=self.base_url,
                        timeout=self.timeout)
        except audio.AudioError as exc:
            raise RunnerError(str(exc)) from exc
        # Peak memory is not observable across HTTP; the server holds the model.
        return out, 0

    def score_kwargs(self) -> dict:
        return {"transcriber": self._transcribe}

    def _transcribe(self, path):
        return audio.transcribe(path, base_url=self.base_url,
                                timeout=self.timeout)
