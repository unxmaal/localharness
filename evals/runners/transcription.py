"""Runner for speech-to-text candidates.

The mirror of the speech runner, and a better measurement than it. There the
prompt is text to speak and the artifact is audio, so the word error rate is
JOINT: it scores the TTS model together with whatever reads it back. Here the
audio is the input and the case's prompt is a transcript a human wrote, so the
number belongs to the STT model alone.

That is why this lane needs a corpus. `evals/corpora.py` generates cases from
LibriSpeech test-clean, which ships human transcripts with the audio.
"""
from __future__ import annotations

from harness import audio

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


class TranscriptionRunner(BaseRunner):
    def __init__(self, model: str = audio.DEFAULT_STT_MODEL,
                 base_url: str = audio.DEFAULT_BASE_URL,
                 timeout: float = 120.0):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.candidate = model.split("/")[-1]

    def generate(self, case: Case):
        if case.audio is None:
            raise RunnerError("case has no audio to transcribe")
        try:
            text = audio.transcribe(case.audio, model=self.model,
                                    base_url=self.base_url,
                                    timeout=self.timeout)
        except audio.AudioError as exc:
            raise RunnerError(str(exc)) from exc
        # Peak memory is not observable across HTTP; the server holds the model.
        return text, 0
