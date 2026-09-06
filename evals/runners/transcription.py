"""Runner for speech-to-text candidates.

The mirror of the speech runner, and a better measurement than it. There the
prompt is text to speak and the artifact is audio, so the word error rate is
JOINT: it scores the TTS model together with whatever reads it back. Here the
audio is the input and the case's prompt is a transcript a human wrote, so the
number belongs to the STT model alone.

That is why this lane needs a corpus. `evals/corpora.py` generates cases from
LibriSpeech test-clean, which ships human transcripts with the audio.

TWO BACKENDS, because one is not enough. Parakeet runs on the audio server and
is English-only; Whisper is multilingual and cannot run on that server at all
-- mlx_audio demands a HuggingFace processor, and the mlx whisper repos ship
weights.npz and config.json without one -- so it is called in-process instead.
"""
from __future__ import annotations

from harness import audio

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


class TranscriptionRunner(BaseRunner):
    def __init__(self, model: str = audio.DEFAULT_STT_MODEL,
                 base_url: str = audio.DEFAULT_BASE_URL,
                 timeout: float = 120.0,
                 backend: str = "server", language: str = ""):
        self.model = model
        self.base_url = base_url
        self.timeout = timeout
        self.backend = backend
        self.language = language
        # Raises on an unknown backend, here rather than on the first of forty
        # clips: a typo must not cost a corpus run.
        self._transcribe = audio.transcriber(
            backend=backend, model=model, language=language,
            base_url=base_url, timeout=timeout)
        # The same weights decoding French and English are two measurements.
        # Sharing a row would average them into a number describing neither.
        name = model.split("/")[-1]
        self.candidate = f"{name}/{language}" if language else name

    def generate(self, case: Case):
        if case.audio is None:
            raise RunnerError("case has no audio to transcribe")
        try:
            text = self._transcribe(case.audio)
        except audio.AudioError as exc:
            raise RunnerError(str(exc)) from exc
        # Peak memory is not observable across HTTP; the server holds the model.
        # The whisper backend does hold it in this process, but what is
        # measurable there is the whole eval process rather than the model, so
        # reporting nothing stays honest in both cases.
        return text, 0
