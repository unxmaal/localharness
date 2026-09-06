"""Judging synthesized speech by whether it can be read back.

This is the suite's first quality axis. The objective checks elsewhere separate
working from broken and cannot order two candidates that both work; word error
rate can.

IT IS A JOINT MEASUREMENT of the TTS model and the STT model reading it, and it
cannot separate them. That is sound for the comparison it is used for -- hold
one side fixed, vary the other, and the ordering is real -- but a WER from here
is not an absolute score for either model on its own. Ranking STT models needs
reference audio with a human transcript (LibriSpeech test-clean), which this
machine does not have yet.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import jiwer
from num2words import num2words

from harness.audio import AudioError

# A WAV header is 44 bytes. Anything near it holds no speech, and transcribing
# it would report the empty result as a terrible TTS model.
MIN_AUDIO_BYTES = 8000

_PUNCT = re.compile(r"[^\w\s]")
_DIGITS = re.compile(r"\d+")


@dataclass
class SpeechResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    wer: float | None = None
    transcript: str = ""

    @property
    def metrics(self) -> dict:
        """What the eval ranks on. Empty when nothing was measured, never 0.0:
        a zero error rate is the best possible score and would rank a candidate
        that never ran at the top."""
        return {} if self.wer is None else {"wer": round(self.wer, 4)}


def normalize(text: str) -> str:
    """Strip everything that would measure transcription style, not speech.

    Parakeet emits no reliable punctuation and inconsistent casing, and it
    spells numbers out: "MLX 200" comes back as "mlx two hundred". Counting
    those as errors ranks models on how they write, not on what they heard.
    """
    text = _DIGITS.sub(lambda m: num2words(int(m.group())), text)
    text = _PUNCT.sub(" ", text.lower())
    return " ".join(text.split())


def wer(reference: str, hypothesis: str) -> float:
    """Word error rate, after normalization. 0.0 is perfect."""
    ref = normalize(reference)
    if not ref:
        raise ValueError("cannot score against an empty reference")
    hyp = normalize(hypothesis)
    if not hyp:
        # jiwer treats an empty hypothesis as every word deleted, which is 1.0;
        # spelling it out keeps the edge case from depending on jiwer's version.
        return 1.0
    return jiwer.wer(ref, hyp)


def check(path: str | Path, reference: str, max_wer: float | None = None,
          transcriber=None) -> SpeechResult:
    """Transcribe the audio at `path` and compare it to `reference`.

    `max_wer` is optional: with no limit the check measures without judging,
    which is what ranking needs.
    """
    from harness import audio  # imported here so tests can inject a stub

    transcriber = transcriber or audio.transcribe
    path = Path(path)
    if not path.exists():
        return SpeechResult(False, f"no audio file at {path}")
    size = path.stat().st_size
    if size < MIN_AUDIO_BYTES:
        return SpeechResult(
            False, f"audio is {size} bytes, too small to hold speech")

    try:
        transcript = transcriber(path)
    except AudioError as exc:
        # An STT outage must not be recorded as this candidate scoring 100%
        # error: that is a broken instrument, not a bad model.
        return SpeechResult(False, f"could not transcribe: {exc}")

    rate = wer(reference, transcript)
    if max_wer is not None and rate > max_wer:
        return SpeechResult(
            False,
            f"word error rate {rate:.2f} over the limit of {max_wer}; "
            f"heard {transcript!r}",
            wer=rate, transcript=transcript)
    return SpeechResult(True, "", wer=rate, transcript=transcript)
