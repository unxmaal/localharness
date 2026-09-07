"""Does a cloned voice sound like the person it was cloned from?

WER CANNOT SEE THIS, and every voice number in this repo is a WER. The tts lane
transcribes the audio and counts word errors, so a perfectly intelligible clone
in a completely different voice scores 0.000 -- the one property cloning exists
to deliver is the one nothing measured. Kokoro's ff_siwis "beat" every
Chatterbox clone on that lane while not being a clone at all.

A speaker embedding does see it: one fixed-size vector per utterance, and the
cosine between two of them is speaker similarity. resemblyzer's encoder is
~17MB and runs on CPU in well under a second, so this is cheap enough to put
beside the WER rather than instead of it.

READ IT AS A COMPARISON, NOT A SCORE. The absolute number depends on the
recording conditions of both clips. Two utterances by one speaker in one
session sit high; the same speaker down a phone line sits lower. What is
meaningful is ranking clones of the SAME reference against each other.
"""
from __future__ import annotations

from pathlib import Path


class SimilarityError(RuntimeError):
    """The two clips could not be compared."""


def _encoder():
    """Built once per process; loading the encoder is most of the cost."""
    global _ENCODER
    try:
        return _ENCODER
    except NameError:
        pass
    try:
        from resemblyzer import VoiceEncoder
    except ImportError as exc:
        raise SimilarityError(
            f"speaker similarity needs the metrics group: {exc} "
            f"(uv run --group metrics ...)") from exc
    _ENCODER = VoiceEncoder("cpu")
    return _ENCODER


def embed(path: str | Path):
    path = Path(path)
    if not path.exists():
        raise SimilarityError(f"no audio file at {path}")
    try:
        from resemblyzer import preprocess_wav
    except ImportError as exc:
        raise SimilarityError(f"speaker similarity needs the metrics group: {exc}") from exc
    try:
        wav = preprocess_wav(path)
    except Exception as exc:  # noqa: BLE001
        raise SimilarityError(f"could not read {path.name}: {exc}") from exc
    if wav.size == 0:
        # preprocess_wav trims silence; an all-silent clip comes back empty and
        # would otherwise embed to a vector of nans.
        raise SimilarityError(f"{path.name} is silent after trimming")
    return _encoder().embed_utterance(wav)


def compare(a: str | Path, b: str | Path) -> float:
    """Cosine similarity between two voices, in [-1, 1]. 1.0 is the same clip."""
    import numpy as np

    va, vb = embed(a), embed(b)
    denom = float(np.linalg.norm(va) * np.linalg.norm(vb))
    if denom == 0:
        raise SimilarityError("one of the clips embedded to a zero vector")
    return float(np.dot(va, vb) / denom)
