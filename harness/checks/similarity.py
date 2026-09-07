"""Does a cloned voice sound like the person it was cloned from?

A WER cannot see this: an intelligible clone in the wrong voice scores 0.000.

NEVER COMPARE ACROSS RECORDING CONDITIONS. Two real recordings score high
against each other whatever their speakers, and two synthesized clips likewise,
so a single global threshold is invalid. The eval only ever asks ref-vs-clone,
which is one condition pair, and the control below is run in that condition.
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


#: Measured 2026-09-07, resemblyzer, ref-vs-synth only, 3 identities.
CONTROL_FLOOR = 0.79


def control(same: list[tuple], different: list[tuple]) -> dict:
    """Score known-same and known-different pairs and report whether they split.

    Run this before quoting any figure. A metric whose classes overlap cannot
    rank anything, and that is invisible in the scores themselves.
    """
    if not same or not different:
        raise SimilarityError("a control needs both same and different pairs")
    s = [compare(a, b) for a, b in same]
    d = [compare(a, b) for a, b in different]
    return {"same_min": min(s), "same_max": max(s), "same_mean": sum(s) / len(s),
            "diff_min": min(d), "diff_max": max(d), "diff_mean": sum(d) / len(d),
            "gap": min(s) - max(d), "separates": min(s) > max(d),
            "n_same": len(s), "n_different": len(d)}
