"""Judging generated music by what can be read back off it.

The tts lane's round-trip, applied to singing. Lyrics go in, a vocal comes out,
a transcriber reads it, and the word error rate orders candidates. Nothing here
is new: `wer_counts` is imported from the speech checker unchanged, so the
music lane and the tts lane compute a rate the same way and their numbers are
at least commensurable.

IT IS THE SAME JOINT MEASUREMENT the speech checker documents. The rate covers
the generator and the ear together and cannot separate them. Holding the ear
fixed while varying the generator gives a real ordering, which is what the
suite is for; a number from here is not an absolute score for either.

WHAT THIS DELIBERATELY DOES NOT MEASURE, both recorded so the gaps can be
argued rather than discovered:

  MUSICAL QUALITY. There is no instrument here that can hear whether a track is
  any good, and a model scoring how good it sounds would score everything the
  same (RULE #211).

  TEMPO. ACE-Step takes a BPM and it was the obvious second axis. It was
  measured and dropped: librosa's estimator returns its own `start_bpm` prior
  on input with no beat in it -- digital silence reads as 120.2 BPM -- so a
  track asked for 120 that comes back "120" tells you nothing. Probed at three
  priors, an instrumental agreed with itself at 89.3 against a request of 90
  and a SUNG track did not agree at all (93.8 / 125.0 / 187.5). A lane whose
  cases are mostly vocal cannot use an axis that only works without a vocal.

Duration is here because a file length cannot be dragged by a prior.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from harness.audio import AudioError
from harness.checks.speech import MIN_AUDIO_BYTES, normalize, wer, wer_counts

#: How far a track may miss its requested length, as a fraction of it. The one
#: measured run landed on 30.00s against a requested 30, so this starts loose
#: rather than pretending a tolerance was derived from more than one number.
DURATION_TOLERANCE = 0.1

#: Words a track asked for NO vocal may still produce before it counts as
#: singing. Not zero: a transcriber will occasionally hallucinate a syllable
#: out of a guitar, and failing a model for that measures the ear. The observed
#: instrumental transcribed to the empty string, so this is headroom and not a
#: fitted value.
INSTRUMENTAL_WORDS_MAX = 2


@dataclass
class MusicResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    wer: float | None = None
    transcript: str = ""
    errors: int = 0
    words: int = 0
    seconds: float | None = None
    asked_seconds: float | None = None

    @property
    def metrics(self) -> dict:
        """What the eval ranks on.

        `wer` is absent rather than 0.0 when nothing was scored: a zero error
        rate is the best possible score, and reporting it for a candidate that
        was never scored would rank it at the top. Same reasoning as the
        speech checker, and the same trap.
        """
        out: dict = {}
        if self.wer is not None:
            out["wer"] = round(self.wer, 4)
            out["wer_errors"] = self.errors
            out["wer_words"] = self.words
        if self.seconds is not None:
            out["seconds"] = round(self.seconds, 2)
            if self.asked_seconds:
                out["duration_error_s"] = round(
                    abs(self.seconds - self.asked_seconds), 2)
        return out


def heard_words(transcript: str, language: str = "en") -> int:
    """How many words the ear got out of the audio, after normalization."""
    return len(normalize(transcript or "", language).split())


def check(path: str | Path, lyrics: str, *, max_wer: float | None = None,
          duration_s: float | None = None, expect_vocals: bool = True,
          transcriber=None, language: str = "en") -> MusicResult:
    """Score one generated track against what was asked for.

    `max_wer` is optional: with no limit the check measures without judging,
    which is what ranking needs. `expect_vocals=False` inverts the question
    from "were the right words sung" to "was anything sung at all".
    """
    from harness import audio  # imported here so tests can inject a stub

    transcriber = transcriber or audio.transcribe
    path = Path(path)
    if not path.exists():
        return MusicResult(False, f"no audio file at {path}")
    size = path.stat().st_size
    if size < MIN_AUDIO_BYTES:
        return MusicResult(
            False, f"audio is {size} bytes, too small to hold music")

    warnings: list[str] = []
    seconds = None
    try:
        seconds = audio.audio_seconds(path)
    except Exception as exc:  # noqa: BLE001
        # A missing ffprobe must not fail a candidate whose vocal is fine.
        warnings.append(f"duration unavailable: {exc}")

    try:
        transcript = transcriber(path)
    except AudioError as exc:
        # An STT outage recorded as this candidate scoring 100% error is a
        # broken instrument reported as a bad model.
        return MusicResult(False, f"could not transcribe: {exc}",
                           warnings=warnings, seconds=seconds,
                           asked_seconds=duration_s)

    out = MusicResult(True, warnings=warnings, transcript=transcript,
                      seconds=seconds, asked_seconds=duration_s)

    if expect_vocals:
        reference = lyric_words(lyrics)
        if not reference:
            return MusicResult(
                False, "a vocal case needs lyrics to score against",
                warnings=warnings, seconds=seconds, asked_seconds=duration_s)
        out.wer = wer(reference, transcript, language)
        out.errors, out.words = wer_counts(reference, transcript, language)
        if max_wer is not None and out.wer > max_wer:
            out.ok = False
            out.reason = (f"word error rate {out.wer:.2f} over the limit of "
                          f"{max_wer}; heard {transcript!r}")
    else:
        # THE NEGATIVE CONTROL, and a requirement in its own right. A model
        # told to produce no vocal that sings anyway has not done what was
        # asked, and neither a duration nor a tempo check can see it.
        got = heard_words(transcript, language)
        if got > INSTRUMENTAL_WORDS_MAX:
            out.ok = False
            out.reason = (f"asked for no vocal and heard {got} words: "
                          f"{transcript!r}")

    if out.ok and duration_s and seconds is not None:
        drift = abs(seconds - duration_s)
        if drift > DURATION_TOLERANCE * duration_s:
            out.ok = False
            out.reason = (f"asked for {duration_s:g}s and got {seconds:.2f}s, "
                          f"off by {drift:.2f}s")
    return out


def lyric_words(lyrics: str) -> str:
    """The words a singer would actually sing.

    Lyrics carry STRUCTURE TAGS -- [Verse], [Chorus], [Instrumental Break] --
    which are directions to the model and are not sung. Scoring them as words
    the vocal failed to produce would charge every candidate for obeying the
    format, which measures the prompt language rather than the singing.
    """
    kept = []
    for line in (lyrics or "").splitlines():
        line = line.strip()
        if not line or (line.startswith("[") and line.endswith("]")):
            continue
        kept.append(line)
    return " ".join(kept)
