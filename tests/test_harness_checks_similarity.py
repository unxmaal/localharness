"""Does the cloned voice sound like the person it was cloned from?

WER CANNOT SEE THIS. The tts lane scores intelligibility: it transcribes the
audio and counts word errors. A perfectly intelligible clone in a completely
different voice scores 0.000, which is the whole point of cloning going
unmeasured. Every French cloning number in this repo is an intelligibility
number and none of them says whether the clone worked.

A speaker embedding does see it: a fixed-size vector per utterance, and the
cosine between two of them is speaker similarity.
"""
from pathlib import Path

import pytest

pytest.importorskip("resemblyzer",
                    reason="needs the metrics group: uv run --group metrics")

from harness.checks import similarity  # noqa: E402


VOICES = Path(__file__).resolve().parents[1] / "harness" / "voices"
#: Two REAL clips of two DIFFERENT French men, shipped with the package. The
#: first version of these tests synthesized a sine wave, which resemblyzer
#: correctly trims to nothing: webrtcvad discards anything that is not speech,
#: so the fixture was the bug rather than the code.
REF_A = VOICES / "fr-male.wav"
REF_B = VOICES / "fr-male-2.wav"

pytestmark = pytest.mark.skipif(not REF_A.exists(),
                                reason="needs the shipped reference clips")


def test_a_clip_is_identical_to_itself():
    assert similarity.compare(REF_A, REF_A) == pytest.approx(1.0, abs=0.02)


def test_two_different_speakers_are_not_the_same_voice():
    """The measurement that matters: these are two different men reading
    French, so the metric has to separate them from a clip and itself."""
    same = similarity.compare(REF_A, REF_A)
    other = similarity.compare(REF_A, REF_B)
    assert other < same - 0.05, f"same {same:.3f} vs other {other:.3f}"


def test_similarity_is_bounded():
    assert -1.0 <= similarity.compare(REF_A, REF_B) <= 1.0


def test_a_missing_clip_is_reported(tmp_path):
    with pytest.raises(similarity.SimilarityError) as e:
        similarity.compare(tmp_path / "nope.wav", REF_A)
    assert "nope.wav" in str(e.value)


def test_silence_is_reported_rather_than_embedded(tmp_path):
    """resemblyzer trims silence, and an all-silent clip comes back empty and
    would embed to a vector of nans."""
    import struct
    import wave
    p = tmp_path / "quiet.wav"
    with wave.open(str(p), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(16000)
        w.writeframes(struct.pack("<h", 0) * 16000)
    with pytest.raises(similarity.SimilarityError) as e:
        similarity.compare(p, REF_A)
    assert "silent" in str(e.value).lower()


def test_the_metric_direction_is_declared():
    """Every metric this suite prints must say which way it runs; `ink` and
    `motion` were both ranked backwards before the registry existed."""
    from evals.core import direction_of
    assert direction_of("speaker_similarity") == "higher"


def test_the_negative_control_is_documented_as_failing():
    """The control that should have been run first: two different men score
    0.827, a clone against its own reference 0.855-0.904. A 0.05 gap is not a
    discriminator, and this module must not read as though it were one."""
    import inspect

    from harness.checks import similarity as mod
    doc = inspect.getdoc(mod)
    assert "0.827" in doc
    assert "NEGATIVE CONTROL" in doc


def test_two_different_men_score_close_to_a_correct_pair():
    """The actual measurement, kept as a test so it cannot quietly stop being
    true. If a future encoder separates these, this test fails and the module
    docstring needs rewriting -- which is the point."""
    same = similarity.compare(REF_A, REF_A)
    different = similarity.compare(REF_A, REF_B)
    assert different > 0.6, "two French men should not be near-orthogonal"
    assert same - different > 0.05, "there is at least SOME separation"
