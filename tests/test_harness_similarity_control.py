"""Issue #5: the control that decides whether the metric may be quoted at all."""
from pathlib import Path

import pytest

from harness.checks import similarity
from harness.checks.similarity import CONTROL_FLOOR, SimilarityError, control


def fake_compare(monkeypatch, table):
    monkeypatch.setattr(similarity, "compare",
                        lambda a, b: table[(a, b)])


def test_clean_separation_is_reported_as_separating(monkeypatch):
    fake_compare(monkeypatch, {("s", "1"): 0.80, ("s", "2"): 0.90,
                               ("d", "1"): 0.60, ("d", "2"): 0.70})
    r = control([("s", "1"), ("s", "2")], [("d", "1"), ("d", "2")])
    assert r["separates"] is True
    assert r["gap"] == pytest.approx(0.10)
    assert r["n_same"] == 2 and r["n_different"] == 2


def test_one_overlapping_pair_fails_the_whole_control(monkeypatch):
    """The failure that was missed the first time: a single wrong-speaker pair
    above a correct one means no ranking can be drawn, however good the means
    look."""
    fake_compare(monkeypatch, {("s", "1"): 0.742, ("s", "2"): 0.904,
                               ("d", "1"): 0.881, ("d", "2"): 0.60})
    r = control([("s", "1"), ("s", "2")], [("d", "1"), ("d", "2")])
    assert r["separates"] is False
    assert r["gap"] < 0
    assert r["same_mean"] > r["diff_mean"], "means alone would have passed it"


def test_a_control_needs_both_classes():
    for same, diff in (([], [("a", "b")]), ([("a", "b")], [])):
        with pytest.raises(SimilarityError):
            control(same, diff)


@pytest.mark.slow
def test_the_real_control_separates_on_ref_versus_clone():
    """Regenerates clones through the live tts server and re-runs the control.

    Marked slow: needs the metrics group and port 8890. This is the assertion
    that stops the metric quietly becoming noise again.
    """
    from harness.audio import resolve_voice, speak

    tmp = Path.home() / "localharness/runs/issue5-control"
    tmp.mkdir(parents=True, exist_ok=True)
    texts = ["The gateway is running and the weights are read from disk.",
             "Stop the containers before starting a download on this machine.",
             "Check the gateway log before you restart anything."]

    refs, clones = {}, {}
    for name in ("fr-male", "fr-male-2"):
        v = resolve_voice(name)
        refs[name] = Path(v.ref_audio)
        clones[name] = []
        for i, text in enumerate(texts):
            p = tmp / f"ctl-{name}-{i}.wav"
            if not p.exists():
                speak(text, p, voice=v.voice, model=v.model,
                      ref_audio=v.ref_audio, lang_code=v.lang_code)
            clones[name].append(p)

    a, b = "fr-male", "fr-male-2"
    same = [(refs[n], c) for n in (a, b) for c in clones[n]]
    diff = [(refs[a], c) for c in clones[b]] + [(refs[b], c) for c in clones[a]]
    r = control(same, diff)
    assert r["separates"], (
        f"speaker similarity no longer separates same from different: {r}. "
        f"No figure from it may be quoted until this passes again.")
    assert r["same_min"] > CONTROL_FLOOR * 0.95
