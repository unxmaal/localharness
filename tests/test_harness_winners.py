"""The typed lane defaults against the receipts that chose them."""
import json

from harness import winners


def pin(monkeypatch, **defaults):
    """Pin the typed defaults this test is about.

    THE SPEECH DEFAULTS ARE CHOSEN BY PLATFORM -- parakeet on a Mac,
    faster-whisper on Windows -- so a test that inherits them is partly a test
    about which operating system ran it. Two of these failed on check-windows
    for exactly that, the same shape as the verdict test that inherited the
    runner's mlx runtime.
    """
    monkeypatch.setattr(winners, "typed", lambda: dict(defaults))


def receipt(tmp_path, name, modality, summary, tier="measure"):
    d = tmp_path / name
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "receipt": {"modality": modality, "tier": tier},
        "summary": summary}), encoding="utf-8")
    return tmp_path


def test_a_default_that_appears_in_no_receipt_is_unmeasured_not_beaten(tmp_path):
    """THE FALSE ALARM THIS EXISTS TO AVOID. The first cut took the best
    candidate across every receipt on disk and reported that extract had been
    won by a model scoring 0.7 -- because the only extract receipt here is a
    three-small-model run the typed default was never part of, while the run
    that scored it 9/10 predates receipts entirely."""
    receipt(tmp_path, "small", "extract",
            {"q3-1.7b": {"pass_rate": 0.7, "median_s": 0.4, "total": 10}})
    rows = {r["modality"]: r for r in winners.disagreements(tmp_path)}
    assert rows["extract"]["state"] == "unmeasured"
    assert rows["extract"]["measured"] == ""


def test_a_default_that_lost_a_run_it_was_in_is_reported(tmp_path):
    receipt(tmp_path, "head-to-head", "extract",
            {"local-large": {"pass_rate": 0.5, "median_s": 1.0, "total": 10},
             "q3-4b": {"pass_rate": 0.9, "median_s": 2.0, "total": 10}})
    rows = {r["modality"]: r for r in winners.disagreements(tmp_path)}
    assert rows["extract"]["state"] == "beaten"
    assert rows["extract"]["measured"] == "q3-4b"


def test_a_default_that_won_is_not_reported_at_all(tmp_path):
    receipt(tmp_path, "head-to-head", "extract",
            {"local-large": {"pass_rate": 0.9, "median_s": 1.0, "total": 10},
             "q3-4b": {"pass_rate": 0.5, "median_s": 2.0, "total": 10}})
    rows = {r["modality"]: r for r in winners.disagreements(tmp_path)}
    assert "extract" not in [r for r in rows if rows[r]["state"] == "beaten"]


def test_a_screen_never_decides_a_lane(tmp_path):
    """A screen answers whether it ran; a measurement answers whether it is
    better. Ranking one against the other compares two different exams."""
    receipt(tmp_path, "screened", "extract",
            {"local-large": {"pass_rate": 0.1, "median_s": 1.0},
             "q3-4b": {"pass_rate": 1.0, "median_s": 2.0}}, tier="screen")
    assert winners.beaten_in(tmp_path) == {}


def test_a_workflow_is_not_a_model_default(tmp_path):
    """`repair:local-large` can win a lane on merit and is still not something
    a --model default can be set to."""
    receipt(tmp_path, "svg", "svg",
            {"local-large": {"pass_rate": 0.6, "median_s": 5.0},
             "repair/local-large": {"pass_rate": 1.0, "median_s": 8.0}})
    assert winners.beaten_in(tmp_path)["svg"]["candidate"] == "local-large"


def test_the_faster_of_two_equals_wins(tmp_path):
    receipt(tmp_path, "svg", "svg",
            {"local-large": {"pass_rate": 1.0, "median_s": 9.0},
             "q3-4b": {"pass_rate": 1.0, "median_s": 2.0}})
    assert winners.beaten_in(tmp_path)["svg"]["candidate"] == "q3-4b"


# --- the constant and the receipt key are different notations -------------

def test_an_engine_spec_and_its_receipt_key_are_the_same_thing():
    """`:` and `/` are one separator in two notations."""
    assert winners.matches("mflux:flux2-klein-4b", "mflux/flux2-klein-4b",
                           "engine") == "exact"


def test_a_quantisation_is_reported_rather_than_smoothed_over():
    """`flux2-klein-4b` and `flux2-klein-4b-q8` are different artifacts. The
    engine hardcodes quantize=8 and the candidate name does not carry it, so
    the constant names something no run here has produced."""
    assert winners.matches("mflux:flux2-klein-4b", "mflux/flux2-klein-4b-q8",
                           "engine") == "quantised"


def test_a_speech_default_and_its_receipt_key_are_trimmed_from_opposite_ends():
    """A default is a HuggingFace id carrying an org prefix; a tts receipt key
    is a model and the voice it used."""
    assert winners.matches("mlx-community/Kokoro-82M-bf16",
                           "Kokoro-82M-bf16/af_sky", "speech") == "exact"
    assert winners.matches("mlx-community/parakeet-tdt-0.6b-v2",
                           "parakeet-tdt-0.6b-v2", "speech") == "exact"


def test_a_different_version_is_not_the_same_model():
    """v2 and v3 are two candidates, and the newer one measured worse here."""
    assert winners.matches("mlx-community/parakeet-tdt-0.6b-v2",
                           "parakeet-tdt-0.6b-v3", "speech") == ""


def test_every_typed_default_declares_which_family_it_belongs_to():
    assert set(winners.typed()) <= set(winners.FAMILIES)
    assert set(winners.typed()) == {"svg", "web", "code", "extract", "image",
                                    "video", "tts", "stt"}


# --- the lane's own metric decides, not a statistic blind to it -----------

def test_the_lanes_metric_beats_pass_rate_and_latency(tmp_path, monkeypatch):
    """NOT HYPOTHETICAL. Ranking on pass rate and then latency reported that
    parakeet-ctc had beaten parakeet-tdt-v2: both passed 300 of 300 and ctc has
    the faster median, so on those two statistics ctc wins and on WER -- the
    one the lane is about -- it loses."""
    pin(monkeypatch, stt="mlx-community/parakeet-tdt-0.6b-v2")
    receipt(tmp_path, "stt", "stt", {
        "parakeet-tdt-0.6b-v2": {"pass_rate": 1.0, "median_s": 0.135,
                                 "total": 300, "metrics": {"wer": 0.0162}},
        "parakeet-ctc-0.6b": {"pass_rate": 1.0, "median_s": 0.116,
                              "total": 300, "metrics": {"wer": 0.0225}}})
    got = winners.beaten_in(tmp_path)["stt"]
    assert got["candidate"] == "parakeet-tdt-0.6b-v2"


def test_a_higher_is_better_metric_runs_the_other_way(tmp_path):
    receipt(tmp_path, "img", "image", {
        "mflux/a": {"pass_rate": 1.0, "median_s": 1.0,
                    "metrics": {"adherence": 0.9}},
        "mflux:flux2-klein-4b": {"pass_rate": 1.0, "median_s": 1.0,
                                 "metrics": {"adherence": 0.4}}})
    assert winners.beaten_in(tmp_path)["image"]["candidate"] == "mflux/a"


def test_a_neutral_metric_is_never_ranked_on(tmp_path):
    """`ink` as higher-is-better once crowned the worst candidate in the svg
    lane, because one big filled blob marks 85% of a canvas."""
    receipt(tmp_path, "svg", "svg", {
        "local-large": {"pass_rate": 1.0, "median_s": 1.0,
                        "metrics": {"ink": 0.02}},
        "q3-4b": {"pass_rate": 1.0, "median_s": 2.0,
                  "metrics": {"ink": 0.85}}})
    assert winners.beaten_in(tmp_path)["svg"]["candidate"] == "local-large"


def test_passing_less_often_loses_whatever_the_metric_says(tmp_path,
                                                          monkeypatch):
    """A model that fails half the cases and scores well on the rest scored on
    a different, easier subset."""
    pin(monkeypatch, stt="mlx-community/parakeet-tdt-0.6b-v2")
    # A receipt key is a model, or a model and the voice it used -- never the
    # org-prefixed id the default carries.
    receipt(tmp_path, "stt", "stt", {
        "parakeet-tdt-0.6b-v2": {"pass_rate": 1.0, "median_s": 1.0,
                                 "metrics": {"wer": 0.05}},
        "other": {"pass_rate": 0.5, "median_s": 1.0,
                  "metrics": {"wer": 0.001}}})
    got = winners.beaten_in(tmp_path)["stt"]
    assert got["candidate"] == "parakeet-tdt-0.6b-v2"


def test_a_speech_default_is_whichever_this_platform_would_actually_use():
    """The branch is deliberate and belongs in ONE test rather than being
    assumed by several: comparing a Mac's receipts against a Windows constant
    would be the accelerator mistake in another costume."""
    import sys
    from harness import audio
    assert winners.typed()["stt"] == audio.DEFAULT_STT_MODEL
    assert winners.typed()["tts"] == audio.DEFAULT_TTS_MODEL
    if sys.platform != "win32":
        assert "parakeet" in winners.typed()["stt"]
