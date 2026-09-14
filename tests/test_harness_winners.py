"""The typed lane defaults against the receipts that chose them."""
import json

from harness import winners


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


def test_this_repo_has_a_typed_default_for_every_lane_it_ships():
    typed = winners.typed()
    assert set(typed) == {"svg", "web", "code", "extract"}
    assert all(typed.values())
