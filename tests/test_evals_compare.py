"""Issues #8 and #57: is a difference between two candidates real?"""
import json

import pytest

from evals import compare


def rows(spec):
    """spec: {candidate: [(errors, words), ...]} -> results.json rows."""
    out = []
    for cand, cases in spec.items():
        for i, (e, w) in enumerate(cases):
            out.append({"candidate": cand, "case_id": f"c{i}",
                        "metrics": {"wer": e / w, "wer_errors": e,
                                    "wer_words": w}})
    return out


def test_a_corpus_rate_is_not_the_mean_of_per_case_rates():
    """A rate is a ratio, and averaging ratios weights a two-word clip like a
    forty-word one. RULE #186, in a test this time."""
    by, cases = compare.paired(rows({"a": [(1, 2), (1, 40)]}), "wer")
    assert compare.rate(by, "a", cases) == pytest.approx(2 / 42)


def test_a_clear_difference_is_separable():
    data = rows({"good": [(0, 10)] * 40, "bad": [(3, 10)] * 40})
    by, cases = compare.paired(data, "wer")
    got = compare.interval(by, cases, "bad", "good", resamples=300)
    assert got["separable"] and got["difference"] > 0
    assert compare.verdict(got, "wer") == "worse"


def test_a_difference_inside_the_noise_is_reported_as_not_separable():
    """The failure this exists to prevent: two point estimates that differ,
    where the gap is noise. n=40 called a significant loss a tie AND a 1.43x
    gap 2x, in the same run."""
    data = rows({"a": [(1, 10), (0, 10), (2, 10), (0, 10)],
                 "b": [(0, 10), (1, 10), (1, 10), (1, 10)]})
    by, cases = compare.paired(data, "wer")
    got = compare.interval(by, cases, "b", "a", resamples=500)
    assert not got["separable"]
    assert compare.verdict(got, "wer") == "not separable"


def test_the_comparison_is_paired_on_the_same_cases():
    """Pairing removes case difficulty, which is the dominant variance: every
    speech model shares the same worst clip."""
    data = rows({"a": [(0, 10)] * 3, "b": [(0, 10)] * 3})
    data.append({"candidate": "a", "case_id": "extra",
                 "metrics": {"wer": 1.0, "wer_errors": 9, "wer_words": 9}})
    by, cases = compare.paired(data, "wer")
    assert "extra" not in cases and len(cases) == 3


def test_the_same_seed_gives_the_same_interval():
    """A confidence interval that moves between runs is not quotable."""
    by, cases = compare.paired(rows({"a": [(1, 10)] * 20,
                                     "b": [(2, 10)] * 20}), "wer")
    kw = dict(resamples=200, seed=7)
    assert (compare.interval(by, cases, "b", "a", **kw)
            == compare.interval(by, cases, "b", "a", **kw))


def test_a_metric_where_higher_is_better_reverses_the_verdict():
    """`worse` has to mean worse, not merely larger. adherence is a real
    higher-is-better metric here; wer is not."""
    row = {"difference": 0.2, "low": 0.1, "high": 0.3, "separable": True}
    assert compare.verdict(row, "wer") == "worse"
    assert compare.verdict(row, "adherence") == "BETTER"


def test_a_run_with_no_such_metric_says_so_rather_than_dividing_by_zero(tmp_path):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"rows": [{"candidate": "a", "case_id": "c",
                                       "metrics": {"seconds": 1.0}}]}))
    assert compare.main([str(p), "--metric", "wer"]) == 1


def test_an_unknown_baseline_names_the_candidates_that_exist(tmp_path, capsys):
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"rows": rows({"a": [(1, 10)], "b": [(2, 10)]})}))
    assert compare.main([str(p), "--baseline", "nope"]) == 1
    assert "a, b" in capsys.readouterr().out


def test_a_run_directory_is_accepted_as_well_as_a_file(tmp_path):
    (tmp_path / "results.json").write_text(
        json.dumps({"rows": rows({"a": [(1, 10)]})}))
    assert compare.load(tmp_path)["rows"]


def test_a_neutral_metric_is_not_a_verdict():
    """svg_bytes is reported and never ranked on: ranking on it would crown
    the blank document. A difference in one is not better or worse."""
    row = {"difference": 0.2, "low": 0.1, "high": 0.3, "separable": True}
    assert compare.verdict(row, "svg_bytes") == "not ranked on"


def test_the_absolute_rate_is_reported_with_its_sample(capsys, tmp_path):
    """The same comparison over a different 300 clips moved the baseline 17%
    while every ranking held. An absolute quoted without its sample travels
    into prose as a property of the model. Issue #88."""
    p = tmp_path / "results.json"
    p.write_text(json.dumps({"rows": rows({"a": [(1, 10)] * 5,
                                           "b": [(2, 10)] * 5})}))
    compare.main([str(p), "--baseline", "a", "--resamples", "50"])
    out = capsys.readouterr().out
    assert "ON THIS SAMPLE" in out and "corpus " in out
    assert "quote the difference" in out
