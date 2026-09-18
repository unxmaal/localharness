"""Issue #90 follow-up: comparing two runs that differ on one axis."""
import json

import pytest

from evals.core import Receipt
from evals.run import compare_runs
from harness import paired


def rows(candidate, verdicts):
    return [{"candidate": candidate, "case_id": f"c{i}", "passed": v}
            for i, v in enumerate(verdicts)]


def test_no_discordant_cells_is_not_a_difference():
    """n=0 must be p=1.0. Returning 0.0 would call two identical runs
    different, which is the failure this whole comparison exists to avoid."""
    assert paired.sign_test(0, 0) == 1.0


def test_an_entirely_one_sided_change_is_significant():
    assert paired.sign_test(0, 7) < 0.05


def test_a_mixed_change_is_not():
    """5 lost against 2 gained is the real #90 result. It looks like a trend
    and is p=0.45."""
    assert paired.sign_test(2, 7) == pytest.approx(0.4531, abs=0.001)


def test_concordant_cells_do_not_dilute_the_test():
    """McNemar uses discordant pairs only. Adding cells that did not change
    must not move p, or a long run of identical cases would bury a real
    effect."""
    few = paired.Cell("m", lost=0, gained=5, unchanged=2)
    many = paired.Cell("m", lost=0, gained=5, unchanged=2000)
    assert few.p == many.p


def test_repeats_of_one_case_are_separate_cells():
    """Three repetitions are three observations. Collapsing them to a single
    per-case verdict would throw away the variance temperature introduces."""
    before = rows("m", [True, True, True])
    after = rows("m", [True, False, False])
    got = paired.cells(before, after)[0]
    assert (got.lost, got.gained, got.unchanged) == (2, 0, 1)


def test_a_candidate_in_only_one_run_is_skipped():
    """Comparing a candidate against nothing is not a paired comparison."""
    got = paired.cells(rows("m", [True]), rows("other", [True]))
    assert got == []


def receipt(**kw):
    base = dict(modality="code", case_ids=["c0"], tier="measure", repeat=1,
                gateway="http://gw", sampling={"code": {"temperature": 0.2}})
    base.update(kw)
    return Receipt(**base)


def write(tmp_path, name, receipt_obj, row_list):
    p = tmp_path / name
    p.write_text(json.dumps({
        "receipt": {"modality": receipt_obj.modality,
                    "case_ids": list(receipt_obj.case_ids),
                    "repeat": receipt_obj.repeat,
                    "sampling": receipt_obj.sampling,
                    "gateway": receipt_obj.gateway,
                    "tier": receipt_obj.tier,
                    "accelerator": receipt_obj.accelerator},
        "summary": {}, "rows": row_list}), encoding="utf-8")
    return str(p)


def test_a_sweep_over_one_axis_is_reported(tmp_path, capsys):
    a = write(tmp_path, "a.json", receipt(), rows("m", [True, True]))
    b = write(tmp_path, "b.json",
              receipt(sampling={"code": {"temperature": 0.7}}),
              rows("m", [True, False]))
    assert compare_runs([a, b], across="sampling") == 0
    out = capsys.readouterr().out
    assert "local" not in out
    assert "no evidence" in out or "differs" in out


def test_a_sweep_that_varies_two_things_is_refused(tmp_path, capsys):
    """The integrity of the whole exception. A run differing on sampling AND
    the accelerator cannot say which one moved the result."""
    a = write(tmp_path, "a.json", receipt(), rows("m", [True]))
    b = write(tmp_path, "b.json",
              receipt(sampling={"code": {"temperature": 0.7}},
                      accelerator="other"),
              rows("m", [False]))
    assert compare_runs([a, b], across="sampling") == 1
    assert "accelerator" in capsys.readouterr().out


def test_a_sweep_where_nothing_varied_is_refused(tmp_path, capsys):
    """Naming an axis the runs agree on would report 'no evidence' for a
    comparison that never happened."""
    a = write(tmp_path, "a.json", receipt(), rows("m", [True]))
    b = write(tmp_path, "b.json", receipt(), rows("m", [False]))
    assert compare_runs([a, b], across="sampling") == 1
    assert "nothing to sweep" in capsys.readouterr().out


def test_an_unknown_axis_is_refused(tmp_path, capsys):
    a = write(tmp_path, "a.json", receipt(), rows("m", [True]))
    b = write(tmp_path, "b.json", receipt(), rows("m", [False]))
    assert compare_runs([a, b], across="nonsense") == 1
    assert "unknown axis" in capsys.readouterr().out


def test_without_across_the_refusal_still_stands(tmp_path, capsys):
    """--across is a named exception, not a weakening. The default path must
    still refuse two sampling settings."""
    a = write(tmp_path, "a.json", receipt(), rows("m", [True]))
    b = write(tmp_path, "b.json",
              receipt(sampling={"code": {"temperature": 0.7}}),
              rows("m", [False]))
    assert compare_runs([a, b]) == 1
    assert "REFUSED" in capsys.readouterr().out
