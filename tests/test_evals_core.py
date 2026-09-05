"""The harness itself: loading cases, running candidates, scoring comparably.

The whole point is answering "which candidate should I use" in twenty minutes
without editing code, so the contract under test is: a case file plus a
candidate name produces a comparable row.
"""
import json

import pytest
import yaml

from evals.core import Case, Result, load_cases, score, summarize


def write_case(d, name, **over):
    body = {"id": name, "modality": "svg",
            "prompt": "Draw a red circle.",
            "assert": {"min_shapes": 1}}
    body.update(over)
    p = d / f"{name}.yaml"
    p.write_text(yaml.safe_dump(body))
    return p


# ---- case loading ----------------------------------------------------------

def test_loads_cases_from_a_directory(tmp_path):
    write_case(tmp_path, "circle")
    write_case(tmp_path, "square", prompt="Draw a square.")
    cases = load_cases(tmp_path)
    assert {c.id for c in cases} == {"circle", "square"}
    assert all(isinstance(c, Case) for c in cases)


def test_cases_are_ordered_deterministically(tmp_path):
    for n in ("c", "a", "b"):
        write_case(tmp_path, n)
    assert [c.id for c in load_cases(tmp_path)] == ["a", "b", "c"]


def test_malformed_case_names_the_file(tmp_path):
    """A broken case must not fail anonymously in a 50-case run."""
    (tmp_path / "bad.yaml").write_text("id: bad\nmodality: svg\n")  # no prompt
    with pytest.raises(ValueError, match="bad.yaml"):
        load_cases(tmp_path)


def test_unknown_modality_is_rejected_at_load(tmp_path):
    """Fail before spending model time, not after."""
    write_case(tmp_path, "x", modality="telepathy")
    with pytest.raises(ValueError, match="telepathy"):
        load_cases(tmp_path)


# ---- scoring ---------------------------------------------------------------

GOOD_SVG = ('<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 2 2">'
            '<circle cx="1" cy="1" r="1"/></svg>')


def test_score_passes_a_good_artifact(tmp_path):
    case = load_cases_one(tmp_path)
    r = score(case, GOOD_SVG)
    assert r.passed
    assert r.detail == ""


def test_score_fails_an_unparseable_artifact(tmp_path):
    case = load_cases_one(tmp_path)
    r = score(case, "<svg><circle</svg>")
    assert not r.passed
    assert "parse" in r.detail.lower()


def test_min_shapes_assertion_is_enforced(tmp_path):
    case = load_cases_one(tmp_path, assert_={"min_shapes": 3})
    r = score(case, GOOD_SVG)
    assert not r.passed
    assert "3" in r.detail


def test_must_contain_assertion(tmp_path):
    case = load_cases_one(tmp_path, assert_={"must_contain": ["circle"]})
    assert score(case, GOOD_SVG).passed
    case2 = load_cases_one(tmp_path, assert_={"must_contain": ["polygon"]})
    assert not score(case2, GOOD_SVG).passed


def load_cases_one(tmp_path, **over):
    a = over.pop("assert_", {"min_shapes": 1})
    write_case(tmp_path, "one", **{"assert": a, **over})
    return load_cases(tmp_path)[0]


# ---- summarizing across candidates ----------------------------------------

def rows():
    return [
        Result("circle", "fast", True, 1.0, 900, ""),
        Result("square", "fast", False, 1.2, 900, "draws nothing"),
        Result("circle", "big", True, 8.0, 4200, ""),
        Result("square", "big", True, 9.0, 4200, ""),
    ]


def test_summary_groups_by_candidate():
    s = summarize(rows())
    assert s["fast"]["passed"] == 1 and s["fast"]["total"] == 2
    assert s["big"]["passed"] == 2 and s["big"]["total"] == 2


def test_summary_reports_median_not_mean_latency():
    """One cold model load must not decide which candidate looks fastest."""
    s = summarize([
        Result("a", "x", True, 1.0, 0, ""),
        Result("b", "x", True, 1.0, 0, ""),
        Result("c", "x", True, 60.0, 0, ""),   # cold start
    ])
    assert s["x"]["median_s"] == 1.0


def test_summary_is_json_serializable():
    json.dumps(summarize(rows()))


def test_summary_of_nothing_is_empty_not_a_crash():
    assert summarize([]) == {}
