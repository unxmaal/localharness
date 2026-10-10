"""The claims lane: schema-enforced claim extraction scored on matched human reviews, cases local only. #654."""
import json
from collections import Counter
import shutil
from pathlib import Path

import pytest

from harness.checks import claims as C

FIXTURES = Path(__file__).parent / "fixtures" / "claims"
EXPORT = FIXTURES / "claims.jsonl"
NEGATIVES = FIXTURES / "claims-negatives.jsonl"


def _line(path=EXPORT) -> dict:
    return json.loads(path.read_text(encoding="utf-8").splitlines()[0])


def _reply(*claims) -> str:
    return json.dumps({"c": [list(c) for c in claims]})


def _good(row):
    return [(r["user"], r["claim"], r["refs"]) for r in row["reviews"] if r["verdict"] == "good"]


def _check(reply, row=None):
    row = row or _line()
    return C.check(reply, row["schema"], row["reviews"], bool(row.get("expect_empty")))


# --- matching --------------------------------------------------------------------

def test_similarity_counts_shared_words_with_their_multiplicity():
    assert C.dice("a a b", "a a c") == pytest.approx(2 * 2 / 6)
    assert C.similarity("fan fan rail", "fan fan pin") == pytest.approx(2 * 2 / 6)
    assert C.similarity("", "") == 0.0 == C.dice("", "")
    assert C.similarity("Jumper J7!", "jumper j7") == 1.0


def test_a_claim_matches_only_its_own_user_and_an_overlapping_ref():
    reviewed = [{"user": "u1", "claim": "fan runs at twelve volts", "refs": [3], "verdict": "good"}]
    assert C.match([("u1", "fan runs at twelve volts", [2, 3])], reviewed, 0.5) == [(0, 0, 1.0)]
    assert C.match([("u2", "fan runs at twelve volts", [3])], reviewed, 0.5) == []
    assert C.match([("u1", "fan runs at twelve volts", [4])], reviewed, 0.5) == []
    assert C.match([("u1", "something else entirely", [3])], reviewed, 0.5) == []


def test_matching_is_one_to_one_and_ignores_reply_order():
    reviewed = [{"user": "u", "claim": "alpha beta gamma", "refs": [1], "verdict": "good"}]
    emitted = [("u", "alpha beta", [1]), ("u", "alpha beta gamma", [1])]
    assert C.match(emitted, reviewed, 0.5) == [(1, 0, 1.0)]


# --- the reply schema --------------------------------------------------------------

def test_the_reviewed_good_claims_pass_and_report_full_recall():
    row = _line()
    got = _check(_reply(*reversed(_good(row))))
    assert got.ok, got.reason
    m = got.metrics
    assert (m["claims_good_found_legacy"], m["claims_good_total_legacy"],
            m["claims_bad_matched_legacy"]) == (4, 4, 0)
    assert m["claims_recall_legacy"] == 1.0 and m["claims_precision_legacy"] == 1.0
    assert m["claims_schema_valid"] == 1
    assert m["claims_judge_calibrated"] == 0


@pytest.mark.parametrize("reply", [
    "Here are the claims: none",
    json.dumps({"c": [["member-a1b2", "x" * 221, [3]]]}),
    json.dumps({"c": [["member-a1b2", "a claim", []]]}),
    json.dumps({"c": [["member-a1b2", "a claim"]]}),
    json.dumps({"c": [], "extra": 1}),
])
def test_a_reply_off_the_schema_fails_hard(reply):
    got = _check(reply)
    assert not got.ok and "schema" in got.reason
    assert got.metrics["claims_schema_valid"] == 0


def test_matching_a_made_up_or_wrong_claim_fails_the_case():
    row = _line()
    bad = [(r["user"], r["claim"], r["refs"]) for r in row["reviews"] if r["verdict"] == "made_up"]
    got = _check(_reply(*_good(row), *bad))
    assert not got.ok and "made_up" in got.reason
    assert got.metrics["claims_bad_matched_legacy"] == 1


def test_missing_most_good_claims_fails_on_recall():
    row = _line()
    got = _check(_reply(_good(row)[0]))
    assert not got.ok and "recall" in got.reason
    assert got.metrics["claims_recall_legacy"] == 0.25


def test_an_unreviewed_claim_is_counted_and_does_not_fail():
    row = _line()
    got = _check(_reply(*_good(row), ("member-c3d4", "Totally novel unreviewed statement.", [8])))
    assert got.ok, got.reason
    assert got.metrics["claims_unreviewed_legacy"] == 1 and got.metrics["claims_emitted"] == 5


def test_a_failure_reason_never_quotes_case_or_reply_text():
    row = _line()
    bad = [(r["user"], r["claim"], r["refs"]) for r in row["reviews"] if r["verdict"] == "wrong"]
    secret = "zebra quokka narwhal"
    for reply in (_reply(*bad), secret, _reply(("member-a1b2", secret, [1]))):
        got = _check(reply)
        assert not got.ok
        assert secret not in got.reason and bad[0][1] not in got.reason
        assert "member-" not in got.reason


# --- expected-empty negatives ---------------------------------------------------------

def test_an_expect_empty_case_passes_only_with_no_claims_and_reports_apart():
    row = _line(NEGATIVES)
    ok = _check(_reply(), row)
    assert ok.ok, ok.reason
    assert ok.metrics == {"claims_schema_valid": 1, "claims_emitted": 0,
                          "claims_empty_kept": 1, "claims_empty_cases": 1, "claims_empty_rate": 1.0,
                          "claims_judge_calibrated": 0, "claims_matcher_version": C.MATCHER_VERSION}
    got = _check(_reply(("member-a1b2", "Regattas need folding chairs.", [3])), row)
    assert not got.ok and "expected no claims" in got.reason
    assert got.metrics["claims_empty_kept"] == 0 and "claims_recall_legacy" not in got.metrics


def test_the_scorer_separates_a_perfect_responder_from_constant_and_echo_responders():
    """Negative control: emitting nothing, or echoing every reviewed claim, must not pass."""
    row = _line()
    everything = [(r["user"], r["claim"], r["refs"]) for r in row["reviews"]]
    assert _check(_reply(*_good(row))).ok
    assert not _check(_reply()).ok
    assert not _check(_reply(*everything)).ok


def test_the_summary_pools_reviewed_and_empty_cases_into_separate_ratios():
    from evals.core import Result, summarize
    good, neg = _line(), _line(NEGATIVES)
    rows = [Result("c1", "m", True, 0.1, 0, "", metrics=_check(_reply(*_good(good))).metrics),
            Result("c2", "m", False, 0.1, 0, "", metrics=_check(_reply(_good(good)[0])).metrics),
            Result("n1", "m", True, 0.1, 0, "", metrics=_check(_reply(), neg).metrics)]
    m = summarize(rows)["m"]["metrics"]
    assert m["claims_recall_legacy"] == pytest.approx(5 / 8)
    assert m["claims_empty_rate"] == 1.0
    assert m["claims_precision_legacy"] == 1.0


# --- case files --------------------------------------------------------------------------

def _yaml_case(tmp_path, **over) -> Path:
    import yaml
    row = _line()
    case = {"id": "claims-x", "modality": "claims", "prompt": row["transcript"],
            "params": {"system": row["system"], "schema": row["schema"]},
            "assert": {"reviews": row["reviews"]}}
    for k, v in over.items():
        case[k] = {**case.get(k, {}), **v} if isinstance(v, dict) else v
    path = tmp_path / "c.yaml"
    path.write_text(yaml.safe_dump(case), encoding="utf-8")
    return tmp_path


def test_a_claims_case_loads_with_its_system_and_schema(tmp_path):
    from evals.core import load_cases
    (case,) = load_cases(_yaml_case(tmp_path))
    assert case.modality == "claims" and case.params["system"].startswith("You read")
    assert case.private is False


@pytest.mark.parametrize("over, why", [
    ({"assert": {"reviews": [{"user": "u", "claim": "c", "refs": [1], "verdict": "great"}]}}, "verdict"),
    ({"assert": {"reviews": [{"user": "u", "claim": "c", "refs": [], "verdict": "good"}]}}, "refs"),
    ({"assert": {"reviews": []}}, "expect_empty"),
    ({"assert": {"expect_empty": True}}, "expect_empty"),
    ({"params": {"schema": None}}, "schema"),
    ({"params": {"system": ""}}, "system"),
])
def test_a_malformed_claims_case_is_refused_at_load(tmp_path, over, why):
    from evals.core import load_cases
    with pytest.raises(ValueError, match=why):
        load_cases(_yaml_case(tmp_path, **over))


# --- the importer -----------------------------------------------------------------------

def test_the_importer_writes_local_cases_with_negatives_carried(tmp_path):
    from evals import claims_import
    from evals.core import load_cases
    out = tmp_path / "home" / "cases" / "claims"
    written = claims_import.run(EXPORT, NEGATIVES, out)
    assert sorted(p.name for p in written) == ["claims-synthetic-1.yaml", "claims-synthetic-neg-1.yaml"]
    cases = {c.id: c for c in load_cases(out, private=True)}
    neg = cases["claims-synthetic-neg-1"]
    assert neg.assertions == {"reviews": [], "expect_empty": True, "basis": "human_irrelevant",
                              "origin": "synthetic"}
    assert cases["claims-synthetic-1"].assertions.get("basis") is None
    assert all(c.private for c in cases.values())
    assert cases["claims-synthetic-1"].prompt == _line()["transcript"]


def test_the_importer_runs_without_a_negatives_file(tmp_path):
    from evals import claims_import
    written = claims_import.run(EXPORT, tmp_path / "absent.jsonl", tmp_path / "out")
    assert [p.name for p in written] == ["claims-synthetic-1.yaml"]


def test_a_reimport_replaces_cases_the_export_dropped(tmp_path):
    from evals import claims_import
    out = tmp_path / "out"
    claims_import.run(EXPORT, NEGATIVES, out)
    claims_import.run(EXPORT, tmp_path / "absent.jsonl", out)
    assert sorted(p.name for p in out.glob("*.yaml")) == ["claims-synthetic-1.yaml"]


def test_the_importer_refuses_to_write_inside_a_git_tree(tmp_path):
    from evals import claims_import
    inside = Path(__file__).resolve().parent / "fixtures" / "claims-out-never"
    with pytest.raises(claims_import.Refused, match="git"):
        claims_import.run(EXPORT, NEGATIVES, inside)
    assert not inside.exists()
    assert claims_import.run(EXPORT, NEGATIVES, tmp_path / "fine")


def test_the_importer_defaults_read_and_write_under_the_home(tmp_path, monkeypatch):
    from evals import claims_import
    from harness import paths
    assert claims_import.default_export() == paths.home() / "import" / "claims.jsonl"
    assert claims_import.default_negatives() == paths.home() / "import" / "claims-negatives.jsonl"
    assert claims_import.default_out() == paths.home() / "cases" / "claims"


def test_the_importer_cli_reports_counts(tmp_path, capsys):
    from evals import claims_import
    assert claims_import.main([str(EXPORT), "--negatives", str(NEGATIVES),
                               "--out", str(tmp_path / "o")]) == 0
    said = capsys.readouterr().out
    assert "1 reviewed" in said and "1 expect_empty" in said and "7 reviews" in said


# --- local-only cases: suite, holdout, digest --------------------------------------------

@pytest.fixture
def imported():
    from evals import claims_import
    return claims_import.run(EXPORT, NEGATIVES, claims_import.default_out())


def test_the_suite_adds_local_cases_as_private(imported):
    from evals import core
    suite = core.load_suite()
    mine = [c for c in suite if c.modality == "claims"]
    assert sorted(c.id for c in mine) == ["claims-synthetic-1", "claims-synthetic-neg-1"]
    assert all(c.private for c in mine)
    assert not any(c.private for c in suite if c.modality != "claims")


def test_the_suite_without_local_cases_is_the_shipped_tree():
    from evals import core
    assert not core.local_root().exists()
    assert len(core.load_suite()) == len(core.load_cases(core.SHIPPED))


def test_private_does_not_change_a_case_digest_or_its_side(imported):
    from dataclasses import replace

    from evals import core
    from harness import holdout
    local = core.load_cases(core.local_root(), private=True)
    for c in local:
        assert core.case_digest(c) == core.case_digest(replace(c, private=False))
    split = holdout.for_lane("claims")
    assert set(split.dev) | set(split.holdout) == {c.id for c in local}
    assert split == holdout.assign("claims", local)


def test_evals_run_reads_local_cases_by_default(imported, monkeypatch):
    from evals import run
    args = run.parse_args(["--modality", "claims", "--candidates", "eval-7b"])
    got = run.load_selected(args)
    assert sorted(c.id for c in got) == ["claims-synthetic-1", "claims-synthetic-neg-1"]


# --- the leak guards -----------------------------------------------------------------------

def test_leaks_names_a_file_holding_private_text_and_a_copied_case(tmp_path, imported):
    from evals import core, private
    cases = core.load_cases(core.local_root(), private=True)
    tree = tmp_path / "tree"
    (tree / "clean").mkdir(parents=True)
    (tree / "clean" / "ok.txt").write_text("nothing private here\n", encoding="utf-8")
    assert private.leaks([tree], cases) == []
    line = _line()["transcript"].splitlines()[2].split(": ", 1)[1]
    (tree / "notes.md").write_text(f"pasted: {line}\n", encoding="utf-8")
    shutil.copy(imported[0], tree / "copied.yaml")
    shutil.copy(imported[0], tree / "renamed.txt")
    got = private.leaks([tree], cases)
    assert {Path(p).name for p, _ in got} == {"notes.md", "copied.yaml", "renamed.txt"}


def test_a_case_digest_match_is_a_leak_even_when_the_text_is_reflowed(tmp_path, imported):
    import yaml

    from evals import core, private
    cases = core.load_cases(core.local_root(), private=True)
    raw = yaml.safe_load(imported[0].read_text(encoding="utf-8"))
    tree = tmp_path / "cases"
    tree.mkdir()
    (tree / "x.yaml").write_text(yaml.safe_dump(raw, width=40), encoding="utf-8")
    got = private.leaks([tree], cases)
    assert [why for _, why in got] and all("digest" in why or "text" in why for _, why in got)


def test_no_shipped_file_matches_a_private_case_on_this_machine():
    """The real local cases, read only; on a machine without them this guards nothing and says so."""
    from evals import core, private
    from harness import paths
    root = paths.DEFAULT_HOME / "cases"
    cases = core.load_cases(root, private=True) if root.is_dir() else []
    repo = Path(__file__).resolve().parents[1]
    assert private.leaks([repo / "evals" / "cases", repo / "tests"], cases) == []


def test_publish_refuses_an_export_carrying_private_case_text(imported):
    from harness import publish
    clean = {"lanes": [{"lane": "claims", "rows": [{"candidate": "eval-7b", "passed": 1}]}]}
    assert publish.checked(clean) == clean
    line = _line()["reviews"][0]["claim"]
    with pytest.raises(publish.ExportRefused, match="private case"):
        publish.checked({**clean, "note": line})


def test_publish_refuses_private_text_however_it_is_escaped(imported):
    from harness import publish
    line = _line()["transcript"].splitlines()[4].split(": ", 1)[1]
    with pytest.raises(publish.ExportRefused, match="private case"):
        publish.checked({"rows": [{"detail": f"model said {line!r}"}]})


# --- the lane --------------------------------------------------------------------------

def test_claims_is_a_wanted_schema_lane_served_by_qwen25_7b_on_llama_server():
    from harness import completion, gateway, lanes, winners
    assert "claims" in lanes.WANTED and "claims" in lanes.GGUF_SERVED
    assert "claims" in gateway.TEXT_LANES and "claims" in gateway.SCHEMA_LANES
    assert winners.typed()["claims"] == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert completion.budget("claims") == 400
    assert completion.SAMPLING["claims"] == {"temperature": 0.0}


def _claims_case():
    from evals.core import Case
    row = _line()
    return Case(id="claims-x", modality="claims", prompt=row["transcript"],
                params={"system": row["system"], "schema": row["schema"]},
                assertions={"reviews": row["reviews"]})


def test_the_runner_asks_with_the_case_system_and_its_schema(monkeypatch):
    from evals.runners.text import CompletionRunner
    from harness import completion, serving
    sent = []

    def fake(prompt, **kw):
        sent.append((prompt, kw))
        return completion.Completion(_reply(*_good(_line())), {"completion_tokens": 40})
    monkeypatch.setattr(completion, "complete_full", fake)
    monkeypatch.setattr(serving, "drops_schema", lambda base: False)
    r = CompletionRunner("http://127.0.0.1:4000", "eval-7b").run(_claims_case())
    assert r.passed, r.detail
    prompt, kw = sent[0]
    assert prompt == _line()["transcript"] and kw["system"] == _line()["system"]
    assert kw["max_tokens"] == 400 and kw["modality"] == "claims"
    assert kw["response_format"] == {"type": "json_schema", "json_schema": {
        "name": "claims", "strict": True, "schema": _line()["schema"]}}


def test_a_claims_case_is_never_scored_on_mlx_lm_server(monkeypatch):
    from evals.runners.text import CompletionRunner
    from harness import completion, reasons, serving
    sent = []
    monkeypatch.setattr(completion, "complete_full", lambda *a, **k: sent.append(k))
    r = CompletionRunner(serving.MLX_URL, "mlx-community/x").run(_claims_case())
    assert not sent and not r.passed and r.failure_class == reasons.REFUSED_BY_GATEWAY


def _export_with(tmp_path, schemas) -> Path:
    rows = []
    for n, schema in enumerate(schemas):
        row = dict(_line(), id=f"synthetic-{n}", schema=schema)
        rows.append(json.dumps(row))
    path = tmp_path / "export.jsonl"
    path.write_text("\n".join(rows) + "\n", encoding="utf-8")
    return path


def _pinned_speaker(schema) -> dict:
    out = json.loads(json.dumps(schema))
    out["properties"]["c"]["items"]["prefixItems"][0]["pattern"] = "^member-[0-9a-f]{4,}$"
    return out


def _delegate_sends(monkeypatch) -> dict:
    from harness import completion, delegate
    got = {}

    def capture(prompt, **kw):
        got.update(kw)
        raise RuntimeError("sent")
    monkeypatch.setattr(completion, "complete_full", capture)
    monkeypatch.setattr(delegate, "_preflight", lambda *a, **k: None)
    with pytest.raises(RuntimeError, match="sent"):
        delegate.complete("claims", "[1] member-a1b2: the fan is 12V")
    return got


def test_the_lane_carries_no_schema_of_its_own():
    assert not hasattr(C, "SCHEMA")


def test_delegating_sends_the_imported_cases_schema_not_the_synthetic_one(tmp_path, monkeypatch):
    from evals import claims_import
    imported = _pinned_speaker(_line()["schema"])
    assert imported != _line()["schema"]
    claims_import.run(_export_with(tmp_path, [imported, imported]), None, claims_import.default_out())
    got = _delegate_sends(monkeypatch)
    assert got["response_format"] == C.response_format(imported)


def test_delegating_before_any_import_refuses_without_sending(monkeypatch):
    from harness import completion, delegate
    sent = []
    monkeypatch.setattr(completion, "complete_full", lambda *a, **k: sent.append(k))
    monkeypatch.setattr(delegate, "_preflight", lambda *a, **k: None)
    with pytest.raises(delegate.Refused, match="claims_import"):
        delegate.complete("claims", "[1] member-a1b2: the fan is 12V")
    assert not sent


def test_the_importer_writes_the_schema_once_beside_the_cases(tmp_path):
    from evals import claims_import
    out = tmp_path / "out"
    claims_import.run(EXPORT, NEGATIVES, out)
    assert json.loads((out / C.SCHEMA_FILE).read_text(encoding="utf-8")) == _line()["schema"]
    assert C.served_schema(out) == _line()["schema"]
    assert C.schema_path() == claims_import.default_out() / C.SCHEMA_FILE


def test_the_importer_refuses_an_export_whose_cases_disagree_on_the_schema(tmp_path):
    from evals import claims_import
    schema = _line()["schema"]
    out = tmp_path / "out"
    assert claims_import.run(_export_with(tmp_path, [schema, schema]), None, out)
    with pytest.raises(claims_import.Refused, match="schema"):
        claims_import.run(_export_with(tmp_path, [schema, _pinned_speaker(schema)]), None, out)
    assert json.loads((out / C.SCHEMA_FILE).read_text(encoding="utf-8")) == schema


def test_the_importer_refuses_negatives_whose_schema_differs_from_the_export(tmp_path):
    from evals import claims_import
    neg = json.loads(NEGATIVES.read_text(encoding="utf-8").splitlines()[0])
    neg["schema"] = _pinned_speaker(neg["schema"])
    path = tmp_path / "neg.jsonl"
    path.write_text(json.dumps(neg) + "\n", encoding="utf-8")
    with pytest.raises(claims_import.Refused, match="schema"):
        claims_import.run(EXPORT, path, tmp_path / "out")


def test_the_claims_thresholds_are_registered_knobs_on_the_receipt():
    from harness import knobs
    got = knobs.settings("claims")
    assert got["claims_match_threshold"] == C.MATCH_THRESHOLD
    assert got["claims_min_recall"] == C.MIN_RECALL


def test_a_case_whose_reviews_hold_no_good_claim_passes_by_avoiding_the_rejected_ones():
    row = _line()
    reviews = [r for r in row["reviews"] if r["verdict"] != "good"]
    bad = [(r["user"], r["claim"], r["refs"]) for r in reviews if r["verdict"] == "wrong"]
    assert C.check(_reply(), row["schema"], reviews).ok
    assert not C.check(_reply(*bad), row["schema"], reviews).ok
    C.validate_case(row["system"], row["schema"], reviews, False)


def test_a_recall_failure_records_the_knob_that_decided_it():
    from evals.core import score
    row = _line()
    got = _check(_reply(_good(row)[0]))
    assert got.limit == f"claims_min_recall>{C.MIN_RECALL:g}"
    case = _claims_case()
    assert score(case, _reply(_good(row)[0])).limit == got.limit
    assert _check(_reply(*_good(row))).limit == ""


def test_a_near_miss_under_the_match_threshold_records_that_knob():
    row = _line()
    good = _good(row)
    user, claim, refs = good[0]
    near = (user, "Jumper J7 picks where the board boots from.", refs)
    weight = C.weights([r["claim"] for r in row["reviews"]])
    assert C.MATCH_THRESHOLD - 0.15 <= C.similarity(near[1], claim, weight) < C.MATCH_THRESHOLD
    got = _check(_reply(near, good[1]))
    assert not got.ok and got.limit == f"claims_match_threshold>{C.MATCH_THRESHOLD:g}"


class _Echo:
    """A runner whose reply and failure detail both quote the case text."""
    candidate = "fake"

    def __init__(self, passed):
        self.passed = passed

    def run(self, case):
        from evals.core import Result
        quoted = case.prompt.splitlines()[2]
        return Result(case.id, self.candidate, self.passed, 0.1, 0,
                      "" if self.passed else f"server said {quoted}", output=f"echo {quoted}")

    def artifact(self, case, suffix):
        return f"{case.id}{suffix}"


@pytest.mark.parametrize("private_case", [True, False])
def test_a_run_keeps_no_text_of_a_private_case_in_its_rows(tmp_path, private_case):
    from dataclasses import replace

    from evals import run
    case = replace(_claims_case(), private=private_case)
    rows = []
    run._run_candidate(run.parse_args([]), "fake", _Echo(False), [case], rows, tmp_path)
    line = case.prompt.splitlines()[2]
    (row,) = rows
    if private_case:
        assert row.output is None and line not in row.detail
        assert "withheld" in row.detail
    else:
        assert line in row.output and line in row.detail
    assert Path(row.artifact_path).read_text(encoding="utf-8").endswith(line)


# --- the per-origin report -----------------------------------------------------------

def test_wilson_bounds_a_small_slice_and_an_empty_one():
    from evals import claims_report as R
    assert R.wilson(0, 0) == (0.0, 1.0)
    lo, hi = R.wilson(5, 10)
    assert lo == pytest.approx(0.2366, abs=1e-4) and hi == pytest.approx(0.7634, abs=1e-4)
    assert R.wilson(10, 10)[1] == 1.0


def _neg_case(cid, origin):
    from evals.core import Case
    return Case(id=cid, modality="claims", prompt="p", params={},
                assertions={"reviews": [], "expect_empty": True, "basis": "human_irrelevant",
                            "origin": origin}, private=True)


def test_the_report_splits_expect_empty_results_by_origin_with_intervals():
    from evals import claims_report as R
    cases = [_neg_case("n1", "undecided"), _neg_case("n2", "undecided"), _neg_case("n3", "lexicon"),
             _claims_case()]
    rows = [{"case_id": "n1", "candidate": "m", "passed": True, "metrics": {}},
            {"case_id": "n2#1", "candidate": "m", "passed": False, "metrics": {}},
            {"case_id": "n3", "candidate": "m", "passed": True, "metrics": {}},
            {"case_id": "claims-x", "candidate": "m", "passed": True,
             "metrics": {"claims_good_found": 3, "claims_good_total": 4,
                         "claims_reviewed_matched": 3, "claims_bad_matched": 0,
                         "claims_schema_valid": 1}}]
    got = R.report(rows, cases)["m"]
    assert got["empty"]["undecided"] == {"kept": 1, "n": 2, "rate": 0.5, "ci95": list(R.wilson(1, 2))}
    assert got["empty"]["lexicon"]["kept"] == 1 and got["empty"]["lexicon"]["n"] == 1
    assert got["empty"]["all"]["n"] == 3
    # A row scored before #661 carries pooled metrics: it reads as the legacy interface.
    assert got["reviewed"] == {"legacy": {"cases": 1, "passed": 1, "good_total": 4, "recall": 0.75, "precision": 1.0,
                                          "bad_matched": 0, "schema_invalid": 0}}
    text = R.render(R.report(rows, cases))
    assert "undecided" in text and "1/2" in text


def test_a_row_for_a_case_this_machine_lacks_is_counted_apart():
    from evals import claims_report as R
    got = R.report([{"case_id": "gone", "candidate": "m", "passed": True, "metrics": {}}], [])
    assert got["m"]["unknown_cases"] == 1


def test_the_report_cli_reads_a_results_file(tmp_path, imported, capsys):
    from evals import claims_report as R
    rows = [{"case_id": "claims-synthetic-neg-1", "candidate": "eval-7b", "passed": True, "metrics": {}}]
    path = tmp_path / "results.json"
    path.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    assert R.main([str(path)]) == 0
    assert "empty synthetic" in capsys.readouterr().out


def test_the_claims_reply_budget_sweeps_its_own_range_and_leaves_code_s_alone():
    from harness import knobs
    k = knobs.KNOBS["reply_budget"]
    assert k.default("claims") in k.values_for("claims") and 400 not in k.values_for("code")
    assert k.values_for("code") == k.values


# --- review interfaces (#661): each one a separate labelling function --------------------

def _tagged(row, *interfaces) -> list[dict]:
    """The row's reviews once per interface, each copy tagged with it."""
    return [{**r, "interface": i} for i in interfaces for r in row["reviews"]]


def _export_tagged(tmp_path, *interfaces) -> Path:
    row = {**_line(), "reviews": _tagged(_line(), *interfaces)}
    path = tmp_path / "claims.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    return path


def test_the_importer_carries_each_review_s_interface(tmp_path):
    from evals import claims_import
    from evals.core import load_cases
    out = tmp_path / "out"
    claims_import.run(_export_tagged(tmp_path, "cited-only", "conversation"), None, out)
    (case,) = load_cases(out, private=True)
    got = Counter(r["interface"] for r in case.assertions["reviews"])
    assert got == {"cited-only": 7, "conversation": 7}


def test_the_importer_cli_counts_reviews_per_interface(tmp_path, capsys):
    from evals import claims_import
    export = _export_tagged(tmp_path, "cited-only", "conversation")
    assert claims_import.main([str(export), "--out", str(tmp_path / "o")]) == 0
    said = capsys.readouterr().out
    assert "cited-only 7" in said and "conversation 7" in said
    assert claims_import.main([str(EXPORT), "--out", str(tmp_path / "p")]) == 0
    assert "legacy 7" in capsys.readouterr().out


def test_a_review_with_an_unknown_interface_is_refused():
    row = _line()
    reviews = [{**r, "interface": "telepathy"} for r in row["reviews"]]
    with pytest.raises(ValueError, match="interface"):
        C.validate_case(row["system"], row["schema"], reviews, False)
    C.validate_case(row["system"], row["schema"], _tagged(row, "conversation"), False)


def test_a_review_without_an_interface_reads_as_legacy():
    assert C.interface_of({"verdict": "good"}) == C.LEGACY
    assert C.interface_of({"interface": "conversation"}) == "conversation"
    got = _check(_reply(*_good(_line())))
    assert got.metrics["claims_recall_legacy"] == 1.0
    assert not any(k.endswith(("_conversation", "_cited_only")) for k in got.metrics)


def test_each_interface_is_scored_on_its_own_reviews_and_never_pooled():
    row = _line()
    reviews = _tagged(row, "cited-only", "conversation")
    got = C.check(_reply(*_good(row)), row["schema"], reviews)
    assert got.ok, got.reason
    m = got.metrics
    for slug in ("cited_only", "conversation"):
        assert (m[f"claims_good_found_{slug}"], m[f"claims_good_total_{slug}"]) == (4, 4)
        assert m[f"claims_recall_{slug}"] == 1.0 and m[f"claims_pass_{slug}"] == 1
    pooled = {"claims_recall", "claims_precision", "claims_good_found", "claims_good_total",
              "claims_bad_matched", "claims_reviewed_matched", "claims_unreviewed"}
    assert not pooled & set(m)


def test_a_case_fails_on_the_one_interface_that_fails_and_says_which():
    row = _line()
    good = _good(row)
    conversation = [{**r, "interface": "conversation"} for r in row["reviews"] if r["verdict"] == "good"]
    cited = [{**r, "interface": "cited-only"} for r in row["reviews"]]
    # cited-only misses most good claims; conversation reviews only the first.
    got = C.check(_reply(good[0]), row["schema"], conversation[:1] + cited)
    assert not got.ok and got.reason.startswith("cited-only: recall")
    assert got.metrics["claims_pass_conversation"] == 1 and got.metrics["claims_pass_cited_only"] == 0
    assert got.limit == f"claims_min_recall>{C.MIN_RECALL:g}"
    made_up = [(r["user"], r["claim"], r["refs"]) for r in row["reviews"] if r["verdict"] == "made_up"]
    got = C.check(_reply(*good, *made_up), row["schema"], conversation + cited)
    assert not got.ok and "cited-only: emitted claims a reviewer rejected: 1 made_up" in got.reason
    assert got.metrics["claims_pass_conversation"] == 1


def test_every_per_interface_metric_declares_its_direction_and_ratios_are_summed():
    from evals.core import METRIC_DIRECTION, RATIO_METRICS
    row = _line()
    got = C.check(_reply(*_good(row)), row["schema"], _tagged(row, *C.INTERFACES))
    ratios = {n for pair in RATIO_METRICS.values() for n in pair}
    assert all(k in METRIC_DIRECTION or k in ratios for k in got.metrics), sorted(got.metrics)
    for slug in ("cited_only", "conversation", "legacy"):
        assert RATIO_METRICS[f"claims_recall_{slug}"] == (f"claims_good_found_{slug}",
                                                          f"claims_good_total_{slug}")


def test_the_summary_keeps_each_interface_s_recall_apart():
    from evals.core import Result, summarize
    row = _line()
    good = _good(row)
    both = _tagged(row, "cited-only", "conversation")
    convo_only = [r for r in both if r["interface"] == "conversation"]
    rows = [Result("c1", "m", True, 0.1, 0, "",
                   metrics=C.check(_reply(*good), row["schema"], both).metrics),
            Result("c2", "m", False, 0.1, 0, "",
                   metrics=C.check(_reply(good[0]), row["schema"], convo_only).metrics)]
    m = summarize(rows)["m"]["metrics"]
    assert m["claims_recall_cited_only"] == 1.0
    assert m["claims_recall_conversation"] == pytest.approx(5 / 8)
    assert "claims_recall" not in m


def _tagged_case(cid, *interfaces):
    from dataclasses import replace
    return replace(_claims_case(), id=cid,
                   assertions={"reviews": _tagged(_line(), *interfaces)})


def test_the_report_prints_one_row_per_interface_and_no_pooled_row():
    from evals import claims_report as R
    row = _line()
    good = _good(row)
    cases = [_tagged_case("a", "cited-only", "conversation"), _tagged_case("b", "conversation")]
    rows = [{"case_id": "a", "candidate": "m", "passed": False,
             "metrics": C.check(_reply(*good), row["schema"], cases[0].assertions["reviews"]).metrics},
            {"case_id": "b", "candidate": "m", "passed": False,
             "metrics": C.check(_reply(good[0]), row["schema"], cases[1].assertions["reviews"]).metrics}]
    got = R.report(rows, cases)["m"]["reviewed"]
    assert set(got) == {"cited-only", "conversation"}
    assert got["cited-only"]["cases"] == 1 and got["cited-only"]["passed"] == 1
    assert got["conversation"] == {"cases": 2, "passed": 1, "good_total": 8, "recall": 0.625, "precision": 1.0,
                                   "bad_matched": 0, "schema_invalid": 0}
    text = R.render(R.report(rows, cases))
    assert "reviewed cited-only" in text and "reviewed conversation" in text
    assert "reviewed  " not in text


def test_a_schema_invalid_reply_counts_against_every_interface_the_case_holds():
    from evals import claims_report as R
    case = _tagged_case("a", "cited-only", "conversation")
    metrics = C.check("not json", _line()["schema"], case.assertions["reviews"]).metrics
    got = R.report([{"case_id": "a", "candidate": "m", "passed": False, "metrics": metrics}],
                   [case])["m"]["reviewed"]
    assert got["cited-only"]["schema_invalid"] == 1 and got["conversation"]["schema_invalid"] == 1
    assert got["conversation"]["passed"] == 0


def test_the_report_states_that_every_reviewed_claim_came_from_one_model():
    from evals import claims_report as R
    text = R.render(R.report([], []))
    assert R.REVIEWED_FROM == "Qwen2.5-7B-Instruct-Q4_K_M"
    assert "Qwen2.5-7B-Instruct-Q4_K_M" in text and "circular" in text


def test_a_run_of_the_renamed_model_is_the_control_under_either_name():
    from evals import claims_report as R
    rows, cases = _control_rows("x")
    renamed = [dict(r, candidate="Qwen2.5-7B-Instruct-Q4_K_M") if r["candidate"] == "eval-7b" else r
               for r in rows]
    assert "control" in R.report(renamed, cases)["Qwen2.5-7B-Instruct-Q4_K_M"]
    assert "control" in R.report(rows, cases)["eval-7b"]


def _control_rows(*claims, reviewed=None):
    from dataclasses import replace
    row = _line()
    case = _tagged_case("a", "conversation")
    if reviewed is not None:
        case = replace(case, assertions={"reviews": case.assertions["reviews"][:reviewed]})
    metrics = C.check(_reply(*claims), row["schema"], case.assertions["reviews"]).metrics
    return [{"case_id": "a", "candidate": cand, "passed": False, "metrics": metrics}
            for cand in ("eval-7b", "eval-4b")], [case]


def test_each_interface_counts_verbatim_matches_and_near_misses():
    row = _line()
    good = _good(row)
    user, claim, refs = good[0]
    near = (user, "Jumper J7 picks where the board boots from.", refs)
    m = C.check(_reply(near, good[1]), row["schema"], _tagged(row, "conversation")).metrics
    assert m["claims_verbatim_conversation"] == 1 and m["claims_near_miss_conversation"] == 1
    m = C.check(_reply(good[1]), row["schema"], _tagged(row, "conversation")).metrics
    assert m["claims_near_miss_conversation"] == 0


def test_the_matcher_control_blames_the_matcher_when_misses_sit_just_under_the_threshold():
    from evals import claims_report as R
    good = _good(_line())
    near = [(u, "Jumper J7 picks where the board boots from.", r) for u, _, r in good[:1]]
    # The first two reviews are the good ones: one reproduced verbatim, one reworded just under 0.5.
    got = R.report(*_control_rows(*near, good[1], reviewed=2))
    assert got["eval-7b"]["control"] == {"interface": "conversation", "recall": 0.5, "good_total": 2,
                                         "verbatim": 1, "near_miss": 1, "matcher_suspect": True}
    assert "control" not in got["eval-4b"]
    assert "matcher, not the model" in R.render(got)


def test_the_matcher_control_does_not_blame_the_matcher_for_claims_the_model_never_made():
    from evals import claims_report as R
    got = R.report(*_control_rows(_good(_line())[0]))
    c = got["eval-7b"]["control"]
    assert c["recall"] == 0.25 and c["near_miss"] == 0 and c["matcher_suspect"] is False
    text = R.render(got)
    assert "matcher control" in text and "did not reproduce" in text
    assert "matcher, not the model" not in text


def test_the_report_rescores_stored_replies_against_the_cases_it_is_given(tmp_path):
    from evals import claims_report as R
    row = _line()
    reply = tmp_path / "eval-7b--claims-x.json"
    reply.write_text(_reply(*_good(row)), encoding="utf-8")
    case = _tagged_case("claims-x", "conversation")
    stale = {"case_id": "claims-x", "candidate": "eval-7b", "passed": False,
             "artifact_path": str(reply), "metrics": {"claims_good_found": 0}}
    missing = {**stale, "case_id": "claims-x#1", "artifact_path": str(tmp_path / "gone.json")}
    got, skipped = R.rescore([stale, missing], [case])
    assert skipped == 1 and len(got) == 1
    assert got[0]["passed"] is True and got[0]["metrics"]["claims_recall_conversation"] == 1.0


# --- the matcher control (#661): rewording versus a different claim ---------------------

PARAPHRASE = FIXTURES / "paraphrase.json"


def _paraphrases():
    return json.loads(PARAPHRASE.read_text(encoding="utf-8"))


def test_the_paraphrase_fixture_holds_three_cases_of_twelve_synthetic_triples():
    items = _paraphrases()
    assert Counter(i["case"] for i in items) == {"quanta": 12, "kestrel": 12, "vireo": 12}
    assert all(set(i) == {"case", "reviewed", "reword", "different"} for i in items)


def test_the_paraphrase_control_matches_every_rewording_and_no_different_claim():
    got = C.paraphrase_control(_paraphrases())
    assert got["n"] == 36
    assert got["reword_matched"] == 36
    # #662: shared subject words carried 6 of the first 12 different claims over plain Dice.
    assert got["different_matched"] == 0


def test_the_control_gap_between_classes_exceeds_the_spread_within_a_class():
    got = C.paraphrase_control(_paraphrases())
    assert got["gap"] == pytest.approx(got["reword_min"] - got["different_max"])
    assert got["gap"] > got["spread"] > 0
    assert got["reword_min"] > C.MATCH_THRESHOLD > got["different_max"]


def test_plain_dice_fails_the_same_control():
    got = C.paraphrase_control(_paraphrases(), plain=True)
    assert got["different_matched"] >= 6
    assert got["gap"] <= got["spread"]


def test_a_case_holding_one_reviewed_claim_has_no_subject_evidence_and_falls_back_to_content_dice():
    items = [{**i, "case": str(n)} for n, i in enumerate(_paraphrases())]
    got = C.paraphrase_control(items)
    # #662: with one reviewed claim every word weighs the same; the classes still order but 0.5 sits inside one.
    assert got["reword_matched"] == 36 and got["different_matched"] == 3
    assert got["gap"] > got["spread"]


def test_the_paraphrase_control_can_fail_at_either_end():
    items = _paraphrases()
    assert C.paraphrase_control(items, threshold=1.01)["reword_matched"] == 0
    assert C.paraphrase_control(items, threshold=0.0)["different_matched"] == 36


def test_a_word_in_every_reviewed_claim_weighs_least_and_an_unseen_one_as_the_rarest():
    w = C.weights(["the Quanta 40 fan spins", "the Quanta 40 ROM boots", "Quanta 40 jumper J7"])
    assert 0 < w("quanta") < w("jumper") == w("never")


def test_stopwords_carry_no_weight_and_plural_or_tense_does_not_split_a_word():
    w = C.weights(["the pins"])
    assert C.content_tokens("The six pins were initialized") == ["6", "pin", "initializ"]
    assert C.similarity("six pins", "six pin", w) == 1.0
    assert C.similarity("the of and", "the of and", w) == 0.0


@pytest.mark.parametrize("a, b", [
    ("User e872 has three SCSI drives in the Quanta 40.", "They have 3 SCSI drives in the Quanta 40."),
    ("The Quanta 40 lacks a parallel port.", "The Quanta 40 does not have a parallel port."),
    ("member-a1b2 says the Quanta 40 doesn't have a parallel port.", "The Quanta 40 has no parallel port."),
])
def test_framing_number_words_and_negation_do_not_split_one_claim(a, b):
    assert C.similarity(a, b) == 1.0


@pytest.mark.parametrize("a, b", [
    ("They have three SCSI drives.", "They have two SCSI drives."),
    ("The Quanta 40 has a parallel port.", "The Quanta 40 has no parallel port."),
    ("The bus runs at 9600 baud.", "The bus runs at 4800 baud."),
])
def test_the_normalisation_keeps_a_different_number_or_a_negation_apart(a, b):
    assert C.similarity(a, b) < 1.0


def test_check_weighs_words_by_the_case_s_own_reviews_so_a_different_claim_is_not_credited():
    items = [i for i in _paraphrases() if i["case"] == "quanta"]
    reviews = [{"user": "u", "claim": i["reviewed"], "refs": [n + 1], "verdict": "good",
                "interface": "conversation"} for n, i in enumerate(items)]
    schema = _line()["schema"]
    rewords = _reply(*[("u", i["reword"], [n + 1]) for n, i in enumerate(items)])
    others = _reply(*[("u", i["different"], [n + 1]) for n, i in enumerate(items)])
    assert C.check(rewords, schema, reviews).metrics["claims_good_found_conversation"] == 12
    assert C.check(others, schema, reviews).metrics["claims_good_found_conversation"] == 0


def test_a_different_claim_is_not_blamed_as_the_rejected_one_it_shares_a_subject_with():
    item = _paraphrases()[3]
    reviews = [{"user": "u", "claim": item["reviewed"], "refs": [1], "verdict": "wrong"},
               {"user": "u", "claim": "The Quanta 40 SCSI bus needs termination.", "refs": [2],
                "verdict": "good"}]
    got = C.check(_reply(("u", item["different"], [1])), _line()["schema"], reviews)
    assert got.metrics["claims_bad_matched_legacy"] == 0


def test_every_claims_row_carries_the_matcher_version():
    got = _check(_reply(*_good(_line())))
    assert got.metrics["claims_matcher_version"] == C.MATCHER_VERSION >= 2
    bad = _check("not json")
    assert bad.metrics["claims_matcher_version"] == C.MATCHER_VERSION


def test_a_matcher_change_changes_a_claims_run_s_digest_but_not_a_case_s_holdout_side(monkeypatch):
    from evals.core import Case, case_digest, cases_digest
    case = _claims_case()
    others = [Case(id="code-x", modality="code", prompt="add two numbers")]
    before = (cases_digest([case]), cases_digest(others), case_digest(case))
    monkeypatch.setattr(C, "MATCHER_VERSION", C.MATCHER_VERSION + 1)
    after = (cases_digest([case]), cases_digest(others), case_digest(case))
    assert after[0] != before[0]
    assert after[1:] == before[1:]


def test_the_report_names_the_matcher_versions_its_rows_were_scored_under():
    from evals import claims_report as R
    rows, cases = _control_rows(_good(_line())[0])
    assert R.report(rows, cases)["eval-7b"]["matcher_versions"] == [C.MATCHER_VERSION]
    old = [{**r, "metrics": {k: v for k, v in r["metrics"].items() if k != "claims_matcher_version"}}
           for r in rows]
    got = R.report(old + rows, cases)
    assert got["eval-7b"]["matcher_versions"] == [1, C.MATCHER_VERSION]
    assert "scored under matcher versions 1 and" in R.render(got)
