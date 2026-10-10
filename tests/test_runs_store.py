"""Runs and their result rows live in the store, written by evals.run. #410.

Every reader asks the tables. Each pin below plants a results.json that was
never stored: a reader that scans runs/ again finds it and fails.
"""
import argparse
import json

import pytest

from evals import core, environment
from evals import run as er
from evals.core import Case, Result
from harness import candidates, memory_store as ms, paths, runs


@pytest.fixture
def conn():
    c = ms.connect()
    yield c
    c.close()


def _unstored(lane="svg", key="local-large", name="20261006-000000-000-0000"):
    """A receipt on disk that nothing recorded. The store already exists."""
    d = paths.runs() / f"{name}-{lane}"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "generated": "2026-10-06T00:00:00",
        "environment": ms.this_machine(),
        "receipt": {"modality": lane, "tier": "measure"},
        "specs": {key: key},
        "summary": {key: {"pass_rate": 1.0, "peak_kb": 99, "total": 1}},
        "rows": [{"case_id": "c", "candidate": key, "passed": True,
                  "seconds": 1.0, "peak_kb": 99 * 1024 ** 2, "detail": ""}]}),
        encoding="utf-8")
    return d


# --- the writer -----------------------------------------------------------

class _Runner:
    candidate = "fake/key"

    def run(self, case):
        return Result(case_id=case.id, candidate=self.candidate,
                      passed=case.id.startswith("a"), seconds=0.5,
                      peak_kb=7, detail="", metrics={"ink": 0.2})


def test_evals_run_writes_the_run_and_its_rows(monkeypatch, tmp_path):
    cases = [Case(id="a", modality="svg", prompt="p"),
             Case(id="b", modality="svg", prompt="q")]
    monkeypatch.setattr(core, "load_cases", lambda path: cases)
    monkeypatch.setattr(er, "build_runner", lambda *a, **k: _Runner())
    monkeypatch.setattr(er, "warn_if_pressed",
                        lambda: argparse.Namespace(as_dict=lambda: {}))
    monkeypatch.setattr(ms.machines, "_THIS_MACHINE", None)
    monkeypatch.setattr(environment, "capture", lambda: {"hw_model": "Test,1",
                                                "os": "t", "arch": "a"})
    out = tmp_path / "out"
    args = argparse.Namespace(
        modality="svg", candidates="fake:spec", cases="x", out=str(out),
        screen=False, repeat=2, adherence="", gateway="http://gw",
        from_winners=False)
    assert er._execute(args) == 0
    c = ms.connect()
    try:
        got = runs.receipt_at(c, out)
        assert got["receipt"]["repeat"] == 2 and got["specs"] == {
            "fake/key": "fake:spec"}
        rows = runs.rows(c, got["run_id"])
        assert [(r["case_id"], r["passed"]) for r in rows] == [
            ("a#1", True), ("a#2", True), ("b#1", False), ("b#2", False)]
        assert {r["candidate_id"] for r in rows} == {
            candidates.get(c, "fake:spec")["id"]}
        row = c.execute("SELECT * FROM runs WHERE id = ?",
                        (got["run_id"],)).fetchone()
        assert (row["lane"], row["tier"], row["repeat_count"]) == (
            "svg", "measure", 2)
        assert row["cases_digest"] and row["generated_at"]
        assert got["summary"]["fake/key"]["passed"] == 2
        # The faked machine is "here" too, so a run made here reads back as made here. #517.
        assert runs.here(c) == [row["machine_id"]]
    finally:
        c.close()
    on_disk = json.loads((out / "results.json").read_text(encoding="utf-8"))
    assert on_disk["summary"] == got["summary"], "the export is the same run"


# --- each converted reader ignores an unstored receipt ---------------------

def test_winners_read_stored_runs_only(conn, store_run):
    from harness import winners
    _unstored()
    assert winners.from_receipts(conn) == {} and winners.beaten_in(conn) == {}
    store_run("r", "svg", {"local-large": {"pass_rate": 1.0}})
    assert winners.from_receipts(conn)["svg"]["candidate"] == "local-large"


def test_rank_reads_stored_lanes_only(conn, store_run):
    from harness import rank
    _unstored(lane="web")
    assert rank.lanes_with_receipts(conn) == set()
    store_run("r", "web", {"q3-4b": {"pass_rate": 1.0}})
    assert rank.lanes_with_receipts(conn) == {"web"}


def test_coverage_dates_a_model_by_its_stored_run(conn, store_run):
    from harness import coverage
    _unstored(key="org/thing")
    assert coverage._measured(conn, {}) == {}
    store_run("r", "image", {"org/thing": {"pass_rate": 1.0}},
              specs={"org/thing": "org/thing"}, generated="2026-01-02T03:04:05")
    assert list(coverage._measured(conn, {})) == ["org/thing"]


def test_the_memory_guard_reads_stored_peaks_only(conn, store_run):
    from harness import memory
    _unstored(key="omnisvg:4B")
    assert memory.measured_peak_gb("omnisvg:4B", conn) is None
    store_run("r", "svg", {"omnisvg:4B": {"peak_kb": 2 * 1024 ** 2}})
    assert memory.measured_peak_gb("omnisvg:4B", conn) == 2.0


def test_from_winners_reads_stored_runs_only(conn, store_run):
    _unstored(lane="svg")
    with pytest.raises(SystemExit):
        er.winner_for("svg", conn)
    store_run("r", "svg", {"local-large": {"pass_rate": 1.0}})
    assert er.winner_for("svg", conn) == "local-large"


def test_the_retraction_migration_reads_stored_runs(conn):
    """The schema 10 retraction derives from stored rows, not the directory."""
    from harness import cli
    bad = "gateway returned HTTP 400: Invalid model name passed in model=org/x"
    runs.record(conn, paths.runs() / "r", {"receipt": {"modality": "code"},
                "rows": [{"case_id": "c", "candidate": "org/x",
                          "passed": False, "detail": bad}]})
    got = list(runs.summaries(conn, tier=""))
    assert cli._all_refused(got[0][2], "org/x")


def test_no_reader_scans_the_runs_directory():
    """The only directory read left is the backfill that imports it."""
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    hits = []
    for path in sorted(root.glob("harness/**/*.py")) + sorted(
            root.glob("evals/*.py")):
        text = path.read_text(encoding="utf-8")
        for needle in ('rglob("results.json")', "_newest_receipt_for",
                       "_latest_receipt", "_newest_run_for", "_run_age_days",
                       "_summary_at", '"runs" / run / "results.json"'):
            if needle in text:
                hits.append(f"{path.name}: {needle}")
    assert hits == ['runs.py: rglob("results.json")'], hits


# --- comparable() from stored digests --------------------------------------

def _receipt(digest):
    return {"generated": "2026-10-06T00:00:00", "environment": {},
            "receipt": {"modality": "svg", "case_ids": ["a"], "repeat": 1,
                        "sampling": {}, "gateway": "g", "tier": "measure",
                        "cases_digest": digest},
            "rows": [{"case_id": "a", "candidate": "k", "passed": True}]}


def test_comparable_refuses_across_stored_case_digests(conn, capsys):
    runs.record(conn, paths.runs() / "one", _receipt("d1"))
    runs.record(conn, paths.runs() / "two", _receipt("d2"))
    assert er.compare_runs([str(paths.runs() / "one"),
                            str(paths.runs() / "two")]) == 1
    assert "different cases: d1 vs d2" in capsys.readouterr().out


def test_comparable_accepts_the_same_stored_digest(conn):
    """Negative control."""
    runs.record(conn, paths.runs() / "one", _receipt("d1"))
    runs.record(conn, paths.runs() / "two", _receipt("d1"))
    assert er.compare_runs([str(paths.runs() / "one" / "results.json"),
                            str(paths.runs() / "two")]) == 0


# --- the backfill -----------------------------------------------------------

def _seed_dir():
    root = paths.runs()
    ok = root / "20261005-000000-000-0000-svg"
    ok.mkdir(parents=True)
    (ok / "results.json").write_text(json.dumps({
        "generated": "2026-10-05T00:00:00",
        "environment": {"hw_model": "Mac17,15", "os": "o", "arch": "a"},
        "receipt": {"modality": "svg", "tier": "measure"},
        "rows": [{"case_id": "c", "candidate": "mlx-community/Qwen2.5-7B-Instruct-4bit", "passed": True},
                 {"case_id": "c", "candidate": "nobody/knows", "passed": False}]
    }), encoding="utf-8")
    (root / "rubric-x").mkdir()
    (root / "rubric-x" / "results.json").write_text(json.dumps({
        "rows": [{"id": 1, "label": "x"}]}), encoding="utf-8")
    (root / "crashed-image").mkdir()


def test_the_backfill_records_what_it_can_and_names_what_it_cannot(conn):
    _seed_dir()
    got = runs.backfill(conn)
    assert got["runs"] == 1 and got["results"] == 2
    assert got["not_eval_receipts"] == ["rubric-x"]
    assert got["no_receipt"] == ["crashed-image"]
    assert got["unlinked_results"] == 1, "nobody/knows has no candidate row"
    run = runs.at(conn, paths.runs() / "20261005-000000-000-0000-svg")
    assert run["generated_at"] is not None and run["machine_id"]


def test_the_backfill_is_idempotent(conn):
    _seed_dir()
    first = runs.backfill(conn)
    ids = [r["id"] for r in runs.find(conn)]
    second = runs.backfill(conn)
    assert (second["runs"], second["results"]) == (first["runs"],
                                                   first["results"])
    assert [r["id"] for r in runs.find(conn)] == ids


def test_the_backfill_links_a_verdict_to_the_run_it_names(conn):
    _seed_dir()
    ms.record(conn, ms.Seen(name="org/x", source="t"))
    vid = ms.decide(conn, "org/x", "broken", tier=ms.SCREEN,
                    run_path=str(paths.runs() / "20261005-000000-000-0000-svg"))
    got = runs.backfill(conn)
    assert got["verdicts_linked"] == 1
    row = conn.execute("SELECT run_id FROM verdicts WHERE id = ?",
                       (vid,)).fetchone()
    assert row["run_id"] == runs.at(
        conn, paths.runs() / "20261005-000000-000-0000-svg")["id"]
    assert not ms.dangling_receipts(conn, exists=lambda p: False)


@pytest.mark.parametrize("here", ["win32", "other"])
def test_the_backfill_links_a_typed_default_by_the_key_its_rows_name(conn, monkeypatch,
                                                                      here):
    """The Mac's stt default ran with no specs map; the migrating platform's default is not used. #516."""
    from harness import audio
    for lane, name in (("tts", "DEFAULT_TTS_MODEL"), ("stt", "DEFAULT_STT_MODEL")):
        monkeypatch.setattr(audio, name, audio.SPEECH_DEFAULTS[lane][here])
    d = paths.runs() / "20261001-000000-000-0000-stt"
    d.mkdir(parents=True)
    (d / "results.json").write_text(json.dumps({
        "receipt": {"modality": "stt", "tier": "measure"},
        "rows": [{"case_id": "c", "candidate": "parakeet-tdt-0.6b-v2", "passed": True}]}),
        encoding="utf-8")
    assert runs.backfill(conn)["unlinked_results"] == 0
    got = {r["spec"] for r in conn.execute("SELECT spec FROM candidates")}
    assert got == {"stt:mlx-community/parakeet-tdt-0.6b-v2"}


def test_a_candidate_created_later_claims_its_unlinked_rows(conn):
    runs.record(conn, paths.runs() / "r", {"receipt": {"modality": "svg"},
                "rows": [{"case_id": "c", "candidate": "late-alias",
                          "passed": True}]})
    assert runs.rows(conn, runs.at(conn, paths.runs() / "r")["id"])[0][
        "candidate_id"] is None
    cid = candidates.ensure(conn, "late-alias", key="late-alias")
    assert runs.rows(conn, runs.at(conn, paths.runs() / "r")["id"])[0][
        "candidate_id"] == cid


def test_a_new_store_imports_an_existing_runs_directory():
    """A store created beside a populated runs/ (a fresh db) backfills once."""
    _seed_dir()
    c = ms.connect()
    try:
        assert len(runs.find(c)) == 1
    finally:
        c.close()


def test_the_screen_verdict_names_its_stored_run(conn, store_run):
    run_id = store_run("screen-1-svg", "svg", {"org/x": {"pass_rate": 0.0}},
                       tier="screen", conn=conn)
    ms.record(conn, ms.Seen(name="org/x", source="t"))
    bare = ms.decide(conn, "org/x", "broken", tier=ms.SCREEN)
    cited = ms.decide(conn, "org/x", "broken", tier=ms.SCREEN, run_id=run_id)
    assert cited != bare, "a verdict with evidence is a new fact, never deduped"
    assert conn.execute("SELECT run_id FROM verdicts WHERE id = ?",
                        (cited,)).fetchone()["run_id"] == run_id


# --- every row carries the candidate it ran as (#429) ----------------------

def _execute(monkeypatch, tmp_path, specs: str, runner=None):
    cases = [Case(id="a", modality="svg", prompt="p")]
    monkeypatch.setattr(core, "load_cases", lambda path: cases)
    monkeypatch.setattr(er, "build_runner", lambda *a, **k: runner or _Runner())
    monkeypatch.setattr(er, "warn_if_pressed",
                        lambda: argparse.Namespace(as_dict=lambda: {}))
    monkeypatch.setattr(ms.machines, "_THIS_MACHINE", None)
    monkeypatch.setattr(environment, "capture", lambda: {"hw_model": "Test,1",
                                                "os": "t", "arch": "a"})
    out = tmp_path / "out"
    args = argparse.Namespace(
        modality="svg", candidates=specs, cases="x", out=str(out),
        screen=False, repeat=1, adherence="", gateway="http://gw",
        from_winners=False)
    return er._execute(args), out


def test_the_export_carries_the_candidate_each_row_ran_as(monkeypatch, tmp_path):
    _, out = _execute(monkeypatch, tmp_path, "fake:spec")
    on_disk = json.loads((out / "results.json").read_text(encoding="utf-8"))
    c = ms.connect()
    try:
        cid = candidates.get(c, "fake:spec")["id"]
    finally:
        c.close()
    assert {r["candidate_id"] for r in on_disk["rows"]} == {cid}


def test_two_specs_that_run_as_one_key_are_refused_before_either_runs(
        monkeypatch, tmp_path):
    ran = []

    class Counting(_Runner):
        def run(self, case):
            ran.append(case.id)
            return super().run(case)

    with pytest.raises(SystemExit, match="both run as fake/key"):
        _execute(monkeypatch, tmp_path, "fake:spec,fake:other",
                 runner=Counting())
    assert ran == []


def test_a_stamped_id_this_store_does_not_hold_is_looked_up_again(conn):
    """An export from another store carries ids that mean nothing here."""
    cid = candidates.ensure(conn, "org/x", key="org/x")
    rows = [{"case_id": "c", "candidate": "org/x", "passed": True,
             "candidate_id": cid + 999}]
    run_id = runs.record(conn, paths.runs() / "r", {"rows": rows})
    assert [r["candidate_id"] for r in runs.rows(conn, run_id)] == [cid]
