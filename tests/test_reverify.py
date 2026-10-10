"""Served models are re-run when their runtime moves, when they age, or when real use regresses. #480, #486."""
import json

import pytest

from harness import adopt, cli, lanes, reverify, runs, workqueue as wq
from harness import memory_store as ms

DAY = 86400.0
NOW = 1_800_000_000.0
VERSIONS = {"mlx": "0.30.0", "mlx-lm": "0.28.0", "mflux": "0.11.0",
            "litellm": "1.77.0", "llama.cpp": "6500"}
M2 = {"fingerprint": "Mac14,12/macOS/arm64", "hw_model": "Mac14,12",
      "os": "macOS-27.0.1-arm64-arm-64bit", "arch": "arm64", "memory_gb": 32.0,
      "accelerator": "unified 32GB", "runtimes": "cpu,mlx", "ceiling_gb": 22.0,
      "versions": dict(VERSIONS)}


@pytest.fixture
def on(monkeypatch):
    """Pin the machine the code runs on, and its runtime versions now."""
    def be(**changes):
        facts = {**M2, **changes}
        monkeypatch.setattr(ms, "this_machine", lambda: dict(facts))
    be()
    return be


@pytest.fixture
def conn(tmp_path, on):
    c = ms.connect(tmp_path / "d.db")
    yield c
    c.close()


_SEQ = iter(range(1, 10_000))


def record_run(conn, lane, results, *, at, versions=None, macos="27.0.1",
               job_id=None, failure_class=""):
    """Store one run: results is [(candidate, case, passed), ...]."""
    env = {"hw_model": "Mac14,12", "os": f"macOS-{macos}-arm64-arm-64bit",
           "arch": "arm64", "macos": macos,
           "versions": dict(VERSIONS if versions is None else versions)}
    rows = [{"candidate": c, "case_id": case, "passed": ok, "seconds": 1.0,
             "failure_class": "" if ok else failure_class}
            for c, case, ok in results]
    specs = {c: c for c, _, _ in results}
    rid = runs.record(conn, f"runs/r{next(_SEQ)}-{lane}",
                      {"receipt": {"modality": lane}, "environment": env,
                       "specs": specs, "rows": rows}, at=at)
    if job_id is not None:
        conn.execute("UPDATE runs SET job_id = ? WHERE id = ?", (int(job_id), rid))
        conn.commit()
    return rid


def cases(spec, passed, total=10):
    return [(spec, f"c{i}", i < passed) for i in range(total)]


def check(conn, **kw):
    kw.setdefault("now", NOW)
    kw.setdefault("lane", "code")
    return reverify.check(conn, **kw)


def queued(conn):
    return [j for j in wq.jobs(conn=conn) if j["requested_by"] == reverify.REQUESTED_BY]


# ---- trigger (a): a runtime version change since the last passing run -------

def test_a_runtime_upgrade_since_the_last_pass_queues_one_reverify(conn, on):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY)
    on(versions={**VERSIONS, "mlx-lm": "0.29.1"})
    got = check(conn)
    [job] = queued(conn)
    assert job["state"] == wq.PENDING
    assert job["argv"][job["argv"].index("--modality") + 1] == "code"
    # The lane serves the renamed id; the old nickname's run is its measured row (#670).
    assert job["argv"][job["argv"].index("--candidates") + 1] == cli.DEFAULT_CODE_MODEL
    [entry] = got["queued"]
    assert entry["triggers"] == [[reverify.VERSIONS, "mlx-lm 0.28.0 -> 0.29.1"]]


def test_the_same_versions_and_a_fresh_pass_queue_nothing(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY)
    got = check(conn)
    assert queued(conn) == []
    [entry] = got["planned"]
    assert entry["triggers"] == [] and entry["skip"] == "nothing moved"


def test_a_runtime_the_lane_does_not_use_is_not_a_trigger(conn, on):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY)
    on(versions={**VERSIONS, "mflux": "0.12.0"})
    check(conn)
    assert queued(conn) == []


def test_a_version_the_old_run_did_not_record_is_unknown_not_a_change(conn, on):
    old = {k: v for k, v in VERSIONS.items() if k != "litellm"}
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY, versions=old)
    check(conn)
    assert queued(conn) == []


def test_a_macos_upgrade_is_a_trigger(conn, on):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY, macos="27.0.1")
    on(os="macOS-27.1-arm64-arm-64bit")
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.VERSIONS, "macos 27.0.1 -> 27.1"]]


def test_the_comparison_is_against_the_last_passing_run_not_the_newest(conn, on):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 3 * DAY)
    record_run(conn, "code", cases("q3-4b", 0), at=NOW - DAY,
               versions={**VERSIONS, "mlx-lm": "0.29.1"})
    on(versions={**VERSIONS, "mlx-lm": "0.29.1"})
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.VERSIONS, "mlx-lm 0.28.0 -> 0.29.1"]]


# ---- trigger (b): age ---------------------------------------------------------

def test_a_pass_older_than_seven_days_is_due(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 8 * DAY)
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.AGE, "last passed 8 days ago"]]


def test_a_pass_inside_seven_days_is_not(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 6 * DAY)
    check(conn)
    assert queued(conn) == []


def test_the_age_is_configurable_per_lane(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 4 * DAY)
    reverify.set_days(conn, "code", 3)
    assert reverify.days_for(conn, "code") == 3.0
    assert reverify.days_for(conn, "web") == reverify.DEFAULT_DAYS
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.AGE, "last passed 4 days ago"]]


def test_a_thin_lane_with_no_passing_run_here_is_queued(conn):
    got = check(conn, lane="web")
    assert got["queued"][0]["triggers"] == [[reverify.AGE, "no passing run on this machine"]]


def test_a_run_that_passed_nothing_is_not_a_pass(conn):
    record_run(conn, "code", cases("q3-4b", 0), at=NOW - DAY)
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.AGE, "no passing run on this machine"]]


def test_another_machines_pass_does_not_count_here(conn, on):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY)
    on(fingerprint="Mac17,15/macOS/arm64", hw_model="Mac17,15")
    ms.remember_machine(conn)
    got = check(conn)
    assert got["queued"][0]["triggers"] == [[reverify.AGE, "no passing run on this machine"]]


# ---- trigger (c): real use regressed after the alias moved -------------------

def _switch_with_errors(conn, lane="code", at=NOW - DAY, before=0, after=20):
    conn.execute("INSERT INTO gateway_switches (lane, old_spec, new_spec, how, "
                 "requested_at, switched_at) VALUES (?,?,?,?,?,?)",
                 (lane, "q3-1.7b", "q3-4b", "idle", at, at))
    for i in range(40):
        side = at - 3600 + i if i < 20 else at + i
        err = (i < before) if i < 20 else (i - 20 < after)
        conn.execute("INSERT INTO gateway_requests (at, alias, lane, error_class) "
                     "VALUES (?,?,?,?)", (side, f"sohot-{lane}", lane,
                                          "APIError" if err else ""))
    conn.commit()


def test_a_real_use_regression_after_a_switch_queues_a_reverify(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 2 * DAY)
    _switch_with_errors(conn)
    got = check(conn)
    [[kind, why]] = got["queued"][0]["triggers"]
    assert kind == reverify.USAGE and "error rate" in why


def test_real_use_within_noise_is_not_a_trigger(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 2 * DAY)
    _switch_with_errors(conn, after=1)
    check(conn)
    assert queued(conn) == []


@pytest.mark.gauntlet("state-is-whichever-row-came-last", site="harness/reverify.py:_latest")
def test_a_usage_regression_already_reverified_since_the_switch_does_not_requeue(conn):
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 2 * DAY)
    _switch_with_errors(conn)
    check(conn)
    job = queued(conn)[0]
    wq._write({**job, "state": wq.DONE, "rc": 0}, conn=conn)
    record_run(conn, "code", cases("q3-4b", 10), at=NOW + 60, job_id=job["id"])
    check(conn, now=NOW + 120)
    assert len(queued(conn)) == 1


# ---- the action: one queued job, deduped, never inline ----------------------

def test_a_second_check_does_not_queue_a_second_job(conn):
    check(conn)
    got = check(conn)
    assert len(queued(conn)) == 1
    assert got["planned"][0]["skip"].startswith("already queued as job")


def test_a_dry_run_says_what_it_would_queue_and_writes_nothing(conn):
    got = check(conn, dry_run=True)
    assert got["planned"][0]["triggers"] and not got["queued"]
    assert queued(conn) == []
    assert conn.execute("SELECT COUNT(*) FROM reverifications").fetchone()[0] == 0


def test_the_job_is_low_priority_on_this_machine_and_nothing_runs_inline(conn, monkeypatch):
    import subprocess
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: pytest.fail("ran inline"))
    check(conn)
    [job] = queued(conn)
    assert job["priority"] < 0
    assert job["machine_id"] == ms.machine_row(conn)
    assert job["kind"] == "command" and job["title"].startswith("reverify code")


def test_every_wanted_lane_with_a_served_model_takes_part(conn):
    got = check(conn, lane="", dry_run=True)
    assert {e["lane"] for e in got["planned"]} == {
        l for l in lanes.WANTED if not lanes.parked(l)[0]}


def test_a_measured_adoption_is_re_run_beside_its_incumbent(conn):
    adopt.record(conn, adopt.Verdict("code", "q3-4b", "q3-8b", True, "won"))
    check(conn)
    [job] = queued(conn)
    got = job["argv"][job["argv"].index("--candidates") + 1]
    assert got == "q3-8b,q3-4b"


def test_a_by_hand_adoption_is_re_run_alone(conn):
    adopt.record(conn, adopt.Verdict("code", "q3-4b", "q3-8b", True, "liked",
                                     adopt.BY_HAND))
    check(conn)
    [job] = queued(conn)
    assert job["argv"][job["argv"].index("--candidates") + 1] == "q3-8b"


# ---- the outcome: settled from the job's run, flagged, never swapped ---------

def _queue_and_finish(conn, results, *, rc=0, state=wq.DONE, failure_class="",
                      lane="code"):
    check(conn, lane=lane)
    job = queued(conn)[-1]
    wq._write({**job, "state": state, "rc": rc}, conn=conn)
    if results is not None:
        record_run(conn, lane, results, at=NOW + 60, job_id=job["id"],
                   failure_class=failure_class,
                   versions={**VERSIONS, "mlx-lm": "0.29.1"})
    return job


def _latest(conn):
    return dict(conn.execute("SELECT * FROM reverifications ORDER BY id DESC").fetchone())


@pytest.fixture
def upgraded(conn, on):
    """A pass before an mlx-lm upgrade, so the check has a baseline and a trigger."""
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - DAY)
    on(versions={**VERSIONS, "mlx-lm": "0.29.1"})
    return conn


def test_a_rerun_that_passes_settles_passed_and_clears_the_trigger(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 10))
    got = check(conn, now=NOW + 120)
    assert [s["outcome"] for s in got["settled"]] == [reverify.PASSED]
    assert len(queued(conn)) == 1
    assert got["planned"][0]["skip"] == "nothing moved"


def test_a_rerun_that_fails_records_reason_and_failure_class(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 0), failure_class="timeout")
    check(conn, now=NOW + 120)
    row = _latest(conn)
    assert row["outcome"] == reverify.FAILED
    assert row["failure_class"] == "timeout" and row["reason"] == "limit"


def test_a_regression_beyond_the_paired_noise_is_flagged(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 2))
    check(conn, now=NOW + 120)
    row = _latest(conn)
    assert row["outcome"] == reverify.REGRESSED and row["reason"] == "candidate"
    assert "8 lost against 0 gained" in row["detail"]


def test_a_drop_within_the_paired_noise_is_not_a_regression(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 8))
    check(conn, now=NOW + 120)
    assert _latest(conn)["outcome"] == reverify.PASSED


def test_a_job_that_stored_no_run_is_flagged_as_unrun_not_as_a_model_failure(upgraded):
    conn = upgraded
    _queue_and_finish(conn, None, rc=2, state=wq.FAILED)
    check(conn, now=NOW + 120)
    row = _latest(conn)
    assert row["outcome"] == reverify.UNRUN and row["reason"] == "harness"
    assert "stored no run" in row["detail"]
    assert set(reverify.flags(conn)) == {"code"}


def test_a_rerun_the_gateway_refused_is_unrun_not_failed(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 0), failure_class="refused_by_gateway")
    check(conn, now=NOW + 120)
    row = _latest(conn)
    assert row["outcome"] == reverify.UNRUN
    assert row["failure_class"] == "refused_by_gateway"


def test_the_job_runs_in_this_checkout_wherever_soh_was_called(conn, tmp_path, monkeypatch):
    from harness import paths
    monkeypatch.chdir(tmp_path)
    check(conn)
    assert queued(conn)[0]["cwd"] == str(paths.REPO)


def test_a_lane_no_table_names_watches_every_runtime():
    assert reverify.runtimes_for("code", "q3-4b") == ("mlx", "mlx-lm", "litellm", "macos")
    got = reverify.runtimes_for("newlane", "x")
    assert {"mlx-lm", "mflux", "llama.cpp", "litellm", "macos"} <= set(got)


def test_a_cancelled_job_is_not_a_verdict_on_the_model(upgraded):
    conn = upgraded
    check(conn)
    wq.cancel(queued(conn)[0]["id"], conn=conn)
    check(conn, now=NOW + 120)
    assert _latest(conn)["outcome"] in (reverify.CANCELLED, "")
    assert reverify.flags(conn) == {}


def test_an_incumbent_that_now_beats_the_served_model_is_flagged(conn):
    adopt.record(conn, adopt.Verdict("code", "q3-4b", "q3-8b", True, "won"))
    check(conn)
    job = queued(conn)[0]
    wq._write({**job, "state": wq.DONE, "rc": 0}, conn=conn)
    record_run(conn, "code", cases("q3-8b", 1) + cases("q3-4b", 10),
               at=NOW + 60, job_id=job["id"])
    check(conn, now=NOW + 120)
    row = _latest(conn)
    assert row["outcome"] == reverify.REGRESSED and "q3-4b" in row["detail"]


def test_a_flagged_lane_keeps_its_served_model_and_is_not_requeued(upgraded):
    conn = upgraded
    before = adopt.current(conn)
    _queue_and_finish(conn, cases("q3-4b", 0), failure_class="crashed")
    got = check(conn, now=NOW + 120)
    assert adopt.current(conn) == before
    assert len(queued(conn)) == 1
    assert got["planned"][0]["skip"].startswith("flagged failed")
    assert set(reverify.flags(conn)) == {"code"}


def test_a_flagged_lane_is_tried_again_after_its_age_threshold(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 0), failure_class="crashed")
    check(conn, now=NOW + 120)
    check(conn, now=NOW + 8 * DAY)
    assert len(queued(conn)) == 2


def test_a_regressed_run_is_not_the_next_baseline(upgraded):
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 2))
    check(conn, now=NOW + 120)
    base = reverify.baseline(conn, "code", reverify.served(conn)["code"]["candidate_id"],
                             runs.here(conn))
    assert base["generated_at"] == NOW - DAY


# ---- the report -------------------------------------------------------------

def test_the_report_flags_the_lane_and_uses_the_lane_threshold(upgraded):
    from harness import report
    conn = upgraded
    _queue_and_finish(conn, cases("q3-4b", 0), failure_class="crashed")
    check(conn, now=NOW + 120)
    reverify.set_days(conn, "web", 0.5)
    rows = {l["lane"]: l for l in report.lanes_state(conn)}
    assert rows["code"]["reverify"]["outcome"] == reverify.FAILED
    assert rows["code"]["flagged"]
    assert not rows["web"]["flagged"]
    html = report._lane_rows(list(rows.values()), {})
    assert "re-verify failed" in html


def test_the_report_lists_lanes_whose_last_run_is_older_than_their_threshold(conn):
    from harness import report
    record_run(conn, "code", cases("q3-4b", 10), at=NOW - 4 * DAY)
    reverify.set_days(conn, "code", 3)
    rows = {l["lane"]: l for l in report.lanes_state(conn, now=NOW)}
    assert rows["code"]["stale"]
    reverify.set_days(conn, "code", 5)
    rows = {l["lane"]: l for l in report.lanes_state(conn, now=NOW)}
    assert not rows["code"]["stale"]


def test_soh_report_names_the_flagged_lane(tmp_path, monkeypatch, capsys):
    from harness import report
    from harness.commands import report as report_cmd
    out = tmp_path / "report.html"
    out.with_suffix(".json").write_text(json.dumps({"lanes": [
        {"lane": "code", "flagged": True, "serves": "q3-4b",
         "reverify": {"outcome": "regressed", "detail": "8 lost against 0 gained"}},
        {"lane": "web", "flagged": False, "reverify": {}}]}), encoding="utf-8")
    monkeypatch.setattr(report, "write", lambda *a, **k: out)
    import argparse
    assert report_cmd.cmd_report(argparse.Namespace(out="", json=False)) == 0
    got = capsys.readouterr().out
    assert "re-verify flagged code (q3-4b): regressed: 8 lost against 0 gained" in got
    assert "web" not in got.split("re-verify")[-1]


# ---- the verb and the schedule ----------------------------------------------

def test_soh_reverify_dry_run_json(on, capsys):
    from harness import cli
    rc = cli.main(["reverify", "--lane", "web", "--dry-run", "--json"])
    out = json.loads(capsys.readouterr().out)
    assert rc == 0 and out["dry_run"] is True
    assert [e["lane"] for e in out["planned"]] == ["web"]
    assert out["planned"][0]["triggers"]
    c = ms.connect()
    try:
        assert queued(c) == []
    finally:
        c.close()


def test_soh_reverify_sets_a_lane_threshold(on, capsys):
    from harness import cli
    assert cli.main(["reverify", "--lane", "web", "--days", "3", "--json"]) == 0
    c = ms.connect()
    try:
        assert reverify.days_for(c, "web") == 3.0
    finally:
        c.close()


def test_soh_reverify_days_needs_a_lane(capsys):
    from harness import cli
    assert cli.main(["reverify", "--days", "3"]) != 0


def test_the_discover_loop_runs_the_check(monkeypatch):
    from harness.commands import loop as loop_cmd
    seen = []
    monkeypatch.setattr(reverify, "check", lambda conn, **kw: seen.append(kw) or
                        {"settled": [], "planned": [], "queued": []})
    loop_cmd._reverify()
    assert seen and not seen[0].get("dry_run")
