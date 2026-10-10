"""Each machine's report as public JSON, and one page from all of them. #432."""
import json
import re
from pathlib import Path

import pytest

from harness.commands import loop as loop_cmd
from harness import memory_store as ms
from harness import privacy, publish, runs

ROOT = Path(__file__).resolve().parent.parent
LOCAL_NAMES = publish._local_names

# privacy-ok: deliberate fixtures, the things the export must never carry.
USER = "patq"
HOST = "patq-tower"
PERSON = "Pat Quinnell"
HOME = f"/Users/{USER}"  # privacy-ok
VOLUME = "/Volumes/Models"  # privacy-ok
LEAKS = (USER, HOST, PERSON, "/Users/", "/Volumes/", "/home/", ".local",  # privacy-ok
         "192.168.", "C:\\")

ENV = {"hw_model": "Mac17,15", "os": "macOS-27.0.1-arm64-arm-64bit-Mach-O",
       "arch": "arm64", "memory_gb": 96,
       "accelerator": {"kind": "unified", "name": "arm64", "total_gb": 96},
       "hostname": f"{HOST}.local", "user": USER, "cwd": f"{HOME}/projects/x",  # privacy-ok
       "HF_HOME": f"{VOLUME}/hf", "lan": "192.168.1.20",  # privacy-ok
       "git_sha": "abc", "versions": {"mlx": "0.30.1", "python": "3.12.3"}}


@pytest.fixture(autouse=True)
def _local(monkeypatch):
    monkeypatch.setattr(publish, "_local_names", lambda: [USER, HOST])
    monkeypatch.setenv("LH_PRIVATE_NAMES", PERSON)


def _receipt(lane, rows, when, specs=None, env=None):
    return {"generated": when,
            "receipt": {"modality": lane, "tier": "measure",
                        "notes": f"ran on {HOST}.local from {HOME}/lh"},  # privacy-ok
            "environment": env or ENV, "specs": specs or {},
            "rows": rows}


def _rows(cand, passed, seconds, art=f"{HOME}/localharness/outputs/a.png"):  # privacy-ok
    return [{"case_id": f"c{i}", "candidate": cand, "passed": p, "seconds": s,
             "peak_kb": 2 * 1024 ** 2, "metrics": {"code_pass": 1.0 if p else 0.0},
             "artifact_path": art, "output": "<svg>out</svg>", "detail": f"wrote {VOLUME}/x for {USER}"}  # privacy-ok
            for i, (p, s) in enumerate(zip(passed, seconds))]


@pytest.fixture
def store():
    conn = ms.connect()
    local = f"{HOME}/models/qwen-local.gguf"  # privacy-ok
    runs.record(conn, f"{HOME}/localharness/runs/r1",  # privacy-ok
                _receipt("code", _rows("q3-4b", [1, 1, 0], [1.0, 1.2, 1.4])
                         + _rows(local, [1, 0, 0], [3.0, 3.1, 3.2])
                         + _rows("claude-code:claude-opus-5-5", [1, 1, 1],
                                 [9.0, 9.5, 10.0]),
                         "2026-10-05T10:00:00",
                         specs={local: f"gguf:{local}"}))
    runs.record(conn, "runs/r2",
                _receipt("decide", _rows("nimble-3b", [1, 1], [0.2, 0.3]),
                         "2026-10-05T11:00:00"))
    conn.commit()
    yield conn
    conn.close()


def _mid(conn):
    return conn.execute("SELECT id FROM machines WHERE hw_model = 'Mac17,15'"
                        ).fetchone()["id"]


def test_an_export_of_a_store_full_of_private_detail_carries_none_of_it(store):
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    text = json.dumps(doc)
    for leak in LEAKS:
        assert leak.lower() not in text.lower(), leak
    assert publish.findings(doc) == []
    assert publish.checked(doc) is doc


def test_the_export_keeps_the_hardware_and_the_results(store):
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    assert doc["machine"]["label"] == "Mac Studio M5 Ultra 96 GB"
    assert doc["machine"]["slug"] == "mac-studio-m5-ultra-96-gb"
    assert doc["machine"]["os"] == "macOS 27.0.1"
    assert doc["schema"] == ms.SCHEMA_VERSION
    assert doc["export_version"] == publish.EXPORT_VERSION
    assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ", doc["generated_at"])
    lanes = {l["lane"]: l for l in doc["lanes"]}
    assert "decide" in lanes and lanes["decide"]["comparison"]
    cands = {r["candidate"]: r for r in lanes["code"]["comparison"]}
    assert cands["claude-code:claude-opus-5-5"]["reference"] is True
    assert cands["claude-code:claude-opus-5-5"]["pass_rate"] == 1.0
    assert "qwen-local.gguf" in cands
    assert lanes["code"]["last_run_at"].startswith("2026-10-05")


def test_no_receipt_environment_artifact_or_detail_is_exported(store):
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    text = json.dumps(doc)
    for key in ('"environment"', '"artifact"', '"artifact_path"', '"output"', '"<svg>', '"detail"', '"receipt"',
                '"hostname"', '"cwd"', '"HF_HOME"', '"git_sha"', '"path"'):
        assert key not in text, key


REFUSED = "crh225/plumb-4b-GGUF"


def _refused_run(store, cls="refused_by_gateway"):
    rows = _rows(REFUSED, [0, 0, 0], [0.07, 0.07, 0.07])
    for r in rows:
        r.update(failure_class=cls, detail="gateway returned HTTP 400")
    runs.record(store, "runs/r4",
                _receipt("decide", _rows("eval-imajev-4b", [1, 0, 1], [0.5, 0.5, 0.5])
                         + rows, "2026-10-07T09:37:05"))
    store.commit()
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    lane = next(l for l in doc["lanes"] if l["lane"] == "decide")
    return doc, {r["candidate"]: r for r in lane["comparison"]}


def test_a_candidate_the_harness_refused_on_every_case_is_exported_as_not_run(store):
    """#583: four GGUF repos refused by the gateway read as 0/36 on the public page."""
    _, rows = _refused_run(store)
    got = rows[REFUSED]
    assert got["not_run"] == "refused_by_gateway"
    assert got["passed"] is None and got["pass_rate"] is None
    assert rows["eval-imajev-4b"]["not_run"] == ""
    assert rows["eval-imajev-4b"]["passed"] == 2


def test_a_candidate_that_ran_and_failed_every_case_is_still_scored(store):
    """Negative control: a model's own failure is a result, not a refusal."""
    _, rows = _refused_run(store, cls="content_failed")
    assert rows[REFUSED]["not_run"] == ""
    assert rows[REFUSED]["passed"] == 0 and rows[REFUSED]["pass_rate"] == 0.0


def test_a_not_run_candidate_is_rendered_as_not_run_never_as_zero_of_n(store):
    from harness import site
    doc, _ = _refused_run(store)
    for html in (publish.render_site([doc], now=1.79e9), site.benchmarks([doc], now=1.79e9)):
        row = next(line for line in html.split("<tr") if REFUSED in line)
        assert "not run" in row and "refused_by_gateway" in row, row
        assert "0/3" not in row and "0.00" not in row, row


def test_a_lane_whose_serving_model_was_refused_has_no_headline_score(store):
    rows = _rows("eval-imajev-4b", [0, 0], [0.07, 0.07])
    for r in rows:
        r.update(failure_class="refused_by_gateway")
    runs.record(store, "runs/r5", _receipt("decide", rows, "2026-10-07T10:00:00",
                                           specs={"eval-imajev-4b": "eval-imajev-4b"}))
    store.commit()
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    lane = next(l for l in doc["lanes"] if l["lane"] == "decide")
    assert lane["serves"] == "imajev-4b-Q8_0"
    assert lane["not_run"] == "refused_by_gateway"
    assert lane["pass_rate"] is None and lane["median_s"] is None
    row = next(line for line in publish.render_site([doc], now=1.79e9).split("<tr")
               if line.startswith("><td>decide</td>"))
    assert "not run" in row and "0.00" not in row, row


def test_a_name_the_scrub_cannot_remove_refuses_the_export(store):
    runs.record(store, "runs/r3",
                _receipt("web", _rows(f"{USER}-finetune", [1], [1.0]),
                         "2026-10-05T12:00:00"))
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    with pytest.raises(publish.ExportRefused):
        publish.checked(doc)
    with pytest.raises(publish.ExportRefused):
        publish.publish(doc, repo="o/r", gh=lambda *a: pytest.fail("published"))


def test_a_discrete_card_is_labelled_by_card_and_os():
    row = {"hw_model": "MS-7D25", "os": "Windows-11-SP0",
           "memory_gb": 64, "accelerator": "discrete 12GB"}
    assert publish.machine_label(row, "NVIDIA GeForce RTX 4070") == "RTX 4070 (Windows)"
    linux = dict(row, os="Linux-6.8.0-45-generic-x86_64-with-glibc2.39")
    assert publish.machine_label(linux, "NVIDIA GeForce RTX 4070") == "RTX 4070 (Linux)"
    assert publish.machine_label({"hw_model": "Mac99,1", "os": "macOS-30",
                                  "memory_gb": 8}) == "Mac99,1 8 GB"


# --- publishing ------------------------------------------------------------

class FakeGh:
    def __init__(self, branch=True, existing=None, conflicts=0):
        self.calls, self.branch, self.existing = [], branch, existing
        self.conflicts = conflicts

    def __call__(self, args, body=None):
        self.calls.append((args, body))
        url = args[0]
        if url.endswith(f"/branches/{publish.BRANCH}"):
            if not self.branch:
                raise publish.GhError("gh: Not Found (HTTP 404)")
            return {}
        if "/contents/" in url and "-X" not in args:
            if self.existing is None:
                raise publish.GhError("gh: Not Found (HTTP 404)")
            return {"sha": self.existing}
        if "/contents/" in url and self.conflicts:
            self.conflicts -= 1
            raise publish.GhError("gh: is at x but expected y (HTTP 409)")
        return {"sha": f"sha{len(self.calls)}"}


DOC = {"export_version": 1, "schema": 1, "generated_at": "2026-10-06T00:00:00Z",
       "machine": {"label": "RTX 4070 (Linux)", "slug": "rtx-4070-linux"},
       "lanes": [], "adoptions": []}


def test_publish_writes_only_its_own_file_and_dispatches_the_page():
    gh = FakeGh(existing="old")
    path = publish.publish(dict(DOC), repo="o/r", gh=gh)
    assert path == "machines/rtx-4070-linux.json"
    puts = [(a, b) for a, b in gh.calls if "PUT" in a]
    assert len(puts) == 1
    args, body = puts[0]
    assert args[0] == "repos/o/r/contents/machines/rtx-4070-linux.json"
    assert body["branch"] == publish.BRANCH and body["sha"] == "old"
    assert not any("force" in json.dumps(b or {}) for _, b in gh.calls)
    assert gh.calls[-1][0][0].endswith(f"/workflows/{publish.WORKFLOW}/dispatches")
    assert gh.calls[-1][1] == {"ref": "main"}


def test_publish_creates_an_orphan_reports_branch_when_there_is_none():
    gh = FakeGh(branch=False)
    publish.publish(dict(DOC), repo="o/r", gh=gh, dispatch=False)
    commit = next(b for a, b in gh.calls if a[0].endswith("/git/commits"))
    assert commit["parents"] == []
    tree = next(b for a, b in gh.calls if a[0].endswith("/git/trees"))
    assert [t["path"] for t in tree["tree"]] == ["machines/rtx-4070-linux.json"]
    ref = next(b for a, b in gh.calls if a[0].endswith("/git/refs"))
    assert ref["ref"] == f"refs/heads/{publish.BRANCH}"


def test_publish_retries_when_another_machine_moved_the_branch():
    gh = FakeGh(existing="old", conflicts=1)
    publish.publish(dict(DOC), repo="o/r", gh=gh, dispatch=False)
    assert sum(1 for a, _ in gh.calls if "PUT" in a) == 2


def test_publishing_is_off_until_a_machine_turns_it_on():
    assert publish.enabled() is False
    publish.set_enabled(True)
    assert publish.enabled() is True
    publish.set_enabled(False)
    assert publish.enabled() is False


# --- the page --------------------------------------------------------------

def _two(tmp_path, store):
    studio = publish.export(store, machine_id=_mid(store), now=1.79e9)
    card = json.loads(json.dumps(DOC))
    card["generated_at"] = "2026-09-01T00:00:00Z"
    card["lanes"] = [{"lane": "code", "serves": "q3-8b", "pass_rate": 0.9,
                      "median_s": 0.5, "measured_at": "2026-09-01T00:00:00Z",
                      "metrics": {}, "comparison": [], "stale": True}]
    for d in (studio, card):
        (tmp_path / f"{d['machine']['slug']}.json").write_text(json.dumps(d), encoding="utf-8")
    (tmp_path / "notes.json").write_text("[]", encoding="utf-8")
    return publish.load(tmp_path)


def test_the_page_has_a_section_per_machine_and_a_lane_table_across_them(tmp_path, store):
    machines = _two(tmp_path, store)
    assert [m["machine"]["slug"] for m in machines] == [
        "mac-studio-m5-ultra-96-gb", "rtx-4070-linux"]
    page = publish.render_site(machines, now=1.79e9)
    assert page.count("<section id=") == 2
    assert "<th>Mac Studio M5 Ultra 96 GB</th><th>RTX 4070 (Linux)</th>" in page
    assert "claude-code:claude-opus-5-5" in page
    assert '<span class="tag warn">stale</span>' in page
    assert "\u2014" not in page
    assert privacy.scan(page, "page") == []


def test_an_empty_set_still_renders(tmp_path):
    page = publish.render_site(publish.load(tmp_path))
    assert "No machine has published yet." in page


def test_the_renderer_cli_writes_the_page(tmp_path, store):
    _two(tmp_path, store)
    out = tmp_path / "site"
    assert publish.main(["render", "--data", str(tmp_path), "--out", str(out)]) == 0
    assert (out / "reports" / "index.html").read_text(encoding="utf-8").count(
        "<section id=") == 2


def test_the_privacy_scanner_fails_on_a_file_that_leaks(tmp_path):
    bad = tmp_path / "index.html"
    bad.write_text(f"<p>{HOME}/runs</p>", encoding="utf-8")  # privacy-ok
    assert privacy.main(["--files", str(bad)]) == 1
    bad.write_text("<p>clean</p>", encoding="utf-8")
    assert privacy.main(["--files", str(bad)]) == 0


# --- the workflow ----------------------------------------------------------

def test_the_pages_workflow_never_runs_on_a_push_and_scans_before_deploy():
    wf = (ROOT / ".github" / "workflows" / publish.WORKFLOW).read_text(encoding="utf-8")
    head = wf.split("jobs:")[0]
    assert "push:" not in head and "pull_request" not in head
    assert "workflow_dispatch" in head
    assert "pages: write" in wf and "id-token: write" in wf
    assert "actions/upload-pages-artifact" in wf and "actions/deploy-pages" in wf
    assert wf.index("harness.privacy --files") < wf.index("upload-pages-artifact")
    assert "unxmaal" not in wf and "github.io" not in wf


# --- the command line and the loop -----------------------------------------

def test_the_loop_publishes_only_on_a_machine_that_turned_it_on(monkeypatch):
    import argparse

    from harness import cli
    sent = []
    monkeypatch.setattr(loop_cmd, "_loop_spend", lambda a, rc, failed=None: rc)
    monkeypatch.setattr(publish, "publish_here", lambda conn=None: sent.append(1) or "p")
    cli._spend_and_settle(argparse.Namespace(), 0)
    assert sent == []
    assert cli.main(["report", "--auto-publish", "on"]) == 0
    cli._spend_and_settle(argparse.Namespace(), 0)
    assert sent == [1]


def test_a_failed_publish_does_not_fail_the_loop(monkeypatch):
    import argparse

    from harness import cli

    def boom(conn=None):
        raise publish.GhError("offline")
    monkeypatch.setattr(loop_cmd, "_loop_spend", lambda a, rc, failed=None: 0)
    monkeypatch.setattr(publish, "publish_here", boom)
    publish.set_enabled(True)
    assert cli._spend_and_settle(argparse.Namespace(), 0) == 0


def test_report_export_writes_checked_json(tmp_path, store, monkeypatch):
    from harness import cli
    monkeypatch.setattr(ms, "machine_row", lambda conn, fp=None: _mid(conn))
    out = tmp_path / "m.json"
    assert cli.main(["report", "--export", "--out", str(out)]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["machine"]["label"] == "Mac Studio M5 Ultra 96 GB"


def test_a_factory_hostname_is_the_product_not_a_person(monkeypatch):
    import socket
    monkeypatch.setattr(socket, "gethostname", lambda: "Mac-Studio.local")
    assert not any(n.lower().startswith("mac-studio") for n in LOCAL_NAMES())
    monkeypatch.setattr(socket, "gethostname", lambda: f"{HOST}.local")  # privacy-ok
    assert HOST in LOCAL_NAMES()


# --- the benchmarks: every comparable run, merged per exam (#621) ----------

OPUS = "claude-code:claude-opus-5-5"
ORNITH = "llamacpp:Ornith-1.5-35B-Q4_K_M"


def _exam(lane, cands, when, digest="d-150", n=3, serving="llama-server",
          sampling=None, env=None):
    """A measure run whose receipt carries the axes comparable() reads."""
    rows = []
    for name, passed in cands.items():
        rows += _rows(name, [i < passed for i in range(n)], [1.0] * n)
    data = _receipt(lane, rows, when, env=env)
    data["receipt"].update(case_ids=[f"c{i}" for i in range(n)], repeat=1,
                           cases_digest=digest, accelerator="unified:arm64",
                           where="host", sampling=sampling or {lane: {"temperature": 0.2}},
                           instruments={"peak": "phys_footprint", "serving": serving})
    return data


def _exams(store, lane="code"):
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    return doc, next(l for l in doc["lanes"] if l["lane"] == lane)["exams"]


def _broad_then_adopt(store):
    runs.record(store, "runs/broad", _exam("code", {OPUS: 3, ORNITH: 3, "q3-coder": 2,
                                                    "q3-30b": 1}, "2026-10-06T04:36:01",
                                           serving="llama-server+mlx_lm.server"))
    runs.record(store, "runs/adopt", _exam("code", {ORNITH: 2, "granite-4.1-8b": 1},
                                           "2026-10-07T11:22:01"))
    store.commit()


def test_a_two_candidate_adopt_run_after_a_broad_run_still_shows_every_candidate(store):
    _broad_then_adopt(store)
    _, exams = _exams(store)
    newest = exams[0]
    got = {r["candidate"]: r for r in newest["rows"]}
    assert set(got) == {OPUS, ORNITH, "q3-coder", "q3-30b", "granite-4.1-8b"}
    assert got[ORNITH]["passed"] == 2, "the latest result for a candidate wins"
    assert got[ORNITH]["run_at"].startswith("2026-10-07")
    assert got["q3-coder"]["run_at"].startswith("2026-10-06")
    assert newest["cases"] == 3 and newest["runs"] == 2
    assert newest["run_at"].startswith("2026-10-07")
    assert newest["first_run_at"].startswith("2026-10-06")


def test_a_run_on_a_different_case_set_is_its_own_table_newest_first(store):
    _broad_then_adopt(store)
    runs.record(store, "runs/old27", _exam("code", {OPUS: 2, "q3-4b": 1}, "2026-10-05T12:00:00",
                                           digest="d-27", n=2))
    store.commit()
    _, exams = _exams(store)
    assert [e["cases"] for e in exams[:2]] == [3, 2]
    assert {r["candidate"] for r in exams[1]["rows"]} == {OPUS, "q3-4b"}
    assert "q3-4b" not in {r["candidate"] for r in exams[0]["rows"]}
    assert exams[1]["run_at"].startswith("2026-10-05")


def test_a_different_sampling_for_the_lane_is_another_exam_but_another_lanes_is_not(store):
    _broad_then_adopt(store)
    runs.record(store, "runs/hot", _exam("code", {"q3-hot": 1}, "2026-10-07T12:00:00",
                                         sampling={"code": {"temperature": 0.7}}))
    runs.record(store, "runs/svg", _exam("code", {"q3-svg": 1}, "2026-10-07T13:00:00",
                                         sampling={"code": {"temperature": 0.2},
                                                   "svg": {"temperature": 0.4}}))
    store.commit()
    _, exams = _exams(store)
    tables = [{r["candidate"] for r in e["rows"]} for e in exams]
    merged = next(t for t in tables if "q3-svg" in t)
    assert {"q3-coder", OPUS, "granite-4.1-8b"} <= merged
    assert "q3-hot" not in merged
    assert {"q3-hot"} in tables


def test_another_machines_run_is_never_merged_into_this_machines_table(store):
    _broad_then_adopt(store)
    linux = dict(ENV, hw_model="MS-7D25", os="Linux-6.8.0-x86_64", memory_gb=64)
    runs.record(store, "runs/linux", _exam("code", {"q3-linux": 3}, "2026-10-07T14:00:00",
                                           env=linux))
    store.commit()
    _, exams = _exams(store)
    assert all("q3-linux" not in {r["candidate"] for r in e["rows"]} for e in exams)


def test_a_reference_row_is_marked_and_never_presented_as_adoptable(store):
    from harness import site
    _broad_then_adopt(store)
    doc, exams = _exams(store)
    rows = exams[0]["rows"]
    assert rows[-1]["reference"] is True, "references sort after candidates"
    assert next(r for r in rows if r["candidate"] == OPUS)["reference"] is True
    code = next(l for l in doc["lanes"] if l["lane"] == "code")
    code["serves"] = OPUS
    html = site.benchmarks([doc], now=1.79e9)
    row = next(line for line in html.split("<tr") if OPUS in line and "<td>" in line)
    assert "reference, not adoptable" in row, row
    assert "serving" not in row, row


def test_the_serving_candidate_is_highlighted_in_its_exam(store):
    from harness import site
    _broad_then_adopt(store)
    doc, _ = _exams(store)
    code = next(l for l in doc["lanes"] if l["lane"] == "code")
    code["serves"] = ORNITH
    html = site.benchmarks([doc], now=1.79e9)
    row = next(line for line in html.split("<tr") if ORNITH in line and "<td>" in line)
    assert row.startswith(' class="serving"'), row
    assert '<span class="tag good">serving</span>' in row


def test_each_exam_is_captioned_with_its_case_count_and_date(store):
    from harness import site
    _broad_then_adopt(store)
    runs.record(store, "runs/old27", _exam("code", {"q3-4b": 1}, "2026-10-05T12:00:00",
                                           digest="d-27", n=2))
    store.commit()
    doc, _ = _exams(store)
    html = site.benchmarks([doc], now=1.79e9)
    new = html.index("3 cases &middot; 2 runs, latest 2026-10-07")
    old = html.index("2 cases &middot; 1 run, 2026-10-05")
    assert new < old


def test_the_export_version_says_the_shape_changed(store):
    assert publish.EXPORT_VERSION >= 2
    doc, exams = _exams(store)
    assert doc["export_version"] == publish.EXPORT_VERSION
    assert exams and exams[0]["rows"]


def test_an_older_export_without_exams_still_renders_its_latest_run(store):
    from harness import site
    _broad_then_adopt(store)
    doc = publish.export(store, machine_id=_mid(store), now=1.79e9)
    doc["export_version"] = 1
    for lane in doc["lanes"]:
        lane.pop("exams", None)
    html = site.benchmarks([doc], now=1.79e9)
    assert "Mac Studio M5 Ultra 96 GB &middot; run 2026-10-07" in html
    assert "granite-4.1-8b" in html and ORNITH in html
