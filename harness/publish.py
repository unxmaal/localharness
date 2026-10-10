"""Each machine's report as public JSON, and one page built from all of them. #432.

The receipts and the store live on each machine, so a GitHub runner has no
measurements of its own. A machine exports its slice keyed by a hardware label,
writes it to machines/<slug>.json on the `reports` branch, and a workflow on
main renders the set into one page.

The repo and its Pages site are public: the export carries hardware, candidate
names, lane results and dates, never paths, hostnames or receipts' environment.
"""
from __future__ import annotations

import base64
import json
import re
import time
from pathlib import Path

#: Bumped when the JSON's shape changes in a way the renderer must know about.
EXPORT_VERSION = 3
BRANCH = "reports"
WORKFLOW = "pages.yml"
#: Per-machine switch for publishing at the end of a discovery loop; absent is off.
SETTINGS = "publish.json"
STALE_DAYS = 7.0

#: Apple model identifier -> product and chip; anything else shows the identifier.
MAC_MODELS = {
    "Macmini9,1": "Mac mini M1",
    "Mac14,3": "Mac mini M2",
    "Mac14,12": "Mac mini M2 Pro",
    "Mac16,10": "Mac mini M4",
    "Mac16,11": "Mac mini M4 Pro",
    "Mac13,1": "Mac Studio M1 Max",
    "Mac13,2": "Mac Studio M1 Ultra",
    "Mac14,13": "Mac Studio M2 Max",
    "Mac14,14": "Mac Studio M2 Ultra",
    "Mac15,14": "Mac Studio M3 Ultra",
    "Mac16,9": "Mac Studio M4 Max",
    "Mac17,15": "Mac Studio M5 Ultra",
}


class ExportRefused(RuntimeError):
    """The export still matched the privacy scanner after scrubbing."""


def _card(name: str) -> str:
    return re.sub(r"^(?:NVIDIA\s+)?(?:GeForce\s+)?", "", (name or "").strip())


def machine_label(row: dict, accelerator_name: str = "") -> str:
    """'Mac Studio M5 Ultra 96 GB' or 'RTX 4070 (Windows)'. Hardware only."""
    from harness import machine
    hw = str(row.get("hw_model") or "")
    family = machine.os_family(str(row.get("os") or "")) or ""
    gb = float(row.get("memory_gb") or 0)
    discrete = str(row.get("accelerator") or "").startswith("discrete")
    if discrete and accelerator_name:
        return f"{_card(accelerator_name)} ({family})" if family else _card(accelerator_name)
    base = MAC_MODELS.get(hw) or hw or str(row.get("arch") or "unknown machine")
    label = f"{base} {gb:.0f} GB" if gb else base
    return label if family in ("macOS", "") else f"{label} ({family})"


def slug(label: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", label.lower()).strip("-") or "machine"


_PATH = re.compile(r"(?<![\w/.~-])(?:[A-Za-z]:[\\/]|~[\\/]|/)"
                   r"(?:[^\s/\\:\"']+[\\/])+")


def scrub(value):
    """Every absolute or home path reduced to its last component, recursively."""
    if isinstance(value, str):
        return _PATH.sub("", value)
    if isinstance(value, dict):
        return {scrub(k): scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [scrub(v) for v in value]
    return value


def _local_names() -> list[str]:
    import getpass
    import socket
    out = []
    try:
        out.append(getpass.getuser())
    except Exception:  # noqa: BLE001
        pass
    host = socket.gethostname() or ""
    out.append(host)
    out.append(host.split(".")[0])
    return [n for n in out if n and len(n) >= 3 and not _generic(n)]


def _generic(name: str) -> bool:
    """A factory hostname such as Mac-Studio names the product, which the label already says."""
    norm = re.sub(r"[-_\s.]+", " ", name.lower()).strip()
    norm = re.sub(r" (?:local|lan|home)$", "", norm)
    products = {" ".join(v.lower().split()[:2]) for v in MAC_MODELS.values()}
    return norm in products | {"localhost", "macbook pro", "macbook air", "imac"}


def private_pattern():
    """The scanner's person-name rule plus this machine's username and hostname."""
    from harness import paths, privacy
    return privacy.name_pattern(paths.REPO, _local_names())


def findings(doc, extra=None) -> list:
    """What harness.privacy finds in the export, one key or value per line."""
    from harness import privacy
    text = json.dumps(doc, indent=1, sort_keys=True)
    return privacy.scan(text, "export", extra if extra is not None else private_pattern())


def _iso(epoch) -> str:
    if epoch is None:
        return ""
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(float(epoch)))


def _metrics(m: dict) -> dict:
    out = {}
    for k, v in (m or {}).items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            continue
        out[str(k)] = round(float(v), 4)
    return out


def _ttft(s: dict) -> dict:
    """Warm TTFT median and p95 as plain numbers, None where unmeasured. #468."""
    out = {}
    for k in ("ttft_median_s", "ttft_p95_s"):
        v = s.get(k)
        out[k] = (round(float(v), 3) if isinstance(v, (int, float))
                  and not isinstance(v, bool) else None)
    return out


def not_run(s: dict, never_ran: tuple[str, ...]) -> str:
    """The class that kept every case from reaching the candidate, or "" if any ran. #583."""
    total = s.get("total", 0)
    passed = s.get("passed", 0)
    failure_classes = s.get("failure_classes") or {}
    if total <= 0 or passed != 0:
        return ""
    counts = {cls: failure_classes.get(cls, 0) for cls in never_ran}
    if sum(counts.values()) != total:
        return ""
    best = ""
    best_count = -1
    for cls in never_ran:
        if counts[cls] > best_count:
            best_count = counts[cls]
            best = cls
    return best


def model_html(row: dict) -> str:
    """A row's model and what it is, never an alias alone; an export from before #670 is labelled now. #670."""
    from harness import models
    from harness.report import _esc
    key = str(row.get("candidate") or "")
    if "model" in row:
        named = {"model": row.get("model") or "", "served_as": row.get("served_as") or ""}
    else:
        named = models.label(key)
    shown = named["model"] or f'{named["served_as"] or key} ({models.UNKNOWN})'
    if named["model"] and named["served_as"]:
        shown += f' (served as {named["served_as"]})'
    if key not in (named["model"], named["served_as"]):
        shown = f"{key}: {shown}"
    about = row.get("about") or models.UNKNOWN_FACT
    link = row.get("source") or ""
    src = f' <a href="{_esc(link)}">source</a>' if link else ""
    return f'{_esc(shown)}<div class="about">{_esc(about)}{src}</div>'


def not_run_text(row: dict) -> str:
    return f'not run ({row.get("not_run")})'


def _row(key: str, s: dict, resolved: dict | None = None, conn=None) -> dict:
    from harness import adopt, models, reasons
    why = not_run(s, reasons.NEVER_RAN)
    named = models.label(key, resolved)
    facts = models.about(conn, named["model"]) if conn is not None else {}
    return {"candidate": key,
            # The model the row measured and the alias it was asked for by; "" model is unknown. #670.
            "model": named["model"], "served_as": named["served_as"],
            "about": models.about_text(facts), "source": facts.get("source", ""),
            "reference": adopt.is_reference(key),
            "not_run": why,
            "passed": None if why else s.get("passed"), "total": s.get("total"),
            "pass_rate": None if why else s.get("pass_rate"),
            "median_s": None if why else s.get("median_s"),
            "first_s": None if why else s.get("first_s"),
            **_ttft({} if why else s),
            "peak_gb": None if why else round((s.get("peak_kb") or 0) / 1024 ** 2, 2),
            "metrics": {} if why else _metrics(s.get("metrics") or {})}


def _accelerator_name(conn, mid: int) -> str:
    row = conn.execute("SELECT environment FROM runs WHERE machine_id = ? AND "
                       "environment LIKE '%accelerator%' ORDER BY "
                       "COALESCE(generated_at, 0) DESC, id DESC LIMIT 1",
                       (mid,)).fetchone()
    try:
        return str((json.loads(row["environment"]).get("accelerator") or {})
                   .get("name") or "") if row else ""
    except (ValueError, AttributeError):
        return ""


def _candidate_id(conn, lane: str, spec: str) -> int | None:
    from harness import screen
    if not spec:
        return None
    from harness import models
    olds = models.old_names(spec)
    for s in (screen.candidate_for(lane, spec) or spec, spec, *olds):
        row = conn.execute("SELECT id FROM candidates WHERE spec = ?",
                           (s,)).fetchone()
        if row and (s not in olds or conn.execute(
                "SELECT 1 FROM results WHERE candidate_id = ?", (row["id"],)).fetchone()):
            return int(row["id"])
    return None


def mutual_groups(items: list, same) -> list[list]:
    """Newest-first items into groups whose members are all `same` as each other."""
    groups: list[list] = []
    for item in items:
        for group in groups:
            if all(same(item, member) for member in group):
                group.append(item)
                break
        else:
            groups.append([item])
    return groups


def exam_receipt(conn, run: dict):
    """A stored run's receipt for comparable(), minus `serving` (the union of its candidates' engines) and other lanes' sampling."""
    from evals.core import Receipt
    try:
        raw = json.loads(run.get("receipt") or "{}")
    except ValueError:
        raw = {}
    lane = str(raw.get("modality") or run.get("lane") or "")
    ids = raw.get("case_ids") or [r["case_id"] for r in conn.execute(
        "SELECT DISTINCT case_id FROM results WHERE run_id = ? ORDER BY case_id",
        (run["id"],)).fetchall()]
    sampling = raw.get("sampling") or {}
    if lane in sampling:
        sampling = {lane: sampling[lane]}
    instruments = {k: v for k, v in (raw.get("instruments") or {}).items()
                   if k != "serving"}
    return Receipt(modality=lane, case_ids=tuple(ids),
                   repeat=int(raw.get("repeat") or run.get("repeat_count") or 1),
                   sampling=sampling, gateway="", adherence=raw.get("adherence", ""),
                   tier=raw.get("tier") or run.get("tier") or "measure",
                   accelerator=raw.get("accelerator", ""), instruments=instruments,
                   where=raw.get("where", ""), cases_digest=raw.get("cases_digest", ""),
                   split=raw.get("split", ""), split_version=raw.get("split_version", ""),
                   devices=raw.get("devices") or {}, launch=raw.get("launch") or {},
                   max_tokens=Receipt.from_dict({**raw, "modality": lane}).max_tokens,
                   knobs=Receipt.from_dict({**raw, "modality": lane}).knobs,
                   router_swaps=raw.get("router_swaps") or {})


def _resolved(run: dict | None) -> dict:
    """The receipt's record of the model each key resolved to when it ran; {} before #670."""
    try:
        return dict(json.loads((run or {}).get("receipt") or "{}").get("resolved") or {})
    except (ValueError, AttributeError):
        return {}


def _rank(r: dict) -> tuple:
    return (bool(r["reference"]), bool(r["not_run"]), -(r["pass_rate"] or 0),
            r["median_s"] or 0)


def exams(conn, mid: int, lane: str) -> list[dict]:
    """Every candidate's latest row per comparable exam on one machine, newest exam first. #621."""
    from evals.core import comparable, contaminated
    from harness import runs
    found = [(run, exam_receipt(conn, run)) for run in
             runs.find(conn, lane=lane, machines=[mid], tier=runs.MEASURE)]
    found = [(run, rec) for run, rec in found if not contaminated(rec)]
    same = (lambda a, b: a[0]["machine_id"] == b[0]["machine_id"]
            and comparable(a[1], b[1])[0])
    out = []
    for group in mutual_groups(found, same):
        rows: dict[str, dict] = {}
        for run, _ in group:
            resolved = _resolved(run)
            for key, s in runs.summarize(runs.rows(conn, run["id"])).items():
                if key not in rows:
                    rows[key] = {**_row(key, s, resolved, conn), "run_at": _iso(run["generated_at"])}
        ids = group[0][1].case_ids
        cases = len({c.partition("#")[0] for c in ids})
        # Repeats as the case ids carry them: a lane that does not repeat ignores --repeat.
        out.append({"cases": cases, "repeat": max(1, round(len(ids) / max(cases, 1))),
                    "runs": len(group),
                    "run_at": _iso(group[0][0]["generated_at"]),
                    "first_run_at": _iso(group[-1][0]["generated_at"]),
                    "rows": sorted(rows.values(), key=_rank)})
    return out


def _lane(conn, mid: int, lane: str, held: dict, typed: dict, now: float) -> dict:
    from harness import lanes as L, runs
    serves = held.get("spec") or typed.get(lane, "")
    cid = held.get("candidate_id") or _candidate_id(conn, lane, serves)
    newest = runs.newest(conn, lane=lane, machines=[mid], tier=runs.MEASURE)
    ran = (runs.newest(conn, lane=lane, machines=[mid], tier=runs.MEASURE,
                       candidate_id=cid) if cid else None)
    here = runs.row_for(conn, ran["id"], cid) if ran else {}
    from harness import reasons
    why = not_run(here, reasons.NEVER_RAN) if here else ""
    if why:
        here = {}
    parked = L.parked(lane)
    age = runs.age_days(newest, now)
    table = []
    if newest:
        summary = runs.summarize(runs.rows(conn, newest["id"]))
        table = sorted((_row(k, s, _resolved(newest), conn) for k, s in summary.items()),
                       key=lambda r: (bool(r["not_run"]), -(r["pass_rate"] or 0),
                                      r["median_s"] or 0))
    return {
        "lane": lane, "wanted": lane in L.WANTED,
        "serves": serves, "adopted": bool(held),
        "serves_model": _serves_model(serves, conn),
        "adopted_how": held.get("how", ""),
        "not_run": why,
        "pass_rate": here.get("pass_rate"), "median_s": here.get("median_s"),
        **_ttft(here),
        "metrics": _metrics(here.get("metrics") or {}),
        "measured_at": _iso(ran["generated_at"]) if ran else "",
        "last_run_at": _iso(newest["generated_at"]) if newest else "",
        "age_days": None if age is None else round(age, 2),
        "parked": parked[0], "parked_until": parked[1],
        "unverified": not newest and not parked[0],
        "stale": bool(age is not None and age > STALE_DAYS and not parked[0]),
        "comparison": table,
        "exams": exams(conn, mid, lane),
    }


def _serves_model(spec: str, conn) -> str:
    """What a lane serves, named by its model id with what it is beside it. #670."""
    from harness import models
    if not spec:
        return ""
    return f"{models.display(spec)}; {models.about_text(models.about(conn, models.resolve(spec)))}"


def _adoptions(conn, mid: int) -> list[dict]:
    from harness import adopt
    rows = conn.execute(
        "SELECT a.*, c.spec, i.spec AS incumbent "
        "FROM adoptions a JOIN candidates c ON c.id = a.candidate_id "
        "LEFT JOIN candidates i ON i.id = a.incumbent_id "
        "WHERE a.machine_id = ? OR a.all_machines = 1 OR a.machine_id IS NULL "
        "ORDER BY a.adopted_at, a.id", (mid,)).fetchall()
    return [{"lane": r["lane"], "candidate": r["spec"],
             "incumbent": r["incumbent"] or "", "how": r["how"],
             "scope": adopt.describe(dict(r), mid),
             "adopted_at": _iso(r["adopted_at"])} for r in rows]


def _real_use(conn, now: float) -> dict:
    """Aggregate real use per lane alias; no client, text or time of any request. #481."""
    from harness import usage
    try:
        return usage.real_use(conn, now=now)
    except Exception:  # noqa: BLE001
        return {}


def export(conn=None, machine_id: int | None = None,
           now: float | None = None) -> dict:
    """One machine's report as plain, scrubbed data. This machine by default."""
    from harness import adopt, lanes as L, winners
    from harness import memory_store as ms
    from harness import runs

    now = time.time() if now is None else now
    with runs.store(conn) as c:
        mid = machine_id if machine_id is not None else ms.machine_row(c)
        if mid is None:
            mid = ms.remember_machine(c)
            c.commit()
        row = dict(c.execute("SELECT * FROM machines WHERE id = ?",
                             (mid,)).fetchone())
        acc = _accelerator_name(c, mid)
        if not acc and machine_id is None:
            from harness import memory
            acc = memory.detect().name if row["accelerator"].startswith("discrete") else ""
        label = machine_label(row, acc)
        real = _real_use(c, now)
        held = adopt.current(c, mid)
        typed = winners.typed()
        try:
            versions = json.loads(row.get("versions") or "{}")
        except ValueError:
            versions = {}
        doc = {
            "export_version": EXPORT_VERSION,
            "schema": ms.SCHEMA_VERSION,
            "generated_at": _iso(now),
            "machine": {
                "label": label, "slug": slug(label),
                "hw_model": row["hw_model"], "os": _os_short(row["os"]),
                "arch": row["arch"], "memory_gb": row["memory_gb"],
                "accelerator": row["accelerator"],
                "accelerator_name": _card(acc) if row["accelerator"].startswith("discrete") else "",
                "runtimes": [r for r in (row.get("runtimes") or "").split(",") if r],
                "versions": {k: v for k, v in versions.items() if v},
                "first_seen": _iso(row["first_seen"]),
                "last_seen": _iso(row["last_seen"]),
            },
            "lanes": [{**_lane(c, mid, lane, held.get(lane) or {}, typed, now),
                       "real_use": real.get(lane)} for lane in L.ALL],
            "adoptions": _adoptions(c, mid),
        }
    return scrub(doc)


def _os_short(os_string: str) -> str:
    """'macOS-27.0.1-arm64-arm-64bit' -> 'macOS 27.0.1'; family and version only."""
    from harness import machine
    fam = machine.os_family(os_string)
    parts = (os_string or "").split("-")
    ver = parts[1] if len(parts) > 1 and re.match(r"^[\d.]+$", parts[1]) else ""
    return f"{fam} {ver}".strip()


def checked(doc: dict) -> dict:
    found = findings(doc)
    if found:
        raise ExportRefused("the export matched the privacy scanner:\n"
                            + "\n".join(str(f) for f in found))
    from evals import private
    if private.found_in(json.dumps(doc, sort_keys=True), private.fragments(private.local_cases())):
        raise ExportRefused("the export carries text from a private case (#654)")
    return doc


def write_export(out=None, conn=None) -> Path:
    """Export, refuse on any privacy finding, write. Default under the home."""
    from harness import paths
    doc = checked(export(conn))
    out = Path(out) if out else (paths.home() / "reports"
                                 / f"{doc['machine']['slug']}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return out


# --- publishing ------------------------------------------------------------

def _gh(args: list[str], body: dict | None = None) -> dict:
    import subprocess
    cmd = ["gh", "api", *args]
    if body is not None:
        cmd += ["--input", "-"]
    r = subprocess.run(cmd, input=json.dumps(body) if body is not None else None,
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise GhError(r.stderr.strip() or r.stdout.strip())
    return json.loads(r.stdout) if r.stdout.strip() else {}


class GhError(RuntimeError):
    pass


def repo_name() -> str:
    import os
    import subprocess
    from harness import paths
    if os.environ.get("SOH_PUBLISH_REPO"):
        return os.environ["SOH_PUBLISH_REPO"]
    r = subprocess.run(["gh", "repo", "view", "--json", "nameWithOwner", "-q",
                        ".nameWithOwner"], cwd=paths.REPO, capture_output=True,
                       text=True)
    if r.returncode != 0 or not r.stdout.strip():
        raise GhError(r.stderr.strip() or "gh could not name the repository")
    return r.stdout.strip()


def _create_branch(repo: str, path: str, content: str, message: str, gh) -> None:
    """An orphan `reports` branch holding only this one file."""
    blob = gh([f"repos/{repo}/git/blobs", "-X", "POST"],
              {"content": content, "encoding": "utf-8"})
    tree = gh([f"repos/{repo}/git/trees", "-X", "POST"],
              {"tree": [{"path": path, "mode": "100644", "type": "blob",
                         "sha": blob["sha"]}]})
    commit = gh([f"repos/{repo}/git/commits", "-X", "POST"],
                {"message": message, "tree": tree["sha"], "parents": []})
    gh([f"repos/{repo}/git/refs", "-X", "POST"],
       {"ref": f"refs/heads/{BRANCH}", "sha": commit["sha"]})


def publish(doc: dict, repo: str | None = None, gh=_gh, dispatch: bool = True) -> str:
    """Write machines/<slug>.json on `reports` through the contents API.

    One file per machine and one API commit per write, so a machine never
    rewrites another's file and nothing is force-pushed.
    """
    doc = checked(doc)
    repo = repo or repo_name()
    path = f"machines/{doc['machine']['slug']}.json"
    content = json.dumps(doc, indent=1, sort_keys=True) + "\n"
    message = f"report: {doc['machine']['label']} at {doc['generated_at']}"
    try:
        gh([f"repos/{repo}/branches/{BRANCH}"])
    except GhError as exc:
        if "404" not in str(exc) and "Not Found" not in str(exc):
            raise
        _create_branch(repo, path, content, message, gh)
    else:
        for attempt in range(3):
            sha = None
            try:
                sha = gh([f"repos/{repo}/contents/{path}?ref={BRANCH}"]).get("sha")
            except GhError as exc:
                if "404" not in str(exc) and "Not Found" not in str(exc):
                    raise
            body = {"message": message, "branch": BRANCH,
                    "content": base64.b64encode(content.encode()).decode()}
            if sha:
                body["sha"] = sha
            try:
                gh([f"repos/{repo}/contents/{path}", "-X", "PUT"], body)
                break
            except GhError as exc:
                # 409: another machine moved the branch between the read and the write.
                if "409" not in str(exc) or attempt == 2:
                    raise
    if dispatch:
        gh([f"repos/{repo}/actions/workflows/{WORKFLOW}/dispatches", "-X", "POST"],
           {"ref": "main"})
    return path


def settings_path() -> Path:
    from harness import paths
    return paths.home() / SETTINGS


def enabled() -> bool:
    try:
        return bool(json.loads(settings_path().read_text(encoding="utf-8"))
                    .get("enabled"))
    except (OSError, ValueError, AttributeError):
        return False


def set_enabled(on: bool) -> Path:
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"enabled": bool(on)}) + "\n", encoding="utf-8")
    return p


def publish_here(conn=None) -> str:
    return publish(export(conn))


# --- the page --------------------------------------------------------------

def load(data_dir) -> list[dict]:
    """Every machines/*.json, skipping anything that is not an export."""
    out = []
    for p in sorted(Path(data_dir).glob("*.json")):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(doc, dict) and isinstance(doc.get("machine"), dict):
            out.append(doc)
    return sorted(out, key=lambda d: d["machine"].get("label", ""))


def _age_days(iso: str, now: float) -> float | None:
    try:
        t = time.mktime(time.strptime(iso, "%Y-%m-%dT%H:%M:%SZ")) - time.timezone
    except (TypeError, ValueError):
        return None
    return max(0.0, (now - t) / 86400.0)


def _num(x, fmt="{:.2f}") -> str:
    return "--" if x is None else fmt.format(x)


def _date(iso: str) -> str:
    return (iso or "")[:10] or "--"


def _cell(lane: dict | None) -> str:
    from harness.report import _esc
    if not lane:
        return '<td class="dim">--</td>'
    if lane.get("parked"):
        return f'<td><span class="tag warn">parked</span> {_esc(lane.get("serves_model") or lane.get("serves"))}</td>'
    if lane.get("unverified"):
        return (f'<td>{_esc(lane.get("serves_model") or lane.get("serves")) or "--"} '
                f'<span class="tag bad">no receipt</span></td>')
    if lane.get("not_run"):
        return (f'<td>{_esc(lane.get("serves_model") or lane.get("serves"))} '
                f'<span class="tag bad">{_esc(not_run_text(lane))}</span></td>')
    stale = ' <span class="tag warn">stale</span>' if lane.get("stale") else ""
    return (f'<td>{_esc(lane.get("serves_model") or lane.get("serves"))}<br><span class="dim">'
            f'{_num(lane.get("pass_rate"))} pass, {_num(lane.get("median_s"))}s, '
            f'{_date(lane.get("measured_at"))}</span>{stale}</td>')


def _ttft_cell(row: dict) -> str:
    from harness.report import ttft_text
    num = (lambda v: v if isinstance(v, (int, float)) and not isinstance(v, bool)
           else None)
    return ttft_text(num(row.get("ttft_median_s")), num(row.get("ttft_p95_s")))


def _metric_text(m: dict) -> str:
    return " ".join(f"{k} {v:.3f}" for k, v in sorted((m or {}).items()))[:60]


def _section(doc: dict) -> str:
    from harness.report import REAL_USE_HEAD, _esc, real_use_rows
    m = doc["machine"]
    rows = []
    for lane in doc.get("lanes") or []:
        tags = []
        if lane.get("adopted"):
            tags.append(f'<span class="tag good">adopted {_esc(lane.get("adopted_how"))}</span>')
        if lane.get("not_run"):
            tags.append(f'<span class="tag bad">{_esc(not_run_text(lane))}</span>')
        if lane.get("parked"):
            tags.append(f'<span class="tag warn">parked: {_esc(lane["parked"])}</span>')
        elif lane.get("unverified"):
            tags.append('<span class="tag bad">no receipt here</span>')
        elif lane.get("stale"):
            tags.append('<span class="tag warn">stale</span>')
        rows.append(
            f'<tr><td>{_esc(lane["lane"])}</td>'
            f'<td>{_esc(lane.get("serves_model") or lane.get("serves")) or "--"}</td>'
            f'<td class="num">{_num(lane.get("pass_rate"))}</td>'
            f'<td class="num">{_num(lane.get("median_s"))}</td>'
            f'<td class="num">{_ttft_cell(lane)}</td>'
            f'<td class="dim">{_esc(_metric_text(lane.get("metrics")))}</td>'
            f'<td>{_date(lane.get("measured_at"))}</td>'
            f'<td>{" ".join(tags)}</td></tr>')
    comps = []
    for lane in doc.get("lanes") or []:
        table = lane.get("comparison") or []
        if not table:
            continue
        body = "".join(
            f'<tr{" class=changed" if r["candidate"] == lane.get("serves") else ""}>'
            f'<td>{model_html(r)}'
            f'{" <span class=dim>(reference)</span>" if r.get("reference") else ""}</td>'
            + (f'<td class="num" colspan="2">{_esc(not_run_text(r))}</td>' if r.get("not_run") else
               f'<td class="num">{r.get("passed")}/{r.get("total")}</td>'
               f'<td class="num">{_num(r.get("pass_rate"))}</td>') +
            f'<td class="num">{_num(r.get("median_s"))}</td>'
            f'<td class="num">{_ttft_cell(r)}</td>'
            f'<td class="num">{_num(r.get("peak_gb"), "{:.1f}")}</td>'
            f'<td class="dim">{_esc(_metric_text(r.get("metrics")))}</td></tr>'
            for r in table)
        comps.append(
            f'<h3>{_esc(lane["lane"])} <span class="dim">latest run '
            f'{_date(lane.get("last_run_at"))}</span></h3>'
            '<div class="wide"><table><tr><th>candidate</th><th>passed</th><th>pass</th>'
            '<th>median s</th><th>TTFT med / p95</th><th>peak GB</th>'
            '<th>metric</th></tr>'
            f'{body}</table></div>')
    real = {l["lane"]: l["real_use"] for l in doc.get("lanes") or [] if l.get("real_use")}
    real_html = (f'<h3>Real use <span class="dim">last 7 days through the gateway</span></h3>'
                 f'<div class="wide">{REAL_USE_HEAD}{real_use_rows(real)}</table></div>'
                 if real else "")
    adopts = "".join(
        f'<tr><td>{_esc(a["lane"])}</td><td>{_esc(a["candidate"])}</td>'
        f'<td>{_esc(a.get("incumbent")) or "--"}</td><td>{_esc(a["how"])}</td>'
        f'<td>{_date(a.get("adopted_at"))}</td></tr>'
        for a in doc.get("adoptions") or [])
    from harness import site
    return site.window(_esc(m.get('label')), f"""
<p class="sub">{_esc(m.get('os'))} &middot; {_esc(m.get('arch'))} &middot;
{_esc(', '.join(m.get('runtimes') or []))} &middot; published {_esc(doc.get('generated_at'))}
&middot; schema {_esc(doc.get('schema'))}</p>
<div class="wide"><table><tr><th>lane</th><th>serves</th><th>pass</th><th>median s</th>
<th>TTFT med / p95</th><th>metric</th><th>measured</th><th></th></tr>
{''.join(rows)}
</table></div>
{''.join(comps)}
{real_html}
<h3>Adoptions</h3>
{f'<div class="wide"><table><tr><th>lane</th><th>adopted</th><th>replaced</th><th>how</th><th>when</th></tr>{adopts}</table></div>' if adopts else '<p class="note">Nothing adopted on this machine.</p>'}
""", tag="section", attrs=f' id="{_esc(m.get("slug"))}"')


def render_site(machines: list[dict], now: float | None = None) -> str:
    """The lane report: freshness, a lane table across machines, a window per machine."""
    from harness import site
    from harness.report import _days, _esc

    now = time.time() if now is None else now
    fresh = []
    for d in machines:
        age = _age_days(d.get("generated_at", ""), now)
        tag = ('<span class="tag warn">stale</span>' if age is None or age > STALE_DAYS
               else '<span class="tag good">current</span>')
        fresh.append(
            f'<tr><td><a href="#{_esc(d["machine"].get("slug"))}">'
            f'{_esc(d["machine"].get("label"))}</a></td>'
            f'<td>{_esc(d.get("generated_at"))}</td><td class="num">{_days(age)}</td>'
            f'<td class="num">{_esc(d.get("schema"))}</td>'
            f'<td class="num">{_esc(d.get("export_version"))}</td><td>{tag}</td></tr>')
    lane_names: list[str] = []
    for d in machines:
        for lane in d.get("lanes") or []:
            if lane["lane"] not in lane_names:
                lane_names.append(lane["lane"])
    by = [{l["lane"]: l for l in d.get("lanes") or []} for d in machines]
    head = "".join(f'<th>{_esc(d["machine"].get("label"))}</th>' for d in machines)
    side = "".join(f'<tr><td>{_esc(n)}</td>{"".join(_cell(b.get(n)) for b in by)}</tr>'
                   for n in lane_names)
    empty = '<p class="note">No machine has published yet.</p>' if not machines else ""
    intro = site.window("Lane report", f"""<p>What each lane serves on each machine,
the receipt behind it, and the latest run's full comparison.</p>{empty}""", "sky")
    machines_win = site.window("Machines", f"""<div class="wide"><table><tr><th>machine</th>
<th>last published</th><th>age</th><th>schema</th><th>export</th><th></th></tr>
{''.join(fresh)}
</table></div>
<p class="note">Each machine publishes its own measurements. One that has not
published in {STALE_DAYS:.0f} days is marked stale, and its numbers are as old as
its publish date.</p>""", "butter")
    across = site.window("Lanes across machines", f"""<div class="wide"><table><tr><th>lane</th>{head}</tr>
{side}
</table></div>
<p class="note">Wall-clock is only comparable within one machine. Across
machines compare pass rates and what each lane serves.</p>""", "teal")
    body = intro + machines_win + across + "".join(_section(d) for d in machines)
    return site.page("SoHoT lane report", body, 1, "reports/", now)


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.publish",
                                 description="Render the published reports page.")
    ap.add_argument("action", choices=["render"])
    ap.add_argument("--data", required=True, help="directory of machine JSON files")
    ap.add_argument("--out", required=True,
                    help="directory for the site: index.html, reports/, benchmarks/")
    a = ap.parse_args(argv)
    from harness import site
    for p in site.write(load(a.data), a.out):
        print(p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
