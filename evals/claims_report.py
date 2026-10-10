"""A claims run read back per candidate: reviewed cases per review interface, expect_empty per origin. #654, #661.

    uv run python -m evals.claims_report [RESULTS.JSON | --run ID] [--rescore] [--cases DIR]

With no argument it reads the newest claims run in the store. Rows are joined to
this machine's local cases by id; nothing here prints case text.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

from harness.checks import claims as claims_check

#: The model every reviewed claim so far was extracted by.
REVIEWED_FROM = "Qwen2.5-7B-Instruct-Q4_K_M"
#: The interface whose reviews REVIEWED_FROM produced under the current prompt at temperature 0.
CONTROL_INTERFACE = "conversation"
#: Control recall under which the matcher, not the model, is the suspect.
CONTROL_RECALL = 0.9
CAVEAT = (f"caveat: every reviewed claim was extracted by {REVIEWED_FROM}, so recall is circular "
          f"and favours {REVIEWED_FROM}-like output until another model's claims are reviewed")


def _reviewed_model(candidate: str) -> bool:
    """True for a run of the model the reviews came from, under its id or its old nickname. #670."""
    from harness import models
    return models.resolve(candidate) == REVIEWED_FROM


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson score interval for k of n, rounded to 4 places."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (max(0.0, round(center - half, 4)), min(1.0, round(center + half, 4)))


def _slice(kept: int, n: int) -> dict:
    return {"kept": kept, "n": n, "rate": round(kept / n, 4) if n else 0.0,
            "ci95": list(wilson(kept, n))}


def _per_interface(row: dict, case) -> dict:
    """interface -> this row's metrics for it; a row scored before #661 reads its pooled ones as legacy."""
    m = row.get("metrics") or {}
    out = {}
    for interface in sorted({claims_check.interface_of(r) for r in case.assertions.get("reviews") or []}):
        def get(name):
            v = m.get(claims_check.metric(name, interface))
            if v is None and interface == claims_check.LEGACY:
                v = m.get(f"claims_{name}")
            return v
        passed = get("pass")
        out[interface] = {"passed": bool(row["passed"]) if passed is None else bool(passed),
                          **{k: int(get(k) or 0) for k in ("good_found", "good_total", "reviewed_matched",
                                                          "bad_matched", "verbatim", "near_miss")},
                          "schema_valid": bool(m.get("claims_schema_valid"))}
    return out


def report(rows: list[dict], cases) -> dict:
    """candidate -> reviewed metrics per interface (never pooled), expect_empty per origin and overall."""
    from harness.holdout import base_id
    by_id = {c.id: c for c in cases}
    out: dict = {}
    for r in rows:
        mine = out.setdefault(r["candidate"], {"reviewed": {}, "empty": {}, "unknown_cases": 0,
                                               "matcher_versions": set()})
        # Rows scored before #662 carry no version: they were matched by plain Dice, version 1.
        mine["matcher_versions"].add(int((r.get("metrics") or {}).get("claims_matcher_version") or 1))
        case = by_id.get(base_id(r["case_id"]))
        if case is None:
            mine["unknown_cases"] += 1
        elif case.assertions.get("expect_empty"):
            origin = str(case.assertions.get("origin") or "unspecified")
            for key in (origin, "all"):
                got = mine["empty"].setdefault(key, [0, 0])
                got[0] += int(bool(r["passed"]))
                got[1] += 1
        else:
            for interface, x in _per_interface(r, case).items():
                mine["reviewed"].setdefault(interface, []).append(x)
    for cand, mine in out.items():
        mine["_raw"] = dict(mine["reviewed"])
        for interface, xs in list(mine["reviewed"].items()):
            total = lambda k: sum(x[k] for x in xs)
            good, matched, wanted = total("good_found"), total("reviewed_matched"), total("good_total")
            mine["reviewed"][interface] = {
                "cases": len(xs), "passed": sum(1 for x in xs if x["passed"]), "good_total": wanted,
                "recall": round(good / wanted, 4) if wanted else 0.0,
                "precision": round(good / matched, 4) if matched else 0.0,
                "bad_matched": total("bad_matched"),
                "schema_invalid": sum(1 for x in xs if not x["schema_valid"])}
        mine["empty"] = {k: _slice(*v) for k, v in sorted(mine["empty"].items())}
        control = mine["reviewed"].get(CONTROL_INTERFACE)
        if _reviewed_model(cand) and control:
            xs = mine["_raw"][CONTROL_INTERFACE]
            missed = control["good_total"] - sum(x["good_found"] for x in xs)
            near = sum(x["near_miss"] for x in xs)
            # The matcher is suspect only when most misses sat just under its threshold.
            mine["control"] = {"interface": CONTROL_INTERFACE, "recall": control["recall"],
                               "good_total": control["good_total"],
                               "verbatim": sum(x["verbatim"] for x in xs), "near_miss": near,
                               "matcher_suspect": control["recall"] < CONTROL_RECALL
                               and missed > 0 and 2 * near >= missed}
        mine.pop("_raw")
        mine["matcher_versions"] = sorted(mine["matcher_versions"])
    return out


def render(got: dict) -> str:
    lines = []
    for cand, mine in sorted(got.items()):
        lines.append(f"{cand}")
        for interface, rv in sorted(mine["reviewed"].items()):
            lines.append(f"  reviewed {interface:<13} {rv['passed']}/{rv['cases']} pass  "
                         f"recall {rv['recall']:.3f}  precision {rv['precision']:.3f}  "
                         f"bad matched {rv['bad_matched']}  schema invalid {rv['schema_invalid']}")
        for origin, s in mine["empty"].items():
            lo, hi = s["ci95"]
            lines.append(f"  empty {origin:<17} {s['kept']}/{s['n']}  {s['rate']:.3f}  "
                         f"95% [{lo:.3f}, {hi:.3f}]")
        if mine.get("control"):
            c = mine["control"]
            if c["recall"] >= CONTROL_RECALL:
                verdict = "the matcher recovers the model's own reviewed claims"
            elif c["matcher_suspect"]:
                verdict = "most misses sit just under the threshold: the matcher, not the model, is failing"
            else:
                verdict = "most misses have no claim near them: the model did not reproduce its reviewed claims"
            lines.append(f"  matcher control: {c['interface']} recall {c['recall']:.3f}, verbatim "
                         f"{c['verbatim']}/{c['good_total']}, near misses {c['near_miss']}; {verdict}")
        versions = mine.get("matcher_versions") or []
        if len(versions) > 1:
            lines.append(f"  rows scored under matcher versions {' and '.join(map(str, versions))}: "
                         "not comparable; rescore with --rescore")
        if mine["unknown_cases"]:
            lines.append(f"  {mine['unknown_cases']} rows name a case this machine does not hold")
    lines.append(CAVEAT)
    return "\n".join(lines)


def rescore(rows: list[dict], cases) -> tuple[list[dict], int]:
    """Rows re-checked from their stored replies against `cases`, and how many had no reply file to read."""
    from evals.core import score
    from harness.holdout import base_id
    by_id = {c.id: c for c in cases}
    out, skipped = [], 0
    for r in rows:
        case = by_id.get(base_id(r["case_id"]))
        path = Path(r.get("artifact_path") or "")
        if case is None or not path.is_file():
            skipped += 1
            continue
        got = score(case, path.read_text(encoding="utf-8"))
        out.append({**r, "passed": bool(got.passed), "metrics": dict(got.metrics or {})})
    return out, skipped


def _rows(a) -> list[dict]:
    if a.results:
        return json.loads(Path(a.results).read_text(encoding="utf-8"))["rows"]
    from harness import runs
    with runs.store(None) as conn:
        run = runs.get(conn, a.run) if a.run else runs.newest(conn, lane="claims")
        if not run:
            raise SystemExit("no claims run in the store")
        return runs.rows(conn, run["id"])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.claims_report")
    ap.add_argument("results", nargs="?", default=None, help="a run's results.json")
    ap.add_argument("--run", type=int, default=None, help="a stored run id")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--rescore", action="store_true",
                    help="re-check each row's stored reply against the cases instead of its stored metrics")
    ap.add_argument("--cases", default=None, help="a claims case directory (default: this machine's)")
    a = ap.parse_args(argv)
    from evals import private
    from evals.core import load_cases
    cases = load_cases(a.cases, private=True) if a.cases else private.local_cases()
    rows = _rows(a)
    if a.rescore:
        rows, skipped = rescore(rows, cases)
        if skipped:
            print(f"{skipped} rows had no stored reply or case to rescore", file=sys.stderr)
    got = report(rows, cases)
    print(json.dumps(got, indent=1, sort_keys=True) if a.json else render(got))
    return 0


if __name__ == "__main__":
    sys.exit(main())
