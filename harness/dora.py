"""The four DORA metrics, from data this repo already produces.

No new instrumentation: every number here comes out of `git` and `gh`, both of
which have been recording since the first push. The gap was never collection,
it was that nothing read it back.

THREE OF THE FOUR ARE HONEST. Deployment frequency, lead time and time to
restore are what they say they are.

CHANGE FAILURE RATE IS NOT, and it is labelled rather than quietly reported
beside them. The cheap definition -- a CI run on the default branch that failed
-- measures CI HEALTH, not change quality: it counts a flaky runner as a failed
change and misses a bad merge that CI could not catch. A real one needs a
convention for "this commit repairs that one". Read it as a floor.

  uv run python -m harness.dora
  uv run python -m harness.dora --days 30 --json
"""
from __future__ import annotations

import json
import statistics
import subprocess
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone

DEFAULT_DAYS = 14
DEFAULT_BRANCH = "main"

#: Below this many samples a median is one event wearing a statistic's clothes,
#: so it is reported WITH its n and marked thin rather than printed bare. The
#: first run of this module produced "time to restore: median 46.6h" off a
#: single incident, which reads as a property of the team.
THIN = 3


def _git(*args: str) -> str:
    return subprocess.run(("git",) + args, capture_output=True, text=True,
                          check=True).stdout.strip()


def _gh_runs(limit: int, run=None) -> list[dict]:
    """CI runs, newest first. Returns [] when gh is absent or unauthenticated,
    because a metric that cannot be gathered is missing, not zero."""
    run = run or (lambda argv: subprocess.run(argv, capture_output=True,
                                              text=True, check=True).stdout)
    try:
        out = run(("gh", "run", "list", "--limit", str(limit), "--json",
                   "conclusion,createdAt,updatedAt,headBranch,event"))
        return json.loads(out)
    except Exception:  # noqa: BLE001 - absent, unauthenticated and offline are one case here
        return []


def _when(iso: str) -> datetime:
    return datetime.fromisoformat(iso.replace("Z", "+00:00"))


@dataclass(frozen=True)
class Change:
    sha: str
    merged_at: str
    first_commit_at: str
    subject: str

    @property
    def lead_hours(self) -> float:
        return (_when(self.merged_at)
                - _when(self.first_commit_at)).total_seconds() / 3600


def changes(days: int = DEFAULT_DAYS, branch: str = DEFAULT_BRANCH,
            git=None) -> list[Change]:
    """Merges into the default branch, with when each one's work STARTED.

    The start is the oldest commit reachable from the merge's second parent and
    not from its first, which is the branch's own history. An earlier cut used
    `--not main~50` and reported a lead time of six days for a branch created
    that morning: the window, not the branch.

    `git` is resolved HERE rather than defaulted in the signature. A default
    argument captures the function at DEFINITION time, so a test rebinding
    `dora._git` changed nothing and the real git ran underneath it: green on a
    machine with the branch checked out, red on three runners without it. The
    same trap is documented at harness/sensitivity.py:84.
    """
    git = git or _git
    since = f"--since={days}.days"
    try:
        raw = git("log", f"{branch}", "--merges", since,
                  "--pretty=%H%x1f%aI%x1f%P%x1f%s")
    except subprocess.CalledProcessError:
        # A shallow clone or a PR ref may not carry the branch locally. No
        # merges to report is the honest answer; an exception is not.
        return []
    out = []
    for line in filter(None, raw.splitlines()):
        sha, merged_at, parents, subject = line.split("\x1f")
        p = parents.split()
        if len(p) < 2:
            continue
        first = git("rev-list", "--reverse", f"{p[0]}..{p[1]}").splitlines()
        if not first:
            continue
        started = git("show", "-s", "--format=%aI", first[0])
        out.append(Change(sha[:12], merged_at, started, subject))
    return out


def restore_times(runs: list[dict], branch: str = DEFAULT_BRANCH) -> list[float]:
    """Hours from a red run on the branch to the next green one.

    Runs arrive newest first, so they are reversed: a restore is forward in
    time, and reading the list as given measures nothing at all.
    """
    mine = [r for r in reversed(runs)
            if r.get("headBranch") == branch and r.get("event") == "push"
            and r.get("conclusion") in ("success", "failure")]
    out, broke_at = [], None
    for r in mine:
        if r["conclusion"] == "failure" and broke_at is None:
            broke_at = _when(r["createdAt"])
        elif r["conclusion"] == "success" and broke_at is not None:
            out.append((_when(r["updatedAt"]) - broke_at).total_seconds() / 3600)
            broke_at = None
    return out


def report(days: int = DEFAULT_DAYS, branch: str = DEFAULT_BRANCH,
           git=None, runs=None) -> dict:
    runs = _gh_runs(300) if runs is None else runs
    merged = changes(days, branch, git=git)
    on_branch = [r for r in runs if r.get("headBranch") == branch
                 and r.get("event") == "push"
                 and r.get("conclusion") in ("success", "failure")]
    failed = [r for r in on_branch if r["conclusion"] == "failure"]
    restores = restore_times(runs, branch)
    leads = sorted(c.lead_hours for c in merged)

    return {
        "window_days": days,
        "branch": branch,
        "deployment_frequency": {
            "merges": len(merged),
            "per_day": round(len(merged) / days, 2) if days else None,
            "active_days": len({c.merged_at[:10] for c in merged}),
        },
        "lead_time_hours": {
            "median": round(statistics.median(leads), 2) if leads else None,
            "p90": round(leads[int(len(leads) * 0.9)], 2) if leads else None,
            "n": len(leads),
            "thin": len(leads) < THIN,
        },
        "change_failure_rate": {
            "value": round(len(failed) / len(on_branch), 3) if on_branch else None,
            "failed": len(failed),
            "runs": len(on_branch),
            "caveat": "CI runs on the branch that failed. This is CI health, "
                      "not change quality: it counts a flaky runner and misses "
                      "a bad merge CI could not catch. A floor, not a rate.",
        },
        "time_to_restore_hours": {
            "median": round(statistics.median(restores), 2) if restores else None,
            "n": len(restores),
            "thin": len(restores) < THIN,
        },
        "ci_history": bool(runs),
    }


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.dora",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--days", type=int, default=DEFAULT_DAYS)
    ap.add_argument("--branch", default=DEFAULT_BRANCH)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)

    got = report(a.days, a.branch)
    if a.json:
        print(json.dumps(got, indent=2))
        return 0

    d, lt, cfr, ttr = (got["deployment_frequency"], got["lead_time_hours"],
                       got["change_failure_rate"], got["time_to_restore_hours"])
    print(f"last {got['window_days']} days on {got['branch']}\n")
    print(f"  deployment frequency   {d['merges']} merges, "
          f"{d['per_day']}/day over {d['active_days']} active days")
    thin = lambda m: "  <- n is small; this is an anecdote" if m["thin"] else ""
    print(f"  lead time              median {lt['median']}h, p90 {lt['p90']}h "
          f"(n={lt['n']}){thin(lt)}")
    print(f"  time to restore        median {ttr['median']}h "
          f"(n={ttr['n']}){thin(ttr)}")
    print(f"  change failure rate    {cfr['value']} "
          f"({cfr['failed']}/{cfr['runs']} runs)")
    print(f"\n  {cfr['caveat']}")
    if not got["ci_history"]:
        print("\n  gh returned nothing, so the last two are MISSING, not zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
