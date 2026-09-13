"""Which lines this change added that nothing runs.

A repo-wide coverage percentage is the whole reported as if it described the
part: it barely moves when a change adds fifty unexercised lines, and it says
nothing at all about the diff in front of a reviewer. The useful question is
narrow and answerable -- of the lines THIS change adds, which does no test
execute?

It is also, deliberately, not a gate. Coverage is blind to whether a line that
ran was asserted about at all, so a threshold buys a number rather than a
property. Report it, read it, decide.

  make coverage                       # writes coverage.json
  uv run python -m harness.covdiff    # the lines this branch added that nothing runs
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

DEFAULT_BASE = "origin/main"
COVERAGE_JSON = "coverage.json"


def added_lines(base: str = DEFAULT_BASE, root: Path | None = None,
                committed_only: bool = False) -> dict[str, set[int]]:
    """Line numbers this change ADDS, per file, from a zero-context diff.

    THE WORKING TREE COUNTS BY DEFAULT. The first cut diffed `base...HEAD` and
    reported "0 added lines that no test runs" for a change consisting of two
    brand-new modules, because neither was committed yet. A differential that
    goes quiet exactly when someone is mid-change is worse than none: it reads
    as a clean bill.

    An UNTRACKED file is entirely new, so every line of it is added.
    """
    root = root or Path.cwd()
    spec = [f"{base}...HEAD"] if committed_only else [base]
    raw = subprocess.run(("git", "-C", str(root), "diff", "--unified=0", *spec),
                         capture_output=True, text=True, check=True).stdout
    out: dict[str, set[int]] = {}
    path = None
    for line in raw.splitlines():
        if line.startswith("+++ b/"):
            path = line[6:]
        elif line.startswith("@@") and path:
            # @@ -old,n +new,m @@
            new = line.split("+")[1].split(" ")[0]
            start, _, count = new.partition(",")
            n = int(count) if count else 1
            if n:
                out.setdefault(path, set()).update(
                    range(int(start), int(start) + n))
    if not committed_only:
        others = subprocess.run(
            ("git", "-C", str(root), "ls-files", "--others", "--exclude-standard"),
            capture_output=True, text=True, check=True).stdout.splitlines()
        for rel in filter(None, others):
            f = root / rel
            try:
                n = len(f.read_text(encoding="utf-8").splitlines())
            except (OSError, UnicodeDecodeError):
                continue
            if n:
                out.setdefault(rel, set()).update(range(1, n + 1))
    return out


def uncovered(base: str = DEFAULT_BASE, root: Path | None = None,
              report: dict | None = None,
              committed_only: bool = False) -> dict[str, list[int]]:
    """Added lines that the coverage run did not execute.

    A file absent from the coverage report is NOT reported as fully uncovered:
    it may be a script, a config, or something coverage was never pointed at,
    and claiming a test gap there is a false positive that gets this switched
    off. Only files coverage actually measured are judged.
    """
    root = root or Path.cwd()
    if report is None:
        p = root / COVERAGE_JSON
        if not p.exists():
            raise FileNotFoundError(
                f"{COVERAGE_JSON} not found. Run `make coverage` first.")
        report = json.loads(p.read_text(encoding="utf-8"))
    files = report.get("files", {})
    out = {}
    for path, lines in sorted(added_lines(base, root, committed_only).items()):
        measured = files.get(path)
        if not measured:
            continue
        missing = sorted(lines & set(measured.get("missing_lines", [])))
        if missing:
            out[path] = missing
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.covdiff",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--base", default=DEFAULT_BASE)
    ap.add_argument("--committed", action="store_true",
                    help="ignore the working tree; compare commits only")
    a = ap.parse_args(argv)

    try:
        found = uncovered(a.base, committed_only=a.committed)
    except FileNotFoundError as exc:
        print(exc)
        return 1
    total = sum(len(v) for v in found.values())
    for path, lines in found.items():
        print(f"{path}: {', '.join(str(n) for n in lines)}")
    print(f"{total} added line(s) that no test runs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
