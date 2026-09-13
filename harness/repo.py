"""Which files a reader would get, which is not the same as which are committed.

BUILT 2026-09-12 for #160, after `git ls-files` produced a check that passed
locally and failed in CI on the same commit (RULE #231). The sequence repeats
for any scanner keyed on the index:

  1. write the scanner, run it, clean
  2. write its tests, which are full of deliberate examples of the thing being
     detected, because that is what a detector's fixtures ARE
  3. `make lint` -> green, because the new file is not in the index yet
  4. commit, push, and watch three runners go red on the same content

The scanner was right both times. Its INPUT differed, and a check whose input
differs between the desk and the runner is not a check.
"""
from __future__ import annotations

import subprocess
from pathlib import Path


def publishable(root: Path) -> list[Path]:
    """Everything a push would publish: tracked, plus untracked and not ignored.

    `git ls-files` alone answers "what is committed". The question a privacy
    scan or an assertions inventory is actually asking is "what would a reader
    see", and the file written thirty seconds ago is the one most likely to
    carry something that should not ship.
    """
    seen, out = set(), []
    for args in (("ls-files",), ("ls-files", "--others", "--exclude-standard")):
        r = subprocess.run(("git", "-C", str(root)) + args,
                           capture_output=True, text=True, check=True)
        for line in r.stdout.splitlines():
            if line and line not in seen:
                seen.add(line)
                out.append(root / line)
    return out
