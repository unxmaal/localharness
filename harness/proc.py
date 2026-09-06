"""Run an external generator and measure what it cost.

Peak memory is the reason this exists. On a 32GB machine the binding constraint
is a single phase's peak, not the model's size on disk: the h3 video runs peak
at 9.5GiB from a 134GiB checkpoint because the DiT streams. Any number that
cannot see that difference is decoration.

The number reported is macOS's `phys_footprint`, via `/usr/bin/time -l`'s "peak
memory footprint" line. It accounts for compressed and swapped pages and is
what Activity Monitor shows.

NOT resource.getrusage(RUSAGE_CHILDREN).ru_maxrss: that is a monotone
high-water mark across every child the process has ever waited on, so a
before/after delta reads 0 for each child after the largest. A published "2x
the memory" comparison rested on that error.
"""
from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# `/usr/bin/time -l` appends its report to ITS OWN stderr, which by default is
# the same stream the child writes to. Interleaving them buries a one-line error
# message in twenty lines of counters, so the child's stderr is diverted and
# only the report is read back from the pipe.
_PEAK = re.compile(r"(\d+)\s+peak memory footprint")


@dataclass(frozen=True)
class Outcome:
    returncode: int
    seconds: float
    peak_kb: int
    stdout: str = ""
    stderr: str = ""

    @property
    def ok(self) -> bool:
        return self.returncode == 0


def run(argv: list[str], timeout: float | None = None,
        stream: bool = False, cwd: str | Path | None = None) -> Outcome:
    """Run argv to completion, returning its exit status, wall time and peak.

    `stream=True` lets the child's output reach the terminal live, which is not
    optional for video: a 40-minute run with no visible progress is
    indistinguishable from a hang.

    Raises FileNotFoundError if the binary is absent and subprocess.TimeoutExpired
    on timeout, so callers can tell "not installed" from "crashed" from "hung".
    """
    if "/" not in argv[0] and shutil.which(argv[0]) is None:
        # Without this the sh wrapper below exits 127 and a missing dependency
        # reads as a mysterious failure of the generator itself.
        raise FileNotFoundError(argv[0])

    started = time.monotonic()
    if stream:
        # The child's stderr joins stdout on the inherited terminal, leaving the
        # pipe carrying nothing but time's report.
        wrapped = ["/usr/bin/time", "-l", "/bin/sh", "-c",
                   'exec "$@" 2>&1', "sh", *argv]
        proc = subprocess.run(wrapped, stderr=subprocess.PIPE, text=True,
                              timeout=timeout, cwd=cwd)
        return Outcome(proc.returncode, round(time.monotonic() - started, 3),
                       _peak_kb(proc.stderr), "", "")

    with tempfile.NamedTemporaryFile("w+", suffix=".err") as errf:
        wrapped = ["/usr/bin/time", "-l", "/bin/sh", "-c",
                   'exec "$@" 2>"$_H_ERR"', "sh", *argv]
        proc = subprocess.run(wrapped, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd,
                              env={**_env(), "_H_ERR": errf.name})
        errf.seek(0)
        child_stderr = errf.read()

    return Outcome(proc.returncode, round(time.monotonic() - started, 3),
                   _peak_kb(proc.stderr), proc.stdout, child_stderr)


def _env() -> dict:
    import os
    return dict(os.environ)


def _peak_kb(time_report: str) -> int:
    m = _PEAK.search(time_report or "")
    return int(m.group(1)) // 1024 if m else 0
