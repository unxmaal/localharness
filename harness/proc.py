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

ON LINUX `/usr/bin/time` is a different program that spells the same request
`-v`, and it is not installed by default at all -- the `time` most shells have
is a builtin with no memory reporting. Rather than depend on a package, the
child is reaped with `os.wait4`, whose rusage belongs to THAT child alone and
so carries none of the accumulation the ru_maxrss note above warns about.
`ru_maxrss` is peak RSS and not a footprint: it cannot see pages that were
swapped out. That makes it a THIRD instrument, and `PEAK_METHOD` says which one
produced a number so two of them are never ranked in one table.

ON WINDOWS there is no `/usr/bin/time -l`, and the obvious substitute is a
trap: GetProcessMemoryInfo answers for an exited process, but the working set
is torn down at exit, so it reported 5.2 MB for a child that had just
allocated 300 MB. The number that survives is the JOB OBJECT's, which the
kernel keeps for the job rather than the process -- measured 306.5 MB for that
same 300 MB child. Each run gets its own job, so this is per-run and not the
monotone mark the ru_maxrss note warns about.
"""
from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path

# `/usr/bin/time -l` appends its report to ITS OWN stderr, which by default is
# the same stream the child writes to. Interleaving them buries a one-line error
# message in twenty lines of counters, so the child's stderr is diverted and
# only the report is read back from the pipe.
_PEAK = re.compile(r"(\d+)\s+peak memory footprint")

#: WHICH INSTRUMENT MEASURED peak_kb. A job object's peak, a phys_footprint and
#: a maxrss are three different quantities, and the same card under two
#: operating systems is otherwise indistinguishable in a receipt. Recorded, so
#: `evals.core.comparable` can refuse rather than average them.
PEAK_METHOD = ("phys_footprint" if sys.platform == "darwin"
               else "job_peak_process" if sys.platform == "win32"
               else "ru_maxrss")


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
    # A bare name is looked up on PATH; anything with a separator in it is
    # already a path and is asked whether it exists. Testing for "/" alone said
    # "not installed" for every absolute Windows path, which uses backslashes
    # -- and shutil.which then refused an extensionless file besides, because
    # it consults PATHEXT.
    head = argv[0]
    separators = [os.sep] + ([os.altsep] if os.altsep else [])
    if any(sep in head for sep in separators):
        if not Path(head).exists():
            raise FileNotFoundError(head)
    elif shutil.which(head) is None:
        # Without this the sh wrapper below exits 127 and a missing dependency
        # reads as a mysterious failure of the generator itself.
        raise FileNotFoundError(head)

    # perf_counter, not monotonic: monotonic is GetTickCount64 on Windows and
    # quantises to 15.6ms, which is coarse enough to report a 0.15s run short.
    started = time.perf_counter()
    if sys.platform == "win32":
        return _run_windows(argv, timeout, stream, cwd, started)
    if sys.platform != "darwin":
        return _run_posix(argv, timeout, stream, cwd, started)
    if stream:
        # The child's stderr joins stdout on the inherited terminal, leaving the
        # pipe carrying nothing but time's report.
        wrapped = ["/usr/bin/time", "-l", "/bin/sh", "-c",
                   'exec "$@" 2>&1', "sh", *argv]
        proc = subprocess.run(wrapped, stderr=subprocess.PIPE, text=True,
                              timeout=timeout, cwd=cwd)
        return Outcome(proc.returncode, round(time.perf_counter() - started, 3),
                       _peak_kb(proc.stderr), "", "")

    with tempfile.NamedTemporaryFile("w+", suffix=".err") as errf:
        wrapped = ["/usr/bin/time", "-l", "/bin/sh", "-c",
                   'exec "$@" 2>"$_H_ERR"', "sh", *argv]
        proc = subprocess.run(wrapped, capture_output=True, text=True,
                              timeout=timeout, cwd=cwd,
                              env={**_env(), "_H_ERR": errf.name})
        errf.seek(0)
        child_stderr = errf.read()

    return Outcome(proc.returncode, round(time.perf_counter() - started, 3),
                   _peak_kb(proc.stderr), proc.stdout, child_stderr)


def _env() -> dict:
    import os
    return dict(os.environ)


def _peak_kb(time_report: str) -> int:
    m = _PEAK.search(time_report or "")
    return int(m.group(1)) // 1024 if m else 0


# ---- Linux, and any other POSIX that is not macOS -------------------------


def _wait4(proc, argv: list[str], timeout: float | None) -> tuple[int, int]:
    """Reap `proc` ourselves and keep its rusage. Returns (returncode, peak_kb).

    `os.wait4` answers for ONE child, so this is that run's peak and not a
    high-water mark across every child ever waited on. subprocess has no way
    to ask for rusage, so the child is reaped here and `proc.returncode` is set
    afterwards to stop Popen reaping a pid that is already gone.
    """
    deadline = None if timeout is None else time.perf_counter() + timeout
    wait = 0.001
    while True:
        pid, status, usage = os.wait4(proc.pid, os.WNOHANG)
        if pid:
            proc.returncode = os.waitstatus_to_exitcode(status)
            # ru_maxrss is KILOBYTES on Linux and BYTES on macOS -- measured
            # here, 13.8 MB reported as 13893632. macOS never reaches this
            # branch, and a platform that reports bytes and is routed here
            # would read a thousandfold high rather than fail.
            return proc.returncode, int(usage.ru_maxrss)
        if deadline is not None and time.perf_counter() > deadline:
            proc.kill()
            os.wait4(proc.pid, 0)
            proc.returncode = -9
            raise subprocess.TimeoutExpired(argv, timeout)
        time.sleep(wait)
        wait = min(wait * 2, 0.05)


def _run_posix(argv: list[str], timeout: float | None, stream: bool,
               cwd: str | Path | None, started: float) -> Outcome:
    """Run argv, reaping it with wait4 so its peak RSS survives the exit.

    Output goes to FILES rather than pipes. Nothing here drains a pipe while
    waiting, so a child that fills one would deadlock; a file has no such
    limit, and the macOS branch already writes the child's stderr to one for
    an unrelated reason.
    """
    if stream:
        proc = subprocess.Popen(argv, cwd=cwd)
        code, peak_kb = _wait4(proc, argv, timeout)
        return Outcome(code, round(time.perf_counter() - started, 3),
                       peak_kb, "", "")

    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "out"
        err_path = Path(tmp) / "err"
        with open(out_path, "w", encoding="utf-8") as out, \
                open(err_path, "w", encoding="utf-8") as err:
            proc = subprocess.Popen(argv, stdout=out, stderr=err, cwd=cwd)
            code, peak_kb = _wait4(proc, argv, timeout)
        return Outcome(code, round(time.perf_counter() - started, 3), peak_kb,
                       out_path.read_text(encoding="utf-8", errors="replace"),
                       err_path.read_text(encoding="utf-8", errors="replace"))


# ---- Windows --------------------------------------------------------------

#: JobObjectExtendedLimitInformation. See the module docstring for why the job
#: is the thing asked rather than the process.
_JOB_EXTENDED_LIMIT_INFORMATION = 9


def _job_structs():
    """The ctypes shapes, built on demand so importing this module on a Mac
    never touches ctypes.wintypes."""
    import ctypes
    from ctypes import wintypes

    class Basic(ctypes.Structure):
        _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64),
                    ("PerJobUserTimeLimit", ctypes.c_int64),
                    ("LimitFlags", wintypes.DWORD),
                    ("MinimumWorkingSetSize", ctypes.c_size_t),
                    ("MaximumWorkingSetSize", ctypes.c_size_t),
                    ("ActiveProcessLimit", wintypes.DWORD),
                    ("Affinity", ctypes.POINTER(ctypes.c_ulong)),
                    ("PriorityClass", wintypes.DWORD),
                    ("SchedulingClass", wintypes.DWORD)]

    class IoCounters(ctypes.Structure):
        _fields_ = [(name, ctypes.c_uint64) for name in
                    ("ReadOperationCount", "WriteOperationCount",
                     "OtherOperationCount", "ReadTransferCount",
                     "WriteTransferCount", "OtherTransferCount")]

    class Extended(ctypes.Structure):
        _fields_ = [("BasicLimitInformation", Basic),
                    ("IoInfo", IoCounters),
                    ("ProcessMemoryLimit", ctypes.c_size_t),
                    ("JobMemoryLimit", ctypes.c_size_t),
                    ("PeakProcessMemoryUsed", ctypes.c_size_t),
                    ("PeakJobMemoryUsed", ctypes.c_size_t)]

    return ctypes.WinDLL("kernel32", use_last_error=True), Extended


def _run_windows(argv: list[str], timeout: float | None, stream: bool,
                 cwd: str | Path | None, started: float) -> Outcome:
    """Run argv under a fresh job object and read the job's peak afterwards.

    The child is assigned to the job immediately after it is created. That is
    a few microseconds after launch rather than before it, so a process that
    exits in less time than that would be missed; nothing this runs comes
    close, and the alternative is creating it suspended, which subprocess does
    not expose a way to resume.
    """
    import ctypes

    kernel32, Extended = _job_structs()
    job = kernel32.CreateJobObjectW(None, None)
    pipe = None if stream else subprocess.PIPE
    proc = subprocess.Popen(argv, stdout=pipe, stderr=pipe, text=True, cwd=cwd)
    try:
        if job:
            kernel32.AssignProcessToJobObject(job, int(proc._handle))
        try:
            out, err = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Leaving it running would hold the pipes open and hang the read.
            proc.kill()
            proc.communicate()
            raise
        peak_kb = 0
        if job:
            info = Extended()
            if kernel32.QueryInformationJobObject(
                    job, _JOB_EXTENDED_LIMIT_INFORMATION, ctypes.byref(info),
                    ctypes.sizeof(info), None):
                # PeakProcessMemoryUsed, not PeakJobMemoryUsed: the macOS side
                # reports one process's footprint, and a number that silently
                # means something wider on one machine defeats the comparison
                # these results exist for.
                peak_kb = info.PeakProcessMemoryUsed // 1024
    finally:
        if job:
            kernel32.CloseHandle(job)

    return Outcome(proc.returncode, round(time.perf_counter() - started, 3),
                   peak_kb, out or "", err or "")
