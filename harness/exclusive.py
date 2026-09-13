"""One large working set at a time, across processes. Issue #137.

harness/jobs.py already enforces this, but only inside ONE process, so it
guards the MCP caller and leaves `lh` free to collide with itself. Every `lh`
invocation is a new process, so serialising them needs the operating system.

MEASURED, NOT ASSUMED. 2026-09-12 on the M2 Pro: `lh image` started while
`lh video` held the GPU. Both went straight at it, Metal returned
kIOGPUCommandBufferCallbackErrorTimeout, swap reached 8.0 GB of 9.2, and both
jobs were killed with no artifact. That is ERROR #33 and assumption C4.

An advisory file lock, NOT a pidfile. A lock the kernel holds is released when
the process dies however it dies -- killed, crashed, OOM -- so there is no
stale-lock path to get wrong. The 2026-09-12 incident ended in exactly the kill
a pidfile scheme handles worst.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
import time
from pathlib import Path

from harness import paths

#: What holds the lock, for a waiter to report. Best-effort and advisory: the
#: LOCK is the truth, this is only how a waiter says what it is waiting behind.
HOLDER = "generation.holder"
LOCK = "generation.lock"

#: Which lanes need the whole machine, declared once rather than tested for at
#: each call site.
#:
#: THE ABSENT ONES ARE THE DELIBERATE PART. svg, web, code and extract go to
#: mlx_lm.server, which already serialises through a single queue of its own;
#: say and hear go to a small resident audio service. Making `lh extract` --
#: one run of one prompt at 0.96s, M2 Pro, 2026-09-12, never repeated -- wait
#: behind a 40-minute video would be a downgrade wearing the clothes of
#: enforcement. The argument does not rest on that number: two orders of
#: magnitude is the claim, and a re-measurement would have to find seconds
#: rather than milliseconds to change it. The resource protected here is a
#: large LOCAL working set, not the machine.
EXCLUSIVE = {"image", "video"}

#: How often a waiter re-reads the holder to refresh what it reports.
POLL_SECONDS = 0.5


def _dir() -> Path:
    d = paths.home() / "queue"
    d.mkdir(parents=True, exist_ok=True)
    return d


def lock_path() -> Path:
    return _dir() / LOCK


def holder_path() -> Path:
    return _dir() / HOLDER


def holder() -> dict:
    """Who has it, or {}. Never raises: a waiter reporting nothing is fine,
    a waiter crashing because the holder exited mid-read is not."""
    try:
        return json.loads(holder_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def identity(record: dict) -> tuple:
    """WHAT is being waited on, with no clock in it.

    `describe` carries the elapsed seconds because a waiter wants to see them,
    and that made it useless for change detection: the string differs on every
    poll, so a caller comparing descriptions reprinted the same line once a
    second for the length of the wait. Measured 2026-09-12: 89 lines for one
    89-second wait behind an image.
    """
    return (record.get("kind"), record.get("pid"))


def describe(record: dict) -> str:
    kind = record.get("kind") or "something"
    pid = record.get("pid")
    since = record.get("since")
    for_ = f" for {time.time() - since:.0f}s" if isinstance(since, (int, float)) else ""
    return f"{kind}{for_}" + (f" (pid {pid})" if pid else "")


def _take(fd: int) -> bool:
    """Try to take the lock without blocking. True if taken."""
    if sys.platform == "win32":
        import msvcrt
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    import fcntl
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except OSError:
        return False


def _release(fd: int) -> None:
    with contextlib.suppress(OSError):
        if sys.platform == "win32":
            import msvcrt
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(fd, fcntl.LOCK_UN)


@contextlib.contextmanager
def held(kind: str, announce=None, poll: float = POLL_SECONDS):
    """Hold the machine for `kind`, waiting if someone else has it.

    `announce` is called with a sentence the first time this has to wait, and
    again whenever what it is waiting behind changes. Waiting silently for
    forty minutes is indistinguishable from hanging.
    """
    if kind not in EXCLUSIVE:
        yield False
        return

    path = lock_path()
    # Opened for the whole block: on Windows the lock is on the open handle,
    # and closing it anywhere would release it early.
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o644)
    try:
        waited = False
        said = None
        while not _take(fd):
            waited = True
            record = holder()
            who = identity(record)
            if announce and who != said:
                announce(f"waiting for {describe(record)} to finish")
                said = who
            time.sleep(poll)

        _write_holder(kind)
        try:
            yield waited
        finally:
            with contextlib.suppress(OSError):
                holder_path().unlink()
            _release(fd)
    finally:
        os.close(fd)


def _write_holder(kind: str) -> None:
    record = {"kind": kind, "pid": os.getpid(), "since": time.time()}
    with contextlib.suppress(OSError):
        holder_path().write_text(json.dumps(record), encoding="utf-8")
