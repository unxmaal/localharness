"""One generation at a time, with the wait attributable.

This machine holds 11.4 GiB for an image and 9.5 GiB for a video, and
`mlx_lm.server` serializes through a single queue while swapping models per
request. Two callers without a queue in front of them means each inserts a full
model load into the other's request, on a box that swaps if both hold weights
at once. A second agent calling in from the LAN makes that a certainty rather
than a hazard.

PLAN.md has said "measurement isolation is assumed, not enforced" since the
eval suite was built. This is the enforcement.

A single worker thread rather than a pool, deliberately: the constraint being
respected is 32 GB of unified memory, and a pool of one is the honest way to
say so. Jobs live in memory and die with the process, which is right while
artifacts are the durable thing and a job record is not.
"""
from __future__ import annotations

import itertools
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field


@dataclass
class Job:
    id: str
    #: What kind of work this is ("image", "video", "svg"...). Carried so a
    #: caller stuck behind something can be told what it is stuck behind.
    kind: str
    state: str = "queued"          # queued | running | done | failed
    result: object = None
    error: str = ""
    submitted: float = field(default_factory=time.monotonic)
    started: float | None = None
    finished: float | None = None
    #: How many jobs are ahead of this one. Only meaningful while queued.
    ahead: int = 0

    @property
    def seconds(self) -> float:
        """Time spent RUNNING, not time since submission: a caller comparing
        this against the eval's median would otherwise be comparing a number
        that includes someone else's job."""
        if self.started is None:
            return 0.0
        return (self.finished or time.monotonic()) - self.started


class Queue:
    """Submit work, get an id back immediately, ask about it later."""

    def __init__(self):
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []      # queued ids, oldest first
        self._pending: queue.Queue = queue.Queue()
        self._lock = threading.Lock()
        self._done = threading.Condition(self._lock)
        self._running: str | None = None
        self._counter = itertools.count()
        self._stop = False
        self._worker = threading.Thread(target=self._run, daemon=True,
                                        name="lh-queue")
        self._worker.start()

    # -- submitting ---------------------------------------------------------

    def submit(self, kind: str, fn) -> str:
        """Queue `fn`. Returns immediately with an id.

        The id is short and unique rather than sequential: a caller quoting
        "job 3" across a restart would be quoting someone else's job.
        """
        job_id = f"{kind}-{next(self._counter)}-{uuid.uuid4().hex[:6]}"
        job = Job(id=job_id, kind=kind)
        with self._lock:
            self._jobs[job_id] = job
            self._order.append(job_id)
            self._renumber()
        self._pending.put((job_id, fn))
        return job_id

    # -- asking -------------------------------------------------------------

    def status(self, job_id: str) -> Job | None:
        with self._lock:
            job = self._jobs.get(job_id)
            if job is not None and job.state == "queued":
                self._renumber()
            return job

    def running(self) -> Job | None:
        with self._lock:
            return self._jobs.get(self._running) if self._running else None

    def wait(self, job_id: str, timeout: float = 300.0) -> Job | None:
        """Block until the job settles. None on timeout, so a caller that is
        willing to wait ten seconds is not committed to forty minutes."""
        deadline = time.monotonic() + timeout
        with self._done:
            while True:
                job = self._jobs.get(job_id)
                if job is not None and job.state in ("done", "failed"):
                    return job
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return None
                self._done.wait(remaining)

    def jobs(self) -> list[Job]:
        with self._lock:
            return list(self._jobs.values())

    # -- internals ----------------------------------------------------------

    def _renumber(self) -> None:
        """Caller holds the lock."""
        for position, queued_id in enumerate(self._order):
            self._jobs[queued_id].ahead = position

    def _run(self) -> None:
        while True:
            item = self._pending.get()
            if item is None:
                return
            job_id, fn = item
            with self._lock:
                if self._stop:
                    return
                job = self._jobs[job_id]
                job.state = "running"
                job.started = time.monotonic()
                job.ahead = 0
                if job_id in self._order:
                    self._order.remove(job_id)
                self._running = job_id
                self._renumber()
            try:
                value = fn()
                outcome, error = "done", ""
            except Exception as exc:          # noqa: BLE001
                # A generator that dies must be a failed job, never a dead
                # worker: one bad prompt would otherwise take the queue with it.
                value, outcome, error = None, "failed", f"{exc}"
            with self._done:
                job.state = outcome
                job.result = value
                job.error = error
                job.finished = time.monotonic()
                self._running = None
                self._done.notify_all()

    def shutdown(self) -> None:
        with self._lock:
            self._stop = True
        self._pending.put(None)
