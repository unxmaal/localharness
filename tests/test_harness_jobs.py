"""One generation at a time, and the wait attributable.

This machine holds 11.4 GiB for an image and 9.5 GiB for a video, and
mlx_lm.server serializes through a single queue while swapping models per
request. Two callers without a queue means each inserts a full model load into
the other's request, on a box that swaps if both hold weights at once.

PLAN.md has said "measurement isolation is assumed, not enforced" since the
eval suite was built. This is the enforcement, and it matters before a second
agent can call in at all.
"""
import threading
import time

import pytest

from harness import jobs


@pytest.fixture
def q():
    queue = jobs.Queue()
    yield queue
    queue.shutdown()


def test_a_job_returns_an_id_immediately(q):
    """Submitting a 40-minute video must not block the caller for 40 minutes."""
    started = time.time()
    job_id = q.submit("video", lambda: time.sleep(0.3) or "done")
    assert job_id
    assert time.time() - started < 0.2


def test_a_finished_job_carries_its_result(q):
    job_id = q.submit("svg", lambda: "<svg/>")
    assert q.wait(job_id, timeout=5).result == "<svg/>"
    assert q.status(job_id).state == "done"


def test_a_failed_job_is_a_state_not_an_exception(q):
    def boom():
        raise RuntimeError("mflux exited 3")
    job_id = q.submit("image", boom)
    job = q.wait(job_id, timeout=5)
    assert job.state == "failed"
    assert "mflux exited 3" in job.error
    assert job.result is None


def test_only_one_job_runs_at_a_time(q):
    """The whole point. Two 11.4 GiB generations at once on 32GB swaps."""
    concurrent, peak = [], []
    lock = threading.Lock()

    def work():
        with lock:
            concurrent.append(1)
            peak.append(len(concurrent))
        time.sleep(0.15)
        with lock:
            concurrent.pop()
        return "ok"

    ids = [q.submit("image", work) for _ in range(4)]
    for job_id in ids:
        assert q.wait(job_id, timeout=10).state == "done"
    assert max(peak) == 1, f"ran {max(peak)} at once"


def test_the_next_job_up_reports_nothing_ahead_of_it(q):
    """Zero has to mean next, or the number cannot be read."""
    job_id = q.submit("svg", lambda: None)
    assert q.wait(job_id, timeout=5).state == "done"
    second = q.submit("svg", lambda: None)
    q.wait(second, timeout=5)
    assert q.status(second).ahead == 0


def test_jobs_run_in_the_order_they_were_submitted(q):
    order = []
    ids = [q.submit("svg", lambda n=n: order.append(n)) for n in range(4)]
    for job_id in ids:
        q.wait(job_id, timeout=10)
    assert order == [0, 1, 2, 3]


def test_a_waiting_job_can_say_how_many_are_ahead_of_it(q):
    """A 90-second wait that reads as 'second in line behind an image' is a
    queue. One that reads as nothing is a slow tool."""
    started = threading.Event()
    release = threading.Event()

    def hold():
        # Wait for the worker to have PICKED UP the first job before asserting
        # anything about the second. Polling for `ahead == 1` on a 1s budget
        # passed alone and failed under load, which is a flaky test rather than
        # a slow queue.
        started.set()
        release.wait(5)

    q.submit("image", hold)
    assert started.wait(5), "the worker never started the first job"
    second = q.submit("svg", lambda: None)
    # ONE ahead, and it is the running one. Counting only the queued jobs told
    # a caller waiting behind a 54-second image that nothing was ahead of it.
    assert q.status(second).ahead == 1
    assert q.status(second).state == "queued"
    release.set()


def test_the_running_job_is_named_so_a_wait_is_attributable(q):
    release = threading.Event()
    running = q.submit("image", lambda: release.wait(5))
    for _ in range(50):
        if q.status(running).state == "running":
            break
        time.sleep(0.02)
    assert q.status(running).kind == "image"
    assert q.running() and q.running().kind == "image"
    release.set()


def test_an_unknown_job_id_is_none_not_a_crash(q):
    assert q.status("no-such-job") is None


def test_wait_returns_none_on_timeout_rather_than_hanging(q):
    release = threading.Event()
    job_id = q.submit("video", lambda: release.wait(5))
    assert q.wait(job_id, timeout=0.1) is None
    release.set()


def test_elapsed_is_measured_so_a_caller_can_report_progress(q):
    job_id = q.submit("svg", lambda: time.sleep(0.1))
    q.wait(job_id, timeout=5)
    assert q.status(job_id).seconds >= 0.1
