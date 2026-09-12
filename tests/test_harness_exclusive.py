"""One large working set at a time, across PROCESSES. Issue #137.

tests/test_harness_jobs.py covers the in-process queue. The gap this closes is
the one that actually bit: two separate `lh` invocations, each with its own
interpreter, both going at the GPU. A test that only spawns threads cannot see
it, so the collision tests here spawn real subprocesses.
"""
import json
import subprocess
import sys
import textwrap
import time
from pathlib import Path

import pytest

from harness import exclusive

REPO = Path(__file__).resolve().parent.parent


@pytest.fixture(autouse=True)
def home(tmp_path, monkeypatch):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    return tmp_path


# ---- what queues, and what deliberately does not --------------------------

def test_the_expensive_lanes_are_the_exclusive_ones():
    assert exclusive.EXCLUSIVE == {"image", "video"}


@pytest.mark.parametrize("lane", ["svg", "web", "code", "extract", "say", "hear"])
def test_a_cheap_lane_does_not_wait(lane):
    """`lh extract` measured 0.96s. Putting it behind a forty-minute video
    would be a downgrade wearing the clothes of enforcement: those lanes go to
    servers that already serialise."""
    started = time.perf_counter()
    with exclusive.held(lane) as waited:
        assert waited is False
    assert time.perf_counter() - started < 1.0


def test_a_cheap_lane_takes_no_lock_even_while_one_is_held():
    with exclusive.held("video"):
        with exclusive.held("extract") as waited:
            assert waited is False


# ---- the lock itself ------------------------------------------------------

def test_the_holder_says_what_it_is_and_who_has_it():
    with exclusive.held("image"):
        record = exclusive.holder()
    assert record["kind"] == "image"
    assert record["pid"] > 0


def test_the_holder_record_is_gone_afterwards():
    with exclusive.held("image"):
        pass
    assert exclusive.holder() == {}


def test_a_torn_holder_file_reads_as_nobody_rather_than_raising():
    """The holder is advisory; the lock is the truth. A waiter that crashes
    because the holder exited mid-write is worse than one that says nothing."""
    exclusive.holder_path().parent.mkdir(parents=True, exist_ok=True)
    exclusive.holder_path().write_text("{not json", encoding="utf-8")
    assert exclusive.holder() == {}


def test_describe_names_the_lane_and_the_wait():
    said = exclusive.describe({"kind": "video", "pid": 42, "since": time.time() - 90})
    assert "video" in said and "90s" in said and "42" in said


def test_describe_survives_an_empty_record():
    assert exclusive.describe({})


def test_identity_ignores_the_clock():
    """`describe` carries elapsed seconds so a waiter can see them, which made
    it useless for change detection: one 89-second wait printed 89 identical
    lines, differing only in the count. Change detection uses this instead."""
    a = {"kind": "image", "pid": 7, "since": 100.0}
    b = {"kind": "image", "pid": 7, "since": 500.0}
    assert exclusive.identity(a) == exclusive.identity(b)
    assert exclusive.describe(a) != exclusive.describe(b)
    assert exclusive.identity(a) != exclusive.identity({"kind": "video", "pid": 7})


# ---- the collision that actually happened ---------------------------------

HOLD = textwrap.dedent("""
    import sys, time
    sys.path.insert(0, {repo!r})
    from harness import exclusive
    with exclusive.held("video"):
        print("held", flush=True)
        time.sleep({seconds})
""")

TAKE = textwrap.dedent("""
    import json, sys, time
    sys.path.insert(0, {repo!r})
    from harness import exclusive
    started = time.perf_counter()
    saw = []
    with exclusive.held("image", announce=saw.append, poll=0.05) as waited:
        print(json.dumps({{"waited": waited,
                           "seconds": time.perf_counter() - started,
                           "said": saw}}), flush=True)
""")


def _script(body, **kw):
    return [sys.executable, "-c", body.format(repo=str(REPO), **kw)]


@pytest.mark.parametrize("seconds", [1.0])
def test_a_second_process_waits_instead_of_colliding(seconds, home, monkeypatch):
    """THE POINT OF THIS FILE. 2026-09-12: `lh image` was started while
    `lh video` held the GPU. Both ran, Metal returned a command-buffer timeout,
    swap hit 8.0 GB of 9.2, and the machine killed both with no artifact.

    The fix is not that the second one fails faster. It is that it waits."""
    env = {**dict(__import__("os").environ), "LOCALHARNESS_HOME": str(home)}
    holder = subprocess.Popen(_script(HOLD, seconds=seconds),
                              stdout=subprocess.PIPE, text=True, env=env)
    assert holder.stdout.readline().strip() == "held"

    taker = subprocess.run(_script(TAKE), capture_output=True, text=True, env=env)
    holder.wait(timeout=30)

    assert taker.returncode == 0, taker.stderr
    got = json.loads(taker.stdout)
    assert got["waited"] is True
    assert got["seconds"] >= seconds * 0.8, "took the lock while it was held"
    assert any("video" in s for s in got["said"]), got["said"]


def test_the_lock_is_released_when_the_holder_is_killed(home):
    """A pidfile scheme handles this case worst, and this case is exactly how
    the 2026-09-12 incident ended: the OOM killer took both processes. An
    advisory lock the kernel holds is released however the process dies."""
    env = {**dict(__import__("os").environ), "LOCALHARNESS_HOME": str(home)}
    holder = subprocess.Popen(_script(HOLD, seconds=60),
                              stdout=subprocess.PIPE, text=True, env=env)
    assert holder.stdout.readline().strip() == "held"
    holder.kill()
    holder.wait(timeout=10)

    started = time.perf_counter()
    with exclusive.held("image") as waited:
        pass
    assert time.perf_counter() - started < 5.0, "a dead holder blocked the lock"
