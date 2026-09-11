"""Running an external generator and measuring what it cost.

This is the layer both the CLI and the eval suite stand on, so it has to report
the same numbers to both. The measurement it exists for is peak memory: on a
32GB machine the binding constraint is one phase's peak, not the model's size
on disk.
"""
import subprocess
import sys
import time

import pytest

from harness import proc
from harness.proc import Outcome, run

PY = sys.executable

#: The POSIX branch does not exist on Windows: there is no os.wait4 there, and
#: the job object is that machine's answer. Everything else in this file runs
#: on all three.
posix_only = pytest.mark.skipif(sys.platform == "win32",
                                reason="the wait4 path is POSIX; Windows uses "
                                       "a job object")


def test_returns_exit_code_and_output():
    r = run([PY, "-c", "import sys; print('hi'); sys.exit(0)"])
    assert isinstance(r, Outcome)
    assert r.returncode == 0
    assert "hi" in r.stdout


def test_nonzero_exit_is_reported_not_raised():
    r = run([PY, "-c", "import sys; sys.exit(3)"])
    assert r.returncode == 3
    assert not r.ok


def test_peak_memory_is_measured():
    r = run([PY, "-c", "x = bytearray(200 * 1024 * 1024); print(len(x))"])
    assert r.peak_kb > 150 * 1024, (
        f"peak {r.peak_kb}KB did not see a 200MB allocation; the instrument is "
        "reading the wrong field")


def test_peak_memory_tracks_the_child_not_a_high_water_mark():
    """The ru_maxrss trap: a monotone mark reads the same for a small child
    after a large one. Two runs of different sizes must differ."""
    big = run([PY, "-c", "x = bytearray(400 * 1024 * 1024); print(len(x))"])
    small = run([PY, "-c", "print(1)"])
    assert small.peak_kb < big.peak_kb / 2


def test_missing_binary_raises_filenotfound_not_a_confusing_exit_code():
    with pytest.raises(FileNotFoundError):
        run(["definitely-not-a-real-binary-xyz"])


def test_timeout_raises():
    import subprocess
    with pytest.raises(subprocess.TimeoutExpired):
        run([PY, "-c", "import time; time.sleep(30)"], timeout=1.0)


def test_elapsed_is_wall_time():
    r = run([PY, "-c", "import time; time.sleep(0.3)"])
    assert r.seconds >= 0.3


def test_stderr_is_captured_without_the_time_instrumentation_leaking():
    """/usr/bin/time -l writes its report to stderr. If that is handed back as
    the child's stderr, every error message is buried in 20 lines of counters."""
    r = run([PY, "-c", "import sys; sys.stderr.write('boom')"])
    assert "boom" in r.stderr
    assert "peak memory footprint" not in r.stderr


def test_streaming_mode_does_not_capture(capfd):
    """Video takes 40 minutes. Progress has to reach the terminal live."""
    run([PY, "-c", "print('live')"], stream=True)
    assert "live" in capfd.readouterr().out


# ---- the third measurement path -------------------------------------------


def test_the_peak_method_is_named():
    """A job object's peak, a phys_footprint and an ru_maxrss are three
    different quantities. A number that does not say which one produced it
    cannot be ranked against one from another machine."""
    assert proc.PEAK_METHOD in {"phys_footprint", "job_peak_process",
                                "ru_maxrss"}


@posix_only
def test_wait4_measures_a_child_that_allocates():
    """The POSIX path, exercised on every POSIX machine rather than only the
    one it was written for. `/usr/bin/time -l` is macOS and GNU time spells it
    -v, so Linux reaps the child itself and keeps its rusage."""
    small = proc._run_posix([PY, "-c", "pass"], None, False, None,
                            time.perf_counter())
    big = proc._run_posix(
        [PY, "-c", "x = bytearray(200 * 1024 * 1024); print(len(x))"],
        None, False, None, time.perf_counter())
    assert small.peak_kb > 0, "a measurement of zero is the failure to avoid"
    assert big.peak_kb > small.peak_kb, (small.peak_kb, big.peak_kb)
    assert "209715200" in big.stdout


@posix_only
def test_wait4_keeps_the_exit_status_and_both_streams():
    r = proc._run_posix(
        [PY, "-c", "import sys; print('out'); sys.stderr.write('boom');"
                   " sys.exit(3)"],
        None, False, None, time.perf_counter())
    assert r.returncode == 3 and not r.ok
    assert "out" in r.stdout and "boom" in r.stderr


@posix_only
def test_wait4_times_out_rather_than_waiting_forever():
    with pytest.raises(subprocess.TimeoutExpired):
        proc._run_posix([PY, "-c", "import time; time.sleep(30)"], 1.0, False,
                        None, time.perf_counter())


@posix_only
def test_wait4_streams_without_capturing(capfd):
    r = proc._run_posix([PY, "-c", "print('live')"], None, True, None,
                        time.perf_counter())
    assert r.returncode == 0
    assert r.peak_kb > 0
    assert "live" in capfd.readouterr().out
