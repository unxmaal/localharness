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
    """A job object's peak, a phys_footprint and a maxrss are three different
    quantities. A number that does not say which one produced it cannot be
    ranked against one from another machine."""
    assert proc.PEAK_METHOD in {"phys_footprint", "job_peak_process",
                                "gnu_time_maxrss"}


def test_a_child_that_allocates_is_measured_higher_than_one_that_does_not():
    """THE TEST THAT CAUGHT THE REAL BUG, and the reason it runs on every
    machine rather than only the one the code was written for.

    The first Linux implementation reaped the child in-process with os.wait4.
    That reported 477124 KB for a child allocating 200 MB and 477124 KB for a
    child doing nothing: the number was the PYTEST PARENT's footprint, because
    Linux carries the forking process's high-water RSS into the child's maxrss.
    Every peak in every results.json would have been the harness's own size.
    """
    small = run([PY, "-c", "pass"])
    big = run([PY, "-c", "x = bytearray(200 * 1024 * 1024); print(len(x))"])
    assert small.peak_kb > 0, "a measurement of zero is the failure to avoid"
    assert big.peak_kb > small.peak_kb + 100_000, (small.peak_kb, big.peak_kb)
    assert "209715200" in big.stdout


@pytest.mark.skipif(sys.platform == "win32", reason="the wrapper is POSIX")
def test_the_wrapper_flag_matches_the_time_on_this_machine():
    """BSD time says -l and GNU time says -v. Asking either for the other's
    flag is a usage error rather than a report, and the peak then reads 0."""
    assert proc.TIME_FLAG == ("-l" if sys.platform == "darwin" else "-v")


def test_both_report_formats_parse_in_kilobytes():
    """macOS reports a phys_footprint in BYTES and GNU time a maxrss in
    KILOBYTES. Reading one as the other is a factor of 1024 either way."""
    assert proc._peak_kb("  213909504  peak memory footprint") == 208896
    assert proc._peak_kb(
        "\tMaximum resident set size (kbytes): 477124\n") == 477124
    assert proc._peak_kb("nothing here") == 0
