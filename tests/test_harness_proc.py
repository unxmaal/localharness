"""Running an external generator and measuring what it cost.

This is the layer both the CLI and the eval suite stand on, so it has to report
the same numbers to both. The measurement it exists for is peak memory: on a
32GB machine the binding constraint is one phase's peak, not the model's size
on disk.
"""
import sys

import pytest

from harness.proc import Outcome, run

PY = sys.executable


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
