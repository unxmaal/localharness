"""Peak memory measurement for subprocess runners.

The first implementation took a before/after delta of
resource.getrusage(RUSAGE_CHILDREN).ru_maxrss. That is a monotone high-water
mark across ALL waited children, so the delta is 0 for every child after the
largest one, and RSS itself falls under memory pressure. The same Z-Image
config measured 6.2 GiB on a quiet machine and 3.3 GiB on a loaded one, and a
documented conclusion ("2x the memory") rested on it.
"""
import sys

import pytest

from evals.runners.process import measure_peak_kb


def test_reports_a_plausible_peak_for_a_known_allocation():
    """Allocate ~200MB in a child and expect the measurement to see it."""
    code = ("import sys;"
            "buf = bytearray(200 * 1024 * 1024);"
            "buf[::4096] = b'x' * (len(buf)//4096 + (1 if len(buf)%4096 else 0));"
            "sys.exit(0)")
    peak, rc = measure_peak_kb([sys.executable, "-c", code])
    assert rc == 0
    assert peak > 150_000, f"expected >150MB, measured {peak}KB"
    assert peak < 2_000_000, f"implausibly large: {peak}KB"


def test_a_second_smaller_child_is_not_attributed_the_first_ones_peak():
    """The exact failure of the ru_maxrss delta approach."""
    big = ("import sys; b = bytearray(300*1024*1024);"
           "b[::4096] = b'x'*(len(b)//4096+1); sys.exit(0)")
    small = "import sys; sys.exit(0)"
    big_peak, _ = measure_peak_kb([sys.executable, "-c", big])
    small_peak, _ = measure_peak_kb([sys.executable, "-c", small])
    assert big_peak > 150_000
    assert small_peak < big_peak / 2, (
        f"small child reported {small_peak}KB against big {big_peak}KB; "
        "the measurement is leaking the previous high-water mark")


def test_exit_code_is_returned():
    _, rc = measure_peak_kb([sys.executable, "-c", "import sys; sys.exit(7)"])
    assert rc == 7


def test_missing_binary_raises_filenotfound():
    with pytest.raises(FileNotFoundError):
        measure_peak_kb(["definitely-not-a-real-binary-xyz"])
