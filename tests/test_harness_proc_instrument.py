"""What `phys_footprint` sees that `maximum resident set size` does not.

Issue #238 was filed on the premise that phys_footprint OVER-reports: the music
lane recorded a 36.3 GiB peak on a 32 GB machine, against 13.82 GiB taken with
`ru_maxrss` inside the same process. The obvious reading is that the larger
number is wrong.

It is the other way round, and these are the controls that settle it. Both
numbers come from ONE `/usr/bin/time -l` run, so they describe the same
process:

    plain 2 GiB bytearray     rss 2.01 GiB   footprint 2.01 GiB   ratio  1.00
    plain 4 GiB bytearray     rss 4.02 GiB   footprint 4.01 GiB   ratio  1.00
    numpy 2 GiB               rss 2.02 GiB   footprint 2.02 GiB   ratio  1.00
    mlx 2 GiB array           rss 0.03 GiB   footprint 2.02 GiB   ratio 62.80
    torch mps 2 GiB           rss 0.21 GiB   footprint 2.16 GiB   ratio 10.05

Where the truth is known, the two instruments AGREE. On a Metal allocation of
exactly the same size, resident set size sees almost nothing while the
footprint is right. RSS cannot see unified-memory buffers, so on this platform
it is the blind instrument, and every MLX lane in this repo would report a
fraction of its real cost if proc.py were "fixed" to use it.

ACE-Step measured the same way: rss 11.26 GiB, footprint 36.27 GiB.

THESE TESTS EXIST TO STOP A PLAUSIBLE FIX. Swapping phys_footprint for maxrss
looks like a correction and would silently under-report every Metal workload
the project runs.
"""
import re
import subprocess
import sys

import pytest

pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="phys_footprint and this argument are macOS-only; Linux and "
           "Windows use their own instruments, see proc.PEAK_METHOD")

#: Small enough to stay quick, large enough to dwarf an interpreter.
SIZE_MIB = 512
_RSS = re.compile(r"(\d+)\s+maximum resident set size")
_FOOT = re.compile(r"(\d+)\s+peak memory footprint")


def both(code: str) -> tuple[int, int]:
    """(maxrss, phys_footprint) in bytes, from one run of one process."""
    out = subprocess.run(["/usr/bin/time", "-l", sys.executable, "-c", code],
                         capture_output=True, text=True).stderr
    rss, foot = _RSS.search(out), _FOOT.search(out)
    assert rss and foot, f"could not parse the report:\n{out[-800:]}"
    return int(rss.group(1)), int(foot.group(1))


def test_the_two_instruments_agree_where_the_truth_is_known():
    """THE POSITIVE CONTROL. Plain anonymous memory is resident and is in the
    footprint, so a disagreement here would mean the parse is wrong rather
    than that the instruments measure different things."""
    rss, foot = both(f"x=bytearray({SIZE_MIB}*1024**2); "
                     f"x[::4096]=b'\\1'*len(x[::4096])")
    want = SIZE_MIB * 1024 ** 2
    assert rss > want * 0.9, f"maxrss {rss} did not see a {SIZE_MIB} MiB array"
    assert foot > want * 0.9, f"footprint {foot} did not see it either"
    assert 0.8 < foot / rss < 1.25, (
        f"footprint {foot} and maxrss {rss} disagree on plain memory, where "
        f"they must not; the parse or the platform has changed")


def test_resident_set_size_is_blind_to_a_metal_allocation():
    """THE FINDING, and the reason phys_footprint stays.

    An MLX array of the same size is allocated in unified memory. The
    footprint sees it; resident set size does not. A receipt built on maxrss
    would report a fraction of what an MLX lane actually costs.
    """
    mlx = pytest.importorskip("mlx.core", reason="MLX is the case under test")
    del mlx
    rss, foot = both(
        f"import mlx.core as mx; "
        f"a=mx.ones(({SIZE_MIB}*1024**2 // 4,), dtype=mx.float32); "
        f"mx.eval(a); print(a[0].item())")
    want = SIZE_MIB * 1024 ** 2
    assert foot > want * 0.9, (
        f"the footprint missed a {SIZE_MIB} MiB MLX array; if this fails, "
        f"phys_footprint has stopped seeing unified memory and the whole "
        f"argument for using it is gone")
    assert rss < want * 0.5, (
        f"maxrss {rss} now sees Metal allocations. If macOS changed this, "
        f"#238 can be revisited -- but check the control above first")


def test_the_receipt_says_which_instrument_produced_the_number():
    """Three platforms, three different quantities. A number without its
    instrument cannot be compared with one from another machine."""
    from harness import proc
    assert proc.PEAK_METHOD == "phys_footprint"
