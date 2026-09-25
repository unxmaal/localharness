"""Current memory signals, against stubbed readers. Issue #283.

The readers are injected so these assert what the module reads rather than what
this particular machine happens to be doing.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import pressure  # noqa: E402

VM_STAT = """Mach Virtual Memory Statistics: (page size of 16384 bytes)
Pages free:                                   290221.
Pages active:                                 478567.
Pages wired down:                             336519.
"""


def readers(**sysctls):
    return (lambda name: str(sysctls.get(name, "")), lambda: VM_STAT)


def test_every_signal_is_read_from_the_surface_that_carries_it():
    sysctl, vm_stat = readers(**{
        "kern.memorystatus_level": 63,
        "kern.memorystatus_vm_pressure_level": 1,
        "vm.compressor.segment.swapout_regular": 42,
    })
    got = pressure.sample(sysctl=sysctl, vm_stat=vm_stat)
    assert got.free_pct == 63
    assert got.level == pressure.NORMAL
    assert got.swapouts == 42
    # 336519 pages * 16384 bytes
    assert got.wired_gb == 5.13, got.wired_gb
    assert got.known


def test_an_unreadable_signal_is_none_rather_than_zero():
    """Zero is a reading. `swap_used_mb` returned 0 when it could not tell, so
    a machine that could not be measured looked like an idle one."""
    got = pressure.sample(sysctl=lambda name: "", vm_stat=lambda: "")
    assert got.free_pct is None
    assert got.level is None
    assert got.swapouts is None
    assert got.wired_gb is None
    assert not got.known


def test_an_unknown_level_is_not_alarming():
    """A guard that blocks when it cannot measure becomes the outage."""
    assert not pressure.Pressure().alarming
    assert not pressure.Pressure(level=pressure.NORMAL).alarming


def test_a_level_past_normal_is_alarming():
    """THE POSITIVE CONTROL. Without it the check above passes on a function
    that returns False unconditionally."""
    assert pressure.Pressure(level=pressure.WARN).alarming
    assert pressure.Pressure(level=pressure.CRITICAL).alarming


def test_the_unmaintained_counter_is_named_so_nobody_reads_it():
    """`/usr/bin/time -l` prints `swaps` and macOS does not maintain it: it is
    0 on every run, so a check built on it fires never."""
    assert "swaps" in pressure.UNMAINTAINED
    assert "swaps" not in pressure.Pressure().as_dict()


def test_the_receipt_dict_carries_every_field():
    got = pressure.Pressure(free_pct=10, level=2, swapouts=3, wired_gb=4.0)
    assert got.as_dict() == {"free_pct": 10, "level": 2, "swapouts": 3,
                             "wired_gb": 4.0}


def test_wired_pages_honour_the_reported_page_size():
    """The page size is in the output and this machine reports 16384, not the
    4096 a reader might assume."""
    four_k = VM_STAT.replace("page size of 16384", "page size of 4096")
    assert pressure.wired_gb(four_k) == round(336519 * 4096 / 1024 ** 3, 2)


def test_vm_stat_without_the_wired_line_is_unknown():
    assert pressure.wired_gb("Pages free: 1.\n") is None


def test_linux_reads_memavailable_and_pswpout(tmp_path):
    """Linux has no jetsam ladder, so `level` stays None rather than invented."""
    meminfo = tmp_path / "meminfo"
    meminfo.write_text("MemTotal:  1000 kB\nMemAvailable:  250 kB\n",
                       encoding="utf-8")
    vmstat = tmp_path / "vmstat"
    vmstat.write_text("pgfault 1\npswpout 77\n", encoding="utf-8")
    got = pressure._linux_signals(meminfo=str(meminfo), vmstat=str(vmstat))
    assert got == {"free_pct": 25, "swapouts": 77}
    assert "level" not in got
