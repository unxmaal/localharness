"""Current memory pressure, unlike vm.swapusage which is a high-water mark.

See README, "Reading a run's memory conditions". Issue #283.
"""
from __future__ import annotations

import platform
import subprocess
from dataclasses import dataclass

#: macOS's own escalation, the ladder jetsam acts on.
NORMAL, WARN, CRITICAL = 1, 2, 4

#: `/usr/bin/time -l` reports these and macOS never sets them: always 0.
UNMAINTAINED = ("swaps",)

_SYSCTLS = {
    "free_pct": "kern.memorystatus_level",
    "level": "kern.memorystatus_vm_pressure_level",
    "swapouts": "vm.compressor.segment.swapout_regular",
}


@dataclass(frozen=True)
class Pressure:
    """A field is None when this platform cannot answer it, never 0."""

    free_pct: int | None = None
    level: int | None = None
    swapouts: int | None = None
    wired_gb: float | None = None

    @property
    def known(self) -> bool:
        return any(v is not None for v in
                   (self.free_pct, self.level, self.swapouts, self.wired_gb))

    @property
    def alarming(self) -> bool:
        """Past normal. Unknown is not alarming: it must not block a run."""
        return self.level is not None and self.level > NORMAL

    def as_dict(self) -> dict:
        return {"free_pct": self.free_pct, "level": self.level,
                "swapouts": self.swapouts, "wired_gb": self.wired_gb}


def _sysctl(name: str) -> str:
    try:
        out = subprocess.run(["sysctl", "-n", name], capture_output=True,
                             text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return ""
    return out.stdout.strip() if out.returncode == 0 else ""


def _vm_stat() -> str:
    try:
        return subprocess.run(["vm_stat"], capture_output=True, text=True,
                              timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return ""


def wired_gb(vm_stat_output: str) -> float | None:
    """Pages wired down, in GB."""
    page, pages = 16384, None
    for line in vm_stat_output.splitlines():
        if "page size of" in line:
            digits = [w for w in line.split() if w.isdigit()]
            if digits:
                page = int(digits[0])
        if line.startswith("Pages wired down"):
            value = line.partition(":")[2].strip().rstrip(".")
            if value.isdigit():
                pages = int(value)
    return None if pages is None else round(pages * page / 1024 ** 3, 2)


def sample(sysctl=_sysctl, vm_stat=_vm_stat) -> Pressure:
    """Every signal this platform offers. Readers injected for the tests."""
    got: dict[str, int | None] = {}
    for field, name in _SYSCTLS.items():
        raw = sysctl(name)
        got[field] = int(raw) if raw.lstrip("-").isdigit() else None
    if not any(v is not None for v in got.values()) and _is_linux():
        got.update(_linux_signals())
    return Pressure(free_pct=got.get("free_pct"), level=got.get("level"),
                    swapouts=got.get("swapouts"),
                    wired_gb=wired_gb(vm_stat()))


def _is_linux() -> bool:
    return platform.system() == "Linux"


def _linux_signals(meminfo="/proc/meminfo", vmstat="/proc/vmstat") -> dict:
    """MemAvailable percent and pswpout. No jetsam ladder, so no `level`."""
    got: dict[str, int | None] = {}
    try:
        with open(meminfo, encoding="utf-8") as fh:
            fields = {}
            for line in fh:
                key, _, value = line.partition(":")
                digits = value.split()
                if digits and digits[0].isdigit():
                    fields[key.strip()] = int(digits[0])
        total, avail = fields.get("MemTotal"), fields.get("MemAvailable")
        if total and avail is not None:
            got["free_pct"] = round(avail * 100 / total)
    except OSError:
        pass
    try:
        with open(vmstat, encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("pswpout "):
                    got["swapouts"] = int(line.split()[1])
    except (OSError, ValueError, IndexError):
        pass
    return got
