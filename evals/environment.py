"""Snapshot of what produced a result set.

A score sheet without its machine is not comparable to anything, and comparing
this mini against the incoming Studio is a stated goal. This also records
memory pressure, because a run competing with something else is pessimistic and
that fact should live in the results rather than in someone's memory.

THE ACCELERATOR IS RECORDED SEPARATELY. On Apple Silicon `hw_model` identifies
the GPU too, because it is the same part. On a PC it does not: an RTX 4070 run
and an M2 Pro run are otherwise indistinguishable in a results.json, which
defeats the one thing this module exists for.

`swap_used_mb` is macOS-only and reads 0 elsewhere. macOS swap has no honest
one-number equivalent on Windows -- commit charge counts resident RAM too --
and inventing one would make the field incomparable across the machines it
exists to compare.
"""
from __future__ import annotations

import platform
import re
import subprocess

from harness import memory


def _sh(*argv: str) -> str:
    try:
        return subprocess.run(argv, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return ""


def _version(module: str) -> str:
    try:
        from importlib.metadata import version
        return version(module)
    except Exception:  # noqa: BLE001 - absent or unreadable is not fatal
        return ""


def _last_word(s: str) -> str:
    parts = s.split()
    return parts[-1] if parts else ""


def capture() -> dict:
    swap = _sh("sysctl", "-n", "vm.swapusage")
    m = re.search(r"used\s*=\s*([\d.]+)M", swap)
    mem_bytes = _sh("sysctl", "-n", "hw.memsize")
    memory_gb = (round(int(mem_bytes) / 1024**3) if mem_bytes.isdigit()
                 else round(memory.system_memory_gb()))
    accelerator = memory.detect()

    return {
        "hw_model": _sh("sysctl", "-n", "hw.model") or platform.processor()
                    or platform.machine(),
        "os": platform.platform(),
        # The macOS version where there is one, empty elsewhere -- rather than
        # a key named `macos` holding a Windows string.
        "macos": platform.mac_ver()[0],
        "arch": platform.machine(),
        "memory_gb": memory_gb,
        "swap_used_mb": round(float(m.group(1))) if m else 0,
        "accelerator": {
            "kind": accelerator.kind,
            "name": accelerator.name or platform.machine(),
            "total_gb": round(accelerator.total_gb, 2),
            "available_gb": round(accelerator.available_gb, 2),
        },
        "git_sha": _sh("git", "rev-parse", "--short=12", "HEAD") or "unknown",
        "git_dirty": bool(_sh("git", "status", "--porcelain")),
        "versions": {
            "python": platform.python_version(),
            "mlx": _version("mlx"),
            "mlx-lm": _version("mlx-lm"),
            # mflux installs as a uv tool, so it is not importable from this
            # venv and has to be asked. Never index into a possibly-empty split.
            "mflux": _version("mflux") or _last_word(
                _sh("mflux-generate", "--version")),
            "litellm": _version("litellm"),
        },
    }
