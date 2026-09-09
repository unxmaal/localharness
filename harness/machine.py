"""What this machine can RUN, which is a different question from which OS it is.

Discovery has one question to answer about a candidate: can this machine run
it. That question is about RUNTIMES. mlx, cuda, rocm and the cpu are the things
a repo depends on; the operating system is only how you find out which of them
are present.

Writing it the other way round is what this module exists to undo. `decide()`
used to read "CUDA is absolute on this machine", which was true of the Mac it
was written on and made every CUDA candidate a rejection on a box bought to run
them, while an MLX repo that cannot start there passed the same gate.

A LINUX BOX WITH AN NVIDIA CARD IS THE SAME MACHINE AS A WINDOWS ONE here, and
that is the property to keep. Nothing downstream should branch on the platform,
so adding Linux, or a ROCm card, or whatever comes after, is a row in
_RUNTIME_PROBES rather than another branch in every caller.
"""
from __future__ import annotations

import shutil
import sys
from dataclasses import dataclass
from functools import lru_cache

from harness import memory
from harness.memory import Accelerator

#: The cpu is always there, so it is never a reason to refuse anything.
ALWAYS = "cpu"


def _has_mlx() -> bool:
    """Apple's array framework. Present means Apple Silicon in practice: the
    package imports on other platforms and then fails to load its library."""
    if sys.platform != "darwin":
        return False
    try:
        import mlx.core  # noqa: F401
    except Exception:  # noqa: BLE001 - an unusable mlx is an absent one
        return False
    return True


def _has_cuda() -> bool:
    """nvidia-smi ships with the driver, so this answers before any toolkit or
    Python binding is installed."""
    return memory._discrete() is not None and shutil.which("nvidia-smi") is not None


def _has_rocm() -> bool:
    """AMD's stack. Untested here, and listed so that adding the machine is a
    row rather than a branch: nothing downstream asks which of these is true,
    only whether the runtime a candidate needs is in the set."""
    return shutil.which("rocm-smi") is not None or shutil.which("rocminfo") is not None


#: runtime -> how to tell whether this machine has it. Order is not meaningful;
#: a machine may have several.
_RUNTIME_PROBES = {
    "mlx": _has_mlx,
    "cuda": _has_cuda,
    "rocm": _has_rocm,
}


@dataclass(frozen=True)
class Machine:
    """The runtimes this machine has, and what it loads weights into."""
    runtimes: frozenset[str]
    accelerator: Accelerator

    def refuses(self, runtime: str) -> str | None:
        """The verdict for a candidate needing `runtime`, or None if it runs.

        The verdict names the RUNTIME rather than the platform, so the same
        sentence reads correctly from either direction: needs-cuda on a Mac and
        needs-mlx on a card are one rule, not two.
        """
        if runtime == ALWAYS or runtime in self.runtimes:
            return None
        return f"needs-{runtime}"

    def describe(self) -> str:
        """One line for a result sheet. Two machines with the same card and the
        same runtimes are comparable; two without are not, and the row should
        say so rather than leaving a reader to infer it from a hostname."""
        return (f"{self.accelerator.name} "
                f"({self.accelerator.kind}, {self.accelerator.total_gb:.0f} GB) "
                f"[{', '.join(sorted(self.runtimes))}]")


@lru_cache(maxsize=1)
def detect() -> Machine:
    """This machine. Cached: the probes shell out, and the answer does not
    change while the process runs."""
    found = {name for name, probe in _RUNTIME_PROBES.items() if probe()}
    found.add(ALWAYS)
    return Machine(runtimes=frozenset(found), accelerator=memory.detect())
