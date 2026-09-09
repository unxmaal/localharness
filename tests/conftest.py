"""Session setup for running this suite on a machine unlike the author's."""
import os
import shutil
import sys
from pathlib import Path

#: Where Git for Windows puts the bash these scripts are written for.
_GIT_BASH_DIRS = (
    r"C:\Program Files\Git\bin",
    r"C:\Program Files (x86)\Git\bin",
)


def _git_bash_dir() -> str | None:
    for candidate in _GIT_BASH_DIRS:
        if Path(candidate, "bash.exe").exists():
            return candidate
    # Installed somewhere else: git.exe sits in <root>/cmd, bash in <root>/bin.
    git = shutil.which("git")
    if git:
        sibling = Path(git).resolve().parent.parent / "bin"
        if (sibling / "bash.exe").exists():
            return str(sibling)
    return None


def pytest_configure(config):
    r"""Put Git's bash ahead of the WSL launcher for the whole session.

    `bash` on a stock Windows PATH is C:\Windows\System32\bash.exe, which
    starts WSL. With no distribution installed it prints "Windows Subsystem for
    Linux has no installed distributions." in UTF-16 and exits 1, so every test
    that sources scripts/env.sh fails on a message that has nothing to do with
    the script under test. Fixing it here rather than at each call site keeps
    the tests reading as `bash`, which is what a person runs.
    """
    if sys.platform != "win32":
        return
    found = _git_bash_dir()
    if found:
        os.environ["PATH"] = found + os.pathsep + os.environ.get("PATH", "")
