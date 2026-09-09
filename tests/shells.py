r"""Resolving a shell name to something this machine will actually run.

`bash` on a stock Windows PATH is C:\Windows\System32\bash.exe, the WSL
launcher. With no distribution installed it exits 1 with a UTF-16 message about
installing a distribution, so a test that sources scripts/env.sh fails on a
message that has nothing to do with the script.

Putting Git's bin first on PATH does NOT fix it. CreateProcess searches the
Windows system directory before any PATH entry, so a bare "bash" handed to
subprocess finds the WSL launcher whatever PATH says. The name has to be
resolved to a path here.
"""
import shutil
import sys
from pathlib import Path

#: Where Git for Windows keeps the shells. usr/bin carries sh and the rest.
_GIT_DIRS = (
    r"C:\Program Files\Git\bin",
    r"C:\Program Files\Git\usr\bin",
    r"C:\Program Files (x86)\Git\bin",
    r"C:\Program Files (x86)\Git\usr\bin",
)


def resolve(name: str) -> str | None:
    """An absolute path to `name`, or None when this machine has no such shell."""
    if sys.platform != "win32":
        return shutil.which(name)

    for directory in _GIT_DIRS:
        found = Path(directory, name + ".exe")
        if found.exists():
            return str(found)

    # Installed somewhere else: git.exe sits in <root>/cmd or <root>/bin.
    git = shutil.which("git")
    if git:
        root = Path(git).resolve().parent.parent
        for sub in ("bin", "usr/bin"):
            found = root / sub / (name + ".exe")
            if found.exists():
                return str(found)

    found = shutil.which(name)
    if found and Path(found).parent.name.lower() == "system32":
        # The WSL launcher, which cannot source a POSIX script.
        return None
    return found


#: None on a machine with no usable bash, which is a skip rather than a failure.
BASH = resolve("bash")
