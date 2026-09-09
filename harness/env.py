"""Where the weights live, resolved by the CLI rather than by the shell.

`scripts/env.sh` has always done this for the services, but it is SOURCED: it
only helps a shell that ran it. Once `lh` is installed on PATH it runs from any
directory with nothing sourced, and `lh image` spawns mflux, which reads
HF_HOME. Unset, huggingface_hub silently falls back to ~/.cache/huggingface and
starts re-downloading weights that are already on the volume -- gigabytes, with
no error, onto the disk this machine has least of.

The rule mirrors env.sh on purpose, and the last test in
tests/test_harness_env.py fails if the two lists drift apart. It is FREE SPACE,
not "is this external": that refuses this mini's internal disk, which sits at
91%, for the reason that actually matters, and it will not refuse the Studio's.
"""
from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

# 160GB because MiniMax-H3's checkpoint alone is 134 GiB.
HF_MIN_FREE_GB = 160

# /Volumes is macOS. A Windows machine names its drives and there is no
# portable guess for which one holds the weights, so the default there is the
# one location that always exists -- point HF_ROOT at a fast drive instead.
# env.sh carries the same two lists; test_the_shipped_candidates_match_the_ones
# _env_sh_searches fails if they drift apart.
if sys.platform == "win32":
    HF_CANDIDATES = (str(Path.home() / ".cache" / "huggingface"),)
else:
    HF_CANDIDATES = ("/Volumes/Models/hf", "/Volumes/T7/hf",
                     str(Path.home() / ".cache" / "huggingface"))


def _anchor(path: str) -> Path | None:
    """The nearest ancestor that exists, since the target may not yet."""
    p = Path(path)
    while not p.exists():
        parent = p.parent
        if parent == p:
            return None
        p = parent
    return p


def free_gb(path: str) -> int:
    anchor = _anchor(path)
    if anchor is None:
        return 0
    try:
        return shutil.disk_usage(anchor).free // (1024 ** 3)
    except OSError:
        return 0


def _writable(anchor: Path) -> bool:
    """Whether a directory can actually be written to.

    `os.access(W_OK)` reflects only the read-only ATTRIBUTE on Windows: it
    answered True for the drive root, which a standard user cannot write to,
    while an actual write raised PermissionError. That matters here because an
    unmounted candidate walks up to the root -- `/Volumes/Models/hf` with
    nothing mounted anchors at `/`, and on macOS the check holds only because
    `/` genuinely is not user-writable. Windows had no such backstop, so every
    bogus candidate resolved to the drive root and was accepted.

    The probe is created and removed. That is not the "litter empty
    directories" this module refuses to do: nothing survives the call.
    """
    if os.name != "nt":
        return os.access(anchor, os.W_OK)
    probe = anchor / f".localharness-write-probe-{os.getpid()}"
    try:
        probe.touch()
    except OSError:
        return False
    try:
        probe.unlink()
    except OSError:
        pass
    return True


def usable(path: str, min_free_gb: int = HF_MIN_FREE_GB) -> bool:
    # THE PARENT MUST EXIST. Walking up to the nearest existing ancestor
    # accepts anything at all on a machine whose filesystem root is
    # writable: /no/such/volume/hf climbs to the drive root, a CI runner can
    # write to the root of its work drive, and the candidate was taken. The
    # weights would then have gone into a four-deep tree invented under it.
    # On macOS the same walk is saved by / not being user-writable, which is
    # a backstop rather than a rule.
    #
    # One level is what a cache root needs: /Volumes/Models has to be mounted
    # before /Volumes/Models/hf is a place, and a drive has to be there
    # before a directory on it is. That is what the mount check was always
    # asking.
    if not Path(path).parent.exists():
        return False
    anchor = _anchor(path)
    if anchor is None or not _writable(anchor):
        return False
    if free_gb(path) < min_free_gb:
        return False
    # READABILITY, which none of the above implies. Under macOS TCC a volume
    # stats fine, reports free space and appears in /Volumes while listing it
    # raises -- and mlx_lm then hangs forever inside os.listdir with no log and
    # 0% CPU. Listing one entry is the cheapest way to find out here instead.
    try:
        next(os.scandir(anchor), None)
    except OSError:
        return False
    return True


def resolve(candidates=HF_CANDIDATES,
            min_free_gb: int = HF_MIN_FREE_GB) -> str | None:
    for cand in candidates:
        if usable(cand, min_free_gb):
            return cand
    return None


def apply(environ=None, candidates=HF_CANDIDATES,
          min_free_gb: int = HF_MIN_FREE_GB) -> str | None:
    """Put HF_HOME in `environ` if it is not already there. Returns what it is.

    Returns None rather than guessing when nothing is usable: a wrong HF_HOME
    is worse than none, because huggingface_hub will happily fill it.
    """
    environ = os.environ if environ is None else environ
    # Cached weights should not depend on the network. mlx_lm.server issues a
    # HEAD to huggingface.co on every model switch even for local files, so a
    # wifi blip turns into a model "failure" mid-run.
    environ.setdefault("HF_HUB_OFFLINE", "1")

    existing = environ.get("HF_HOME")
    if existing:
        return existing
    found = resolve(candidates, min_free_gb)
    if found:
        environ["HF_HOME"] = found
    return found
