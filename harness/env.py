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
from pathlib import Path

# 160GB because MiniMax-H3's checkpoint alone is 134 GiB.
HF_MIN_FREE_GB = 160
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


def usable(path: str, min_free_gb: int = HF_MIN_FREE_GB) -> bool:
    anchor = _anchor(path)
    if anchor is None or not os.access(anchor, os.W_OK):
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
