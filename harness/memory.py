"""Refusing to load a model that will take the machine down.

WRITTEN AFTER TAKING THE MACHINE DOWN. An eval sweeping seven candidates
reached Qwen3-30B-A3B-4bit -- 16 GB of weights -- while Qwen3-14B was still
resident and a Docker VM held 7.2 GB, on a 32 GB Mac. The eval log ends
mid-line at the candidate header. There was no panic report and no jetsam
entry; the machine simply stopped.

Nothing in this repo checked a model's size against available memory, on a
machine whose entire design constraint is 32 GB of unified memory. The eval
measured peak memory AFTER the fact and printed it in a column, which is a
postmortem, not a guard.

Two numbers matter and they are different:

  * TOTAL RAM is not the budget. Apple's Metal driver reports a recommended
    working set well below it (25.0 GiB of 32 on this mini, per tools/h3probe),
    and exceeding that is where things go wrong rather than at the RAM figure.
  * WHAT IS FREE RIGHT NOW is not the budget either. mlx_lm.server hot-swaps
    models per request and the outgoing weights are not guaranteed to be freed
    before the incoming ones are read, so a swap can hold both at once.

So the check is deliberately pessimistic: it budgets for the resident model
staying put, and reserves headroom for everything that is not us.
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

#: Fraction of physical RAM Metal will hand out as a working set. Apple does
#: not document the formula and it is version-dependent; 0.75 matches what
#: h3probe reported here (25.0 of 32 GiB) and is conservative on larger
#: machines, where the real fraction is higher.
GPU_FRACTION = 0.75

#: For WindowServer, the browser, a Docker VM, the audio server, the MCP
#: server -- everything that is not the model. Measured against a real crash:
#: Docker alone held 7.2 GB.
DEFAULT_RESERVE_GB = 6.0


def total_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True,
                             text=True, timeout=5).stdout.strip()
        return int(out) / 1024 ** 3
    except (OSError, ValueError, subprocess.SubprocessError):
        return 0.0


def ceiling_gb() -> float:
    """The working set to budget against, not the RAM figure on the box."""
    return total_gb() * GPU_FRACTION


def available_gb() -> float:
    """Free plus inactive, which is what the system can actually hand over."""
    try:
        out = subprocess.run(["vm_stat"], capture_output=True, text=True,
                             timeout=5).stdout
    except (OSError, subprocess.SubprocessError):
        return total_gb()
    page = 16384
    counts = {}
    for line in out.splitlines():
        if ":" not in line:
            continue
        key, _, value = line.partition(":")
        value = value.strip().rstrip(".")
        if value.isdigit():
            counts[key.strip()] = int(value)
        if "page size of" in line:
            digits = [w for w in line.split() if w.isdigit()]
            if digits:
                page = int(digits[0])
    # Free plus everything macOS will hand back without paging anything out.
    # Free + inactive ALONE understates badly right after reading a large model
    # off disk: most of what shows as active is file cache. On an idle machine
    # that `memory_pressure` called 92% free, free+inactive read 12.8 GB.
    free = sum(counts.get(k, 0) for k in
               ("Pages free", "Pages inactive", "Pages speculative",
                "Pages purgeable"))
    if not free:
        return total_gb()
    return free * page / 1024 ** 3


def fits(need_gb: float, available_gb: float, ceiling_gb: float,
         resident_gb: float = 0.0,
         reserve_gb: float = DEFAULT_RESERVE_GB) -> tuple[bool, str]:
    """Can `need_gb` of weights be loaded without taking the machine down?

    `resident_gb` is what a hot-swapping server may still be holding. Budgeting
    as if only one model is resident is exactly how this went wrong.
    """
    budget = min(available_gb, ceiling_gb) - reserve_gb - resident_gb
    if need_gb > budget:
        return False, (
            f"needs {need_gb:.1f} GB but only {budget:.1f} GB is safely "
            f"available ({available_gb:.1f} GB free, {ceiling_gb:.1f} GB GPU "
            f"ceiling, {reserve_gb:.1f} GB reserved for the system"
            + (f", {resident_gb:.1f} GB still resident" if resident_gb else "")
            + ")")
    return True, f"{need_gb:.1f} GB into {budget:.1f} GB"


def cache_path(repo: str) -> str | None:
    """Where huggingface_hub put this repo, or None if it is not cached."""
    root = Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface"))
    d = root / "hub" / ("models--" + repo.replace("/", "--"))
    return str(d) if d.exists() else None


def size_gb(path: str) -> float | None:
    """Weight size on disk: the best estimate of load cost, and free to get.

    None when unknown. NOT zero -- zero reads as "fits easily", which is the
    dangerous default for a number nobody could determine.
    """
    root = Path(path)
    if not root.exists():
        return None
    # Deduplicate by inode. The HuggingFace cache keeps the real weights in
    # blobs/ and symlinks them into snapshots/, so walking naively counts
    # every file TWICE -- which reported a 4.3 GB model as 8.6 GB and had the
    # guard refusing models that fit with room to spare.
    total = 0
    seen: set[tuple[int, int]] = set()
    for f in root.rglob("*"):
        try:
            st = f.stat()          # follows symlinks, which is what we want
            if not f.is_file():
                continue
            key = (st.st_dev, st.st_ino)
            if key in seen:
                continue
            seen.add(key)
            total += st.st_size
        except OSError:
            continue
    return total / 1024 ** 3 if total else None


#: Bits per weight in an unquantised checkpoint. Used to scale the on-disk
#: size when a caller will quantise on load.
_BASE_BITS = 16


def check_model(repo: str, resident_gb: float = 0.0,
                reserve_gb: float = DEFAULT_RESERVE_GB,
                quantize: int | None = None) -> tuple[bool, str]:
    """One call: will loading `repo` be safe on this machine right now?

    `quantize` is the bit width the CALLER will load at. Without it this used
    the on-disk size, which is right for a pre-quantised MLX repo and badly
    wrong for a bf16 checkpoint loaded with `--quantize 4`: it refused
    FLUX.1-dev at "needs 31.4 GB" while a stage using exactly that model ran
    successfully on the same machine. A guard that blocks working work gets
    switched off, which is worse than no guard at all.
    """
    path = cache_path(repo)
    need = size_gb(path) if path else None
    if need is not None and quantize:
        need = need * quantize / _BASE_BITS
    if need is None:
        # Allowed, but said out loud. Refusing everything uncached would make
        # the guard the thing that breaks the workflow.
        return True, f"size of {repo} is unknown (not cached); proceeding unchecked"
    return fits(need, available_gb(), ceiling_gb(), resident_gb, reserve_gb)
