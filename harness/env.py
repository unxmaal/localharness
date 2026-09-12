"""Where the weights live, resolved by the CLI rather than by the shell.

`scripts/env.sh` has always done this for the services, but it is SOURCED: it
only helps a shell that ran it. Once `lh` is installed on PATH it runs from any
directory with nothing sourced, and `lh image` spawns mflux, which reads
HF_HOME. Unset, huggingface_hub silently falls back to ~/.cache/huggingface and
starts re-downloading weights that are already on the volume -- gigabytes, with
no error, onto the disk this machine has least of.

THERE IS NO CANDIDATE LIST ANY MORE. This used to hunt through
two named external volumes and ~/.cache/huggingface, which is one person's
Mac written into the repo: the Windows port had to fork the list
in Python, env.sh had to fork it again in shell, and a test existed whose only
job was to notice when the two forks drifted. The default is now a path inside
the working tree, and a machine that wants a fast drive says so with HF_ROOT.

WHAT IS SAID IS STILL CHECKED. An explicit HF_ROOT goes through the same
writability and free-space tests as the default; it does NOT bypass them. That
was a real bug once -- an unwritable and a nonexistent path both passed with
exit 0 -- and "the caller says where" must not become "and stop looking".
The rule is FREE SPACE, not "is this external": that refuses this machine's
internal disk, which sits at 91%, for the reason that actually matters, and it
will not refuse the Studio's.
"""
from __future__ import annotations

import os
import shutil
from pathlib import Path

#: Enough to pull a normal model without filling the disk. NOT enough for
#: MiniMax-H3's 134 GiB checkpoint, and deliberately so: this used to be 160
#: for that one model, which made every location on every ordinary machine
#: unusable -- including the in-tree default and every CI runner -- to guard a
#: download that already guards itself. scripts/fetch-h3-weights.sh demands its
#: own 160 GB and setup-omnisvg.sh its own 40, at the point where the size is
#: actually known. A global floor cannot know what is about to be fetched.
HF_MIN_FREE_GB = 20

#: Where weights land when nobody says otherwise. Inside the checkout, so it
#: exists on every machine, needs no drive letter and no mount, and is the same
#: answer on every machine, needing no drive letter and no mount. Gitignored.
#: env.sh computes the same path; test_harness_env.py fails if they disagree.
DEFAULT_ROOT = "hf_root"

#: This machine's real weights volume, set in the environment rather than
#: shipped in the source. See README for the HF_ROOT examples.
ROOT_VAR = "HF_ROOT"


def default_root() -> str:
    """The in-tree default, absolute. Relative would scatter caches into
    whatever directory `lh` happened to be run from, which is the bug
    harness/paths.py exists to remember."""
    return str(Path(__file__).resolve().parent.parent / DEFAULT_ROOT)


def configured() -> str:
    """What this machine says, or the default. One place that decides."""
    return os.environ.get(ROOT_VAR) or default_root()


def beside() -> Path:
    """Where the things too big for the hub cache go: corpora, and the 134 GiB
    of H3 weights. Beside the weights root, so pointing HF_ROOT at a drive
    moves them together and the default stays inside the checkout."""
    return Path(configured()).resolve().parent


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
    unmounted candidate walks up to the root -- `/Volumes/FAST/hf` with
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
    """Whether weights may be written here. Asked of ONE location.

    The ancestor walk below is safe now in a way it was not as part of a
    search. A guessed candidate on an unmounted volume climbed to the
    filesystem root, and on a machine whose root is writable -- a CI runner,
    or Windows -- it was accepted, so weights went into a tree invented under
    it. Nothing is guessed any more: this path was either configured by a
    person or is the in-tree default, and creating it is the intent either way.
    """
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


def resolve(root: str | None = None,
            min_free_gb: int = HF_MIN_FREE_GB) -> str | None:
    """The weights location, or None if it is not usable.

    One location, checked. Not a search: a search is how the wrong disk gets
    chosen quietly, and how "which of three mounts answered today" became a
    property of the run.
    """
    root = configured() if root is None else root
    return root if usable(root, min_free_gb) else None


def apply(environ=None, root: str | None = None,
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
    found = resolve(root if root is not None else environ.get(ROOT_VAR),
                    min_free_gb)
    if found:
        environ["HF_HOME"] = found
    return found
