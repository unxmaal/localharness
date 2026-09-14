"""The seam between the code and the image that runs it.

GAUNTLET, "each half verified against its own spec, the seam against nothing".
The code was tested. The image built. Neither knew about the other, so
`harness/github.py` shelled out to `gh` while the Dockerfile installed only
`git`, and the first real fan-out failed all 125 candidates with

    [Errno 2] No such file or directory: 'gh'

Nobody owned the agreement, so nobody tested it. This does: the binaries the
source invokes BY BARE NAME are extracted mechanically and checked against what
the Dockerfile installs.

TIER 1 on purpose. One side is machine-readable, which is more often true of a
seam than it looks, and a scan covers tools nobody has added yet.
"""
import ast
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
DOCKERFILE = REPO / "Dockerfile"

#: Only what the DISCOVERY image must carry. The media lanes shell out to far
#: more (mflux, ffmpeg, rsvg-convert) and run on a workstation, not in this
#: image, so scanning every module would make this an inventory rather than a
#: gate. These are the modules the chart's workloads actually execute.
SCANNED = ("harness/github.py", "harness/inspect.py", "harness/feeds.py",
           "harness/fetching.py", "harness/memory_store.py", "harness/store.py")

#: Present in any POSIX image; asserting them would be noise.
ASSUMED = {"sh", "env", "python", "python3"}


def invoked(path: Path) -> set[str]:
    """Bare-name executables this module hands to subprocess.

    A bare name is the interesting case: it goes through PATH, so it is the
    one the image must provide. An absolute path fails loudly on its own.
    """
    out = set()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        name = getattr(node.func, "attr", None) or getattr(node.func, "id", None)
        if name not in ("run", "Popen", "check_output", "call", "check_call"):
            continue
        if not node.args:
            continue
        argv = node.args[0]
        if not isinstance(argv, (ast.List, ast.Tuple)) or not argv.elts:
            continue
        first = argv.elts[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            exe = first.value
            if "/" not in exe and exe not in ASSUMED:
                out.add(exe)
    return out


def installed() -> set[str]:
    """What the Dockerfile puts on PATH: apt packages plus anything installed
    into /usr/local/bin."""
    text = DOCKERFILE.read_text(encoding="utf-8")
    got = set()
    for m in re.finditer(r"apt-get install[^\n]*(?:\\\n[^\n]*)*", text):
        for word in re.split(r"[\s\\]+", m.group(0)):
            if word and not word.startswith("-") and word not in (
                    "apt-get", "install", "&&"):
                got.add(word)
    got |= set(re.findall(r"install -m \S+ \S+ /usr/local/bin/(\S+)", text))
    return got


def test_the_image_provides_every_bare_name_the_code_runs():
    """The seam assertion. It fails when EITHER side changes alone: a new
    subprocess call in the code, or a package dropped from the image."""
    needed = set().union(*(invoked(REPO / m) for m in SCANNED))
    assert needed, "nothing scanned, so this asserts nothing"
    missing = sorted(needed - installed())
    assert not missing, (
        f"the code invokes {missing} and the Dockerfile installs none of them. "
        f"This is the gh failure again: the pod discovers it, not the build.")


def test_the_scan_actually_finds_the_known_tools():
    """A negative control for the extractor. If the AST walk silently matched
    nothing, the test above would pass forever."""
    needed = set().union(*(invoked(REPO / m) for m in SCANNED))
    assert {"git", "gh"} <= needed, f"expected git and gh among {sorted(needed)}"


def test_an_absolute_path_is_not_the_image_s_problem():
    """`/usr/bin/time` fails loudly by itself; only PATH lookups can silently
    find the wrong thing or nothing."""
    src = "subprocess.run(['/usr/bin/time', '-v', 'x'])"
    p = REPO / "harness" / "github.py"
    tree = ast.parse(src)
    found = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and node.args:
            argv = node.args[0]
            if isinstance(argv, ast.List):
                first = argv.elts[0]
                if isinstance(first, ast.Constant) and "/" not in first.value:
                    found.add(first.value)
    assert found == set()


def test_the_dockerfile_reader_is_not_matching_everything():
    """A parser that returned every word would make the gate vacuous."""
    got = installed()
    assert "git" in got and "gh" in got
    assert "FROM" not in got and "RUN" not in got
