"""A reference must resolve.

THE VIOLATION: a name that pointed at something real when it was written. It
is the same decay as a number without its configuration -- true once, read as
true forever, and nothing notices the move.

Scoped to repo-relative paths, because those are the references this repo can
actually check. A URL needs the network and a model id needs a registry; both
belong in a slower job than `make check`.
"""
import re
from pathlib import Path

import pytest

from harness import repo

REPO = Path(__file__).resolve().parent.parent

#: Only directories this repo owns, so `scripts/env.sh` is checked and
#: `/usr/bin/time` is not.
OURS = ("harness/", "evals/", "scripts/", "docs/", "gateway/", "tests/",
        "tools/", "voice/", ".github/")

#: The lookbehind is the rule that makes this work: a BARE repo-relative path
#: means this repo, and a foreign one carries its repo in front of it. So
#: `scripts/env.sh` is checked and `EnviousWispr/scripts/eval/registry.py` is
#: not, which is also how a reader tells them apart.
REFERENCE = re.compile(
    r"(?<![\w/])(?:" + "|".join(re.escape(d) for d in OURS) + r")"
    r"[A-Za-z0-9_./-]*[A-Za-z0-9_]")

#: Prose names things that do not exist yet, on purpose. A runbook step that
#: CREATES a file is the clearest case, and so is a glob.
ALLOW = re.compile(r"\*|<|\$|\{")

#: Paths the documentation names BEFORE they exist, each with the step that
#: creates it. Enumerated rather than inferred: "it is gitignored" would excuse
#: a typo in any ignored path, which is most of what a build produces.
BUILT_BY_A_DOCUMENTED_STEP = {
    "tools/h3probe": "built by the h3probe step in PLAN.md; gitignored binary",
}


def documents():
    return [p for p in repo.publishable(REPO)
            if p.suffix == ".md" and p.is_file()]


def references(path: Path):
    out = []
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        for m in REFERENCE.finditer(line):
            ref = m.group(0).rstrip(".,:;)")
            if not ALLOW.search(ref):
                out.append((n, ref))
    return out


def shipped():
    """What a READER gets, not what happens to be on this disk.

    The first cut of this test called Path.exists(), passed here and failed on
    all three runners naming `tools/h3probe` -- a gitignored binary that exists
    on the machine that built it and on no clone. That is violation 5 from
    #160, committed inside the test written to catch violation 6, which is a
    fair measure of how easy it is: the check must see what a reader sees.
    """
    return {p.relative_to(REPO).as_posix() for p in repo.publishable(REPO)}


@pytest.mark.parametrize("doc", documents(),
                         ids=lambda p: p.relative_to(REPO).as_posix())
def test_every_path_named_in_prose_reaches_the_reader(doc):
    have = shipped()
    dirs = {d for p in have for d in _parents(p)}
    missing = [(n, r) for n, r in references(doc)
               if r not in have and r not in dirs
               and r not in BUILT_BY_A_DOCUMENTED_STEP]
    assert not missing, (
        f"{doc.relative_to(REPO)} names paths a clone does not have: "
        + ", ".join(f"line {n}: {r}" for n, r in missing))


def _parents(rel: str):
    parts = rel.split("/")
    return ["/".join(parts[:i]) for i in range(1, len(parts))]


def test_the_exceptions_are_still_exceptions():
    """An allowlist that outlives its reason is how a check rots. Every entry
    must still be absent from a clone; one that starts shipping belongs in the
    checked set, not in the excuses."""
    have = shipped()
    for path, why in BUILT_BY_A_DOCUMENTED_STEP.items():
        assert path not in have, \
            f"{path} ships now, so it no longer needs the exception: {why}"


def test_another_project_s_path_is_not_read_as_ours():
    """PLAN.md cites EnviousWispr's comparable() by path. Written bare it reads
    as a file in this repo, and this test found exactly that on its first run."""
    assert REFERENCE.findall(
        "borrowed from EnviousWispr/scripts/eval/model_registry.py") == []
    assert REFERENCE.findall("see scripts/env.sh") == ["scripts/env.sh"]


def test_the_check_would_notice_a_move():
    """A negative control. Without it a regex that matched nothing would pass
    every document in the repo and look like proof of health."""
    assert not (REPO / "harness/a_module_that_does_not_exist.py").exists()
    found = REFERENCE.findall("see harness/a_module_that_does_not_exist.py for it")
    assert found == ["harness/a_module_that_does_not_exist.py"]


def test_the_documents_were_actually_read():
    """Same reason: an empty corpus passes parametrized tests silently."""
    assert len(documents()) >= 3
    assert sum(len(references(d)) for d in documents()) > 20
