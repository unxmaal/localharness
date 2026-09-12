"""The scanner behind ASSERTIONS.md.

Same discipline as tests/test_harness_privacy.py: the negative cases matter
more than the positive ones, because a scanner that flags every line gets
turned off.
"""
from pathlib import Path

import pytest

from harness import assertions


def kinds(text, name="t.md"):
    return [c.kinds for c in assertions.scan(text, Path(name), name)]


# ---- what must be caught --------------------------------------------------

@pytest.mark.parametrize("line,kind", [
    ("# it holds 11.4 GiB while running", "NUM"),
    ("# measured at 40.5 minutes on this machine", "NUM"),
    ("# the request took 250ms", "NUM"),
    ("# tesseract read 6/18 where Vision read 18/18", "NUM"),
    ("# this never returns None", "ABS"),
    ("# nothing else on the machine can do it", "ABS"),
    ("# measured 2026-09-11 on a runner", "SAYS-MEASURED"),
    ("# verified against the vendor guide", "SAYS-MEASURED"),
])
def test_a_claim_in_a_comment_is_found(line, kind):
    assert kind in kinds(line, "t.py")[0]


def test_a_number_and_a_provenance_word_are_both_reported():
    got = kinds("# measured at 11.4 GiB on the M2 Pro", "t.py")[0]
    assert "NUM" in got and "SAYS-MEASURED" in got


# ---- what must NOT be caught ----------------------------------------------

def test_code_is_not_prose():
    """A number in an expression is a value. A number in a sentence is a claim
    about the world, and only the second kind can be wrong about reality."""
    assert kinds("MEMORY_CEILING = 22 * GIB", "t.py") == []
    assert kinds("timeout = 15.0", "t.py") == []


def test_a_fenced_block_in_markdown_is_not_prose():
    assert kinds("```", "t.md") == []
    assert kinds("    peak 23.9 GiB", "t.md") == []


def test_prose_without_a_claim_is_left_alone():
    assert kinds("# Resolve the path and return it.", "t.py") == []


# ---- the archives are excluded on purpose ---------------------------------

def test_the_dated_records_are_skipped_by_default():
    """A validation log SHOULD be full of numbers from one afternoon. What
    rots is a number a reader takes as current."""
    assert "docs/validation-log.md" in assertions.ARCHIVES
    assert "PLAN.md" in assertions.ARCHIVES
    assert "assumptions.md" in assertions.ARCHIVES
    assert "ASSERTIONS.md" in assertions.ARCHIVES


def test_archives_can_be_asked_for():
    root = Path(__file__).resolve().parent.parent
    assert len(assertions.collect(root, include_archives=True)) \
        > len(assertions.collect(root))


# ---- the repo still has claims, which is not a failure --------------------

def test_the_scan_finds_the_repo_it_documents():
    """Unlike harness/privacy.py this is an INVENTORY, not a gate. Zero would
    mean the scanner broke, not that the repo got honest."""
    found = assertions.collect(Path(__file__).resolve().parent.parent)
    assert len(found) > 100
    assert any("cli.py" in c.path for c in found)
