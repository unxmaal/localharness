"""A check must see what CI will see.

THE VIOLATION: harness/privacy.py read `git ls-files`, so the file written
thirty seconds ago was invisible to it. The scanner went green on this desk and
red on three runners, on the same commit, and the scanner was right both times.
Its INPUT differed. RULE #231.

A detector's fixtures are deliberate examples of the thing being detected, so
this is not a rare corner: it is what happens every time someone writes a
detector and its tests in one sitting.
"""
import subprocess
from pathlib import Path

import pytest

from harness import repo


@pytest.fixture
def tree(tmp_path):
    subprocess.run(("git", "init", "-q", str(tmp_path)), check=True)
    (tmp_path / "committed.md").write_text("x", encoding="utf-8")
    subprocess.run(("git", "-C", str(tmp_path), "add", "committed.md"),
                   check=True)
    (tmp_path / ".gitignore").write_text("ignored.md\nbuild/\n", encoding="utf-8")
    (tmp_path / "ignored.md").write_text("x", encoding="utf-8")
    (tmp_path / "written-just-now.md").write_text("x", encoding="utf-8")
    return tmp_path


def names(root):
    return {p.relative_to(root).as_posix() for p in repo.publishable(root)}


def test_the_file_written_just_now_is_visible(tree):
    """The whole point. `git ls-files` alone answers "what is committed"; the
    question a scanner asks is "what would a reader see"."""
    assert "written-just-now.md" in names(tree)


def test_what_is_committed_is_still_included(tree):
    assert "committed.md" in names(tree)


def test_an_ignored_file_is_not_published(tree):
    """--exclude-standard, so .privacy-names and hf_root/ stay out. An ignored
    file is not going anywhere, and scanning it would flag the one file the
    author deliberately keeps local."""
    assert "ignored.md" not in names(tree)


def test_nothing_is_listed_twice(tree):
    """A staged file appears in both git commands. Counting it twice would
    double every claim in the assertions inventory."""
    got = [p for p in repo.publishable(tree)]
    assert len(got) == len(set(got))


def test_the_scanners_use_it():
    """Both of them, because the next scanner will copy whichever it reads."""
    from harness import assertions, privacy
    assert privacy.tracked.__module__ == "harness.privacy"
    for mod in (assertions, privacy):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        assert "repo.publishable" in src, \
            f"{mod.__name__} answers 'what would a reader see' its own way again"
