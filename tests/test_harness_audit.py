"""The out-of-sample audit, and above all its FALSE POSITIVES.

Its first run reported twelve findings in this repo and precision was 0/12.
Every one was a call that merely shares the name `open`. The conclusion built
on them -- "the encoding gate has a hole" -- was one step from being published,
and the cohort rates computed from the same matcher were noise.

So the negative cases here are not decoration. They are the test.
"""
import ast
from pathlib import Path

import pytest

from harness import audit


def opens(src: str) -> int:
    """How many text opens the matcher counts in this source."""
    t = audit.Tally()
    p = Path("x.py")
    tree = ast.parse(src)
    # scan_file reads from disk; exercise the matcher directly instead.
    n = 0
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and audit._is_text_open(node):
            n += 1
    return n


# ---- the four shapes that made the first run worthless -------------------

@pytest.mark.parametrize("src,why", [
    ("Image.open(path)", "PIL; nothing to do with text encoding"),
    ("wave.open(str(path))", "stdlib audio"),
    ("os.open(path, os.O_RDWR)", "a file descriptor; no encoding parameter exists"),
    ("gzip.open(path)", "compressed; its own encoding rules"),
    ("socket.open()", "not a file at all"),
])
def test_a_call_that_merely_shares_the_name_is_not_a_text_open(src, why):
    assert opens(src) == 0, why


def test_a_method_open_is_not_guessed_at():
    """`path.open()` IS a text open and `sock.open()` is not, and an AST cannot
    tell them apart without the receiver's type. In a measuring instrument a
    false negative costs a smaller number; a false positive costs the result."""
    assert opens('path.open("rb")') == 0
    assert opens("path.open()") == 0


# ---- what must still be caught -------------------------------------------

def test_a_bare_builtin_open_is_caught():
    assert opens("open(path)") == 1
    assert opens("open(path, 'w')") == 1
    assert opens("data = open(path).read()") == 1


def test_read_text_and_write_text_are_caught():
    assert opens("p.read_text()") == 1
    assert opens("p.write_text(body)") == 1


def test_a_literal_binary_mode_is_not_a_text_open():
    assert opens("open(path, 'rb')") == 0
    assert opens("open(path, mode='wb')") == 0


def test_a_computed_mode_is_treated_as_text():
    """The conservative direction: a false positive here costs an argument, a
    false negative costs a UnicodeDecodeError on somebody else's machine."""
    assert opens("open(path, mode)") == 1


def test_an_encoded_open_is_not_a_finding(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("open(p, encoding='utf-8')\np.read_text(encoding='utf-8')\n",
                 encoding="utf-8")
    t = audit.Tally()
    assert audit.scan_file(f, t) == []
    assert t.opens == 2 and t.unencoded == 0


def test_an_unencoded_open_is(tmp_path):
    f = tmp_path / "m.py"
    f.write_text("open(p)\n", encoding="utf-8")
    t = audit.Tally()
    found = audit.scan_file(f, t)
    assert len(found) == 1 and t.unencoded == 1


# ---- mutable defaults ----------------------------------------------------

@pytest.mark.parametrize("src", ["def f(a=[]): pass", "def f(a={}): pass",
                                 "def f(a=set()): pass", "def f(a=list()): pass",
                                 "def f(*, a=[]): pass"])
def test_a_mutable_default_is_caught(src, tmp_path):
    f = tmp_path / "m.py"
    f.write_text(src + "\n", encoding="utf-8")
    t = audit.Tally()
    assert audit.scan_file(f, t) and t.mutable_default == 1


@pytest.mark.parametrize("src", ["def f(a=None): pass", "def f(a=()): pass",
                                 "def f(a=0): pass", "def f(a=CONST): pass"])
def test_an_immutable_default_is_not(src, tmp_path):
    f = tmp_path / "m.py"
    f.write_text(src + "\n", encoding="utf-8")
    t = audit.Tally()
    assert audit.scan_file(f, t) == [] and t.mutable_default == 0


# ---- the denominator -----------------------------------------------------

def test_rates_are_per_opportunity_not_raw_counts():
    """Three opens with one bad is not healthier than three hundred with ten."""
    t = audit.Tally()
    assert t.rate(1, 3) == 333.3
    assert t.rate(10, 300) == 33.3


def test_no_opportunities_reports_nothing_rather_than_zero():
    """A repo that never opens a file has no rate. Reporting 0.0 would rank it
    as the healthiest codebase measured."""
    assert audit.Tally().rate(0, 0) is None


def test_unparseable_source_is_counted_not_swallowed(tmp_path):
    """Python 2, or a template. It must show in the denominator rather than
    quietly shrinking the sample."""
    f = tmp_path / "m.py"
    f.write_text("def f(:\n", encoding="utf-8")
    t = audit.Tally()
    assert audit.scan_file(f, t) == [] and t.parse_errors == 1


def test_vendored_trees_are_skipped(tmp_path):
    (tmp_path / ".venv" / "lib").mkdir(parents=True)
    (tmp_path / ".venv" / "lib" / "dep.py").write_text("open(p)\n", encoding="utf-8")
    (tmp_path / "mine.py").write_text("open(p)\n", encoding="utf-8")
    t, found = audit.scan(tmp_path)
    assert t.files == 1 and len(found) == 1
