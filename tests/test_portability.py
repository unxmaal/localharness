"""Rules that only a machine unlike the author's would notice being broken.

Everything here reads the SOURCE rather than running it. That is deliberate:
these are defects that cannot fail on macOS, so no amount of running the suite
there would catch one being reintroduced.
"""
import ast
import re
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
#: The tests too. `open("scripts/env.sh").read()` in a test fails on
#: Windows for exactly the same reason the library code did.
PACKAGES = ("harness", "evals", "tests")

#: `Path.read_text(encoding="utf-8")` and `.write_text()` with no encoding use
#: `locale.getencoding()`. That is UTF-8 on macOS and cp1252 on a stock
#: Windows install, so a file holding any byte outside cp1252 raises
#: UnicodeDecodeError there and nowhere else.
_TEXT_IO = ("read_text", "write_text")


def _sources():
    for package in PACKAGES:
        yield from sorted((REPO / package).rglob("*.py"))


def test_no_text_file_is_read_or_written_without_an_explicit_encoding():
    """CAUGHT 17 FAILING TESTS ON WINDOWS AT ONCE.

    `charmap codec can't decode byte 0x9d` out of a YAML case file, a cached
    feed and a cloned candidate's source. The author's machine cannot produce
    this failure, which is the whole reason it is asserted against the source.
    """
    offenders = []
    for path in _sources():
        # Parsed, not grepped: the call may be spread over several lines, and a
        # line-wise check reported three already-fixed sites as offenders.
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if (isinstance(node, ast.Call)
                    and isinstance(node.func, ast.Attribute)
                    and node.func.attr in _TEXT_IO
                    and not any(kw.arg == "encoding" for kw in node.keywords)):
                offenders.append(
                    f"{path.relative_to(REPO)}:{node.lineno}: .{node.func.attr}()")
    assert not offenders, (
        "text I/O without an explicit encoding defaults to the locale codepage "
        "on Windows:\n  " + "\n  ".join(offenders))


#: `df -g` is a BSD flag. GNU coreutils -- Git Bash on Windows, and Linux --
#: answers "unknown option -- g", so the function reads empty, every candidate
#: is judged unusable, and env.sh reports "no writable location with 0GB free".
#: That reads as a machine with no disk rather than as a wrong flag.
#: `df -Pk` forces 1024-byte blocks on both.
_BSD_DF = re.compile(r"df\s+-[A-Za-z]*g")


def test_no_script_uses_a_bsd_only_df_flag():
    offenders = []
    for path in sorted((REPO / "scripts").glob("*.sh")):
        for number, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if line.lstrip().startswith("#"):
                continue
            if _BSD_DF.search(line):
                offenders.append(
                    f"{path.relative_to(REPO)}:{number}: {line.strip()}")
    assert not offenders, (
        "df -g is BSD-only; use `df -Pk` and divide by 1048576 -- "
        + "; ".join(offenders))
