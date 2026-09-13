"""Does a flaw class have power on code it was not derived from?

The gauntlet's classes were abstracted from defects in one repo, which makes
every hit in that repo uninformative: the classes were fitted to it. The only
honest test is out of sample, against a cohort nobody here wrote.

Runs the classes with UNAMBIGUOUS ground truth -- where a finding is a defect by
definition rather than by judgement -- so the count needs no adjudication.

    uv run python -m harness.audit A=/path/to/one B=/path/to/another

Rates are per 1000 OPPORTUNITIES, never raw counts: a numerator with no
denominator is the shape this whole exercise is about. A repo with three text
opens and one bad is not healthier than one with three hundred and ten.

MEASURED 2026-09-13, text opens with no encoding=, per 1000 opens:

    derivation repo, gated                0.0     (193 opens)
    second repo, src/, gated              0.0     ( 21 opens)
    second repo, tests/, NOT gated      177.7     (197 opens)
    four third-party codebases      692 - 1000

The control cohort being far WORSE than the ungated same-author code is the
result: the class is general rather than one author's habit. And the gated and
ungated halves of one repo, same week, differ by the gate alone.

RATE IS NOT HARM, and the difference matters before anyone quotes this. Most
code omitting encoding= works, because the locale default is UTF-8 on Linux and
macOS. The class bites where non-ASCII data meets a non-UTF-8 platform, which
is a real but conditional failure -- here it cost seventeen simultaneous
Windows test failures, which is why it is in the corpus at all.
"""
from __future__ import annotations

import ast
import json
import sys
from dataclasses import dataclass
from pathlib import Path

SKIP = {".venv", "node_modules", "site-packages", ".git", "build", "dist",
        "__pycache__", ".tox", ".mypy_cache"}

MUTABLE = (ast.List, ast.Dict, ast.Set)
MUTABLE_CALLS = {"list", "dict", "set", "collections.OrderedDict"}


@dataclass
class Tally:
    files: int = 0
    parse_errors: int = 0
    opens: int = 0          # denominator for the encoding class
    defaults: int = 0       # denominator for the mutable-default class
    unencoded: int = 0
    mutable_default: int = 0

    def rate(self, hits: int, opportunities: int) -> float | None:
        return round(1000 * hits / opportunities, 1) if opportunities else None


def _is_text_open(node: ast.Call) -> bool | None:
    """True for a text-mode builtin open or Path.read_text/write_text.

    THIS FUNCTION IS WHY THE FIRST RUN OF THIS AUDIT WAS WORTHLESS. It matched
    any call named `open`, and Python is full of unrelated ones:

        Image.open(path)     PIL, nothing to do with text encoding
        wave.open(str(path)) stdlib audio
        os.open(path, flags) a file descriptor, no encoding parameter exists
        path.open("rb")      binary, and the mode is at index 0 on a METHOD,
                             not index 1 as on the builtin

    All twelve findings it reported for the derivation repo were of those four
    shapes: precision 0/12. The cohort rates built on it were noise, and
    "the gate has a hole" was about to be published off them.

    So: only a BARE `open` resolves to the builtin. An attribute `.open()`
    cannot be resolved without knowing the receiver's type, so it is counted
    as unclassifiable rather than guessed at -- a false negative in a
    measuring instrument costs a smaller number, a false positive costs the
    whole result.
    """
    f = node.func
    if isinstance(f, ast.Attribute) and f.attr in ("read_text", "write_text"):
        return True
    if isinstance(f, ast.Name) and f.id == "open":
        for i, a in enumerate(node.args):
            if i == 1:
                return not (isinstance(a, ast.Constant)
                            and isinstance(a.value, str) and "b" in a.value)
        for kw in node.keywords:
            if kw.arg == "mode" and isinstance(kw.value, ast.Constant) \
                    and isinstance(kw.value.value, str) and "b" in kw.value.value:
                return False
        return True
    return None


def scan_file(path: Path, t: Tally) -> list[str]:
    try:
        tree = ast.parse(path.read_text(encoding="utf-8", errors="replace"))
    except (SyntaxError, ValueError):
        t.parse_errors += 1
        return []
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            kind = _is_text_open(node)
            if kind:
                t.opens += 1
                if not any(k.arg == "encoding" for k in node.keywords):
                    t.unencoded += 1
                    out.append(f"{path}:{node.lineno} text open, no encoding=")
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            for d in list(node.args.defaults) + [
                    x for x in node.args.kw_defaults if x is not None]:
                t.defaults += 1
                bad = isinstance(d, MUTABLE) or (
                    isinstance(d, ast.Call)
                    and getattr(d.func, "id", None) in MUTABLE_CALLS)
                if bad:
                    t.mutable_default += 1
                    out.append(f"{path}:{node.lineno} mutable default in "
                               f"{node.name}()")
    return out


def scan(root: Path, limit: int = 4000) -> tuple[Tally, list[str]]:
    t, findings = Tally(), []
    for p in sorted(root.rglob("*.py")):
        if SKIP & set(p.parts):
            continue
        t.files += 1
        findings += scan_file(p, t)
        if t.files >= limit:
            break
    return t, findings


def main(argv: list[str]) -> int:
    if not argv:
        print(__doc__)
        return 2
    rows = []
    for spec in argv:
        label, _, path = spec.partition("=")
        t, findings = scan(Path(path).expanduser())
        rows.append((label, t, findings))

    w = max(len(r[0]) for r in rows)
    print(f"{'cohort':<{w}}  {'files':>6} {'opens':>6} {'no-enc':>7} "
          f"{'/1k':>6}   {'defaults':>8} {'mutable':>8} {'/1k':>6}")
    for label, t, _ in rows:
        print(f"{label:<{w}}  {t.files:>6} {t.opens:>6} {t.unencoded:>7} "
              f"{str(t.rate(t.unencoded, t.opens)):>6}   "
              f"{t.defaults:>8} {t.mutable_default:>8} "
              f"{str(t.rate(t.mutable_default, t.defaults)):>6}")
    print()
    for label, t, findings in rows:
        if findings:
            print(f"-- {label}: {len(findings)} finding(s), first 5")
            for f in findings[:5]:
                print(f"   {f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
