"""Find claims stated as fact, so they can be re-checked. ASSERTIONS.md.

assumptions.md enumerates what this project BELIEVES without proof: constants,
defaults, untested premises. This finds the other half -- what it STATES as
true, in prose a reader will act on.

The difference matters because the two rot differently. An assumption is
honestly labelled and stays put. An assertion was true when it was written,
reads as true forever, and carries no date, no configuration and no way to tell
whether anyone has checked it since.

WHAT MADE THIS WORTH BUILDING, 2026-09-12: `lh image` was measured at 23.9 GiB
against a recorded 11.4, and two issues were filed claiming the record was
stale. The record was right. The runs were at 1024x1024 because the CLI passes
no resolution and inherits mflux's default; every eval case pins 512. Neither
number carried the configuration that produced it, so two correct measurements
looked like a regression. See RULE on "a number without its configuration".
"""
from __future__ import annotations

import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

#: A number with a unit is the decaying kind of claim: it was measured once, on
#: one machine, in one configuration, and nothing about the sentence says so.
NUMBER = re.compile(
    r"\b\d+(?:[.,]\d+)?\s*"
    r"(?:GiB|GB|MiB|MB|KB|ms|s\b|%|x\b|/\d+|minutes?|hours?|seconds?)\b")

#: An absolute is the falsifiable kind: one counter-example ends it.
ABSOLUTE = re.compile(
    r"\b(?:never|always|cannot|the only|nothing else|no other|every|impossible)\b",
    re.I)

#: A provenance word is a promise that someone checked. Worth finding precisely
#: because the promise is usually the last record of the checking.
PROVENANCE = re.compile(
    r"\b(?:measured|verified|confirmed|proven?|reproduc\w+)\b", re.I)

#: What separates a MEASUREMENT from a NUMBER, which is the distinction this
#: whole file is about. Matching any one of these near a NUM claim means the
#: sentence says something about the conditions it was taken in: when, on what,
#: at what size, with which tool.
#:
#:   README:219  "Measured on the 4070, a request takes VRAM from 2462 to 3296
#:               MiB and fifteen idle seconds return it to 2473"   qualified
#:   README:33   "an image, ~19s"                                  not
#:
#: Deliberately generous. A scanner that flags everything gets switched off
#: within a week, and the cost of missing one is a row nobody re-checks, while
#: the cost of a false positive is an edit that makes prose worse.
QUALIFIER = re.compile(
    r"\b20\d\d-\d\d-\d\d\b"                    # a date
    r"|\b\d+\s*x\s*\d+\b"                        # a resolution or a grid
    r"|\b(?:M1|M2|M3|M4|M5)\b"                     # which Apple part
    r"|\bRTX\b|\b\d0[679]0\b"                     # which card
    r"|\b(?:macOS|Windows|Linux|Ubuntu|runner|unified|VRAM)\b"
    r"|\b\d+\s*-?\s*bit\b"                        # which quantisation
    r"|\bmeasured\s+(?:on|at|with|against)\b"
    r"|\bon (?:an?|the|this) \w+",                 # on an M2 Pro, on the 4070
    re.I)

#: How far from the number the conditions may sit. A paragraph, roughly: the
#: date is often on the line above and the machine on the line below.
QUALIFIER_WINDOW = 2

#: Dated records rather than live claims. A validation log SHOULD be full of
#: numbers from one afternoon; that is what it is for. The claims that rot are
#: the ones a reader takes as current.
ARCHIVES = {"docs/validation-log.md", "PLAN.md", "assumptions.md",
            "ASSERTIONS.md"}

TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml"}


@dataclass(frozen=True)
class Claim:
    path: str
    line: int
    kinds: tuple[str, ...]
    text: str
    qualified: bool = True

    @property
    def config_less(self) -> bool:
        """A number with nothing said about where it came from.

        Only NUM decays this way. An absolute needs a counter-example rather
        than a configuration, and a provenance word is itself the claim.
        """
        return "NUM" in self.kinds and not self.qualified

    def __str__(self) -> str:
        marks = list(self.kinds)
        if self.config_less:
            marks.append("CONFIG-LESS")
        return f"{self.path}:{self.line}\t{','.join(marks)}\t{self.text}"


def is_prose(path: Path, line: str) -> bool:
    """Comments and markdown only. A number in an expression is a value; a
    number in a sentence is a claim about the world."""
    s = line.strip()
    if path.suffix == ".md":
        # The INDENT test reads the raw line. An earlier cut tested the
        # stripped one, where leading spaces cannot exist, so the
        # indented-code-block rule silently never fired and every number in a
        # fenced-by-indentation block counted as a claim.
        if line.startswith("    ") or line.startswith("\t"):
            return False
        return not s.startswith(("```", "|---"))
    return s.startswith("#") or s.startswith('"""') or '"""' in s


def scan(text: str, path: Path, origin: str) -> list[Claim]:
    out = []
    lines = text.splitlines()
    for n, line in enumerate(lines, 1):
        if not is_prose(path, line):
            continue
        kinds = tuple(k for k, pat in (("NUM", NUMBER), ("ABS", ABSOLUTE),
                                       ("SAYS-MEASURED", PROVENANCE))
                      if pat.search(line))
        if kinds:
            lo, hi = max(0, n - 1 - QUALIFIER_WINDOW), n + QUALIFIER_WINDOW
            near = "\n".join(lines[lo:hi])
            out.append(Claim(origin, n, kinds, line.strip()[:160],
                             bool(QUALIFIER.search(near))))
    return out


def tracked(root: Path) -> list[Path]:
    r = subprocess.run(("git", "-C", str(root), "ls-files"),
                       capture_output=True, text=True, check=True)
    return [root / line for line in r.stdout.splitlines() if line]


def collect(root: Path, include_archives: bool = False) -> list[Claim]:
    out: list[Claim] = []
    for p in tracked(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix not in TEXT_SUFFIXES:
            continue
        if not include_archives and rel in ARCHIVES:
            continue
        if p.name == Path(__file__).name:
            continue
        out.extend(scan(p.read_text(encoding="utf-8", errors="replace"), p, rel))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.assertions",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--archives", action="store_true",
                    help="include the dated records, which are meant to be full of numbers")
    ap.add_argument("--kind", choices=("NUM", "ABS", "SAYS-MEASURED"),
                    help="only this kind of claim")
    ap.add_argument("--config-less", action="store_true",
                    help="only numbers with nothing said about the conditions "
                         "they were taken in, which is the kind that rots")
    a = ap.parse_args(argv)

    found = collect(Path(__file__).resolve().parent.parent, a.archives)
    if a.kind:
        found = [c for c in found if a.kind in c.kinds]
    if a.config_less:
        found = [c for c in found if c.config_less]
    for c in found:
        print(c)
    print(f"{len(found)} claim(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
