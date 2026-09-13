"""Find one person's home setup in text meant for strangers. Issue #136.

This repo is public. Every issue, PR body and doc is published, and prose
describing the author's desk -- a named volume, a spare drive in a particular
tower, which box plays games -- teaches a reader nothing and dates fast.

A MEASUREMENT KEEPS ITS HARDWARE. "Measured on an RTX 4070" is provenance and
must survive; the card is part of the instrument. What this refuses is the
domestic arrangement around it.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

from harness import repo

#: Opt out when a match is genuinely generic or is the anti-pattern being
#: quoted: put the marker on the line, or on the line above it for something
#: that will not take a trailing comment. Costs a visible marker, which is the
#: point -- an exemption should be readable as a decision.
ALLOW = re.compile(r"privacy-ok")

#: (name, pattern, what to write instead). Ordered most specific first so a
#: line reports the sharpest reason rather than the broadest.
PATTERNS: list[tuple[str, re.Pattern[str], str]] = [
    (
        "named-volume",
        # A real volume name carries a lowercase letter. ALL-CAPS is the
        # placeholder convention here, so /Volumes/FAST passes and
        # /Volumes/Models does not. /System/Volumes is Apple's, not a drive.
        re.compile(r"(?<!System)/Volumes/(?=\w*[a-z])\w+"),
        "one person's drive; use an ALL-CAPS placeholder or $HF_ROOT",
    ),
    (
        "home-path",
        re.compile(r"/(?:Users|home)/(?!<)(?!\w*\b(?:you|user|USER)\b)\w+/"),
        "a real account name; use ~/ or a placeholder",
    ),
    (
        "private-host",
        re.compile(r"\b(?:10|192\.168|172\.(?:1[6-9]|2\d|3[01]))\.\d+\.\d+\b"
                   r"|(?<![\w<])[a-z0-9-]+\.local\b"),
        "a LAN address; use <host> or a documented placeholder",
    ),
    (
        "spare-drive",
        re.compile(r"\bspare\s+(?:nvme|ssd|drive|disk)\b", re.I),
        "how one machine was built; say what the machine IS, not how it got there",
    ),
    (
        "same-box",
        re.compile(r"\b(?:in|on)\s+(?:the|that)\s+same\s+"
                   r"(?:desktop|tower|case|chassis)\b", re.I),
        "a physical arrangement nobody else has",
    ),
    (
        "gaming-box",
        re.compile(r"\b(?:plays?\s+games|main\s+job\s+is\s+games|"
                   r"gaming\s+(?:box|rig|desktop|machine))\b", re.I),
        "what this machine does after hours; say 'a machine with a day job'",
    ),
    (
        "machine-nickname",
        re.compile(r"\b(?:the|this|my|our)\s+mini\b", re.I),
        "a nickname for one box; name the runtime or the role",
    ),
]

#: People's names cannot ship in this file -- the checker would publish what it
#: exists to hide. One name per line in an untracked file, or in LH_PRIVATE_NAMES
#: separated by commas. Absent means the structural patterns still run.
NAMES_FILE = ".privacy-names"


def name_pattern(root: Path) -> tuple[str, re.Pattern[str], str] | None:
    import os

    raw = os.environ.get("LH_PRIVATE_NAMES", "")
    f = root / NAMES_FILE
    if f.exists():
        raw += "," + f.read_text(encoding="utf-8")
    names = [n.strip() for n in raw.replace("\n", ",").split(",") if n.strip()]
    if not names:
        return None
    alt = "|".join(re.escape(n) for n in sorted(names, key=len, reverse=True))
    return ("person-name", re.compile(rf"\b(?:{alt})\b", re.I),
            "a real person; say the role, or restructure to passive")


#: What git tracks, because that is exactly what gets published. An ignored
#: fixture full of local paths is nobody's business but this machine's.
TEXT_SUFFIXES = {".md", ".py", ".sh", ".yaml", ".yml", ".toml", ".txt", ".cfg"}


@dataclass(frozen=True)
class Finding:
    origin: str
    line: int
    name: str
    advice: str
    excerpt: str

    def __str__(self) -> str:
        return f"{self.origin}:{self.line}: {self.name}: {self.excerpt}\n    -> {self.advice}"


def scan(text: str, origin: str,
         extra: tuple[str, re.Pattern[str], str] | None = None) -> list[Finding]:
    out: list[Finding] = []
    patterns = PATTERNS + ([extra] if extra else [])
    lines = text.splitlines()
    for n, raw in enumerate(lines, 1):
        if ALLOW.search(raw) or (n > 1 and ALLOW.search(lines[n - 2])):
            continue
        for name, pat, advice in patterns:
            m = pat.search(raw)
            if m:
                out.append(Finding(origin, n, name, advice, m.group(0).strip()))
                break
    return out


#: A detector's own fixtures must contain what it detects, so these two are
#: never scanned. Nothing else gets an exemption without a privacy-ok marker.
SELF = {"harness/privacy.py", "tests/test_harness_privacy.py"}


def tracked(root: Path) -> list[Path]:
    """What a push would publish, not what a commit already did. See
    harness/repo.publishable: a check that cannot see the file you just wrote
    goes green on your desk and red on the runner, on the same content."""
    return repo.publishable(root)


def scan_paths(root: Path) -> list[Finding]:
    extra = name_pattern(root)
    out: list[Finding] = []
    for p in tracked(root):
        rel = p.relative_to(root).as_posix()
        if p.suffix not in TEXT_SUFFIXES or rel in SELF:
            continue
        out.extend(scan(p.read_text(encoding="utf-8", errors="replace"), rel, extra))
    return out


def _gh(*args: str) -> list[dict]:
    r = subprocess.run(("gh", *args), capture_output=True, text=True, check=True)
    return json.loads(r.stdout)


def scan_github(limit: int = 300,
                extra: tuple[str, re.Pattern[str], str] | None = None) -> list[Finding]:
    """Issues and PRs, which are published the moment they are filed."""
    out: list[Finding] = []
    for kind, cmd in (("issue", "issue"), ("pr", "pr")):
        rows = _gh(cmd, "list", "--state", "all", "--limit", str(limit),
                   "--json", "number,title,body")
        for r in rows:
            body = f"{r['title']}\n{r['body'] or ''}"
            out.extend(scan(body, f"{kind}#{r['number']}", extra))
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(prog="python -m harness.privacy",
                                 description=__doc__.splitlines()[0])
    ap.add_argument("--github", action="store_true",
                    help="also scan every issue and PR body")
    a = ap.parse_args(argv)

    root = Path(__file__).resolve().parent.parent
    found = scan_paths(root)
    if a.github:
        found += scan_github(extra=name_pattern(root))
    for f in found:
        print(f)
    print(f"{len(found)} finding(s)")
    return 1 if found else 0


if __name__ == "__main__":
    raise SystemExit(main())
