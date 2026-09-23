"""Verdicts a program cannot reach, collected from a person. Issue #273.

Not every capability has a metric. Per the 2026 literature there is no
per-clip style-similarity measure, Frechet Audio Distance is distributional
and cannot score one clip, and human preference studies remain the ground
truth for music generation. This project hit the same wall three times --
cover similarity, musical quality, svg aesthetics -- and each time answered by
not building the capability. The capability is wanted; only the AUTOMATIC
verdict is impossible.

WHAT THIS IS NOT: a research protocol. One person judges their own project.
There is no panel, no inter-rater agreement and no self-consistency control,
because with n=1 those measure nothing and cost real time.

The one thing kept from the machine path is that a comparison does not name
its candidates until it has been answered. Not ceremony -- it is less work to
print "A" than the model id, and knowing which one is the incumbent is the
cheapest way to accidentally confirm what you already believed.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

from harness import lanes, paths

#: How many answers before a pairing counts. One is a draw, not a measurement:
#: ACE-Step varies run to run at a pinned seed (RULE #280), and the same is
#: true of every diffusion lane here.
ENOUGH = 3

#: What a person may answer. "tie" is a real finding and not an evasion -- the
#: image lane has already recorded a statistical tie as a legitimate verdict,
#: and a forced choice would manufacture a winner from noise.
ANSWERS = ("a", "b", "tie")


def _file() -> Path:
    return paths.home() / "human-verdicts.json"


def _load() -> list[dict]:
    f = _file()
    if not f.is_file():
        return []
    try:
        return json.loads(f.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return []


def _save(rows: list[dict]) -> None:
    f = _file()
    f.parent.mkdir(parents=True, exist_ok=True)
    f.write_text(json.dumps(rows, indent=1), encoding="utf-8")


def pairings(receipt: dict) -> list[dict]:
    """Every A/B a person could be asked about, from one run's receipt.

    Candidates are compared WITHIN a case, because two different prompts are
    two different questions and preferring one says nothing about the models.
    """
    by_case: dict[str, list[dict]] = {}
    for row in receipt.get("rows") or []:
        if row.get("artifact") and row.get("passed"):
            by_case.setdefault(row["case_id"].split("#")[0], []).append(row)
    out = []
    for case, rows in sorted(by_case.items()):
        seen: dict[str, dict] = {}
        for r in rows:                     # one artifact per candidate per case
            seen.setdefault(r["candidate"], r)
        names = sorted(seen)
        for i, a in enumerate(names):
            for b in names[i + 1:]:
                out.append({"case": case, "a": a, "b": b,
                            "a_file": seen[a]["artifact"],
                            "b_file": seen[b]["artifact"]})
    return out


def key(lane: str, case: str, a: str, b: str) -> str:
    """One pairing's identity, order-independent: asking B-vs-A is the same
    question as A-vs-B and its answers belong in the same pile."""
    lo, hi = sorted((a, b))
    return f"{lanes.canonical(lane)}|{case}|{lo}|{hi}"


def record(lane: str, case: str, a: str, b: str, answer: str,
           shown_first: str = "") -> None:
    """Store one answer. `shown_first` is which candidate was on the left, so
    a later reader can tell whether the side was randomised."""
    if answer not in ANSWERS:
        raise ValueError(f"answer must be one of {ANSWERS}, got {answer!r}")
    lo, hi = sorted((a, b))
    winner = {"a": a, "b": b}.get(answer, "")
    rows = _load()
    rows.append({"key": key(lane, case, a, b), "lane": lanes.canonical(lane),
                 "case": case, "left": lo, "right": hi, "winner": winner,
                 "shown_first": shown_first})
    _save(rows)


def tally(lane: str, case: str, a: str, b: str) -> Counter:
    """Answers so far for one pairing, keyed by winner ("" means tie)."""
    k = key(lane, case, a, b)
    return Counter(r["winner"] for r in _load() if r["key"] == k)


def decided(lane: str, case: str, a: str, b: str) -> str | None:
    """The winner, "" for a tie, or None when it has not been asked enough.

    A plurality decides. A pairing that ties or splits evenly IS decided --
    as a tie -- because three answers that disagree is the finding, not a
    reason to keep asking.
    """
    counts = tally(lane, case, a, b)
    if sum(counts.values()) < ENOUGH:
        return None
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return ""
    return top[0][0]


def lane_verdict(lane: str, pairs: list[dict]) -> tuple[str | None, str]:
    """Who won the LANE, and why, from the per-case answers.

    Returns (winner, why). `None` means not enough has been judged to say;
    `""` means judged and no preference.

    EVERY PAIRING MUST BE DECIDED FIRST. Adopting on a partial set lets the
    order somebody happened to click in pick the winner, and the cases are
    not interchangeable -- this lane's own first run split `cover` to one
    candidate and `lyrics-dense` to the other, unanimously each way.

    A PLURALITY OF CASES WINS, and a tie is a real outcome rather than a
    reason to keep asking. Cases where the answer was "no preference" count
    toward the total and for nobody, so a candidate that wins one case out of
    four does not carry the lane on a technicality.
    """
    from collections import Counter

    decided_by = {}
    for p in pairs:
        got = decided(lane, p["case"], p["a"], p["b"])
        if got is None:
            return None, (f"{p['case']} has fewer than {ENOUGH} answers, so "
                          f"the lane is not judged yet")
        decided_by[p["case"]] = got
    if not decided_by:
        return None, "nothing to judge"

    counts = Counter(w for w in decided_by.values() if w)
    ties = sum(1 for w in decided_by.values() if not w)
    detail = ", ".join(f"{c}: {w.split('@')[-1] if w else 'no preference'}"
                       for c, w in sorted(decided_by.items()))
    if not counts:
        return "", f"no case had a preference ({detail})"
    top = counts.most_common()
    if len(top) > 1 and top[0][1] == top[1][1]:
        return "", (f"the cases disagree {top[0][1]}-{top[1][1]} with no "
                    f"majority ({detail})")
    won, n = top[0]
    return won, f"won {n} of {len(decided_by)} case(s) ({detail})" + (
        f", {ties} with no preference" if ties else "")


def pending(lane: str, pairs: list[dict]) -> list[dict]:
    """Pairings still short of ENOUGH answers, each with what it still needs
    and the side to show first, shuffled."""
    out = []
    for p in pairs:
        counts = tally(lane, p["case"], p["a"], p["b"])
        asked = sum(counts.values())
        if asked >= ENOUGH:
            continue
        left, right = ((p["a"], p["b"]) if random.random() < 0.5
                       else (p["b"], p["a"]))
        out.append({**p, "asked": asked, "needs": ENOUGH - asked,
                    "left": left, "right": right,
                    "left_file": p["a_file"] if left == p["a"] else p["b_file"],
                    "right_file": p["b_file"] if right == p["b"] else p["a_file"]})
    return out
