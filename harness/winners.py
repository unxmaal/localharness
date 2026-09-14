"""What the receipts say won each lane, against what the code has typed in.

harness/cli.py carries four constants -- DEFAULT_SVG_MODEL and friends -- above
a thirty-line comment recording the run that chose them. The run is real and
the comment is careful. It is still a HAND COPY of a number that lives
somewhere else, which is this project's most-bitten failure class: one question
with two answers, and nothing that notices when they drift.

WHY NOT DERIVE THE DEFAULT DIRECTLY. A default that changes because somebody
ran an eval last night is a CLI whose behaviour depends on local history, and
two machines would then disagree about what `lh svg` does. The constant stays:
it is stable, reviewable, and arrives with the argument for it. What changes is
that a drift is now VISIBLE instead of silent -- `lh discover --winners` shows
both answers side by side, and a test reports disagreement.

AN INVENTORY, NOT A GATE. Receipts live in a directory that no clone has, so on
CI this compares nothing and must not fail; on a machine that has run the
suite, it is the only place the two answers meet.
"""
from __future__ import annotations

import json

#: A candidate spec that is not a bare model: `repair:local-large` is a
#: WORKFLOW built on a model, and `trace/mflux/...` a composition. Both can win
#: a lane on merit, and neither is what a `--model` default can be set to.
_COMPOSITE = (":", "/")


def is_plain_model(candidate: str) -> bool:
    return not any(mark in candidate for mark in _COMPOSITE)


def from_receipts(runs=None, plain_only: bool = True) -> dict[str, dict]:
    """The best candidate per modality, read from every run receipt on disk.

    Best is the highest pass rate, ties broken by the faster median -- the same
    order the lane table in cli.py was chosen with, so the two are comparable
    rather than merely both being opinions.
    """
    from harness import paths
    try:
        receipts = sorted((runs or paths.runs()).rglob("results.json"))
    except OSError:
        return {}
    best: dict[str, dict] = {}
    for f in receipts:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue        # a half-written run must not decide a lane
        receipt = data.get("receipt") or {}
        modality = (receipt.get("modality") or "").strip().lower()
        if not modality:
            continue
        # A SCREEN IS NOT A MEASUREMENT. Ranking one against the other compares
        # two different exams, which comparable() refuses for the same reason.
        if (receipt.get("tier") or "measure") != "measure":
            continue
        for candidate, row in (data.get("summary") or {}).items():
            if plain_only and not is_plain_model(candidate):
                continue
            rate = float(row.get("pass_rate") or 0.0)
            median = float(row.get("median_s") or 0.0)
            got = {"candidate": candidate, "pass_rate": rate,
                   "median_s": median, "total": int(row.get("total") or 0),
                   "run": f.parent.name}
            held = best.get(modality)
            if held is None or (rate, -median) > (held["pass_rate"],
                                                  -held["median_s"]):
                best[modality] = got
    return best


def typed() -> dict[str, str]:
    """What the CLI has written down, by the modality each default serves."""
    from harness import cli
    return {"svg": cli.DEFAULT_SVG_MODEL, "web": cli.DEFAULT_WEB_MODEL,
            "code": cli.DEFAULT_CODE_MODEL, "extract": cli.DEFAULT_EXTRACT_MODEL}


def beaten_in(runs=None) -> dict[str, dict]:
    """Per modality, the best candidate from runs THE TYPED DEFAULT WAS IN.

    ONLY WITHIN ONE COMPARISON, and that restriction is the whole instrument.
    The first cut took the best candidate across every receipt on disk and
    reported that `extract` had been won by q3-1.7b at 0.7 -- because the only
    extract receipt in the tree is a three-small-model run that local-large was
    never part of, while the run that scored it 9/10 predates receipts and
    lives in a directory no longer read.

    So "the best on disk" is not "the best measured" when the receipt set is
    incomplete, and a default that was never in the room did not lose. Same
    rule comparable() enforces: compare within one exam.
    """
    from harness import paths
    try:
        receipts = sorted((runs or paths.runs()).rglob("results.json"))
    except OSError:
        return {}
    wanted = typed()
    best: dict[str, dict] = {}
    for f in receipts:
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        receipt = data.get("receipt") or {}
        modality = (receipt.get("modality") or "").strip().lower()
        summary = data.get("summary") or {}
        if modality not in wanted or wanted[modality] not in summary:
            continue
        if (receipt.get("tier") or "measure") != "measure":
            continue
        for candidate, row in summary.items():
            if not is_plain_model(candidate):
                continue
            rate = float(row.get("pass_rate") or 0.0)
            median = float(row.get("median_s") or 0.0)
            held = best.get(modality)
            if held is None or (rate, -median) > (held["pass_rate"],
                                                  -held["median_s"]):
                best[modality] = {"candidate": candidate, "pass_rate": rate,
                                  "median_s": median, "run": f.parent.name,
                                  "total": int(row.get("total") or 0)}
    return best


def disagreements(runs=None) -> list[dict]:
    """Where a typed default lost a comparison it was actually in.

    A modality whose default appears in no receipt is NOT a disagreement:
    nothing measured it here, and reporting silence as conflict is how an
    inventory becomes noise nobody reads. It is reported as `unmeasured`
    instead, which is a different and also useful thing to know.
    """
    best = beaten_in(runs)
    out = []
    for modality, name in sorted(typed().items()):
        got = best.get(modality)
        if not got:
            out.append({"modality": modality, "typed": name,
                        "state": "unmeasured", "measured": "", "run": ""})
        elif got["candidate"] != name:
            out.append({"modality": modality, "typed": name, "state": "beaten",
                        "measured": got["candidate"],
                        "pass_rate": got["pass_rate"],
                        "median_s": got["median_s"], "run": got["run"]})
    return out
