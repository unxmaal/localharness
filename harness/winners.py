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


#: HOW EACH FAMILY OF DEFAULT IS SPELLED, because the constant and the receipt
#: key are not the same string and pretending otherwise is the identity-by-name
#: defect this project has already paid for.
#:
#:   gateway alias   local-large                      -> local-large
#:   engine spec     mflux:flux2-klein-4b             -> mflux/flux2-klein-4b-q8
#:   speech model    mlx-community/Kokoro-82M-bf16    -> Kokoro-82M-bf16/af_sky
#:
#: Two of those differences are SPELLING and one is not. `:` and `/` are the
#: same separator in two notations, and the receipt drops the org prefix a
#: HuggingFace id carries -- both are safe to normalise. A `-q8` suffix is NOT:
#: `flux2-klein-4b` and `flux2-klein-4b-q8` are different artifacts, and the
#: constant naming the first while every run measured the second is a finding
#: rather than a mismatch to smooth over.
FAMILIES = {
    "svg": "alias", "web": "alias", "code": "alias", "extract": "alias",
    "image": "engine", "video": "engine",
    "tts": "speech", "stt": "speech",
}


def _normal(name: str, family: str) -> str:
    """The comparable form of a name, per family. Lowercased, because the
    receipt and the constant disagree on case in places too."""
    got = name.strip().lower()
    if family == "engine":
        got = got.replace(":", "/")
    return got


def matches(typed_name: str, candidate: str, family: str) -> str:
    """"" if these are different things, else "exact" or "quantised".

    `quantised` is reported rather than hidden: it means the receipts only ever
    measured a quantisation of the thing the constant names, so the default is
    under-specified rather than wrong. mflux hardcodes quantize=8 and the
    candidate name does not carry it, which is how a lane default came to name
    an artifact no run here has ever produced.
    """
    want = _normal(typed_name, family)
    got = _normal(candidate, family)
    if family == "speech":
        # THE TWO ARE TRIMMED FROM OPPOSITE ENDS, which is the whole reason
        # this needs saying out loud. A default is a HuggingFace id carrying an
        # org prefix; a receipt key is a model and, for tts, the voice it used.
        #   mlx-community/Kokoro-82M-bf16  ->  Kokoro-82M-bf16   (drop prefix)
        #   Kokoro-82M-bf16/af_sky         ->  Kokoro-82M-bf16   (drop voice)
        want = want.split("/")[-1]
        got = got.split("/")[0]
    if got == want:
        return "exact"
    import re as _re
    if _re.fullmatch(_re.escape(want) + r"-q\d+", got):
        return "quantised"
    return ""


def from_receipts(runs=None, plain_only: bool = True) -> dict[str, dict]:
    """The best candidate per modality, read from every run receipt on disk.

    Best is the lane's own metric first. See order_key().
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
            got = {"candidate": candidate, "key": order_key(row),
                   "pass_rate": float(row.get("pass_rate") or 0.0),
                   "median_s": float(row.get("median_s") or 0.0),
                   "total": int(row.get("total") or 0), "run": f.parent.name}
            held = best.get(modality)
            if held is None or got["key"] > held["key"]:
                best[modality] = got
    return best


def order_key(row: dict) -> tuple:
    """How good a summary row is, best LAST so `max` picks the winner.

    THE LANE'S OWN METRIC COMES FIRST, and getting that wrong is not
    hypothetical: ranking on pass rate and then latency reported that
    parakeet-ctc had beaten parakeet-tdt-v2 in the stt lane. Both passed 300 of
    300, and ctc has the faster median -- so on those two statistics ctc wins,
    and on the one the lane is actually about it loses:

        parakeet-tdt-0.6b-v2   wer 0.0162   median 0.135   <- adopted
        parakeet-ctc-0.6b      wer 0.0225   median 0.116

    A statistic blind to the effect under test reports a null, or worse an
    inversion, with no visible symptom. METRIC_DIRECTION exists for exactly
    this and says which way each metric runs; `neutral` ones are reported and
    never ranked on.
    """
    from evals.core import direction_of

    metrics = row.get("metrics") or {}
    ranked = []
    for name in sorted(metrics):
        if direction_of(name) == "neutral":
            continue
        value = float(metrics[name] or 0.0)
        ranked.append(value if direction_of(name) == "higher" else -value)
    rate = float(row.get("pass_rate") or 0.0)
    median = float(row.get("median_s") or 0.0)
    # A candidate that passed less often is worse whatever its metric says: a
    # model that fails half the cases and scores well on the rest is scoring on
    # a different, easier subset.
    return (rate, tuple(ranked), -median)


def typed() -> dict[str, str]:
    """What the code has written down, by the modality each default serves.

    ALL OF THEM, not the four that happened to be in one file. The image engine
    is the one #148 phase 5 names by hand and the one #141 and #142 are both
    about, and it lived two modules away from the others.

    The speech defaults are chosen by platform, so this reports the one THIS
    machine would use: comparing a Mac's receipts against a Windows constant
    would be the accelerator mistake in another costume.
    """
    from harness import audio, cli
    return {"svg": cli.DEFAULT_SVG_MODEL, "web": cli.DEFAULT_WEB_MODEL,
            "code": cli.DEFAULT_CODE_MODEL,
            "extract": cli.DEFAULT_EXTRACT_MODEL,
            "image": cli.DEFAULT_IMAGE_ENGINE,
            "video": cli.DEFAULT_VIDEO_ENGINE,
            "tts": audio.DEFAULT_TTS_MODEL, "stt": audio.DEFAULT_STT_MODEL}


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
        if modality not in wanted:
            continue
        if (receipt.get("tier") or "measure") != "measure":
            continue
        family = FAMILIES.get(modality, "alias")
        # WAS THE TYPED DEFAULT IN THIS RUN AT ALL? Spelled per family, because
        # the constant and the receipt key are different notations for the same
        # thing in two of the three families.
        how = {c: matches(wanted[modality], c, family) for c in summary}
        if not any(how.values()):
            continue
        for candidate, row in summary.items():
            # A composition can win a lane on merit and still not be something
            # a --model default can be set to. An engine spec is not composite
            # in that sense: `mflux/flux2-klein-4b-q8` IS the engine default.
            if family != "engine" and not is_plain_model(candidate) \
                    and not how.get(candidate):
                continue
            key = order_key(row)
            held = best.get(modality)
            if held is None or key > held["key"]:
                best[modality] = {
                    "candidate": candidate, "key": key,
                    "pass_rate": float(row.get("pass_rate") or 0.0),
                    "median_s": float(row.get("median_s") or 0.0),
                    "metrics": dict(row.get("metrics") or {}),
                    "run": f.parent.name,
                    "total": int(row.get("total") or 0),
                    "match": how.get(candidate, ""), "family": family}
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
        elif got["match"] == "quantised":
            # NOT a disagreement about which candidate is better. The constant
            # names an artifact no run here has produced, because the engine
            # quantises and the candidate name does not say so.
            out.append({"modality": modality, "typed": name,
                        "state": "under-specified",
                        "measured": got["candidate"],
                        "pass_rate": got["pass_rate"],
                        "median_s": got["median_s"], "run": got["run"]})
        elif not got["match"]:
            out.append({"modality": modality, "typed": name, "state": "beaten",
                        "measured": got["candidate"],
                        "pass_rate": got["pass_rate"],
                        "median_s": got["median_s"], "run": got["run"]})
    return out
