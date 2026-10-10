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

AN INVENTORY, NOT A GATE. Stored runs exist only where the suite ran, so on CI
this compares nothing and must not fail.
"""
from __future__ import annotations

#: A candidate spec that is not a bare model: `repair:<model>` is a
#: WORKFLOW built on a model, and `trace/mflux/...` a composition. Both can win
#: a lane on merit, and neither is what a `--model` default can be set to.
_COMPOSITE = (":", "/")


def is_plain_model(candidate: str) -> bool:
    return not any(mark in candidate for mark in _COMPOSITE)


#: Which lanes a default is an engine spec for, so a composite key can win.
FAMILIES = {
    "svg": "alias", "web": "alias", "code": "alias", "extract": "alias",
    "decide": "alias", "agent": "alias", "claims": "alias",
    "image": "engine", "video": "engine", "music": "engine", "ocr": "engine",
    "retrieval": "engine", "pii": "engine",
    "tts": "speech", "stt": "speech",
}


def from_receipts(conn, plain_only: bool = True) -> dict[str, dict]:
    """The best candidate per lane over every stored measure run. #410."""
    from harness import runs
    best: dict[str, dict] = {}
    for run, summary, _ in runs.summaries(conn, tier=runs.MEASURE):
        for candidate, row in summary.items():
            if plain_only and not is_plain_model(candidate):
                continue
            got = {"candidate": candidate, "key": order_key(row),
                   "pass_rate": float(row.get("pass_rate") or 0.0),
                   "median_s": float(row.get("median_s") or 0.0),
                   "total": int(row.get("total") or 0), "run": run["path"],
                   "run_id": run["id"]}
            held = best.get(run["lane"])
            if held is None or got["key"] > held["key"]:
                best[run["lane"]] = got
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
    from harness.models import build_here
    # A text default is a model id; a machine whose gateway serves another build of it uses that. #670.
    return {"svg": build_here(cli.DEFAULT_SVG_MODEL), "web": build_here(cli.DEFAULT_WEB_MODEL),
            "code": build_here(cli.DEFAULT_CODE_MODEL),
            "extract": build_here(cli.DEFAULT_EXTRACT_MODEL),
            "decide": build_here(cli.DEFAULT_DECIDE_MODEL),
            "claims": build_here(cli.DEFAULT_CLAIMS_MODEL),
            "agent": build_here(cli.DEFAULT_AGENT_MODEL),
            "image": cli.DEFAULT_IMAGE_ENGINE,
            "video": cli.DEFAULT_VIDEO_ENGINE,
            "music": cli.DEFAULT_MUSIC_ENGINE,
            "ocr": cli.DEFAULT_OCR_ENGINE,
            "retrieval": cli.DEFAULT_RETRIEVAL_ENGINE,
            "pii": cli.DEFAULT_PII_ENGINE,
            "tts": audio.DEFAULT_TTS_MODEL, "stt": audio.DEFAULT_STT_MODEL}


def typed_anywhere() -> dict[str, tuple[str, ...]]:
    """lane -> every value typed() returns on any platform, here first. #516."""
    from harness import audio
    out = {}
    for lane, name in typed().items():
        others = sorted(set(audio.SPEECH_DEFAULTS.get(lane, {}).values()) - {name})
        out[lane] = (name, *others)
    return out


def served_ids(conn) -> dict[str, int]:
    """lane -> the candidates row of what it serves: adopted here, else typed.

    #429; the adopted row comes from `adoptions`, by id. #412.
    """
    from harness import adopt, candidates, screen
    held = adopt.current(conn)
    out = {lane: row["candidate_id"] for lane, row in held.items()}
    for lane, name in typed().items():
        if lane in out:
            continue
        spec = screen.candidate_for(lane, name) or name
        cid = candidates.served(conn, spec, lane=lane)
        if cid:
            out[lane] = cid
    return out


def beaten_in(conn) -> dict[str, dict]:
    """Per lane, the best candidate from stored runs THE SERVED DEFAULT WAS IN.

    Only within one comparison: a default that was never in the room did not
    lose. "In the room" is its candidate id among the run's rows. #429.
    """
    from harness import runs
    wanted = served_ids(conn)
    best: dict[str, dict] = {}
    for run, _, rows in runs.summaries(conn, tier=runs.MEASURE):
        lane = run["lane"]
        if lane not in wanted or not any(
                r["candidate_id"] == wanted[lane] for r in rows):
            continue
        family = FAMILIES.get(lane, "alias")
        groups: dict = {}
        for r in rows:
            groups.setdefault(r["candidate_id"] or ("key", r["candidate"]),
                              []).append(r)
        for cid, group in groups.items():
            candidate, row = next(iter(runs.summarize(group).items()))
            served = cid == wanted[lane]
            if family != "engine" and not is_plain_model(candidate) \
                    and not served:
                continue
            key = order_key(row)
            held = best.get(lane)
            if held is None or key > held["key"]:
                best[lane] = {
                    "candidate": candidate, "key": key,
                    "pass_rate": float(row.get("pass_rate") or 0.0),
                    "median_s": float(row.get("median_s") or 0.0),
                    "metrics": dict(row.get("metrics") or {}),
                    "run": run["path"], "run_id": run["id"],
                    "total": int(row.get("total") or 0),
                    "match": "exact" if served else "",
                    "candidate_id": cid if isinstance(cid, int) else None,
                    "family": family}
    return best


def disagreements(conn) -> list[dict]:
    """Where a typed default lost a comparison it was actually in.

    A default in no stored run is `unmeasured`, not a disagreement.
    """
    from harness import adopt
    best = beaten_in(conn)
    out = []
    for modality, name in sorted(adopt.lane_defaults(conn).items()):
        got = best.get(modality)
        if not got:
            out.append({"modality": modality, "typed": name,
                        "state": "unmeasured", "measured": "", "run": ""})
        elif not got["match"]:
            out.append({"modality": modality, "typed": name, "state": "beaten",
                        "measured": got["candidate"],
                        "pass_rate": got["pass_rate"],
                        "median_s": got["median_s"], "run": got["run"]})
    return out
