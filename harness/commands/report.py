"""`soh report` and the discover reports on the queue, coverage, winners and the export."""
from __future__ import annotations

import json
import sys

from harness.commands import loop as loop_cmd


def cmd_report(a) -> int:
    """Write the status page. Issue #231."""
    from harness import report

    if getattr(a, "json", False):
        print(json.dumps(report.state(), indent=1, default=str))
        return 0
    if getattr(a, "auto_publish", ""):
        from harness import publish
        on = a.auto_publish == "on"
        print(f"{publish.set_enabled(on)}: publishing after each discovery "
              f"loop is {'on' if on else 'off'}")
        return 0
    if getattr(a, "export", False) or getattr(a, "publish", False):
        return _report_export(a)
    out = report.write(getattr(a, "out", "") or None)
    state = report._load_state(out.with_suffix(".json"))
    lanes = state.get("lanes") or []
    unverified = [l["lane"] for l in lanes if l.get("unverified")]
    stale = [l["lane"] for l in lanes if l.get("stale")]
    print(f"\n{out}")
    if state.get("loop"):
        print(f"  {state['loop']['line']}")
    for l in lanes:
        if l.get("adoption"):
            print(f"  {l['lane']}: {report._model(l['serves'])} adopted {l['adoption']}")
        if l.get("refused"):
            print(f"  {l['lane']}: {l['refused']}")
    if unverified:
        print(f"  {len(unverified)} lane(s) with no receipt on this machine: "
              f"{', '.join(unverified)}")
    # A PARKED LANE IS NOT A GAP. Reported separately so the two reasons for
    # having no receipt do not read as one. #244.
    for l in lanes:
        if l.get("parked"):
            print(f"  {l['lane']} is parked: {l['parked']}; "
                  f"revisit when {l['parked_until']}")
    saturated = [l["lane"] for l in lanes if l.get("saturated")]
    if saturated:
        print(f"  {len(saturated)} lane(s) saturated: the incumbent passes "
              f">= 95% of holdout, so its cases cannot separate candidates: "
              f"{', '.join(saturated)}")
    for f in state.get("failed_switches") or []:
        print(f"  sohot-{f['lane']} did not switch to {f['new_spec']} after "
              f"{f['attempts']} attempt(s): {f['reason']}")
    if stale:
        print(f"  {len(stale)} lane(s) not measured within their re-verify "
              f"threshold: {', '.join(stale)}")
    for l in lanes:
        if l.get("flagged"):
            rv = l.get("reverify") or {}
            print(f"  re-verify flagged {l['lane']} ({l.get('serves', '')}): "
                  f"{rv.get('outcome', '')}: {rv.get('detail', '')}; the served "
                  f"model was not changed")
    return 0


def _report_export(a) -> int:
    """This machine's report as public JSON, written locally or published. #432."""
    from harness import publish
    try:
        if a.publish:
            print(f"wrote {publish.publish_here()} on the {publish.BRANCH} branch")
        else:
            print(publish.write_export(getattr(a, "out", "") or None))
    except publish.ExportRefused as exc:
        print(exc, file=sys.stderr)
        return 1
    except publish.GhError as exc:
        print(f"publish failed: {exc}", file=sys.stderr)
        return 1
    return 0


def _report_queue(a) -> int:
    """What a screen would teach us, best first. Issue #175.

    NOT A PREDICTION OF WHO WINS. The judge tier tried that and could not: two
    controls shaped like this data both failed to separate six models with known
    opposite outcomes, because the fact that separated them was produced by
    RUNNING them and a registry card has never held it.

    Arithmetic over what the store already knows, so it is deterministic and
    needs no gateway -- which is worth as much as the ordering, given the
    instrument it stands in for had to be run three times to be believed.
    """
    from harness import rank
    from harness import memory_store as ms

    want = (getattr(a, "lane", "") or "").strip().lower()
    store = ms.connect()
    try:
        rows, waiting = loop_cmd._queueable(store, want)
        retests = ms.retest_counts(store)
    finally:
        store.close()
    if not a.json:
        print(retest_line(retests))
    if not rows:
        print(f"nothing queued in the {want} lane that a screen has not answered"
              if want else "nothing queued that a screen has not answered")
        return 0
    # RANK THEM ALL, THEN TAKE THE TOP. rank() drops what no screen can answer
    # -- a laneless row, an adapter -- so the denominator has to come from
    # after that or it counts rows that will never be offered. The image lane
    # held 11 waiting of which 6 were LoRAs. Issue #209.
    ranked = rank.rank(rows, serving=rank.serving(),
                       measured_lanes=rank.lanes_with_receipts())
    waiting, ranked = len(ranked), ranked[:getattr(a, "top", 25)]
    if a.json:
        print(json.dumps({"queue": ranked, "waiting": waiting,
                          "retests": retests}, indent=2))
        return 0
    print(f"\n{len(ranked)} of {waiting} waiting, by what a screen would teach:")
    for r in ranked:
        print(f"\n  {r['value']:+6.1f}  {r['name']}")
        print(f"          {r['value_why'] or 'nothing known about it'}")
    return 0


def retest_line(c: dict) -> str:
    """The queue report's retest summary. #431."""
    return (f"retests: {c['due']} due, {c['pending']} scheduled, "
            f"{c['final']} final after 3; {c['recovered']} recovered false "
            f"negative(s)")


def _report_coverage(a) -> int:
    """What discovery never saw. Issue #99.

    Extraction precision measures the quality of what is CAUGHT. This measures
    reach: of the things this project actually adopted, which did a configured
    source ever surface. A source list is not a measurement of coverage.
    """
    from harness import coverage
    from harness import memory_store as ms

    store = ms.connect()
    try:
        got = coverage.report(store)
    finally:
        store.close()
    if a.json:
        print(json.dumps(got, indent=2))
        return 0
    print(f"\n{got['adopted']} things this machine runs or measured:")
    print(f"  {len(got['found']):3d} surfaced by a discovery source first")
    print(f"  {len(got['late']):3d} surfaced only after they were already run")
    print(f"  {len(got['holes']):3d} never surfaced by any source")
    if got["by_source"]:
        print("\n  credited with a find:")
        for source, n in got["by_source"].items():
            print(f"    {source:24} {n}")
    if got["proposed"]:
        # A source producing plenty that nobody has run is a different problem
        # from one producing nothing, and they want opposite fixes.
        print("\n  proposals produced (adopted or not):")
        for source, n in got["proposed"].items():
            print(f"    {source:24} {n}")
    holes = [h for h in got["holes"] if "why" not in h]
    ours = [h for h in got["holes"] if "why" in h]
    if ours:
        print(f"\n  {len(ours)} in the store only because one of our own tiers "
              f"put it there:")
        for h in ours[:12]:
            print(f"    {h['name']}")
    if holes:
        print(f"\n  {len(holes)} no configured source ever produced:")
        for h in holes[:20]:
            print(f"    {h['name']:44} {'/'.join(h['how'])}")
    if got["unlinked"]:
        print(f"\n  {len(got['unlinked'])} receipt key(s) on "
              f"{sum(got['unlinked'].values())} result rows name no candidate "
              f"and are not counted")
    return 0


def _report_winners(a) -> int:
    """What the receipts say won each lane, against what this file has typed in.

    The four DEFAULT_*_MODEL constants above are a hand copy of a measurement
    that lives in the stored runs. Both are worth having -- a default that
    moved because somebody ran an eval last night is a CLI two machines
    disagree about -- but a hand copy with nothing watching it is this
    project's most-bitten failure class.
    """
    from harness import adopt, winners

    from harness import memory_store as ms
    store = ms.connect()
    try:
        best = winners.beaten_in(store)
        rows = winners.disagreements(store)
        serves = adopt.lane_defaults(store)
    finally:
        store.close()
    if a.json:
        print(json.dumps({"typed": winners.typed(), "serves": serves,
                          "measured": best, "disagreements": rows}, indent=2))
        return 0
    #: exact agreement needs no mark; the other two each say which they are.
    MARK = {"exact": " ", "": "*"}
    print(f"\n  {'lane':9} {'serves':34} {'measured here':30} run")
    for lane, name in sorted(serves.items()):
        got = best.get(lane)
        if not got:
            print(f"  {lane:9} {name:34} {'-- not in any receipt':30}")
        else:
            mark = MARK.get(got["match"], "*")
            print(f" {mark}{lane:9} {name:34} "
                  f"{got['candidate'] + ' ' + str(got['pass_rate']):30} "
                  f"{got['run']}")
    print("\n  * beaten in a run it was in")
    beaten = [r for r in rows if r["state"] == "beaten"]
    unmeasured = [r for r in rows if r["state"] == "unmeasured"]
    if beaten:
        print(f"\n  {len(beaten)} default(s) lost a comparison they were in:")
        for r in beaten:
            print(f"    {r['modality']}: {r['measured']} beat {r['typed']} "
                  f"in {r['run']}")
    if unmeasured:
        # NOT a disagreement. A default that appears in no receipt was never in
        # the room, and reporting silence as conflict is how an inventory
        # becomes noise nobody reads.
        print(f"\n  {len(unmeasured)} default(s) appear in no receipt here, so "
              f"nothing on this machine can check them:")
        for r in unmeasured:
            print(f"    {r['modality']}: {r['typed']}")
    return 0
