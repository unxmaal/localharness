"""Does each lane actually work? Run its OWN default and find out.

Asked 2026-09-19: whether any of this works is not known, because most lanes
have never been run here.

The status page says the same: three lanes appear in no receipt on
this machine at all and four more were last measured twelve days ago. A lane
default is a claim until something runs it here.

THIS IS NOT THE SCREEN AND NOT THE MEASURE TIER. Both of those exist to judge
a CHALLENGER. This runs the INCUMBENT alone, to establish that the lane works
at all and to give every later comparison a control that is known to pass.
"""
from __future__ import annotations

from dataclasses import dataclass, field

#: Roughly what one full pass of a lane costs here, in seconds, so a plan can
#: be read before it is spent. MEASURED ON THIS MACHINE and therefore an
#: estimate everywhere else; the point is the ORDER OF MAGNITUDE, which is what
#: decides whether a lane belongs in an unattended pass.
COST_S = {
    "extract": 20, "code": 60, "web": 120, "svg": 30,
    "tts": 30, "stt": 120, "image": 200, "video": 2700,
    # 55s of model init then ~37s per 30s case, three cases. Measured
    # 2026-09-20 on the M2 Pro, turbo at 8 steps.
    "music": 180,
}

#: A lane nobody should start without meaning to. Video is ~40 minutes for ONE
#: case and its engine carries a six-hour timeout; including it in a default
#: pass is how a machine ends up unusable overnight.
DELIBERATE = ("video",)


@dataclass
class Task:
    lane: str
    candidate: str
    argv: list[str] = field(default_factory=list)
    cost_s: int = 0
    why: str = ""
    skip: str = ""


def plan(lanes_state, *, only: str = "", include_stale: bool = True,
         force: bool = False) -> list[Task]:
    """What to run, cheapest first.

    Cheapest first because the point is to find out WHETHER the lanes work,
    and a cheap lane that fails tells you that sooner. A pass that opens with
    forty minutes of video learns nothing for forty minutes.
    """
    from harness import lanes as L
    from harness import screen

    want = L.canonical(only)
    out = []
    for lane in lanes_state:
        name = lane["lane"]
        if want and name != want:
            continue
        if not (lane["unverified"] or (include_stale and lane["stale"])
                or force):
            continue
        default = lane.get("serves") or ""
        task = Task(lane=name, candidate=default,
                    cost_s=COST_S.get(name, 60),
                    why=("no receipt on this machine" if lane["unverified"]
                         else f"last measured {lane['age_days']:.0f} days ago"))
        if not default:
            task.skip = "the lane names no default to run"
        elif name in DELIBERATE and not want:
            # Nameable, never automatic: --lane video opts in explicitly.
            task.skip = (f"~{COST_S.get(name, 0) // 60} minutes for one case; "
                         f"ask for it by name with --lane {name}")
        else:
            spec = screen.candidate_for(name, default) or default
            task.argv = ["uv", "run", "python", "-m", "evals.run",
                         "--modality", name, "--candidates", spec]
            route = screen.routed_gateway(default)
            if route:
                task.argv += ["--gateway", route]
        out.append(task)
    return sorted(out, key=lambda t: (bool(t.skip), t.cost_s))


def summarise(lane: str, data: dict) -> str:
    """One line per lane from its receipt, or why there is nothing to say."""
    summary = (data or {}).get("summary") or {}
    if not summary:
        return "the run wrote no summary"
    bits = []
    for key, got in sorted(summary.items()):
        passed, total = got.get("passed", 0), got.get("total", 0)
        metric = " ".join(f"{k} {v:.3f}" for k, v in
                          sorted((got.get("metrics") or {}).items()))
        med = got.get("median_s")
        bits.append(f"{key} {passed}/{total}"
                    + (f" median {med:.2f}s" if med else "")
                    + (f" {metric}" if metric else ""))
    return "; ".join(bits)


def verdict(data: dict) -> str:
    """works / partial / broken, from the receipt alone.

    A lane whose own default passes NOTHING is broken here whatever the reason,
    and that is the finding the pass exists to produce. Partial is kept
    separate because a lane can legitimately fail a hard case: `web` has cases
    no small model passes, and calling that broken would be wrong.
    """
    summary = (data or {}).get("summary") or {}
    if not summary:
        return "broken"
    passed = sum(g.get("passed", 0) for g in summary.values())
    total = sum(g.get("total", 0) for g in summary.values())
    if not total:
        return "broken"
    if passed == 0:
        return "broken"
    return "works" if passed == total else "partial"
