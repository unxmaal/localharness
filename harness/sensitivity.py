"""Does this constant do anything?

Fourteen numbers in this project decide what gets proposed, screened and
fetched, and nobody knows whether any of them matters. That is issue #98, and
the question is not "what is the best value". It is the cheaper and more useful
one: is the output sensitive to this at all? Three answers, all worth having.

    inert   nothing downstream moves -> delete it, or find the case it should
            have caught
    band    it moves at the extremes but not near the default -> record the
            margin so the next person knows how much room there is
    edge    the value next door changes the answer -> the default is one commit
            away from behaving differently. For a THRESHOLD that is the
            constant doing its job and not a defect; for a tuning constant it
            is a warning. The verdict describes; it does not judge.

A distance is not a quality. "The ranking moved" says nothing about whether it
moved somewhere better, so a probe may carry a `quality` callable -- for the
crowd ranking that is the existing two-directional control -- and its answer is
printed beside the distance. Sensitivity without it says only that a knob is
connected to something.

WHAT THIS MODULE EXISTS TO PREVENT is a false `inert`, and there are two ways
to get one.

The first is a blind statistic. HALF_LIFE_DAYS was swept across 90/365/1095/off
and reported inert on the strength of three numbers -- how many known-good
repos landed in the top ten, how many popular ones leaked in, and the name at
position one. Every one of those is a count over a SET, and a count over a set
cannot see an ordering by construction, which is exactly what was being tested.
Ten of the top fifteen positions had in fact changed. So every distance here is
zero if and only if the two outputs are identical, and rankings are compared
position by position rather than as membership.

The second is a patch that does not take. A module constant used as a DEFAULT
ARGUMENT is captured when the function is defined, so rebinding the constant
later changes nothing at all -- silently. `bound()` is the fix, and both
POPULATION and HALF_LIFE_DAYS reach their callers that way.
"""
from __future__ import annotations

import contextlib
import functools
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

#: A reading that could not be taken. Kept distinct from a distance of zero,
#: which is a measurement saying "nothing moved" rather than the absence of one.
FAILED = -1


@dataclass(frozen=True)
class Reading:
    value: Any
    distance: int
    detail: str = ""
    error: str = ""
    #: What a probe's own check says about the output at this value, if it has
    #: one. Free text: "separates", "FAILS CONTROL", "3 lanes".
    quality: str = ""

    @property
    def taken(self) -> bool:
        return self.distance != FAILED


@dataclass(frozen=True)
class Finding:
    name: str
    default: Any
    verdict: str
    readings: tuple[Reading, ...] = ()
    #: The contiguous run of values, around the default, that all produce the
    #: baseline output. The margin, stated.
    band: tuple = ()
    note: str = ""


@contextlib.contextmanager
def bound(module, name: str, **pinned):
    """Pin arguments of a module-level function for the duration.

    Rebinding the CONSTANT does not work: `def f(x, population=POPULATION)`
    captures the value at definition time, so a sweep that sets
    `neighbors.POPULATION = 1e7` and calls a function that never reads it again
    measures nothing and reports inert.
    """
    original = getattr(module, name)
    setattr(module, name, functools.partial(original, **pinned))
    try:
        yield
    finally:
        setattr(module, name, original)


def kendall(base: Sequence, other: Sequence) -> float | None:
    """Rank correlation over the items both orderings contain, or None when
    fewer than two are shared."""
    present = set(other)
    common = [x for x in base if x in present]
    if len(common) < 2:
        return None
    at = {x: i for i, x in enumerate(other)}
    agree = disagree = 0
    for a, b in itertools.combinations(common, 2):
        if at[a] < at[b]:
            agree += 1
        else:
            disagree += 1
    total = agree + disagree
    return (agree - disagree) / total if total else None


def rank_distance(base: Sequence, other: Sequence) -> tuple[int, str]:
    """Positions whose occupant differs. Zero only for an identical ordering.

    Reported alongside set membership and rank correlation because the three
    are blind to different things: membership cannot see a reordering, and a
    correlation over shared items cannot see an item that left entirely.
    """
    base, other = list(base), list(other)
    width = max(len(base), len(other))
    moved = sum(1 for i in range(width)
                if (base[i] if i < len(base) else None)
                != (other[i] if i < len(other) else None))
    entered = [x for x in other if x not in set(base)]
    left = [x for x in base if x not in set(other)]
    tau = kendall(base, other)
    bits = [f"{moved}/{width} positions"]
    if entered or left:
        bits.append(f"+{len(entered)} -{len(left)}")
    if tau is not None:
        bits.append(f"tau {tau:+.2f}")
    if left:
        bits.append("out: " + ", ".join(str(x) for x in left[:3]))
    return moved, "  ".join(bits)


def map_distance(base: Mapping, other: Mapping) -> tuple[int, str]:
    """Keys whose value differs. Zero only for an identical mapping."""
    keys = sorted(set(base) | set(other))
    changed = [k for k in keys if base.get(k) != other.get(k)]
    detail = f"{len(changed)}/{len(keys)} changed"
    if changed:
        shown = ", ".join(f"{k}: {base.get(k)}->{other.get(k)}"
                          for k in changed[:3])
        detail += "  " + shown
    return len(changed), detail


def distance(base, other) -> tuple[int, str]:
    if isinstance(base, Mapping) or isinstance(other, Mapping):
        return map_distance(base or {}, other or {})
    if isinstance(base, (int, float, str, bool)) or base is None:
        return (0 if base == other else 1), f"{base!r} -> {other!r}"
    return rank_distance(base, other)


def verdict_for(default, values: Sequence, readings: Sequence[Reading]) -> tuple[str, tuple]:
    """inert, band or edge, plus the contiguous run of values that match.

    The default must be one of the values swept. Without it there is no way to
    ask the question this module is for -- whether the value NEXT DOOR to the
    one in use behaves differently -- and a sweep that cannot ask it would
    report a band whose relationship to the default is unknown.
    """
    taken = [r for r in readings if r.taken]
    if not taken:
        return "unmeasurable", ()
    if all(r.distance == 0 for r in taken):
        return "inert", tuple(r.value for r in taken)

    by_value = {r.value: r for r in readings}
    order = [v for v in values if v in by_value and by_value[v].taken]
    if default not in order:
        raise ValueError(
            f"the default {default!r} is not among the values swept "
            f"({', '.join(repr(v) for v in order)}); without it a band cannot "
            f"be placed relative to the value actually in use")
    i = order.index(default)
    lo = hi = i
    while lo > 0 and by_value[order[lo - 1]].distance == 0:
        lo -= 1
    while hi < len(order) - 1 and by_value[order[hi + 1]].distance == 0:
        hi += 1
    band = tuple(order[lo:hi + 1])
    return ("edge" if len(band) == 1 else "band"), band


def measure(name: str, default, values: Sequence,
            run: Callable[[Any], Any], *, note: str = "",
            quality: Callable[[Any], str] | None = None) -> Finding:
    """Run `run` at each value and say whether the output moved.

    `run(default)` is the baseline and is measured like any other value, so a
    run that is not deterministic shows up as a nonzero distance against
    itself rather than as a silently shifted baseline.
    """
    try:
        base = run(default)
    except Exception as exc:  # noqa: BLE001 - a probe that cannot start is a reading
        return Finding(name, default, "unmeasurable",
                       (Reading(default, FAILED, error=str(exc)[:160]),),
                       note=note)
    readings = []
    for value in values:
        try:
            out = run(value)
        except Exception as exc:  # noqa: BLE001
            readings.append(Reading(value, FAILED, error=str(exc)[:160]))
            continue
        d, detail = distance(base, out)
        mark = ""
        if quality is not None:
            try:
                mark = quality(value)
            except Exception as exc:  # noqa: BLE001 - a check is not the measurement
                mark = f"check failed: {type(exc).__name__}"
        readings.append(Reading(value, d, detail, quality=mark))
    verdict, band = verdict_for(default, values, readings)
    return Finding(name, default, verdict, tuple(readings), band, note)


def report(findings: Sequence[Finding]) -> str:
    lines = []
    for f in findings:
        head = f"{f.name} = {f.default!r}   {f.verdict.upper()}"
        if f.band and f.verdict == "band":
            head += f"   band {f.band[0]!r}..{f.band[-1]!r}"
            # A band the default sits at the END of is one-sided: there is
            # room in one direction and none in the other, which is a
            # different thing to know than a margin on both sides.
            if f.default == f.band[0]:
                head += "  (default at the LOW edge)"
            elif f.default == f.band[-1]:
                head += "  (default at the HIGH edge)"
        lines.append(head)
        if f.note:
            lines.append(f"    {f.note}")
        for r in f.readings:
            if not r.taken:
                lines.append(f"    {r.value!r:>12}  --  {r.error}")
                continue
            body = "same" if r.distance == 0 else r.detail
            mark = f"   [{r.quality}]" if r.quality else ""
            lines.append(f"    {r.value!r:>12}  {body}{mark}")
        lines.append("")
    return "\n".join(lines)
