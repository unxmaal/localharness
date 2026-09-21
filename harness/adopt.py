"""The loop's closing step: a measured winner becomes the lane's default.

Without this the loop cannot change anything. A candidate can be swept,
inspected, ranked, fetched, screened, measured, beat the incumbent on the
lane's own metric, and the lane goes on serving the constant somebody typed
into a source file. `lh discover --winners` reported that disagreement and
stopped, which makes the gap visible and closes none of it.

WHAT ADOPTION IS NOT. It is not "the challenger scored higher". A run is a
sample, and this project has twice drawn a conclusion from a difference that
was not there: the crowd decay called inert on statistics blind to ordering,
and a temperature trend of 12/27 to 9/27 that is 5 lost against 2 gained,
p=0.45. So a challenger must beat the incumbent on the LANE'S OWN METRIC and
the cell-by-cell comparison must be significant. Anything less is recorded as
a loss, which is the more useful row: it stops the same candidate being
re-proposed every sweep.

THE ADOPTED WINNER IS A STORE FACT, not an edit to a source file. A loop that
rewrites constants would need a commit to take effect and could not be undone
without another one. The lane reads the record and falls back to the typed
constant when nothing has been adopted, so a fresh checkout behaves exactly as
it does today.
"""
from __future__ import annotations

from dataclasses import dataclass

#: The tier that records an adoption. See memory_store.TIERS.
TIER = "adopt"

#: Below this the difference is not established and the incumbent stays. The
#: exact McNemar test on discordant cells; see harness/paired.py.
ALPHA = 0.05


@dataclass(frozen=True)
class Verdict:
    lane: str
    incumbent: str
    challenger: str
    adopt: bool
    why: str


def better(incumbent: dict, challenger: dict) -> bool:
    """Is the challenger's summary row better on the lane's own metric?

    Delegates to winners.order_key so there is one ordering rule in the
    project rather than two that can disagree. RULE #254.
    """
    from harness import winners

    return winners.order_key(challenger) > winners.order_key(incumbent)


def decide(lane: str, incumbent: dict, challenger: dict,
           rows: list[dict] | None = None) -> Verdict:
    """Whether this challenger replaces this incumbent.

    Two gates, both required. The metric gate answers "better on what this lane
    is about". The paired gate answers "by more than the noise". A challenger
    that passes one and not the other is a loss.
    """
    from harness import paired

    name_i = incumbent.get("candidate", "")
    name_c = challenger.get("candidate", "")
    if not better(incumbent, challenger):
        return Verdict(lane, name_i, name_c, False,
                       "does not beat the incumbent on the lane's metric")
    if not rows:
        return Verdict(lane, name_i, name_c, False,
                       "no paired rows, so the difference is unmeasured")
    cell = paired.head_to_head(rows, name_i, name_c)
    if not cell.discordant and not cell.unchanged:
        return Verdict(lane, name_i, name_c, False,
                       f"{name_i} and {name_c} share no case in this run, so "
                       f"there is nothing to compare")
    if cell.p > ALPHA:
        return Verdict(lane, name_i, name_c, False,
                       f"better on the metric, but {cell.lost} lost against "
                       f"{cell.gained} gained is p={cell.p:.2f}, so the "
                       f"difference is not established")
    return Verdict(lane, name_i, name_c, True,
                   f"beats {name_i} on the lane's metric, {cell.gained} gained "
                   f"against {cell.lost} lost, p={cell.p:.2f}")


def record(conn, verdict: Verdict) -> None:
    """Write an adoption, or the loss, so neither is rediscovered."""
    from harness import memory_store as ms

    outcome = "measured" if verdict.adopt else "declined"
    detail = f"{verdict.lane}: {verdict.why}"
    try:
        ms.decide(conn, verdict.challenger, outcome, tier=TIER,
                  detail=detail[:200])
        return
    except KeyError:
        pass

    # NO PROPOSAL ROW, AND THE DECISION STILL HAS TO SURVIVE. This used to
    # swallow the KeyError with a note saying the decision had happened
    # anyway. It had not: `adopted()` reads verdict rows, so an adoption whose
    # winner arrived any way other than through a sweep was discarded on the
    # spot and the lane went on serving the incumbent. A candidate named on a
    # command line, or reached through `lh verify`, could beat the incumbent on
    # the metric AND the paired test and change nothing.
    #
    # AN ADOPTION IS A FACT ABOUT THE LANE, not about a proposal, so the row
    # exists to carry it rather than the other way round. `source` says where
    # it came from, so a later sweep seeing the same name adds a sighting to
    # this row instead of starting a second one.
    ms.record(conn, ms.Seen(name=verdict.challenger, source=TIER, url="",
                            why=f"named in a {verdict.lane} comparison",
                            lane=verdict.lane, resolved=verdict.challenger))
    ms.decide(conn, verdict.challenger, outcome, tier=TIER,
              detail=detail[:200])


def adopted(conn) -> dict[str, str]:
    """The lane -> candidate the loop has adopted, newest per lane."""
    rows = conn.execute(
        "SELECT p.name, v.detail, v.id FROM verdicts v "
        "JOIN proposals p ON p.id = v.proposal_id "
        "WHERE v.tier = ? AND v.outcome = 'measured' ORDER BY v.id",
        (TIER,)).fetchall()
    out = {}
    for row in rows:
        lane = str(row["detail"]).split(":", 1)[0].strip()
        if lane:
            out[lane] = row["name"]
    return out


def default_for(lane: str, fallback: str, conn=None) -> str:
    """What this lane serves right now: the adopted winner, else the constant.

    RESOLVED AT CALL TIME. `DEFAULT_TTS_MODEL` reaches its callers as a default
    ARGUMENT, captured when the function is defined, so rebinding the constant
    changes nothing -- the same trap that made SpeechRunner's declared voice
    unreachable (#195). An adoption that cannot take effect without a restart
    is a report with extra steps.

    Never raises and never blocks a generation. A store that is missing,
    locked or older than this code leaves the lane on its typed constant, which
    is what a fresh checkout does anyway.
    """
    if not lane:
        return fallback
    close = conn is None
    try:
        if conn is None:
            from harness import memory_store as ms
            conn = ms.connect()
        got = adopted(conn).get(lane.strip().lower())
    except Exception:  # noqa: BLE001
        return fallback
    finally:
        if close and conn is not None:
            try:
                conn.close()
            except Exception:  # noqa: BLE001
                pass
    return got or fallback
