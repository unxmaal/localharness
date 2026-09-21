"""Every lane driven through the fake ladder, and an inventory of the gaps.

Issue #264. Every defect #262 fixed was found in `image` or `code`, which are
the two lanes that have recorded fixtures. That is not a coincidence: a lane
nothing drives is not a lane known to work, it is a lane nobody has looked at.

This file asks ONE question of every lane -- can a candidate be spelled and
planned for it -- and reports the lanes that cannot by name. It starts as an
INVENTORY rather than a gate, because several lanes genuinely cannot, and each
converts to a gate the moment it lands.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))

from harness import lanes, screen  # noqa: E402

#: A candidate id per lane, shaped the way a sweep leaves one: a bare repo id
#: with an owner. Deliberately NOT a name any engine's table knows, because
#: that is what discovery produces and what the image lane could not handle.
DISCOVERED = "someorg/a-discovered-model"


@pytest.mark.parametrize("lane", lanes.ALL)
def test_every_lane_can_spell_a_discovered_candidate(lane):
    """A lane with no spelling cannot screen anything, ever, and says nothing.

    `video` had no entry here while its h3 engine, its diffusers-video engine
    and evals/cases/video/fox-run.yaml all existed. Being PARKED is a separate
    fact -- that one is about spend, is stated in lanes.PARKED, and is enforced
    in the fetch tier. A lane can be too expensive to run this month and still
    have to be nameable.
    """
    spec = screen.candidate_for(lane, DISCOVERED)
    assert spec, (
        f"the {lane} lane cannot spell a candidate, so nothing discovery finds "
        f"for it can ever be screened. screen.LANE_CANDIDATES has no entry.")


@pytest.mark.parametrize("lane", lanes.ALL)
def test_every_lane_has_a_runner_for_what_it_spells(lane):
    """Spelling it is not running it. RULE #286: the image lane spelled every
    candidate `mflux:` and mflux takes an enumerated set of families, so the
    spec was well-formed and no runner took it."""
    spec = screen.candidate_for(lane, DISCOVERED)
    gap = screen.no_runner(spec)
    assert not gap, (
        f"the {lane} lane spells a discovered candidate `{spec}` and no engine "
        f"of its can run it: {gap}")


@pytest.mark.parametrize("lane", lanes.ALL)
def test_every_lane_plans_a_discovered_candidate(lane):
    """The whole screen tier, not only the spelling. A lane that reaches
    NO_RUNNER here is one where the loop stops dead."""
    row = {"name": DISCOVERED, "lane": lane, "description": ""}
    got = screen.plan([row], missing=lambda n: [])[0]
    assert got["state"] != screen.NO_RUNNER, (
        f"the {lane} lane plans a discovered candidate as no-runner: "
        f"{got['why_not']}")


def test_a_lane_with_cases_has_somewhere_to_send_them():
    """The other direction. A lane with eval cases and no spelling can be
    MEASURED by hand and never REACHED by the loop, which is how a lane looks
    finished while the circuit through it is open."""
    root = Path(__file__).resolve().parents[1] / "evals" / "cases"
    with_cases = {p.name for p in root.iterdir()
                  if p.is_dir() and any(p.glob("*.yaml"))}
    unspellable = sorted(
        lane for lane in with_cases
        if not screen.candidate_for(lanes.canonical(lane), DISCOVERED))
    assert not unspellable, (
        f"these lanes have eval cases and no way to name a candidate, so the "
        f"loop cannot reach what a person can run by hand: {unspellable}")


def test_an_engine_refuses_a_model_it_would_ignore():
    """THE ONE THAT WOULD NOT HAVE FAILED LOUDLY.

    `h3:someorg/a-model` resolved to `h3/minimax-h3`: h3 reads its weights
    from $H3_MODEL_DIR and simply dropped the repo id. Spelling the video lane
    `h3:{model}` -- the obvious fix for its missing entry -- would have run
    MiniMax-H3 against every discovered candidate and written the result into
    a receipt under the candidate's name.

    The image lane's version of this failure produced an error. This one would
    have produced a plausible number, which is worse: a receipt for an
    experiment that never happened is not distinguishable from one that did.
    """
    from harness import engines

    with pytest.raises(ValueError, match="takes no model"):
        engines.resolve("h3:someorg/a-discovered-video-model")
    # The incumbent's own spelling still works, or the lane has no default.
    assert engines.resolve("h3").modality == "video"


def test_the_video_lane_is_parked_and_still_nameable():
    """Two different facts that were being carried by one missing entry.

    PARKED is about spend: 40.5 minutes for 22 frames on this machine, stated
    in lanes.PARKED and enforced in the fetch tier. UNSPELLABLE is about
    capability and says nothing at all. A lane can be too expensive to run
    this month and still have to be nameable, or unparking it silently yields
    a lane that cannot screen.
    """
    why, until = lanes.parked("video")
    assert why and until, "this test is about the parked lane; video is not one"
    assert screen.candidate_for("video", DISCOVERED)


# --- the inventory: which lanes the fake world can actually drive -----------

def _lane_of_fixture(name):
    from harness import inspect as ins
    import fakes
    return ins.lane_for(fakes.card(name)) or ""


def test_every_lane_is_covered_by_a_recorded_card_or_is_served_by_one():
    """THE ANSWER TO "have you mocked out each of the other lanes", as a
    command rather than a claim.

    Every defect #262 fixed was found in `image` or `code` -- the only two
    lanes with fixtures at the time. A lane nothing drives is not a lane known
    to work; it is a lane nobody has looked at.

    Two lanes cannot be covered this way and are named rather than faked.
    lane_for reads ONE lane off a card and a text model reads as `code`, so no
    `web` or `extract` card can exist. Inventing one would re-create RULE #269
    (one engine serving four lanes makes three look empty), which this project
    has already reported as a discovery hole once before measuring it.
    """
    import fakes

    covered = {_lane_of_fixture(n) for n in fakes.WHY}
    covered.discard("")
    expected = set(lanes.ALL) - set(fakes.SERVED_NOT_SIGHTED)
    missing = sorted(expected - covered)
    assert not missing, (
        f"no recorded registry card lands in these lanes, so nothing in the "
        f"fake world exercises what the ladder does with one: {missing}. "
        f"Record one with scripts/record_fixtures.py and say in fakes.WHY "
        f"what it proves.")


@pytest.mark.parametrize("lane", ["web", "extract"])
def test_a_served_lane_is_reachable_even_though_no_card_names_it(lane):
    """The other half of the pair above. These are covered by being spellable
    and runnable, not by a sighting -- so assert that, rather than letting
    them sit in an exception list nobody re-checks."""
    spec = screen.candidate_for(lane, DISCOVERED)
    assert spec == DISCOVERED, (
        f"{lane} is served by a text model through mlx_lm.server, which takes "
        f"the request's model as a live repo id, so its spec carries no "
        f"prefix: got {spec!r}")
    assert not screen.no_runner(spec)
