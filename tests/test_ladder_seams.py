"""The ladder's seams, driven end to end against a fake world. Issue #257.

Every defect found on 2026-09-20/21 lived BETWEEN two tiers that each worked
alone. 111 test files stayed green through all of them, and each one was found
instead by running the real loop: real downloads, a live gateway, a live model
server, and hours.

These run in under a second and would have caught them. Each test names the
issue it reproduces, so a failure says which defect came back rather than
leaving somebody to work it out.

The registry cards are RECORDED (tests/fixtures/registry, refreshed by
scripts/record_fixtures.py). An invented card would encode what I expected a
registry to say, and three of these defects exist because the real one said
something else.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakes  # noqa: E402

from harness import fetching, inspect as ins, lanes, rank, screen  # noqa: E402
from harness import machine as _machine  # noqa: E402
from harness import memory_store as ms  # noqa: E402
from harness.memory import Accelerator  # noqa: E402


def mac():
    """Pinned, so a verdict does not vary with whichever runner asks.
    RULE #249: an MLX-native card is `fits` on a Mac and `needs-mlx` elsewhere.
    """
    return _machine.Machine(frozenset({"mlx", "cpu"}),
                            Accelerator("unified", 32.0, 22.0, "Mac14,12"))


@pytest.fixture
def store(tmp_path):
    conn = ms.connect(tmp_path / "s.db")
    yield conn
    conn.close()


# --- the card reaches the decision that reads it -------------------------

def test_an_adapter_is_caught_through_the_real_card(store):
    """#241. `lora` is the seventh tag on this card and card_description kept
    six, so the word that decides whether a runner can load it never reached
    screen.is_attachment. It ranked TOP of the queue at +5.3."""
    desc = ins.card_description(fakes.card(
        "Xanthius/Ace-Step-1.5-XL-Concept-Sliders"))
    assert screen.is_attachment(desc), desc


def test_a_supertype_pipeline_tag_yields_to_the_specific_one():
    """#246. Declares text-generation, tagged Text-to-SVG three times, and was
    filed under `code` where the lane hands it to mlx_lm.server."""
    assert ins.lane_for(fakes.card("OmniSVG/OmniSVG1.1_8B")) == "svg"


def test_a_foreign_runtime_is_refused_before_a_download():
    """#245. Tagged gemlite and cuda. It ranked, and would have been fetched
    and handed to a runner that cannot load it."""
    desc = ins.card_description(fakes.card(
        "prism-ml/bonsai-image-binary-4B-gemlite-1bit"))
    row = {"name": "prism-ml/bonsai-image-binary-4B-gemlite-1bit",
           "lane": "image", "description": desc}
    assert rank.unrunnable(row, mac()) == "needs-cuda"


def test_an_ordinary_card_survives_all_three():
    """THE NEGATIVE CONTROL, and the half that decides whether any of this can
    ship. Three filters that fire on an ordinary model empty the queue and
    read exactly like a queue that ran out."""
    c = fakes.card("openbmb/MiniCPM5-1B")
    desc = ins.card_description(c)
    assert not screen.is_attachment(desc), desc
    assert ins.lane_for(c) == "code"
    assert not rank.unrunnable(
        {"name": "openbmb/MiniCPM5-1B", "lane": "code", "description": desc},
        mac())


# --- one tier's output is the next tier's input ---------------------------

def test_the_fetch_tier_reads_the_queue_in_rank_order(store):
    """#249 and RULE #275. Two tiers ordering one queue by different keys
    means the producer sizes rows the consumer never reaches, and both halves
    report success."""
    fakes.seeded_store(store, [
        ("org/tiny", "code", 0.1, 2),
        ("org/big", "code", 11.5, 2),
        ("org/parked", "video", 16.0, 9),
    ])
    rows = fetching.queued(store, needs_lane=False)
    ranked = [r["name"] for r in fetching.in_rank_order(rows, store)]
    assert ranked[-1] == "org/parked", (
        f"a parked lane's candidate must rank last, got {ranked}")


def test_a_parked_lane_never_reaches_a_download(store, tmp_path):
    """#249. Four of the top twelve were video candidates for a lane nothing
    will run, and --top 4 would have downloaded 16.2 GiB for it."""
    fakes.seeded_store(store, [("org/parked", "video", 16.0, 9)])
    downloads = fakes.Downloads(tmp_path / "hub")
    ranked = fetching.in_rank_order(
        fetching.queued(store, needs_lane=False), store)
    for row in ranked:
        if rank.unrunnable(row, mac()) or lanes.parked(row.get("lane"))[0]:
            continue
        downloads(repo_id=row["name"])
    assert downloads.asked == [], downloads.asked


def test_an_adapter_never_reaches_a_download(store, tmp_path):
    """#249 again, one tier along: rank drops attachments, but rank is an
    ORDERING and fetch is a SPEND."""
    name = "Xanthius/Ace-Step-1.5-XL-Concept-Sliders"
    desc = ins.card_description(fakes.card(name))
    fakes.seeded_store(store, [(name, "music", 0.7, 2)])
    store.execute("UPDATE proposals SET description = ? WHERE name = ?",
                  (desc, name))
    downloads = fakes.Downloads(tmp_path / "hub")
    got = fetching.run(store, {name: int(0.7 * 1024 ** 3)}, limit=1,
                       snapshot=downloads)
    assert downloads.asked == [], "an adapter was downloaded"
    assert got and "attaches to a model" in got[0]["why"]


def test_a_terminal_verdict_is_not_handed_back_to_the_screen(store):
    """#253, now FIXED. GPT-X2.5-135M was screened `broken` twice and was
    ready for a third; LFM2.5-350M was measured, lost, recorded `declined`,
    and measured again an hour later against the same incumbent, which is
    three of the four adopt verdicts in the real store.

    The mark on this test XPASSed the moment ms.judgeable learned to exclude
    a candidate whose LATEST verdict is terminal, which is what strict=True is
    for: an inventory that becomes a gate says so rather than rotting."""
    from harness.cli import _queueable

    fakes.seeded_store(store, [("org/answered", "code", 0.5, 2),
                               ("org/fresh", "code", 0.5, 2)])
    ms.decide(store, "org/answered", "broken", tier="screen",
              detail="it ran and passed nothing || generated 0 of 3")
    names = [r["name"] for r in _queueable(store, "code")[0]]
    assert "org/fresh" in names, "the unanswered candidate went missing"
    assert "org/answered" not in names, (
        f"a candidate screened `broken` is still queueable, so the loop will "
        f"screen it again: {names}")


# --- the fakes must not agree with themselves -----------------------------

def test_an_unrecorded_card_raises_rather_than_answering_empty():
    """A fake that returns an empty card for anything it does not know makes
    every test pass while measuring nothing."""
    with pytest.raises(KeyError, match="[Rr]ecord one"):
        fakes.card("org/never-recorded")


def test_every_recorded_card_says_why_it_is_kept():
    """A fixture directory nobody prunes becomes a directory nobody reads."""
    recorded = {p.stem.replace("_", "/", 1)
                for p in fakes.FIXTURES.glob("*.json")}
    assert recorded == set(fakes.WHY), (
        f"unexplained: {sorted(recorded - set(fakes.WHY))}; "
        f"explained but absent: {sorted(set(fakes.WHY) - recorded)}")
