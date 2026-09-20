"""Can discovery reach every lane? Issue #240.

Three lanes had no source at all -- video, music and stt -- and the gap was
invisible because nothing anywhere asked the question. The answer was not hard
to work out; it was simply not askable, which is the same shape as the four
disagreeing lane vocabularies in #207.

A GATE, NOT AN INVENTORY. Every gap it names was closed in the same change, so
it starts green, and holding zero costs nothing.
"""
import pytest

from harness import feeds, lanes


def test_every_lane_can_be_reached_by_some_source_family():
    """A lane no family can reach has a permanently empty queue, and reads to
    anyone looking at it as a lane nobody publishes models for.

    ACROSS FAMILIES. Asking only about feeds would report `music` as a hole:
    it has no usable feed and five registry queries, and sending somebody to
    find a feed for it would be work with no answer at the end.
    """
    from harness import discover
    missing = discover.unreachable_lanes(feeds.DEFAULT_SOURCES, machine=_mac())
    assert not missing, (
        f"{sorted(missing)} can never fill a queue: no feed and no registry "
        f"query reaches them. Add a source, or argue the lane out of lanes.ALL")


def test_music_is_reached_by_the_registry_and_not_by_a_feed():
    """Pins the shape that made the cross-family question necessary.

    ACE-Step publishes a releases feed that parses and carries ten entries, so
    every health check here would pass it. It still cannot be used: a
    `releases` source reports DRIFT, and ACE-Step tags v0.1.8 while its
    package is 1.5.0, so `behind()` is False forever and the source would
    propose nothing while looking fine.
    """
    from harness import discover
    how = discover.lane_reach(feeds.DEFAULT_SOURCES, machine=_mac())["music"]
    assert how["feeds"] == [], (
        "a music feed was added; check it can actually report drift before "
        "trusting it, and delete this test if it can")
    assert how["registry"], "nothing at all would find a music model"


def _mac():
    """Pin the machine, so reach does not vary with whichever runner asks."""
    from harness import machine
    from harness.memory import Accelerator
    return machine.Machine(frozenset({"mlx", "cpu"}),
                           Accelerator("unified", 32.0, 25.0, "Mac14,12"))


def test_coverage_counts_a_text_source_for_all_four_text_lanes():
    """THE ACCOUNTING THAT MAKES THE GATE HONEST.

    One text model serves code, web, svg and extract. Counting by the filed
    lane alone reports web and svg as unreachable and sends somebody hunting
    for feeds about vector graphics that do not exist -- which is exactly the
    wrong conclusion #207 had to retract, where web and svg read as having
    zero candidates while holding 103.
    """
    text_source = feeds.Source("x", "http://example.invalid/f", lane="code")
    reached = feeds.source_lanes(text_source)
    for lane in lanes.TEXT_SERVED:
        assert lane in reached, f"a text source must reach {lane}"


def test_an_image_source_does_not_count_for_the_text_lanes():
    """THE NEGATIVE CONTROL. A gate that counts every source for every lane
    passes forever and measures nothing. mflux does not write markup."""
    image_source = feeds.Source("x", "http://example.invalid/f", lane="image")
    reached = feeds.source_lanes(image_source)
    assert reached == ("image",)
    for lane in lanes.TEXT_SERVED + ("video", "music", "tts", "stt"):
        assert lane not in reached


def test_a_source_claiming_no_lane_covers_nothing_rather_than_everything():
    """`all` is a source saying it makes no claim. Counting it as coverage
    would make the gate pass for every lane forever -- and recording it as a
    candidate's lane is the defect that put 243 of 323 proposals in a lane
    that does not exist (#207)."""
    assert feeds.source_lanes(
        feeds.Source("x", "http://example.invalid/f", lane="all")) == ()
    covered = feeds.reach([
        feeds.Source("catch-all", "http://example.invalid/f", lane="all")])
    assert all(not hits for hits in covered.values())


def test_a_source_can_reach_more_lanes_than_it_is_filed_under():
    """mlx-audio is filed `tts` and Kokoro speaks through it while parakeet
    listens through it. `stt` read as a lane with no source while its only
    source sat right there."""
    by_name = {s.name: s for s in feeds.DEFAULT_SOURCES}
    reached = feeds.source_lanes(by_name["mlx-audio-releases"])
    assert "tts" in reached and "stt" in reached


def test_the_recorded_lane_stays_a_single_lane():
    """`lane` is ALSO recorded as the candidate's lane at four sites in
    discover.py, so it must never become a list. That is why `reaches` is a
    separate field rather than a comma-separated spelling of this one: a
    source's coverage claim is not a candidate's lane."""
    for source in feeds.DEFAULT_SOURCES:
        assert "," not in source.lane, (
            f"{source.name} puts several lanes in `lane`; that string is "
            f"recorded as a proposal's lane. Use `reaches` instead")
        assert lanes.known(source.lane) or lanes.canonical(source.lane) == "", (
            f"{source.name} is filed under {source.lane!r}, which is not a lane")


def test_a_disabled_source_does_not_count_as_coverage():
    """Switching a source off is how a lane quietly loses its only reach."""
    off = feeds.Source("x", "http://example.invalid/f", lane="music",
                       enabled=False)
    assert feeds.reach([off])["music"] == []


@pytest.mark.parametrize("name,url", [
    ("diffusers-releases", "https://github.com/huggingface/diffusers/releases.atom"),
])
def test_the_new_sources_are_spelled_the_way_they_were_probed(name, url):
    """Fetched and parsed before being added -- 10 entries -- where two of the
    six candidates tried that day served no feed at all. This
    pins the URL that was actually verified, so an edit to it is deliberate
    rather than a typo nobody notices until a sweep returns nothing.

    It does NOT reach the network: a test that fails when GitHub has a bad
    afternoon teaches people to skip tests.
    """
    by_name = {s.name: s for s in feeds.DEFAULT_SOURCES}
    assert by_name[name].url == url
