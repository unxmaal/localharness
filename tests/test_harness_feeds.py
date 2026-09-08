"""Issue #33: community aggregation feeds as a discovery source.

Fixtures are real reddit responses captured 2026-09-07, so a layout change
fails here loudly instead of silently returning nothing.
"""
import json
import time
from pathlib import Path

import pytest

from harness import discover, feeds
from harness.feeds import FeedError, Source

FIX = Path(__file__).parent / "fixtures" / "feeds"
WEEK = FIX / "reddit-top-week.xml"
RECAP = FIX / "reddit-recap-search.xml"


@pytest.fixture
def week():
    return feeds.parse(WEEK.read_text())


@pytest.fixture
def recap():
    return feeds.parse(RECAP.read_text())


def test_the_week_feed_parses(week):
    assert len(week) == 25
    assert all(e.title and e.link for e in week)


def test_the_recap_body_arrives_whole(recap):
    """The reason this is one request and not two: the entire post is in the
    Atom <content>, so a recap needs no second fetch and no HTML scrape."""
    assert len(recap) >= 4
    assert len(recap[0].body) > 5000
    assert "MiniMax" in recap[0].body


def test_a_block_page_is_an_error_not_an_empty_feed():
    """Reddit's block page arrives looking like a normal response. Parsing is
    the check that catches it; the status code is not."""
    with pytest.raises(FeedError) as exc:
        feeds.parse("<!doctype html><html><head><title>Blocked</title>")
    assert "not a feed" in str(exc.value)


def test_linked_repos_are_extracted_as_ids(week):
    """A linked repo is a real id. It beats a name lifted from prose, and it
    needs no registry round-trip to be worth offering."""
    repos = {p.name for p in feeds.candidates(week) if p.kind == "repo"}
    assert "inclusionAI/LLaDA-Image" in repos
    assert "KennethFal/vh5tape-vhs-lora-minimax-h3" in repos


def test_the_recap_yields_named_candidates_with_descriptions(recap):
    got = {p.name: p.why for p in feeds.candidates(recap[:1])}
    assert "MiniMax-H3-Turbo-Lora" in got
    assert "5x" in got["MiniMax-H3-Turbo-Lora"]
    assert len(got) > 50


def test_titles_with_spaced_version_names_are_caught(week):
    """A hyphen-only pattern found ONE candidate in twenty-five posts, because
    titles write names with spaces: "Krea 2", "Wan 2.2"."""
    names = {p.name for p in feeds.candidates(week)}
    assert "Krea 2" in names


def test_unknown_hosts_are_proposed_as_sources(week):
    hosts = {p.name for p in feeds.candidate_sources(week)}
    assert "civitai.red" in hosts
    # Already read directly, so pointing at them is not a new source.
    assert not {h for h in hosts if "huggingface" in h or "reddit" in h}


def test_media_hosts_are_not_proposed_as_sources(week):
    hosts = {p.name for p in feeds.candidate_sources(week)}
    assert not {h for h in hosts if "imgur" in h or "redd.it" in h}


def test_a_known_source_is_not_proposed_back(week):
    known = [Source("x", "https://civitai.red/models/1")]
    hosts = {p.name for p in feeds.candidate_sources(week, known)}
    assert "civitai.red" not in hosts


# ---- schedule -------------------------------------------------------------

def test_a_never_read_source_is_stale(tmp_path):
    rows = feeds.staleness([Source("s", "u")], path=tmp_path / "state.json")
    assert rows[0]["stale"] and rows[0]["last_fetched"] is None


def test_staleness_turns_over_at_the_interval(tmp_path):
    state = tmp_path / "state.json"
    now = time.time()
    feeds.record_fetch("s", when=now - 20 * 86400, path=state)
    fresh = feeds.staleness([Source("s", "u")], now=now, days=30, path=state)
    assert not fresh[0]["stale"]
    stale = feeds.staleness([Source("s", "u")], now=now, days=10, path=state)
    assert stale[0]["stale"]
    assert stale[0]["age_days"] == pytest.approx(20, abs=0.01)


def test_the_interval_is_a_user_setting(monkeypatch):
    monkeypatch.setenv(feeds.INTERVAL_ENV, "7")
    assert feeds.interval_days() == 7
    monkeypatch.setenv(feeds.INTERVAL_ENV, "nonsense")
    assert feeds.interval_days() == feeds.DEFAULT_INTERVAL_DAYS
    monkeypatch.setenv(feeds.INTERVAL_ENV, "0")
    assert feeds.interval_days() == feeds.DEFAULT_INTERVAL_DAYS


def test_sources_come_from_config_when_there_is_one(tmp_path):
    cfg = tmp_path / "sources.json"
    cfg.write_text(json.dumps({"sources": [
        {"name": "mine", "url": "https://example.invalid/f.rss"}]}))
    got = feeds.load_sources(cfg)
    assert [s.name for s in got] == ["mine"]


def test_a_missing_or_broken_config_falls_back_to_the_defaults(tmp_path):
    assert feeds.load_sources(tmp_path / "nope.json") == feeds.DEFAULT_SOURCES
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert feeds.load_sources(bad) == feeds.DEFAULT_SOURCES


def test_save_then_load_round_trips(tmp_path):
    cfg = tmp_path / "s.json"
    feeds.save_sources([Source("a", "u1"), Source("b", "u2", enabled=False)], cfg)
    got = feeds.load_sources(cfg)
    assert [(s.name, s.enabled) for s in got] == [("a", True), ("b", False)]


# ---- fetching -------------------------------------------------------------

def test_the_rate_limit_is_retried_not_surrendered_to():
    """A rapid second request returns 429 with an empty body. Giving up there
    would make the feed unreadable in normal use."""
    import urllib.error

    url = "https://example.invalid/f.rss"
    calls = []
    seq = [urllib.error.HTTPError(url, 429, "Too Many", {}, None),
           urllib.error.HTTPError(url, 429, "Too Many", {}, None)]

    def urlopen(req, timeout=None):
        calls.append(1)
        if seq:
            raise seq.pop(0)

        class R:
            def read(self):
                return b"<feed/>"

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False
        return R()

    import harness.feeds as m
    old = m.urllib.request.urlopen
    m.urllib.request.urlopen = urlopen
    try:
        assert m.fetch(url, sleep=lambda s: None) == "<feed/>"
    finally:
        m.urllib.request.urlopen = old
    assert len(calls) == 3


def test_a_permanent_error_is_not_retried():
    import urllib.error

    import harness.feeds as m
    calls = []

    url = "https://example.invalid/f.rss"

    def urlopen(req, timeout=None):
        calls.append(1)
        raise urllib.error.HTTPError(url, 404, "gone", {}, None)

    old = m.urllib.request.urlopen
    m.urllib.request.urlopen = urlopen
    try:
        with pytest.raises(FeedError):
            m.fetch(url, sleep=lambda s: None)
    finally:
        m.urllib.request.urlopen = old
    assert len(calls) == 1


def test_read_caches_and_never_caches_a_block_page(tmp_path):
    src = Source("s", "https://example.invalid/f.rss")
    hits = []

    def fetcher(url):
        hits.append(url)
        return WEEK.read_text()

    a = feeds.read(src, cache_dir=tmp_path, fetcher=fetcher)
    b = feeds.read(src, cache_dir=tmp_path, fetcher=fetcher)
    assert len(a) == len(b) == 25
    assert len(hits) == 1, "second read should come from cache"

    src2 = Source("blocked", "https://example.invalid/b.rss")
    with pytest.raises(FeedError):
        feeds.read(src2, cache_dir=tmp_path, fetcher=lambda u: "<html>no</html>")
    assert not (tmp_path / "blocked.xml").exists(), "cached a block page"


# ---- discover wiring ------------------------------------------------------

def test_from_feeds_offers_linked_repos_without_a_registry_call(week):
    src = Source("fix", "https://example.invalid", lane="image")
    got = discover.from_feeds([src], reader=lambda s: week, verify=False)
    names = {c.name for c in got}
    assert "inclusionAI/LLaDA-Image" in names
    assert all(c.kind == "proposal" for c in got)


def test_prose_names_are_dropped_when_the_registry_does_not_know_them(week):
    """"RTX 3070" and "Diablo 4" come out of titles. A name nothing can resolve
    must not be offered as a candidate."""
    src = Source("fix", "https://example.invalid", lane="image")
    got = discover.from_feeds([src], reader=lambda s: week, verify=False)
    assert not {c.name for c in got} & {"RTX 3070", "Diablo 4", "The 1967"}


def test_a_schemeless_url_in_config_is_a_finding_not_a_traceback():
    with pytest.raises(FeedError):
        feeds.fetch("not-a-url", sleep=lambda s: None)


def test_well_formed_html_is_still_not_a_feed():
    """"<html>no</html>" parses perfectly well as XML. Checking that it parsed
    is not the same as checking it is a feed."""
    with pytest.raises(FeedError) as exc:
        feeds.parse("<html><body>blocked</body></html>")
    assert "not a feed" in str(exc.value)


def test_a_dead_source_is_a_finding_not_a_crash():
    def boom(src):
        raise FeedError("no network")

    got = discover.from_feeds([Source("dead", "u")], reader=boom)
    assert len(got) == 1
    assert got[0].blocked and not got[0].present


def test_a_disabled_source_is_not_read():
    def boom(src):
        raise AssertionError("read a disabled source")

    assert discover.from_feeds([Source("off", "u", enabled=False),
                                ], reader=boom) == []


def test_huggingface_non_repo_paths_are_not_offered_as_models(week):
    """huggingface.co/blog/<slug> matches the owner/name shape. "blog/zuanfilm"
    was offered as a model candidate in the first live run."""
    names = {p.name for p in feeds.candidates(week)}
    assert not {n for n in names if n.split("/")[0] in ("blog", "docs",
                                                        "spaces", "datasets")}


def test_github_links_are_tools_not_model_candidates(week):
    kinds = {p.name: p.kind for p in feeds.candidates(week)}
    assert kinds.get("Merserk/dlss5-visual-enhancer") == "tool"
    assert kinds.get("inclusionAI/LLaDA-Image") == "repo"


# ---- apple silicon relevance (#45) ----------------------------------------

def test_native_mlx_outranks_a_generic_model():
    """Measured 2026-09-07: 0/25 and 1/25 of the reddit weekly entries mention
    Apple Silicon at all, so an MLX result must not be ranked identically to a
    generic GGUF one."""
    mlx = feeds.relevance("SDMLX - Speeds up SDXL workflows on Mac using native MLX.")
    plain = feeds.relevance("Ling-3.0-tiny - Low-cost local AI reasoning model.")
    assert mlx > plain


def test_hardware_this_machine_does_not_have_scores_negative():
    assert feeds.relevance("ninfer-4090 - Runs Qwen3.8-27B on one RTX 4090.") < 0
    assert feeds.relevance("needs CUDA and 24GB VRAM") < 0


def test_relevance_is_zero_when_the_text_says_neither_way():
    assert feeds.relevance("A compact model that tops benchmarks.") == 0


def test_a_term_repeated_does_not_inflate_the_score():
    once = feeds.relevance("mlx")
    many = feeds.relevance("mlx mlx mlx mlx mlx")
    assert once == many


def test_candidates_carry_their_relevance(recap):
    props = {p.name: p for p in feeds.candidates(recap[:1])}
    assert any(p.relevance > 0 for p in props.values())


# ---- release feeds (#45) ---------------------------------------------------

RELEASES = """<feed xmlns="http://www.w3.org/2005/Atom">
  <entry><title>v0.5.3</title><link href="https://x/1"/><updated>2026-09-01</updated></entry>
  <entry><title>v0.5.10</title><link href="https://x/2"/><updated>2026-08-01</updated></entry>
  <entry><title>v0.5.1</title><link href="https://x/3"/><updated>2026-07-01</updated></entry>
</feed>"""


def test_newest_release_compares_numerically_not_as_text():
    """0.5.10 is newer than 0.5.3, and string ordering says the opposite."""
    assert feeds.newest_release(feeds.parse(RELEASES)) == "0.5.10"


def test_behind_is_false_when_either_version_is_unreadable():
    assert not feeds.behind("", "0.5.1")
    assert not feeds.behind("0.5.3", "")
    assert feeds.behind("0.5.3", "0.5.1")
    assert not feeds.behind("0.5.1", "0.5.3")
    assert not feeds.behind("0.5.1", "0.5.1")


def test_a_pinned_service_version_is_found_in_versions_sh(tmp_path):
    pins = tmp_path / "versions.sh"
    pins.write_text('MLX_AUDIO_PIN="mlx-audio==0.5.1"\n'
                    'MISAKI_PIN="misaki[en]==0.9.4"\n')
    assert feeds.installed_version("misaki", pins) == "0.9.4"


def test_every_releases_source_declares_what_it_tracks():
    for s in feeds.DEFAULT_SOURCES:
        if s.kind == "releases":
            assert s.name in feeds.TRACKS, s.name


def test_a_releases_source_reports_drift_not_the_repo_it_watches(monkeypatch):
    """Proposing `ml-explore/mlx` to a project built on MLX is noise. The
    useful signal is that we are behind."""
    src = Source("mlx-releases", "https://example.invalid/r.atom",
                 kind="releases", lane="all")
    monkeypatch.setitem(feeds.TRACKS, "mlx-releases", "mlx")
    monkeypatch.setattr(feeds, "installed_version", lambda p, pins=None: "0.5.1")
    got = discover.from_feeds([src], reader=lambda s: feeds.parse(RELEASES))
    assert [c.name for c in got] == ["mlx"]
    assert got[0].kind == "update"
    assert "0.5.10" in got[0].note


def test_a_current_releases_source_reports_nothing(monkeypatch):
    src = Source("mlx-releases", "https://example.invalid/r.atom",
                 kind="releases", lane="all")
    monkeypatch.setitem(feeds.TRACKS, "mlx-releases", "mlx")
    monkeypatch.setattr(feeds, "installed_version", lambda p, pins=None: "0.5.10")
    assert discover.from_feeds([src], reader=lambda s: feeds.parse(RELEASES)) == []


def test_the_platform_filter_drops_cuda_only_proposals(week):
    src = Source("fix", "https://example.invalid", lane="image")
    everything = discover.from_feeds([src], reader=lambda s: week, verify=False)
    filtered = discover.from_feeds([src], reader=lambda s: week, verify=False,
                                   min_relevance=1)
    assert len(filtered) < len(everything)
    assert all(c.relevance >= 1 for c in filtered)


# ---- issue #49: a repo name in prose is a claim -----------------------------

class FakeGH:
    """Stands in for harness.github.Client.exists."""

    def __init__(self, known=(), unreachable=False):
        self.known, self.unreachable = set(known), unreachable
        self.asked = []

    def exists(self, full_name):
        self.asked.append(full_name)
        if self.unreachable:
            raise RuntimeError("no network")
        return full_name in self.known


def test_a_github_proposal_that_does_not_exist_is_dropped(week):
    """Held to the same bar as a model name: HF ids were verified from the
    start and repo names were not."""
    src = Source("fix", "https://example.invalid", lane="image")
    gh = FakeGH(known=())
    got = discover.from_feeds([src], reader=lambda s: week, verify=True, gh=gh)
    assert not [c for c in got if c.kind == "proposal" and "/" in c.name
                and c.how.startswith("https://github.com")]


def test_an_unreachable_api_keeps_the_proposal_rather_than_emptying_the_sweep(week):
    """Fails OPEN. An outage that silently returns nothing looks identical to
    a quiet week, which is the worst way for a discovery tool to break."""
    src = Source("fix", "https://example.invalid", lane="image")
    linked = discover.from_feeds([src], reader=lambda s: week, verify=False)
    survived = discover.from_feeds([src], reader=lambda s: week, verify=True,
                                   gh=FakeGH(unreachable=True))
    assert {c.name for c in survived} >= {c.name for c in linked if c.kind == "tool"}
