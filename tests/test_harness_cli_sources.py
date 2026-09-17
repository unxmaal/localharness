"""The source report must probe the URL, not the instruction for adding it."""
import json
from pathlib import Path

import pytest

from harness import cli, discover, feeds


@pytest.mark.parametrize("how", [
    "add it to discovery-sources.json to start reading it", "",
])
@pytest.mark.parametrize("is_feed", [True, False])
def test_sources_probes_the_source_url(monkeypatch, tmp_path, capsys,
                                      how, is_feed):
    source = tmp_path / "source.xml"
    if is_feed:
        fixture = Path(__file__).parent / "fixtures/feeds/reddit-top-week.xml"
        source.write_text(fixture.read_text(encoding="utf-8"), encoding="utf-8")
    else:
        source.write_text("<html><body>Not a feed</body></html>", encoding="utf-8")
    proposed = discover.Capability("source", "example", "all", source.as_uri(), how)
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(feeds, "staleness", lambda: [])
    monkeypatch.setattr(discover, "feed_sources", lambda: [proposed])

    # Keep probe, fetch and parse real: urllib reads the local fixture URL.
    assert cli.main(["discover", "--sources"]) == 0
    out = capsys.readouterr().out
    if is_feed:
        assert "  FEED  example" in out
        assert "25 entries" in out
    else:
        assert "  none  example" in out
        assert "not a feed: root element is <html>" in out
    assert "unknown url type" not in out


def test_sources_json_reports_proposals_without_probing(monkeypatch, tmp_path, capsys):
    proposed = discover.Capability("source", "example", "all",
                                   "https://example.invalid/feed", "add it")
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    monkeypatch.setattr(feeds, "staleness", lambda: [])
    monkeypatch.setattr(discover, "feed_sources", lambda: [proposed])

    def unexpected_probe(*args, **kwargs):
        pytest.fail("JSON reporting must not probe sources")

    monkeypatch.setattr(feeds, "probe", unexpected_probe)
    assert cli.main(["discover", "--sources", "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == {
        "sources": [], "proposed": [vars(proposed)],
    }
