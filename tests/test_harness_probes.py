"""The probes issue #98 asks for, and the parts of them that can be tested
without this Mac's cache.

The sweeps themselves need a populated GitHub cache and clones on disk. What is
tested here is everything around them: that the offline client cannot reach the
network, that a missing size is an absent size rather than a fetch, and that
the list of constants NOT covered stays honest.
"""
import json

import pytest

from harness import github, probes


def test_the_offline_client_never_fetches(tmp_path):
    """A zero budget and an enormous TTL together mean every answer comes off
    the disk. An eval or a sweep that quietly re-downloads is not measuring
    what it claims to."""
    c = probes.offline(cache=tmp_path)
    assert c.budget == 0
    (tmp_path / f"{github._slug('repos/a/b')}.json").write_text(
        json.dumps({"fetched": 0, "path": "repos/a/b", "data": {"id": 1}}))
    # Ancient by any real TTL, and served anyway.
    assert c.repo("a/b") == {"id": 1}
    assert c.spent == 0


def test_a_missing_cache_entry_is_an_error_not_a_download(tmp_path):
    c = probes.offline(cache=tmp_path)
    with pytest.raises(github.BudgetError):
        c.repo("nobody/nothing")
    assert c.spent == 0


def test_an_unsized_model_is_unsized_rather_than_fetched():
    assert probes.cached_facts("who/knows", {}) == {"size": 0, "lane": ""}


def test_a_size_cache_holding_bare_integers_still_reads():
    """hf-sizes.json stores an int per model, not a dict."""
    assert probes.cached_facts("a/b", {"a/b": 4096})["size"] == 4096


def test_a_size_cache_entry_may_carry_a_lane():
    got = probes.cached_facts("a/b", {"a/b": {"size": 8, "lane": "stt"}})
    assert got == {"size": 8, "lane": "stt"}


def test_clone_directory_names_map_back_to_repo_names(tmp_path, monkeypatch):
    monkeypatch.setattr(probes, "clones", lambda: tmp_path)
    (tmp_path / "owner__repo").mkdir()
    (tmp_path / "not-a-repo").mkdir()
    (tmp_path / "a-file").write_text("x")
    assert probes._repos_on_disk() == ["owner/repo"]


def test_every_probe_is_callable_by_name():
    assert probes.PROBES
    assert all(callable(f) for f in probes.PROBES.values())


def test_the_uncovered_list_names_a_reason_for_each():
    """A sweep that silently omitted a constant would read as coverage."""
    assert probes.UNCOVERED
    assert all(reason.strip() for reason in probes.UNCOVERED.values())


def test_no_constant_is_both_covered_and_listed_as_uncovered():
    covered = {n.lower() for n in probes.PROBES}
    for name in probes.UNCOVERED:
        assert name.lower().replace(" ", "_") not in covered
