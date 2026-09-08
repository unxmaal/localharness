"""Issue #58: the cached GitHub client the discovery loop leans on."""
import json
import time

import pytest

from harness import github


def runner(replies, fail_after=None):
    seq = {"n": 0}

    def call(path):
        seq["n"] += 1
        if fail_after is not None and seq["n"] > fail_after:
            raise github.GitHubError("service is gone")
        return json.dumps(replies.get(path, {"path": path}))
    call.count = lambda: seq["n"]
    return call


def test_a_second_read_is_served_from_the_cache(tmp_path):
    r = runner({"repos/a/b": {"stargazers_count": 7}})
    c = github.Client(cache=tmp_path, runner=r)
    assert c.repo("a/b")["stargazers_count"] == 7
    assert c.repo("a/b")["stargazers_count"] == 7
    assert c.spent == 1


def test_a_dead_service_still_answers_from_what_was_already_fetched(tmp_path):
    """The reason this project computes similarity itself. A hosted recommender
    in the same prior-art search is already archived and end-of-service; what
    we fetched before that happens has to keep working."""
    r = runner({"repos/a/b": {"stargazers_count": 7}}, fail_after=1)
    c = github.Client(cache=tmp_path, runner=r)
    c.repo("a/b")
    dead = github.Client(cache=tmp_path, runner=r, ttl_hours=0)
    assert dead.repo("a/b")["stargazers_count"] == 7
    assert dead.stale == ["repos/a/b"]


def test_a_failure_with_nothing_cached_is_an_error_not_an_empty_answer(tmp_path):
    """Returning nothing would read as 'this repo has no neighbours'."""
    c = github.Client(cache=tmp_path, runner=runner({}, fail_after=0))
    with pytest.raises(github.GitHubError):
        c.repo("a/b")


def test_a_stale_entry_is_refreshed_when_the_service_is_up(tmp_path):
    r = runner({"repos/a/b": {"stargazers_count": 9}})
    github.Client(cache=tmp_path, runner=r).repo("a/b")
    c = github.Client(cache=tmp_path, runner=r, ttl_hours=0)
    assert c.repo("a/b")["stargazers_count"] == 9
    assert c.spent == 1 and c.stale == []


def test_the_budget_is_refused_rather_than_exceeded(tmp_path):
    """5000 requests an hour is shared with every other tool on this machine."""
    c = github.Client(cache=tmp_path, runner=runner({}), budget=2)
    c.get("one")
    c.get("two")
    with pytest.raises(github.BudgetError):
        c.get("three")


def test_a_spent_budget_still_reads_the_cache(tmp_path):
    c = github.Client(cache=tmp_path, runner=runner({}), budget=1)
    c.get("one")
    with pytest.raises(github.BudgetError):
        c.get("two")
    assert c.get("one") is not None


def test_cache_hits_do_not_count_against_the_budget(tmp_path):
    c = github.Client(cache=tmp_path, runner=runner({}), budget=1)
    for _ in range(5):
        c.get("one")
    assert c.spent == 1


def test_a_corrupt_cache_file_is_refetched_not_fatal(tmp_path):
    r = runner({"repos/a/b": {"stargazers_count": 3}})
    c = github.Client(cache=tmp_path, runner=r)
    c.repo("a/b")
    next(tmp_path.glob("*.json")).write_text("{not json")
    assert github.Client(cache=tmp_path, runner=r).repo("a/b") == \
        {"stargazers_count": 3}


def test_a_path_becomes_one_safe_filename(tmp_path):
    c = github.Client(cache=tmp_path, runner=runner({}))
    c.get("users/x/starred?per_page=100&page=2")
    assert len(list(tmp_path.glob("*.json"))) == 1


def test_paged_endpoints_read_the_pages_asked_for(tmp_path):
    r = runner({"repos/a/b/contributors?per_page=100&page=1": [{"login": "x"}],
                "repos/a/b/contributors?per_page=100&page=3": [{"login": "y"}]})
    c = github.Client(cache=tmp_path, runner=r)
    assert c.contributors("a/b", (1, 3)) == ["x", "y"]


def test_the_client_offers_no_way_to_list_who_starred_a_repo(tmp_path):
    """Measured #58: repos/<r>/stargazers is 404 with a token and 401 without,
    for every repo. A method that always fails is worse than no method, and the
    constant is what stops the next person rediscovering it."""
    assert "stargazers" in github.NO_REPO_TO_PEOPLE
    assert not hasattr(github.Client(cache=tmp_path), "stargazers")


def test_rows_without_the_field_are_skipped(tmp_path):
    """A private or deleted account comes back as a partial row."""
    r = runner({"users/u/starred?per_page=100&page=1":
                [{"full_name": "a/b"}, {}, {"full_name": None}]})
    assert github.Client(cache=tmp_path, runner=r).starred("u") == ["a/b"]


def test_the_cache_records_when_it_was_fetched(tmp_path):
    """The store is a record of facts, so each one carries its date."""
    c = github.Client(cache=tmp_path, runner=runner({}))
    c.get("one")
    payload = json.loads(next(tmp_path.glob("*.json")).read_text())
    assert abs(payload["fetched"] - time.time()) < 60
    assert payload["path"] == "one"


# ---- issue #77: one TTL for every endpoint made a shorter sweep a no-op -----

def test_the_star_ttl_is_shorter_than_the_discovery_interval():
    """THE relationship this all turns on. With everything cached for 30 days
    and the interval at 30 days, a weekly sweep re-read a month-old answer,
    found what it found last time, and reported success. Two constants that
    happened to be compatible; asserted now."""
    from harness import feeds
    assert github.TTL_BY_ENDPOINT_HOURS["/starred"] < \
        feeds.DEFAULT_INTERVAL_DAYS * 24


def test_what_someone_starred_expires_fastest():
    """It is the signal the crowd source exists for."""
    star = github.ttl_for("users/x/starred?per_page=100")
    assert star < github.ttl_for("users/x/following?per_page=100")
    assert star < github.ttl_for("repos/a/b/contributors")


def test_the_longest_matching_endpoint_wins():
    """`repos/a/b/contributors` is a contributors call, not a repos call."""
    assert github.ttl_for("repos/a/b/contributors") == \
        github.TTL_BY_ENDPOINT_HOURS["/contributors"]


def test_an_unlisted_endpoint_falls_back(tmp_path):
    assert github.ttl_for("something/new") == github.DEFAULT_TTL_HOURS


def test_a_star_list_older_than_its_ttl_is_refetched(tmp_path):
    """The bug in one line: the answer was reused for a month."""
    import json as _json
    r = runner({"users/u/starred?per_page=100&page=1": [{"full_name": "a/b"}]})
    c = github.Client(cache=tmp_path, runner=r)
    c.starred("u")
    stale = next(tmp_path.glob("*.json"))
    payload = _json.loads(stale.read_text())
    payload["fetched"] = time.time() - (
        github.TTL_BY_ENDPOINT_HOURS["/starred"] + 1) * 3600
    stale.write_text(_json.dumps(payload))
    fresh = github.Client(cache=tmp_path, runner=r)
    fresh.starred("u")
    assert fresh.spent == 1


def test_contributors_that_old_are_still_served_from_cache(tmp_path):
    """Who writes a project changes over months, so re-fetching it weekly is
    requests spent on an answer that did not change."""
    import json as _json
    r = runner({"repos/a/b/contributors?per_page=100&page=1": [{"login": "x"}]})
    c = github.Client(cache=tmp_path, runner=r)
    c.contributors("a/b")
    p = next(tmp_path.glob("*.json"))
    payload = _json.loads(p.read_text())
    payload["fetched"] = time.time() - 8 * 24 * 3600     # a week old
    p.write_text(_json.dumps(payload))
    again = github.Client(cache=tmp_path, runner=r)
    again.contributors("a/b")
    assert again.spent == 0


def test_an_explicit_ttl_still_overrides_the_table(tmp_path):
    c = github.Client(cache=tmp_path, runner=runner({}), ttl_hours=0)
    c.get("users/u/starred")
    c.get("users/u/starred")
    assert c.spent == 2
