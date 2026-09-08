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
