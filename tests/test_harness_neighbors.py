"""Issue #58: repo similarity from a crowd's stars, computed here."""
import json

import pytest

from harness import github, neighbors as nb


class FakeAPI:
    """A GitHub with a known shape, so the ranking can be checked exactly."""

    def __init__(self, repos=None, contributors=None, following=None,
                 starred=None):
        self.repos = repos or {}
        self.contributors = contributors or {}
        self.following = following or {}
        self.starred = starred or {}
        self.calls = []

    def __call__(self, path):
        self.calls.append(path)
        head = path.split("?")[0]
        if head.startswith("repos/") and head.endswith("/contributors"):
            name = head[len("repos/"):-len("/contributors")]
            return json.dumps([{"login": u}
                               for u in self.contributors.get(name, [])])
        if head.startswith("repos/") and head.endswith("/stargazers"):
            raise github.GitHubError("404 Not Found")
        if head.startswith("repos/"):
            name = head[len("repos/"):]
            if name not in self.repos:
                raise github.GitHubError(f"404 {name}")
            return json.dumps(self.repos[name])
        if head.startswith("users/") and head.endswith("/starred"):
            login = head[len("users/"):-len("/starred")]
            if login not in self.starred:
                raise github.GitHubError(f"404 {login}")
            return json.dumps([{"full_name": r}
                               for r in self.starred[login]])
        if head.startswith("users/") and head.endswith("/following"):
            login = head[len("users/"):-len("/following")]
            return json.dumps([{"login": u}
                               for u in self.following.get(login, [])])
        raise github.GitHubError(f"unexpected {path}")


def client(api, tmp_path, **kw):
    return github.Client(cache=tmp_path / "c", runner=api, budget=10_000, **kw)


# ---- the metric ------------------------------------------------------------

def test_the_overlap_is_exact_not_estimated():
    """Every member of the crowd is checked, so the intersection is counted,
    not inferred from a sample."""
    assert nb.jaccard(40, 200, 900) == pytest.approx(40 / (200 + 900 - 40))


def test_popularity_is_divided_out():
    """THE point of the metric. A repo with half a million stars shares its
    crowd with everything and must not outrank a niche one."""
    niche = nb.jaccard(40, 200, 900)
    huge = nb.jaccard(20, 200, 500_000)
    assert huge < niche / 100


def test_a_count_larger_than_the_crowd_is_clamped():
    """Defensive: a union below the intersection would score above 1 and read
    as a spectacular result rather than as an arithmetic error."""
    assert nb.jaccard(999, 200, 900) <= 1.0


@pytest.mark.parametrize("args", [(5, 0, 10), (5, 10, 0), (0, 10, 10)])
def test_nothing_to_divide_by_scores_zero_rather_than_raising(args):
    assert nb.jaccard(*args) == 0.0


# ---- building the crowd ----------------------------------------------------

def test_the_crowd_starts_from_who_writes_what_we_run(tmp_path):
    """Not a hand-kept list of people: the seed follows what is installed."""
    api = FakeAPI(contributors={"org/tool": ["ann", "bob"]},
                  following={"ann": ["carol"], "bob": []})
    got = nb.cohort(["org/tool"], client(api, tmp_path))
    assert got[:2] == ["ann", "bob"]
    assert "carol" in got


def test_the_second_hop_can_be_turned_off(tmp_path):
    api = FakeAPI(contributors={"org/tool": ["ann"]}, following={"ann": ["carol"]})
    assert nb.cohort(["org/tool"], client(api, tmp_path), follow=False) == ["ann"]


def test_nobody_is_counted_twice(tmp_path):
    """A person who contributes to two of our tools would otherwise get two
    votes, which is a popularity metric with extra steps."""
    api = FakeAPI(contributors={"a/one": ["ann"], "b/two": ["ann", "bob"]},
                  following={"ann": ["bob"], "bob": ["ann"]})
    got = nb.cohort(["a/one", "b/two"], client(api, tmp_path))
    assert sorted(got) == ["ann", "bob"]


def test_bots_are_not_people(tmp_path):
    api = FakeAPI(contributors={"a/one": ["ann", "dependabot[bot]"]})
    assert "dependabot[bot]" not in nb.cohort(["a/one"], client(api, tmp_path),
                                              follow=False)


def test_the_crowd_is_capped(tmp_path):
    api = FakeAPI(contributors={"a/one": [f"u{i}" for i in range(50)]},
                  following={f"u{i}": [f"v{i}"] for i in range(50)})
    assert len(nb.cohort(["a/one"], client(api, tmp_path), per_repo=50,
                         limit=20)) == 20


def test_a_seed_repo_that_cannot_be_read_is_skipped_not_fatal(tmp_path):
    api = FakeAPI(contributors={"b/two": ["bob"]})
    assert nb.cohort(["gone/away", "b/two"], client(api, tmp_path),
                     follow=False) == ["bob"]


def test_the_crowd_is_deterministic(tmp_path):
    api = FakeAPI(contributors={"a/one": ["ann", "bob"]},
                  following={"ann": ["carol"], "bob": ["dave"]})
    c = client(api, tmp_path)
    assert nb.cohort(["a/one"], c) == nb.cohort(["a/one"], c)


# ---- the ranking -----------------------------------------------------------

def _world():
    """Fifty people. Forty starred a small sibling project; all fifty starred a
    giant that everybody stars."""
    people = [f"u{i}" for i in range(50)]
    repos = {
        "org/sibling": {"stargazers_count": 900},
        "big/everything": {"stargazers_count": 500_000, "archived": True,
                           "language": "JavaScript", "topics": ["awesome"],
                           "pushed_at": "2020-01-01T00:00:00Z",
                           "description": "everyone stars this"},
    }
    starred = {u: ["big/everything"] for u in people}
    for u in people[:40]:
        starred[u].append("org/sibling")
    return FakeAPI(repos=repos, starred=starred), people


def test_the_sibling_outranks_the_repo_everybody_stars(tmp_path):
    api, people = _world()
    got = nb.neighbors(people, client(api, tmp_path))
    assert [n.repo for n in got] == ["org/sibling", "big/everything"]
    assert got[0].score > got[1].score * 100


def test_raw_overlap_would_have_got_it_backwards(tmp_path):
    """The giant is starred by MORE of the crowd than the sibling. Anything
    ranking on that count alone recommends it."""
    api, people = _world()
    counts, asked = nb.crowd(people, client(api, tmp_path))
    assert counts.most_common(1)[0][0] == "big/everything"
    assert asked == 50


def test_the_crowd_size_travels_with_every_score(tmp_path):
    """A score whose sample is invisible gets quoted as if it were measured."""
    api, people = _world()
    got = nb.neighbors(people, client(api, tmp_path))
    assert all(n.crowd == 50 and n.shared > 0 for n in got)


def test_a_thin_overlap_is_dropped_rather_than_scored(tmp_path):
    api, people = _world()
    api.starred["u0"].append("org/oneoff")
    got = nb.neighbors(people, client(api, tmp_path), min_shared=3)
    assert "org/oneoff" not in [n.repo for n in got]


def test_what_we_already_run_can_be_excluded(tmp_path):
    """Proposing mlx to a project built on mlx is noise, the same reason the
    release feeds are a separate source kind."""
    api, people = _world()
    got = nb.neighbors(people, client(api, tmp_path), exclude=["org/sibling"])
    assert [n.repo for n in got] == ["big/everything"]


def test_metadata_comes_from_github_not_from_a_third_party(tmp_path):
    """Topics, archived and pushed date are what let the judge tell a live
    project from an abandoned one, and they are GitHub's own fields."""
    api, people = _world()
    got = nb.neighbors(people, client(api, tmp_path))
    big = [n for n in got if n.repo == "big/everything"][0]
    assert big.archived and big.language == "JavaScript"
    assert big.topics == ["awesome"] and big.pushed == "2020-01-01"


def test_a_candidate_whose_metadata_is_gone_is_skipped_not_fatal(tmp_path):
    """Repos get deleted and renamed between the star and the sweep."""
    api, people = _world()
    for u in people[:40]:
        api.starred[u].append("gone/away")
    got = nb.neighbors(people, client(api, tmp_path))
    assert "gone/away" not in [n.repo for n in got]
    assert "org/sibling" in [n.repo for n in got]


def test_a_person_who_cannot_be_read_does_not_count_as_asked(tmp_path):
    """Dividing by a crowd that never answered inflates every score."""
    api, people = _world()
    counts, asked = nb.crowd(people + ["ghost"], client(api, tmp_path))
    assert asked == 50


# ---- the control -----------------------------------------------------------

def test_the_control_passes_when_the_expected_ranks_and_no_decoy_does(tmp_path):
    api, people = _world()
    got = nb.control(["org/sibling"], people, client(api, tmp_path), top=1,
                     decoys=("big/everything",))
    assert got["expected_found"] == ["org/sibling"]
    assert got["decoys_in_top"] == []
    assert got["separates"] is True


def test_a_decoy_in_the_top_fails_the_control(tmp_path):
    """The negative direction. Passing the positive alone is how a metric that
    ranks popularity ships as a metric that ranks relevance."""
    api, people = _world()
    got = nb.control(["org/sibling"], people, client(api, tmp_path), top=5,
                     decoys=("big/everything",))
    assert got["decoys_in_top"] == ["big/everything"]
    assert got["separates"] is False


def test_the_control_reports_what_it_failed_to_find(tmp_path):
    api, people = _world()
    got = nb.control(["org/sibling", "never/seen"], people,
                     client(api, tmp_path))
    assert got["missing"] == ["never/seen"]


def test_the_control_shows_the_ranking_it_replaced(tmp_path):
    """raw_top against top is the evidence that normalising changed the
    answer, rather than a claim that it did."""
    api, people = _world()
    got = nb.control(["org/sibling"], people, client(api, tmp_path))
    assert got["raw_top"][0] == "big/everything"
    assert got["top"][0] == "org/sibling"
