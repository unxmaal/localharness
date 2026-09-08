"""Which repos share a crowd with the people this project already relies on. #58.

The mechanism behind every "similar repositories" service: things starred by the
same people are probably about the same thing. We compute it here rather than
calling one, because the hosted versions rate-limit, gate results behind
starring them, and die -- one found in the same prior-art search is already
archived and marked end-of-service.

THE CROWD IS PEOPLE, NOT STARGAZERS, AND NOT BY CHOICE. `repos/<r>/stargazers`
returns 404 to an authenticated token and 401 to none, for every repo, so who
starred a thing cannot be read. What can be read is who WROTE it and who they
follow, which turns out to be the better seed anyway: contributors to mflux and
mlx-audio are a stronger signal than whoever bookmarked them.

NORMALISATION IS THE WHOLE JOB. Ranking by how many of the crowd starred
something just ranks by popularity and returns whatever huge repo everyone
stars. Dividing by the union fixes it, and `control()` proves that on real data
rather than assuming it, because a metric without a negative control is noise.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

from harness import github

#: Repos this project actually runs. Their contributors are the starting crowd,
#: so the seed follows what is installed rather than a hand-kept list of people.
DEFAULT_SEEDS = ("ml-explore/mlx", "ml-explore/mlx-lm", "Blaizzy/mlx-audio",
                 "mflux-community/mflux")
#: Below this, a score rests on one or two people and means nothing.
MIN_SHARED = 3
#: Repos that share a crowd with everything. If one of these outranks the real
#: neighbours, the normalisation is not working.
DECOYS = ("freeCodeCamp/freeCodeCamp", "torvalds/linux", "facebook/react",
          "huggingface/transformers", "ollama/ollama", "vllm-project/vllm")


@dataclass
class Neighbor:
    repo: str
    score: float
    shared: int
    crowd: int
    stars: int = 0
    language: str = ""
    topics: list[str] = field(default_factory=list)
    archived: bool = False
    pushed: str = ""
    description: str = ""


def jaccard(shared: int, crowd_size: int, stars: int) -> float:
    """Overlap between the crowd and everyone who starred the candidate.

    Exact, not estimated: `shared` IS the intersection, because every member of
    the crowd is checked, and `stars` is the candidate's own star count from
    GitHub. Union is `crowd + stars - shared`.

    So a repo with half a million stars and twenty of our crowd scores far
    below a niche one with nine hundred stars and forty, which is the entire
    point: the question is how concentrated a repo's audience is in the crowd
    we trust, not how many people like it.
    """
    if crowd_size <= 0 or stars <= 0 or shared <= 0:
        return 0.0
    shared = min(shared, crowd_size, stars)
    union = crowd_size + stars - shared
    return shared / union if union > 0 else 0.0


def cohort(seeds=DEFAULT_SEEDS, client: github.Client | None = None, *,
           per_repo: int = 12, follow: bool = True, limit: int = 250
           ) -> list[str]:
    """People to ask, from the repos this project runs.

    Contributors first, then who those people follow. The second hop is what
    turns a handful of maintainers into a community: it reaches the people they
    read, which is where a technique shows up before it reaches a subreddit.

    Deterministic order, so two sweeps are comparable.
    """
    client = client or github.Client()
    people: list[str] = []
    seen: set[str] = set()

    def add(login):
        if login and login not in seen and not login.endswith("[bot]"):
            seen.add(login)
            people.append(login)

    core = []
    for repo in seeds:
        try:
            found = client.contributors(repo)[:per_repo]
        except github.GitHubError:
            continue
        for login in found:
            add(login)
            core.append(login)
    if follow:
        for login in core:
            if len(people) >= limit:
                break
            try:
                for other in client.following(login):
                    add(other)
            except github.GitHubError:
                continue
    return people[:limit]


def crowd(people, client: github.Client | None = None) -> tuple[Counter, int]:
    """What the crowd starred, and how many of them actually answered."""
    client = client or github.Client()
    counts: Counter = Counter()
    asked = 0
    for login in people:
        try:
            starred = client.starred(login)
        except github.GitHubError:
            continue
        asked += 1
        counts.update(set(starred))
    return counts, asked


def neighbors(people=None, client: github.Client | None = None, *,
              seeds=DEFAULT_SEEDS, min_shared: int = MIN_SHARED,
              top: int = 25, exclude=()) -> list[Neighbor]:
    """Repos concentrated in this crowd, best first."""
    client = client or github.Client()
    people = list(people) if people is not None else cohort(seeds, client)
    counts, asked = crowd(people, client)
    skip = set(exclude)
    out = []
    for repo, shared in counts.most_common():
        if shared < min_shared:
            break
        if repo in skip:
            continue
        try:
            meta = client.repo(repo)
        except github.GitHubError:
            continue
        stars = int(meta.get("stargazers_count") or 0)
        out.append(Neighbor(
            repo=repo, shared=shared, crowd=asked,
            score=jaccard(shared, asked, stars), stars=stars,
            language=meta.get("language") or "",
            topics=list(meta.get("topics") or []),
            archived=bool(meta.get("archived")),
            pushed=(meta.get("pushed_at") or "")[:10],
            description=(meta.get("description") or "")[:200]))
    out.sort(key=lambda n: n.score, reverse=True)
    return out[:top]


def control(expect, people=None, client: github.Client | None = None, *,
            top: int = 10, decoys=DECOYS, **kw) -> dict:
    """Rank a known crowd and report whether the ranking is real.

    Two directions, because either alone passes on a broken metric:

    POSITIVE -- things already known to belong here have to rank. The tools this
    project runs are the honest test set: they were chosen before the metric
    existed.
    NEGATIVE -- a repo everybody stars must NOT rank, which is exactly what
    counting shared stars alone does and the reason for dividing by the union.

    `raw_top` is the ranking by shared count alone, kept so the difference
    between the two orderings is visible rather than asserted.
    """
    client = client or github.Client()
    people = list(people) if people is not None else cohort(client=client)
    found = neighbors(people, client, top=top, **kw)
    counts, asked = crowd(people, client)
    names = [n.repo for n in found]
    hits = [e for e in expect if e in names]
    decoyed = [d for d in decoys if d in names]
    return {"crowd": asked, "top": names, "expected_found": hits,
            "missing": [e for e in expect if e not in names],
            "decoys_in_top": decoyed,
            "raw_top": [r for r, _ in counts.most_common(top)],
            "separates": bool(hits) and not decoyed}
