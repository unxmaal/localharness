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

import math
import time
from collections import Counter
from datetime import datetime, timezone
from dataclasses import dataclass, field

from harness import github

#: Repos this project actually runs. Their contributors are the starting crowd,
#: so the seed follows what is installed rather than a hand-kept list of people.
DEFAULT_SEEDS = ("ml-explore/mlx", "ml-explore/mlx-lm", "Blaizzy/mlx-audio",
                 "mflux-community/mflux")
#: Below this, a score rests on one or two people and means nothing.
MIN_SHARED = 3
#: How many people could plausibly have starred any of this. NOT a free
#: constant: it sets how much evidence counts against how much enrichment, and
#: the control only passes between 1e6 and 2e7. Measured, see control().
POPULATION = 5_000_000
#: Score halves after this long without a push. A live niche project beats a
#: dead one with the same overlap, because the question is what to try NOW.
HALF_LIFE_DAYS = 365.0
#: A repo this large is starred by everyone and tells us nothing. One in the
#: top means popularity has leaked back into the ranking. Structural rather
#: than a list of names, because the first hand-written list missed every repo
#: that actually leaked (openclaw, opencode, hermes-agent) and named repos this
#: crowd does not star.
POPULAR_STARS = 100_000


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


def enrichment(shared: int, crowd_size: int, stars: int,
               population: float = POPULATION) -> float:
    """How much more this crowd stars a repo than the world does, times the
    evidence for saying so.

        shared * log( (shared / crowd) / (stars / population) )

    BOTH HALVES ARE NECESSARY AND THIS WAS MEASURED, not reasoned. Ranking on
    `shared` alone returns whatever giant everyone stars: five repos over
    100k stars led the first real run. Ranking on the ratio alone -- plain
    Jaccard, lift, or a Wilson bound, all three tried -- inverts the error and
    returns 33-star repos that four people happen to share, putting nothing
    this project runs in the top ten. Multiplying the log ratio by the count
    is the standard fix and the only one of six that passed both directions.

    `population` decides the balance. The control passes between 1e6 and 2e7
    and fails outside it, so the default sits in the middle of that window
    rather than at its edge.
    """
    if crowd_size <= 0 or stars <= 0 or shared <= 0:
        return 0.0
    shared = min(shared, crowd_size)
    ratio = (shared / crowd_size) / (stars / population)
    return shared * math.log(ratio) if ratio > 1 else 0.0


def recency(pushed: str, now: float | None = None,
            half_life: float = HALF_LIFE_DAYS) -> float:
    """Weight from how long ago the repo was last pushed, halving per half-life.

    Unknown or unparseable dates weigh 1.0 rather than 0: a missing field is
    not evidence of abandonment, and scoring it as such would silently drop
    every repo whose metadata came back thin.
    """
    if not pushed:
        return 1.0
    try:
        when = datetime.strptime(pushed[:10], "%Y-%m-%d").replace(
            tzinfo=timezone.utc).timestamp()
    except ValueError:
        return 1.0
    now = time.time() if now is None else now
    days = max(0.0, (now - when) / 86400.0)
    return 0.5 ** (days / half_life)


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
              top: int = 25, exclude=(), keep_archived: bool = False
              ) -> list[Neighbor]:
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
        # An archived repo is finished by definition, so it cannot be a thing
        # to try next however well it scores. A 2020 physics course ranked
        # sixth on the first real run.
        if bool(meta.get("archived")) and not keep_archived:
            continue
        stars = int(meta.get("stargazers_count") or 0)
        pushed = (meta.get("pushed_at") or "")[:10]
        out.append(Neighbor(
            repo=repo, shared=shared, crowd=asked,
            score=enrichment(shared, asked, stars) * recency(pushed),
            stars=stars,
            language=meta.get("language") or "",
            topics=list(meta.get("topics") or []),
            archived=bool(meta.get("archived")),
            pushed=pushed,
            description=(meta.get("description") or "")[:200]))
    out.sort(key=lambda n: n.score, reverse=True)
    return out[:top]


def control(expect, people=None, client: github.Client | None = None, *,
            top: int = 10, popular: int = POPULAR_STARS, **kw) -> dict:
    """Rank a known crowd and report whether the ranking is real.

    Two directions, because either alone passes on a broken metric:

    POSITIVE -- things already known to belong here have to rank. The tools this
    project runs are the honest test set: they were chosen before the metric
    existed.
    NEGATIVE -- nothing enormous may rank, because a repo with 300k stars is
    starred by every crowd and carries no information about this one. Measured
    by size rather than by a list of names: the hand-written list passed a
    ranking whose top five were all repos over 100k stars, because it did not
    happen to name those five.

    `raw_top` is the ranking by shared count alone, kept so the difference
    between the two orderings is visible rather than asserted.

    ONE HONEST LIMIT: `POPULATION` was chosen by running this control, so a
    pass now confirms the metric has not regressed rather than proving it
    generalises. A fresh crowd is the real test.
    """
    client = client or github.Client()
    people = list(people) if people is not None else cohort(client=client)
    found = neighbors(people, client, top=top, **kw)
    counts, asked = crowd(people, client)
    names = [n.repo for n in found]
    hits = [e for e in expect if e in names]
    leaks = [(n.repo, n.stars) for n in found if n.stars > popular]
    return {"crowd": asked, "top": names, "expected_found": hits,
            "missing": [e for e in expect if e not in names],
            "popularity_leaks": leaks,
            "raw_top": [r for r, _ in counts.most_common(top)],
            "separates": bool(hits) and not leaks}
