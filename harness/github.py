"""GitHub's API, cached on disk. Issue #58.

Everything here goes through `gh`, so authentication is whatever the human
already set up: 5000 requests an hour rather than 60.

TWO PROPERTIES THE REST OF THE DISCOVERY LOOP DEPENDS ON.

A response is cached permanently and a FAILED fetch falls back to the stale
copy. A hosted recommender can vanish -- one in the same search was already
archived and marked end-of-service -- and the answer to that is to keep every
fact we ever fetched rather than to pick a more reliable service.

Requests are counted against a budget and the budget is refused, not silently
exceeded. A sweep that quietly burns an hour's rate limit blocks every other
tool on this machine that speaks to GitHub.
"""
from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from harness import paths

#: Star lists move slowly and a month-old neighbourhood is still a
#: neighbourhood. The cache is a store of facts, not a performance trick.
DEFAULT_TTL_HOURS = 24 * 30
PER_PAGE = 100
#: repos/<r>/stargazers and /subscribers return 404 to this token and 401 to no
#: token at all, for every repo, so who starred a repo cannot be listed. The
#: crowd has to be assembled from people we can name instead. Measured #58.
NO_REPO_TO_PEOPLE = ("stargazers", "subscribers")


class GitHubError(RuntimeError):
    """The API could not be reached and no cached copy existed."""


class BudgetError(GitHubError):
    """The caller's request budget is spent. Raised, never exceeded."""


def cache_dir() -> Path:
    return paths.home() / "cache" / "github"


def _slug(path: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]", "_", path)[:180]


def _gh(path: str) -> str:
    """One `gh api` call. Replaced wholesale in tests."""
    proc = subprocess.run(["gh", "api", path], capture_output=True, text=True,
                          timeout=60)
    if proc.returncode != 0:
        raise GitHubError(f"gh api {path}: {proc.stderr.strip()[:200]}")
    return proc.stdout


@dataclass
class Client:
    cache: Path | None = None
    ttl_hours: float = DEFAULT_TTL_HOURS
    runner: Callable[[str], str] = _gh
    budget: int = 400
    #: Requests actually sent. Cache hits are free and do not count.
    spent: int = 0
    #: Paths served from a cache entry older than the TTL because the fetch
    #: failed. Reported rather than hidden: the data is real but not fresh.
    stale: list[str] = field(default_factory=list)

    def _path(self, api_path: str) -> Path:
        d = Path(self.cache) if self.cache is not None else cache_dir()
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{_slug(api_path)}.json"

    def get(self, api_path: str) -> Any:
        """Cached JSON. Fresh cache wins; a failed fetch falls back to stale."""
        cached = self._path(api_path)
        payload = None
        if cached.exists():
            try:
                payload = json.loads(cached.read_text())
            except (OSError, ValueError):
                payload = None
        if payload is not None:
            age = time.time() - float(payload.get("fetched", 0))
            if age < self.ttl_hours * 3600:
                return payload["data"]
        if self.spent >= self.budget:
            if payload is not None:
                self.stale.append(api_path)
                return payload["data"]
            raise BudgetError(f"request budget {self.budget} spent at {api_path}")
        try:
            self.spent += 1
            data = json.loads(self.runner(api_path))
        except Exception as exc:  # noqa: BLE001 - any failure falls back
            if payload is not None:
                self.stale.append(api_path)
                return payload["data"]
            raise GitHubError(f"{api_path}: {exc}") from exc
        cached.write_text(json.dumps({"fetched": time.time(),
                                      "path": api_path, "data": data}))
        return data

    # ---- the endpoints this project uses -----------------------------------

    def repo(self, full_name: str) -> dict:
        return self.get(f"repos/{full_name}")

    def contributors(self, full_name: str, pages: tuple[int, ...] = (1,)) -> list[str]:
        """Who writes this repo, most commits first.

        This is where a crowd starts, because the stargazer list is not
        readable. It is also the better seed: people who build a tool are a
        stronger signal than people who bookmarked it.
        """
        out = []
        for p in pages:
            rows = self.get(f"repos/{full_name}/contributors"
                            f"?per_page={PER_PAGE}&page={p}")
            out += [r["login"] for r in rows
                    if isinstance(r, dict) and r.get("login")]
        return out

    def starred(self, login: str, pages: tuple[int, ...] = (1,)) -> list[str]:
        """Repos this person starred, most recently starred first.

        The default is one page. That is a deliberate recency bias: what
        somebody starred lately is a better signal about what is worth trying
        now than what they starred in 2019.
        """
        out = []
        for p in pages:
            rows = self.get(f"users/{login}/starred?per_page={PER_PAGE}&page={p}")
            out += [r["full_name"] for r in rows
                    if isinstance(r, dict) and r.get("full_name")]
        return out

    def following(self, login: str, pages: tuple[int, ...] = (1,)) -> list[str]:
        out = []
        for p in pages:
            rows = self.get(f"users/{login}/following?per_page={PER_PAGE}&page={p}")
            out += [r["login"] for r in rows
                    if isinstance(r, dict) and r.get("login")]
        return out
