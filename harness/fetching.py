"""Download queued weights in the background, durably. Issue #62.

WHY NOT harness.jobs.Queue: its own docstring says jobs "live in memory and die
with the process", and it is instantiated once inside the MCP server. A
download queue that exists only while a server happens to be running is not
automatic, which was the requirement.

The durable queue already existed and was unused for this. `memory_store` has a
non-terminal `queued` verdict; marking a candidate queued is a fact that
survives a restart, and it is already the vocabulary the loop speaks. This
module is the worker for that queue, nothing more.

ONE AT A TIME, for the same reason generation is: this machine holds one
working set, and a download competing with an eval distorts the measurement it
was queued to make possible.

NOTHING IS EXECUTED. Fetching weights is not running them.
"""
from __future__ import annotations

import re
import shutil
from dataclasses import dataclass
from pathlib import Path

from harness import memory_store as ms

GIB = 1024 ** 3
#: Never fill the volume. A download that leaves no room is a download that
#: breaks the next eval rather than enabling it.
DISK_FLOOR = 50 * GIB
#: Refuse anything larger in one go, whatever the disk says.
MAX_DOWNLOAD = 60 * GIB


class FetchError(RuntimeError):
    """The download could not be started. Never a verdict about the model."""


@dataclass
class Plan:
    repo: str
    size: int
    ok: bool
    why: str


def free_bytes(path: str | Path | None = None) -> int:
    import os
    target = Path(path) if path else Path(
        os.environ.get("HF_HOME") or Path.home() / ".cache" / "huggingface")
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free


def plan(repo: str, size: int, *, free: int | None = None,
         floor: int = DISK_FLOOR, cap: int = MAX_DOWNLOAD) -> Plan:
    """Whether this download may start, and why not when it may not.

    A SIZE IS REQUIRED. The inspect tier produces it, and an unknown size is
    refused rather than attempted: the whole point of inspecting before
    fetching is not to find out how big something is by downloading it.
    """
    if size <= 0:
        return Plan(repo, size, False, "no measured size; inspect it first")
    if size > cap:
        return Plan(repo, size, False,
                    f"{size / GIB:.1f} GiB is over the {cap / GIB:.0f} GiB cap")
    have = free_bytes() if free is None else free
    if have - size < floor:
        return Plan(repo, size, False,
                    f"{size / GIB:.1f} GiB would leave under the "
                    f"{floor / GIB:.0f} GiB floor ({have / GIB:.0f} GiB free)")
    return Plan(repo, size, True, f"{size / GIB:.1f} GiB, {have / GIB:.0f} GiB free")


#: Only these tiers may put something in the download queue. The JUDGE tier
#: also writes `queued`, and it judges a description: it queued a 122B model on
#: a 32 GB machine. Nothing is downloaded on the strength of prose.
FETCHABLE_TIERS = ("inspect", "fetch")
#: And only WEIGHTS are downloadable. The inspect tier queued the GitHub repos
#: it read, and this worker calls snapshot_download, which wants a HuggingFace
#: model id: every queued name 401'd. A repo is something to install and screen,
#: a weight is something to fetch, and they are not the same queue.
FETCHABLE_KIND = "weights"


def have(model_id: str, root: Path | None = None) -> bool:
    """Already in the HuggingFace cache, so there is nothing to download."""
    import os
    home = Path(root or os.environ.get("HF_HOME")
                or Path.home() / ".cache" / "huggingface")
    return (home / "hub" / f"models--{model_id.replace('/', '--')}").exists()


def queued(conn, tiers=FETCHABLE_TIERS, kind: str = FETCHABLE_KIND,
           needs_lane: bool = True) -> list[dict]:
    """Weights the inspect tier queued, best score first.

    A weight NO LANE CAN TEST is returned but never counted as fetchable:
    downloading is only justified by a measurement that follows it, and an
    automated loop would otherwise fill the disk with models that are
    unmeasurable by construction. Issue #81.
    """
    rows = conn.execute("""
        SELECT p.name, p.resolved, p.kind, p.lane,
               (SELECT v.outcome FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS outcome,
               (SELECT v.tier FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS tier,
               (SELECT v.detail FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS detail,
               -- The judge scores REPOS; the queue holds WEIGHTS, and a weight
               -- is never judged (judging a model id in isolation is the
               -- copywriting problem the rubric exists to avoid). So a weight
               -- inherits the score of the repo that named it, through the
               -- `needs` edge. Without this every row sorted at 0 and "largest
               -- first" was whatever order the query happened to return.
               COALESCE((SELECT v2.score FROM edges e
                           JOIN verdicts v2 ON v2.proposal_id = e.src
                          WHERE e.dst = p.id AND e.relation = 'needs'
                            AND v2.score IS NOT NULL
                          ORDER BY v2.id DESC LIMIT 1), 0) AS score
        FROM proposals p""").fetchall()
    out = [dict(r) for r in rows
           if r["outcome"] == "queued" and r["tier"] in tiers
           and (not kind or r["kind"] == kind)
           and not have(r["resolved"] or r["name"])]
    return sorted(out, key=lambda r: -(r["score"] or 0))


def size_of(row: dict) -> int:
    """The size the inspect tier measured, carried on the verdict.

    Read from the store rather than asked for again: the registry rate-limits,
    and a size already measured is a fact.
    """
    detail = (row.get("detail") or "")
    m = re.search(r"bytes=(\d+)", detail)
    return int(m.group(1)) if m else 0


def download(repo: str, snapshot=None) -> str:
    """Weights into the shared cache. Returns the path.

    HF_HUB_OFFLINE is 1 everywhere else in this project on purpose: an eval
    that silently re-downloads a model turns a network hiccup into a model
    "failure" mid-run. Fetching is the ONE operation whose whole job is to go
    online, so it lifts the guard for the length of the call and puts it back.
    """
    import os
    was = os.environ.get("HF_HUB_OFFLINE")
    os.environ["HF_HUB_OFFLINE"] = "0"
    constants = restore = None
    try:
        if snapshot is None:
            # The env var alone is not enough: huggingface_hub reads it ONCE at
            # import into a module constant, and harness.env has already set it
            # by then. Set both, and put both back.
            from huggingface_hub import constants
            from huggingface_hub import snapshot_download as snapshot
            restore = constants.HF_HUB_OFFLINE
            constants.HF_HUB_OFFLINE = False
        return str(snapshot(repo_id=repo))
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"{repo}: {str(exc)[:200]}") from exc
    finally:
        if constants is not None:
            constants.HF_HUB_OFFLINE = restore
        if was is None:
            os.environ.pop("HF_HUB_OFFLINE", None)
        else:
            os.environ["HF_HUB_OFFLINE"] = was


def run(conn, sizes: dict[str, int], *, limit: int = 1, snapshot=None,
        free: int | None = None) -> list[dict]:
    """Fetch up to `limit` queued candidates, recording what happened.

    A refused plan is recorded as `declined`, terminal, so the same oversized
    model is not re-queued every sweep. A failed download is NOT: a network
    error says nothing about the candidate.
    """
    done = []
    fetched = 0
    for row in queued(conn):
        if not row.get("lane"):
            continue      # nothing here could measure it, so nothing fetches it
        # `limit` bounds DOWNLOADS, not decisions. Counting refusals against it
        # let one unsized entry at the head of the queue consume the whole
        # budget, so nothing was ever fetched.
        if fetched >= limit:
            break
        name = row["resolved"] or row["name"]
        p = plan(name, sizes.get(name, 0), free=free)
        if not p.ok:
            ms.decide(conn, row["name"], "declined", tier="fetch", detail=p.why)
            done.append({"repo": name, "ok": False, "why": p.why})
            continue
        try:
            where = download(name, snapshot=snapshot)
        except FetchError as exc:
            done.append({"repo": name, "ok": False, "why": str(exc)})
            continue
        fetched += 1
        ms.decide(conn, row["name"], "queued", tier="fetch",
                  detail=f"downloaded to {where}", run_path=where)
        done.append({"repo": name, "ok": True, "why": where})
    return done
