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


def queued(conn, tiers=FETCHABLE_TIERS) -> list[dict]:
    """Candidates the inspect tier queued, best score first."""
    rows = conn.execute("""
        SELECT p.name, p.resolved,
               (SELECT v.score FROM verdicts v WHERE v.proposal_id = p.id
                 AND v.score IS NOT NULL ORDER BY v.id DESC LIMIT 1) AS score,
               (SELECT v.outcome FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS outcome,
               (SELECT v.tier FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS tier
        FROM proposals p""").fetchall()
    out = [dict(r) for r in rows
           if r["outcome"] == "queued" and r["tier"] in tiers]
    return sorted(out, key=lambda r: -(r["score"] or 0))


def download(repo: str, snapshot=None) -> str:
    """Weights into the shared cache. Returns the path."""
    if snapshot is None:
        from huggingface_hub import snapshot_download as snapshot
    try:
        return str(snapshot(repo_id=repo))
    except Exception as exc:  # noqa: BLE001
        raise FetchError(f"{repo}: {str(exc)[:200]}") from exc


def run(conn, sizes: dict[str, int], *, limit: int = 1, snapshot=None,
        free: int | None = None) -> list[dict]:
    """Fetch up to `limit` queued candidates, recording what happened.

    A refused plan is recorded as `declined`, terminal, so the same oversized
    model is not re-queued every sweep. A failed download is NOT: a network
    error says nothing about the candidate.
    """
    done = []
    for row in queued(conn):
        if len(done) >= limit:
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
        ms.decide(conn, row["name"], "queued", tier="fetch",
                  detail=f"downloaded to {where}", run_path=where)
        done.append({"repo": name, "ok": True, "why": where})
    return done
