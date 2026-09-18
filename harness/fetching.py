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


#: Config keys whose value names a repo you must ALSO have on disk. Kept as an
#: allowlist rather than "anything repo-shaped", because the first real repo
#: scanned proved that wrong. Issue #196.
REQUIRES_KEYS = ("text_tokenizer", "tokenizer_name", "audio_tokenizer",
                 "codec_model", "vocoder", "base_model")

#: Repo-shaped and NOT a requirement, with the evidence. `_name_or_path` is
#: HuggingFace boilerplate recording the checkpoint a config was derived from:
#: microsoft/wavlm-base-plus-sv names microsoft/wavlm-base-plus, which is not on
#: this machine, and the model loads anyway. Treating it as a dependency marks
#: a working model unready.
PROVENANCE_KEYS = {
    "_name_or_path": "where the config came from, not what it needs",
}

#: `org/name`, and not a path, a mime type or a ratio.
_REPO = re.compile(r"^[A-Za-z0-9][\w.-]*/[\w.-]+$")


def _config(model_id: str, root: Path | None = None) -> dict:
    """The cached config.json for a model, or {} if there is not one."""
    import json
    import os

    home = Path(root or os.environ.get("HF_HOME")
                or Path.home() / ".cache" / "huggingface")
    d = home / "hub" / f"models--{model_id.replace('/', '--')}"
    for cfg in sorted(d.glob("snapshots/*/config.json")):
        try:
            return json.loads(cfg.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
    return {}


def requires(model_id: str, root: Path | None = None) -> list[str]:
    """Other repos this model's own config says it needs.

    A REPO BEING COMPLETE IS NOT A MODEL BEING LOADABLE. Marvis-AI's 8-bit MLX
    repo is whole -- every symlink resolving, no .incomplete files -- and names
    `text_tokenizer: Marvis-AI/marvis-tts-250m-v0.2`, which is a different repo.
    With HF_HUB_OFFLINE=1 the load fails rather than fetching it. Issue #196.
    """
    data = _config(model_id, root)
    out = []
    for key in REQUIRES_KEYS:
        value = data.get(key)
        if isinstance(value, str) and _REPO.match(value) and value != model_id:
            out.append(value)
    return sorted(set(out))


def missing(model_id: str, root: Path | None = None) -> list[str]:
    """Everything this model needs that is not on disk, itself included."""
    wanted = [model_id] + requires(model_id, root)
    return [m for m in wanted if not have(m, root)]


def queued(conn, tiers=FETCHABLE_TIERS, kind: str = FETCHABLE_KIND,
           needs_lane: bool = True, lane: str = "") -> list[dict]:
    """Weights the inspect tier queued, best score first.

    A weight NO LANE CAN TEST is returned but never counted as fetchable:
    downloading is only justified by a measurement that follows it, and an
    automated loop would otherwise fill the disk with models that are
    unmeasurable by construction. Issue #81.
    """
    rows = conn.execute("""
        SELECT p.name, p.resolved, p.kind, p.lane, p.registry,
               (SELECT v.outcome FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS outcome,
               (SELECT v.tier FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS tier,
               (SELECT v.detail FROM verdicts v WHERE v.proposal_id = p.id
                 ORDER BY v.id DESC LIMIT 1) AS detail,
               -- The newest verdict that CARRIES a size, which is not always
               -- the newest verdict: a later row saying why a fetch was
               -- refused has no size in it, and neither does the newest
               -- inspect format. Issue #211.
               (SELECT v.detail FROM verdicts v WHERE v.proposal_id = p.id
                 AND (v.detail LIKE '%bytes=%' OR v.detail LIKE '%GiB%')
                 ORDER BY v.id DESC LIMIT 1) AS sized,
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
           # snapshot_download wants a HuggingFace id, and this queue once held
           # GitHub repo names: every one of them 401'd. `kind` was the only
           # thing standing between the two, and the store now says outright
           # which registry a name belongs to. An empty registry is a row from
           # before that column, so `kind` still answers for it.
           and r["registry"] != ms.GITHUB
           and not have(r["resolved"] or r["name"])]
    if lane:
        # A LANE-SCOPED LOOP MUST SCOPE THE STEP THAT SPENDS THE DISK. The loop
        # printed "(spending only on the image lane)" and then considered
        # whisper-tiny, wav2vec2, gpt2 and vicuna, because --lane reached the
        # queue report and not the fetch. Issue #211.
        from harness import lanes
        out = [r for r in out if lanes.serves(r.get("lane"), lane)]
    return sorted(out, key=lambda r: -(r["score"] or 0))


#: The two spellings the inspect tier has used for a measured size. `bytes=` is
#: exact and machine-readable; the later `fits: weights from 5.5 to 8.9 GiB` is
#: a range for humans, and it REPLACED the first without anything reading it.
_BYTES = re.compile(r"bytes=(\d+)")
_GIB = re.compile(r"([\d.]+)\s*GiB")


def size_of(row: dict) -> int:
    """The size the inspect tier measured, carried on the verdict.

    Read from the store rather than asked for again: the registry rate-limits,
    and a size already measured is a fact.

    BOTH SPELLINGS, AND THE NEWEST ROW THAT HAS ONE. queued() returns the
    latest verdict, and the latest verdict format dropped `bytes=`, so the one
    row this read was the one that does not carry the number. Every candidate
    came back unsized and was declined -- terminally -- for a size sitting in
    the row above. Issue #211.

    `bytes=` is preferred where it exists because it is the SUM of the repo's
    weights, which is what a download costs. The GiB range is smallest-to-
    largest of the individual files, so the upper bound is taken: over-
    estimating a budget refuses a fetch, and under-estimating fills a disk.
    """
    for detail in (row.get("sized") or "", row.get("detail") or ""):
        m = _BYTES.search(detail)
        if m:
            return int(m.group(1))
    for detail in (row.get("sized") or "", row.get("detail") or ""):
        found = [float(g) for g in _GIB.findall(detail)]
        if found:
            return int(max(found) * GIB)
    return 0


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


#: Reasons a fetch did not start that are about THIS HARNESS rather than about
#: the candidate. Recording one as a verdict settles a real model permanently
#: on the strength of our own gap. screen.NOT_THE_CANDIDATE is the same list
#: one tier along.
NOT_THE_CANDIDATE = ("no measured size",)


def refused_by_harness(why: str) -> str:
    """The phrase saying this refusal is ours, or "" when it is the model's."""
    text = (why or "").lower()
    for phrase in NOT_THE_CANDIDATE:
        if phrase in text:
            return phrase
    return ""


def run(conn, sizes: dict[str, int], *, limit: int = 1, snapshot=None,
        free: int | None = None, lane: str = "") -> list[dict]:
    """Fetch up to `limit` queued candidates, recording what happened.

    A refused plan is recorded as `declined`, terminal, so the same oversized
    model is not re-queued every sweep. A failed download is NOT: a network
    error says nothing about the candidate. NEITHER IS A REFUSAL THAT IS ABOUT
    THIS HARNESS: "no measured size" says the store could not answer a question
    about itself, and writing that down as declined settled 16 real candidates
    permanently, including the only upgrade the image lane had. Issue #211,
    the same class as #206 a day earlier.
    """
    done = []
    fetched = 0
    for row in queued(conn, lane=lane):
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
            outcome = "queued" if refused_by_harness(p.why) else "declined"
            ms.decide(conn, row["name"], outcome, tier="fetch", detail=p.why)
            done.append({"repo": name, "ok": False, "why": p.why})
            continue
        try:
            where = download(name, snapshot=snapshot)
        except FetchError as exc:
            done.append({"repo": name, "ok": False, "why": str(exc)})
            continue
        fetched += 1
        # WHAT IT NEEDS BESIDE ITSELF. Only readable once the config is on
        # disk, so this is after the download rather than in the plan. A model
        # whose tokenizer lives in another repo is `ready` and unloadable
        # without it, and HF_HUB_OFFLINE turns that into a hard failure at run
        # time rather than a slow first call. Issue #196.
        for dep in requires(name):
            if have(dep):
                continue
            try:
                download(dep, snapshot=snapshot)
                done.append({"repo": dep, "ok": True,
                             "why": f"needed by {name}"})
            except FetchError as exc:
                done.append({"repo": dep, "ok": False,
                             "why": f"needed by {name}: {exc}"})
        ms.decide(conn, row["name"], "queued", tier="fetch",
                  detail=f"downloaded to {where}", run_path=where)
        done.append({"repo": name, "ok": True, "why": where})
    return done
