"""Read a candidate's source before downloading its weights. Issue #61.

The tiers used to jump from JUDGE (a one-line description, no GPU, about a
second) straight to SCREEN (a real run, which for anything new means gigabytes
of weights first). This is the tier in between, and it is nearly free: a source
clone is single-digit megabytes and answers questions the judge was guessing at.

WHAT THE SOURCE SAYS THAT THE DESCRIPTION DOES NOT: which weights it loads and
how big they are, whether it needs CUDA, whether it is MLX-native or torch with
an MPS fallback, whether there is anything to call, and when it was really last
touched.

NOTHING HERE IS EXECUTED. No install, no setup.py, no import of the cloned
code. The tree is read as text and nothing else, because every byte of it is
written by strangers -- the same standing rule as the feeds.

DISK IS NOT THE CONSTRAINT: 611 GiB free on the models volume when this was
written. Memory is, and it is the thing a download cannot tell you afterwards.
"""
from __future__ import annotations

import json
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

GIB = 1024 ** 3
#: Weights above this cannot run here. 32 GB unified, and macOS gives the GPU
#: roughly 70-75% of it by default, so the honest ceiling is well under 32.
#: Overridable: the M5 Ultra arriving with 96 GB moves this, nothing else.
MEMORY_CEILING = 22 * GIB
#: A "source" repo past this is carrying weights or datasets in git, and
#: cloning it is the download this tier exists to avoid.
CLONE_KB_CAP = 250_000
#: No commits in this long and it is not where this month's technique lives.
DEAD_DAYS = 730

VERDICTS = ("fits", "too-big", "needs-cuda", "no-entry-point", "dead", "unknown")

#: Imports and pins that mean it will not run on this machine at all.
CUDA_MARKERS = re.compile(
    r"\b(torch\.cuda|cuda_is_available|bitsandbytes|flash[_-]attn|xformers|"
    r"triton|nvidia-[a-z0-9-]+|tensorrt|cupy|deepspeed|vllm|auto-?gptq|awq)\b",
    re.I)
MLX_MARKERS = re.compile(r"\b(import mlx|from mlx|mlx[_-]lm|mlx[_-]audio|"
                         r"mlx[_-]vlm|mlx\.core|MLX)\b")
MPS_MARKERS = re.compile(r"""device\s*=\s*['"]mps['"]|torch\.backends\.mps""")
#: A HuggingFace id as it appears in code: quoted, owner/name.
HF_ID = re.compile(r"""['"]([A-Za-z0-9][\w.-]{1,38}/[\w.-]{1,80})['"]""")
#: Directories that are never the project's own source. Dot-directories are
#: skipped wholesale: an agent skills folder full of helper scripts was being
#: read as the project's entry points.
SKIP_DIRS = {"node_modules", "venv", "__pycache__", "dist", "build",
             "target", "Pods", "test", "tests", "examples", "docs"}
#: Where a HARD dependency is declared, as opposed to merely mentioned.
DEPENDENCY_FILES = {"requirements.txt", "pyproject.toml", "setup.py",
                    "setup.cfg", "environment.yml", "Pipfile", "poetry.lock",
                    "requirements-dev.txt", "Package.resolved"}
READ_SUFFIXES = {".py", ".toml", ".cfg", ".txt", ".json", ".yaml", ".yml",
                 ".swift", ".md", ".sh", ".rs", ".c", ".m", ".mm", ".h"}
#: Owners that are infrastructure, not weights, so an id under them is noise.
NOT_WEIGHTS = {"actions", "github", "pypa", "astral-sh", "docker"}


class InspectError(RuntimeError):
    """The candidate could not be read. Never a verdict about the candidate."""


@dataclass
class Fit:
    repo: str
    verdict: str = "unknown"
    why: str = ""
    weights: dict[str, int] = field(default_factory=dict)
    largest: int = 0
    #: What decides whether it can run here. See decide().
    smallest: int = 0
    #: Weights it names that could not be sized. Kept apart from naming none
    #: at all: "unknown" and "none" are different answers.
    unsized: list[str] = field(default_factory=list)
    #: CUDA named in code but NOT in any dependency file. Weaker evidence: a
    #: `torch.cuda.is_available()` guard is compatible with running elsewhere.
    cuda_mentioned: list[str] = field(default_factory=list)
    mlx: bool = False
    mps: bool = False
    cuda: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    last_commit: str = ""
    source_kb: int = 0

    @property
    def largest_gib(self) -> float:
        return self.largest / GIB


def _run(argv: list[str], cwd: Path | None = None, timeout: float = 180.0) -> str:
    proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                          timeout=timeout)
    if proc.returncode != 0:
        raise InspectError(f"{argv[0]}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def clone(repo: str, dest: Path, run=_run) -> Path:
    """Shallow, single branch, no tags, no history. Source only."""
    dest = Path(dest)
    if dest.exists():
        return dest
    run(["git", "clone", "--depth", "1", "--single-branch", "--no-tags",
         f"https://github.com/{repo}.git", str(dest)])
    return dest


def files(tree: Path) -> list[Path]:
    out = []
    for p in Path(tree).rglob("*"):
        if not p.is_file() or p.suffix.lower() not in READ_SUFFIXES:
            continue
        parts = set(p.relative_to(tree).parts[:-1])
        if SKIP_DIRS & parts or any(d.startswith(".") for d in parts):
            continue
        out.append(p)
    return sorted(out)


def scan(tree: Path) -> dict:
    """Everything readable from the tree, with nothing run."""
    required: set[str] = set()
    mentioned: set[str] = set()
    ids: set[str] = set()
    mlx = mps = False
    entries: list[str] = []
    for p in files(tree):
        try:
            text = p.read_text(errors="ignore")
        except OSError:
            continue
        hits = {m.group(0).lower() for m in CUDA_MARKERS.finditer(text)}
        if p.name in DEPENDENCY_FILES:
            required |= hits
        else:
            mentioned |= hits
        mlx = mlx or bool(MLX_MARKERS.search(text))
        mps = mps or bool(MPS_MARKERS.search(text))
        rel = p.relative_to(tree).as_posix()
        if p.suffix in (".py", ".json", ".yaml", ".yml", ".toml", ".swift"):
            ids |= {m.group(1) for m in HF_ID.finditer(text)
                    if m.group(1).split("/")[0].lower() not in NOT_WEIGHTS}
        if rel == "pyproject.toml" and "[project.scripts]" in text:
            entries.append("pyproject scripts")
        if "__main__" in text and p.suffix == ".py":
            entries.append(rel)
        if p.name in ("Package.swift", "setup.py", "Makefile"):
            entries.append(rel)
    return {"cuda": sorted(required), "cuda_mentioned": sorted(mentioned - required),
            "hf_ids": sorted(ids), "mlx": mlx, "mps": mps,
            "entry_points": sorted(set(entries))[:8]}


def hf_size(model_id: str, fetch=None) -> int:
    """Total bytes of a HuggingFace repo, or -1 when it cannot be told.

    -1 rather than 0, and callers must keep the two apart: an unknown size
    reported as zero reads as "small enough", which is the opposite of what is
    known. HuggingFace rate-limits this endpoint to 429 under a sweep, so it
    goes through the feed fetcher, which already retries with backoff.
    """
    from harness import feeds
    fetch = fetch or feeds.fetch
    url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
    try:
        data = json.loads(fetch(url))
    except Exception:  # noqa: BLE001 - unreachable is unknown, never zero
        return -1
    sizes = [s.get("size") or 0 for s in data.get("siblings", [])]
    return sum(sizes) if sizes else -1


def decide(fit: Fit, ceiling: int = MEMORY_CEILING, dead_days: int = DEAD_DAYS,
           now: float | None = None) -> Fit:
    """Turn what was read into one verdict and the reason for it.

    Order matters. CUDA first because it is absolute on this machine: a repo
    that cannot run here at any size is not a size question.
    """
    import time
    from datetime import datetime, timezone
    # Only a DECLARED dependency disqualifies. apple/coreai-models mentions
    # torch.cuda in one export recipe and is an Apple on-device repo; calling
    # that "needs CUDA" threw away the most relevant candidate in the sweep.
    if fit.cuda:
        fit.verdict, fit.why = "needs-cuda", f"depends on {', '.join(fit.cuda[:3])}"
        return fit
    # THE SMALLEST decides, not the largest, and this was measured the hard
    # way: Blaizzy/nativ names a 1774 GiB model and was reported as too big for
    # this machine. It is a Mac app with a CATALOGUE of models it can serve.
    # Source cannot tell a requirement from an option, so the honest question
    # is whether ANYTHING it names could run here.
    if fit.smallest > ceiling:
        fit.verdict = "too-big"
        fit.why = (f"smallest weight it names is {fit.smallest / GIB:.1f} GiB, "
                   f"over the {ceiling / GIB:.0f} GiB ceiling")
        return fit
    if fit.last_commit:
        try:
            when = datetime.fromisoformat(fit.last_commit).timestamp()
            days = ((time.time() if now is None else now) - when) / 86400.0
            if days > dead_days:
                fit.verdict = "dead"
                fit.why = f"last commit {days / 365.0:.1f} years ago"
                return fit
        except ValueError:
            pass
    if not fit.entry_points:
        fit.verdict, fit.why = "no-entry-point", "nothing here to call"
        return fit
    if fit.unsized and not fit.weights:
        fit.verdict = "unknown"
        fit.why = (f"names {len(fit.unsized)} weight(s) and none could be "
                   f"sized, so whether it fits is not known")
        return fit
    size = (f"weights from {fit.smallest / GIB:.1f} to {fit.largest / GIB:.1f} GiB"
            if fit.largest > 0 else "no weights named in the source")
    fit.verdict = "fits"
    fit.why = f"{'MLX-native, ' if fit.mlx else ''}{size}"
    return fit


def inspect(repo: str, workdir: Path, *, meta: dict | None = None,
            sizer=hf_size, run=_run, ceiling: int = MEMORY_CEILING,
            kb_cap: int = CLONE_KB_CAP) -> Fit:
    """Clone a candidate's source, read it, and say whether it can run here."""
    fit = Fit(repo=repo, source_kb=int((meta or {}).get("size") or 0))
    if fit.source_kb and fit.source_kb > kb_cap:
        fit.verdict = "too-big"
        fit.why = (f"source tree is {fit.source_kb / 1000:.0f} MB, which is "
                   f"weights in git, not a source repo")
        return fit
    tree = clone(repo, Path(workdir) / repo.replace("/", "__"), run=run)
    found = scan(tree)
    try:
        fit.last_commit = run(["git", "log", "-1", "--format=%cI"],
                              cwd=tree).strip()
    except InspectError:
        fit.last_commit = ""
    fit.mlx, fit.mps = found["mlx"], found["mps"]
    fit.cuda, fit.entry_points = found["cuda"], found["entry_points"]
    fit.cuda_mentioned = found["cuda_mentioned"]
    for model_id in found["hf_ids"]:
        size = sizer(model_id)
        if size > 0:
            fit.weights[model_id] = size
        else:
            fit.unsized.append(model_id)
    fit.largest = max(fit.weights.values(), default=0)
    fit.smallest = min(fit.weights.values(), default=0)
    return decide(fit, ceiling=ceiling)
