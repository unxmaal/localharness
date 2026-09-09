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


def ceiling_bytes(acc=None) -> int:
    """The largest weight this machine could load at all, in bytes.

    A CONSTANT CANNOT ANSWER THIS ANY MORE. MEMORY_CEILING describes one 32 GB
    mini; a 12 GB card and a 24 GB card give the same candidate opposite
    verdicts, and both are Windows. So a discrete card is asked directly and
    its VRAM is the wall.

    Unified memory deliberately keeps the measured constant. Deriving it too
    (32 * GPU_FRACTION is 24 GiB) would move the mini's ceiling from 22 to 24
    and change which candidates it accepts -- a decision about the Mac, which a
    Windows port has no business making on its way past.
    """
    from harness import memory

    acc = memory.detect() if acc is None else acc
    if acc.kind == "discrete":
        return int(acc.total_gb * GIB)
    return MEMORY_CEILING
#: A "source" repo past this is carrying weights or datasets in git, and
#: cloning it is the download this tier exists to avoid.
CLONE_KB_CAP = 250_000
#: No commits in this long and it is not where this month's technique lives.
DEAD_DAYS = 730
#: How many named weights to size. A repo that lists a hundred models is
#: showing a CATALOGUE, and the smallest few decide the verdict anyway, so
#: sizing all of them buys nothing and costs an hour.
SIZE_LIMIT = 12
#: The registry 429s under a sweep. Retrying at the feed cadence (4 tries, 20s
#: apart) turns one repo naming 130 models into three hours, which defeats a
#: tier whose whole justification is that it costs seconds. One quick retry,
#: then record the size as unknown and move on.
SIZE_RETRIES = 1
SIZE_DELAY = 2.0

VERDICTS = ("fits", "too-big", "needs-cuda", "no-entry-point", "dead", "unknown")

#: HuggingFace's own task label -> the lane that can actually measure it.
#: `pipeline_tag` is frequently absent (3 of 6 real models checked), so the
#: repo's free-text tags are read too. Anything not here has NO LANE, and that
#: is a fact about this harness rather than about the model: silero-vad and
#: MossFormer2 are both good and neither can be scored by anything here.
PIPELINE_LANES = {
    "automatic-speech-recognition": "stt", "text-to-speech": "tts",
    "text-to-audio": "tts", "text-to-image": "image",
    "text-to-video": "video", "image-to-video": "video",
    "text-generation": "code",
}
TAG_LANES = {"asr": "stt", "speech-recognition": "stt", "stt": "stt",
             "tts": "tts", "text-to-speech": "tts",
             "text-to-image": "image", "diffusion": "image"}


def lane_for(meta: dict) -> str:
    """Which lane could measure this, or "" when nothing here can.

    Empty is not a rejection. It means the eval suite has no case, no runner
    and no metric for this kind of model, which is a gap in the harness and
    sometimes the work worth doing (language ID is issue #2).
    """
    tag = (meta.get("pipeline_tag") or "").strip().lower()
    if tag in PIPELINE_LANES:
        return PIPELINE_LANES[tag]
    for t in (meta.get("tags") or []):
        got = TAG_LANES.get(str(t).strip().lower())
        if got:
            return got
    return ""

#: Imports and pins that mean it will not run on this machine at all.
#: NOT awq, gptq or vllm: awq is a QUANTISATION FORMAT that mlx-lm implements
#: natively, and matching it flagged ml-explore/mlx-lm as CUDA-dependent on the
#: strength of an entry point named `mlx_lm.awq`.
CUDA_MARKERS = re.compile(
    r"\b(torch\.cuda|cuda_is_available|bitsandbytes|flash[_-]attn|xformers|"
    r"triton|nvidia-[a-z0-9-]+|tensorrt|cupy|deepspeed)\b", re.I)
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
#: requirements-dev.txt is deliberately absent: a dev requirement is not a
#: runtime one, and this list is read to decide whether a thing can RUN here.
DEPENDENCY_FILES = {"requirements.txt", "pyproject.toml", "setup.py",
                    "setup.cfg", "environment.yml", "Pipfile", "poetry.lock",
                    "Package.resolved"}
#: Sections of a dependency file that are NOT runtime requirements. Reading a
#: pyproject wholesale called starvector CUDA-dependent partly on `deepspeed`,
#: which sits in `[project.optional-dependencies] train` -- a TRAINING extra.
#: The verdict happened to be right for another reason, which is worse than
#: being wrong: it hid the defect. Third time this rule has needed narrowing,
#: after largest-vs-smallest weight and mention-vs-dependency.
OPTIONAL_SECTIONS = re.compile(
    r"^\s*\[(project\.optional-dependencies|tool\.poetry\.(group|dev-dependencies)"
    r"[^\]]*|options\.extras_require)\]", re.M)
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
    #: model id -> the lane that could measure it, "" when nothing here can.
    lanes: dict[str, str] = field(default_factory=dict)
    #: Named weights ranked by how likely each is the thing the repo is FOR.
    headline: list[str] = field(default_factory=list)
    #: CUDA named in code but NOT in any dependency file. Weaker evidence: a
    #: `torch.cuda.is_available()` guard is compatible with running elsewhere.
    cuda_mentioned: list[str] = field(default_factory=list)
    mlx: bool = False
    mps: bool = False
    cuda: list[str] = field(default_factory=list)
    entry_points: list[str] = field(default_factory=list)
    last_commit: str = ""
    source_kb: int = 0
    #: GitHub's own one-line description, carried so the judge can be shown the
    #: prose AND the source facts in one place.
    description: str = ""

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
    counts: dict[str, int] = {}
    in_readme: set[str] = set()
    mlx = mps = False
    entries: list[str] = []
    for p in files(tree):
        try:
            text = p.read_text(errors="ignore", encoding="utf-8")
        except OSError:
            continue
        if p.name in DEPENDENCY_FILES:
            # Everything from the first optional/extras section onward is not
            # a runtime requirement.
            cut = OPTIONAL_SECTIONS.search(text)
            runtime, extra = ((text[:cut.start()], text[cut.start():])
                              if cut else (text, ""))
            required |= {m.group(0).lower()
                         for m in CUDA_MARKERS.finditer(runtime)}
            mentioned |= {m.group(0).lower()
                          for m in CUDA_MARKERS.finditer(extra)}
        else:
            mentioned |= {m.group(0).lower() for m in CUDA_MARKERS.finditer(text)}
        mlx = mlx or bool(MLX_MARKERS.search(text))
        mps = mps or bool(MPS_MARKERS.search(text))
        rel = p.relative_to(tree).as_posix()
        readme = p.name.lower().startswith("readme")
        if readme or p.suffix in (".py", ".json", ".yaml", ".yml", ".toml",
                                  ".swift"):
            found = [m.group(1) for m in HF_ID.finditer(text)
                     if m.group(1).split("/")[0].lower() not in NOT_WEIGHTS]
            for i in found:
                counts[i] = counts.get(i, 0) + 1
                if readme:
                    in_readme.add(i)
            ids |= set(found)
        if rel == "pyproject.toml" and "[project.scripts]" in text:
            entries.append("pyproject scripts")
        if "__main__" in text and p.suffix == ".py":
            entries.append(rel)
        if p.name in ("Package.swift", "setup.py", "Makefile"):
            entries.append(rel)
    return {"cuda": sorted(required), "cuda_mentioned": sorted(mentioned - required),
            "hf_ids": sorted(ids), "counts": counts, "in_readme": sorted(in_readme),
            "mlx": mlx, "mps": mps, "entry_points": sorted(set(entries))[:8]}


def _sizes_path() -> Path:
    from harness import paths
    d = paths.home() / "cache" / "github"
    d.mkdir(parents=True, exist_ok=True)
    return d / "hf-sizes.json"


def _size_cache() -> dict:
    try:
        return json.loads(_sizes_path().read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def headline(repo: str, ids, counts: dict, in_readme) -> list[str]:
    """The models a repo is FOR, best first, separated from the ones it merely
    touches.

    The first queue built from "smallest named weight" filled with tokenizers,
    speaker-embedding helpers and a 0.6B somebody used in a test, because the
    smallest id in a repo is almost never the headline. Three signals, none
    conclusive alone:

      README        a model named where the project introduces itself
      repetition    the model a repo is ABOUT is named again and again
      name overlap  mlx-video naming Lightricks/LTX-2 beats it naming umt5

    A ranking, not a filter: everything is still returned, in order.
    """
    want = {t for t in re.split(r"[^a-z0-9]+", repo.lower()) if len(t) > 2}
    readme = set(in_readme)

    def score(i: str) -> tuple:
        tokens = {t for t in re.split(r"[^a-z0-9]+", i.lower()) if len(t) > 2}
        return (3 * (i in readme) + min(counts.get(i, 1), 5)
                + 2 * bool(tokens & want), -len(i))
    return sorted(ids, key=score, reverse=True)


def hf_facts(model_id: str, fetch=None, cache: dict | None = None) -> dict:
    """Total bytes and measurable lane, from ONE registry call.

    Both come out of the same response, and the registry rate-limits, so
    asking twice for one model is a request spent on nothing.
    """
    from harness import feeds
    if cache is not None and model_id in cache:
        got = cache[model_id]
        if isinstance(got, dict):
            return got
        # A size-only entry predates lanes. Returning it would report every
        # already-sized model as unmeasurable, which is how this first read:
        # 0 queued and 14 orphans, several of them plainly STT and image
        # models. Fall through and upgrade the entry instead.
    if fetch is None:
        def fetch(url):
            return feeds.fetch(url, retries=SIZE_RETRIES, delay=SIZE_DELAY)
    url = f"https://huggingface.co/api/models/{model_id}?blobs=true"
    try:
        data = json.loads(fetch(url))
    except Exception:  # noqa: BLE001 - unreachable is unknown, never zero
        return {"size": -1, "lane": ""}
    sizes = [s.get("size") or 0 for s in data.get("siblings", [])]
    out = {"size": sum(sizes) if sizes else -1, "lane": lane_for(data)}
    # Only a real answer is worth keeping. Caching a failure would freeze a
    # rate-limit into a permanent "unknown".
    if cache is not None and out["size"] > 0:
        cache[model_id] = out
    return out


def hf_size(model_id: str, fetch=None, cache: dict | None = None) -> int:
    """Total bytes of a HuggingFace repo, or -1 when it cannot be told.

    -1 rather than 0, and callers must keep the two apart: an unknown size
    reported as zero reads as "small enough", which is the opposite of what is
    known. HuggingFace rate-limits this endpoint to 429 under a sweep, so it
    goes through the feed fetcher, which already retries with backoff.
    """
    return hf_facts(model_id, fetch=fetch, cache=cache)["size"]


def decide(fit: Fit, ceiling: int | None = None, dead_days: int = DEAD_DAYS,
           now: float | None = None, machine=None) -> Fit:
    """Turn what was read into one verdict and the reason for it.

    Order matters. The runtime comes first because it is absolute: a repo that
    cannot run here at any size is not a size question.

    THE VERDICT NAMES THE RUNTIME, NOT THE PLATFORM. This used to read "CUDA is
    absolute on this machine", which was true of the Mac it was written on and
    made every CUDA candidate a rejection on a box bought to run them, while an
    MLX repo that cannot start there passed the same gate. Asking which runtime
    a candidate needs, and whether this machine has it, reads correctly from
    either direction and makes a third kind of machine a row of data.
    """
    import time
    from datetime import datetime, timezone

    # Resolved here rather than as a default argument: a default is bound at
    # import time, which would pin the ceiling to whatever machine imported
    # the module first.
    if ceiling is None:
        ceiling = ceiling_bytes()
    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()

    # Which runtimes the repo OFFERS. Only a DECLARED dependency counts:
    # apple/coreai-models mentions torch.cuda in one export recipe and is an
    # Apple on-device repo, and calling that "needs CUDA" threw away the most
    # relevant candidate in a sweep. ml-explore/mlx itself was reported the
    # same way, from nvidia-* inside `if toolkit == 12:`.
    offered = []
    if fit.mlx:
        offered.append("mlx")
    if fit.cuda:
        offered.append("cuda")

    # A repo offering more than one runs wherever ONE of them lands. An MLX
    # import beside a CUDA pin is a project with two paths, and each machine
    # has one of them; refusing it on either was the old rule's mistake in the
    # one case it got right for the wrong reason.
    if offered and all(machine.refuses(r) for r in offered):
        fit.verdict = machine.refuses(offered[0])
        detail = ", ".join(fit.cuda[:3]) if "cuda" in offered else "mlx"
        fit.why = f"depends on {detail}, and this machine has no {offered[0]}"
        return fit

    # Kept as a mention rather than a requirement once the machine can satisfy
    # it: the field is read downstream as evidence, and a satisfied dependency
    # is not evidence against.
    if fit.cuda and "cuda" not in machine.runtimes:
        fit.cuda_mentioned = sorted(set(fit.cuda_mentioned) | set(fit.cuda))
        fit.cuda = []
    # THE SMALLEST decides, not the largest, and this was measured the hard
    # way: Blaizzy/nativ names a 1774 GiB model and was reported as too big for
    # this machine. It is a Mac app with a CATALOGUE of models it can serve.
    # Source cannot tell a requirement from an option, so the honest question
    # is whether ANYTHING it names could run here.
    # A truncated size scan cannot support "too-big": the smallest of twelve
    # sized ids out of nearly three hundred named is an upper bound on the
    # floor, not the floor. Blaizzy/mlx-video was refused on exactly this.
    if fit.smallest > ceiling and len(fit.unsized) > len(fit.weights):
        fit.verdict = "unknown"
        fit.why = (f"smallest of {len(fit.weights)} sized weights is "
                   f"{fit.smallest / GIB:.1f} GiB, but {len(fit.unsized)} more "
                   f"were never sized, so the floor is not known")
        return fit
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


def _takes_cache(fn) -> bool:
    import inspect as _i
    try:
        return "cache" in _i.signature(fn).parameters
    except (TypeError, ValueError):
        return False


def inspect(repo: str, workdir: Path, *, meta: dict | None = None,
            sizer=None, facts=hf_facts, run=_run,
            ceiling: int | None = None,
            dead_days: int = DEAD_DAYS,
            machine=None,
            kb_cap: int = CLONE_KB_CAP) -> Fit:
    """Clone a candidate's source, read it, and say whether it can run here."""
    if sizer is not None:      # older callers and tests pass a size-only stub
        def facts(model_id, cache=None):
            n = (sizer(model_id, cache=cache) if _takes_cache(sizer)
                 else sizer(model_id))
            return {"size": int(n), "lane": ""}
    fit = Fit(repo=repo, source_kb=int((meta or {}).get("size") or 0),
              description=((meta or {}).get("description") or "")[:200])
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
    cache = _size_cache()
    before = len(cache)
    fit.headline = headline(repo, found["hf_ids"], found.get("counts") or {},
                            found.get("in_readme") or [])
    # HALF the budget to the smallest ids and half to the headline ones. The
    # smallest decide whether anything here can run at all; the headline ones
    # decide what is worth downloading, and they are rarely the same models.
    half = max(1, SIZE_LIMIT // 2)
    picked = list(dict.fromkeys(
        sorted(found["hf_ids"], key=len)[:half] + fit.headline[:half]))
    for model_id in picked:
        got = (facts(model_id, cache=cache) if _takes_cache(facts)
               else facts(model_id))
        size = got["size"] if isinstance(got, dict) else int(got)
        if isinstance(got, dict) and got.get("lane"):
            fit.lanes[model_id] = got["lane"]
        if size > 0:
            fit.weights[model_id] = size
        else:
            fit.unsized.append(model_id)
    fit.unsized += [i for i in found["hf_ids"] if i not in picked]
    if len(cache) > before:
        try:
            _sizes_path().write_text(json.dumps(cache, indent=1, sort_keys=True), encoding="utf-8")
        except OSError:
            pass
    fit.largest = max(fit.weights.values(), default=0)
    fit.smallest = min(fit.weights.values(), default=0)
    return decide(fit, ceiling=ceiling, dead_days=dead_days, machine=machine)
