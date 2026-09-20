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
import shutil
import subprocess
from dataclasses import dataclass, field
from pathlib import Path

from harness import lanes
from harness import memory_store as ms

GIB = 1024 ** 3
#: Weights above this cannot run here. 32 GB unified, and macOS gives the GPU
#: roughly 70-75% of it by default, so the honest ceiling is well under 32.
#: Overridable: the M5 Ultra arriving with 96 GB moves this, nothing else.
#:
#: IT IS A WEIGHT CEILING, NOT A JOB CEILING, and the difference bit once. The
#: default image job peaks at 11.4 GiB at 512x512 and 23.9 at 1024 (M2 Pro,
#: 2026-09-12), so while the CLI inherited mflux's 1024 the tool's own default
#: exceeded the gate it screens candidates with. #157 gave the CLI a default of
#: 512 so both numbers come from the same exam.
MEMORY_CEILING = 22 * GIB


def ceiling_bytes(acc=None) -> int:
    """The largest weight this machine could load at all, in bytes.

    A CONSTANT CANNOT ANSWER THIS ANY MORE. MEMORY_CEILING describes one 32 GB
    mini; a 12 GB card and a 24 GB card give the same candidate opposite
    verdicts, and both are Windows. So a discrete card is asked directly and
    its VRAM is the wall.

    Unified memory deliberately keeps the measured constant. Deriving it too
    (32 * GPU_FRACTION is 24 GiB) would move an Apple Silicon machine's ceiling from 22 to 24
    and change which candidates it accepts -- a decision about the Mac, which a
    Windows port has no business making on its way past.
    """
    from harness import machine

    # Through machine.detect(), which is cached: memory.detect() shells out to
    # sysctl or nvidia-smi, and decide() calls this once per candidate. A
    # sensitivity sweep spawned about five hundred of them.
    acc = machine.detect().accelerator if acc is None else acc
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

#: Every verdict decide() can return. The needs-* half is DERIVED, because a
#: hand-written list said "needs-cuda" long after decide() had learned to say
#: needs-mlx and needs-rocm, and a stale list of this kind reads as
#: authoritative.
def verdicts() -> tuple[str, ...]:
    from harness import machine
    return (("fits", "too-big", "no-entry-point", "dead", "unknown")
            + tuple(f"needs-{r}" for r in sorted(machine._RUNTIME_PROBES)))

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
             "text-to-image": "image", "diffusion": "image",
             # A text-to-SVG model IS a text-generation model, so its
             # pipeline_tag is honest and useless for routing. #246.
             "svg": "svg", "text-to-svg": "svg", "image-to-svg": "svg",
             "text-to-music": "music", "music": "music"}

#: The one pipeline_tag that is a SUPERTYPE of other lanes, so a more specific
#: tag on the same card may overrule it.
#:
#: `text-generation` covers code, web, svg and extract at once -- they are all
#: one prompt to a text model -- so a publisher naming it is telling the truth
#: and telling us nothing. Every other entry in PIPELINE_LANES already names
#: exactly one lane and is never overridden. #246.
GENERIC_PIPELINE = "text-generation"


def lane_for(meta: dict, prose: str = "") -> str:
    """Which lane could measure this, or "" when nothing here can.

    Empty is not a rejection. It means the eval suite has no case, no runner
    and no metric for this kind of model, which is a gap in the harness and
    sometimes the work worth doing (language ID is issue #2).

    THE REGISTRY OUTRANKS THE PROSE. `pipeline_tag` is the publisher's own
    answer; `prose` is a one-line description written by whoever mentioned it,
    and is read only when the registry said nothing. Against the 37 store rows
    where both speak, they agree 34 times. #207.
    """
    tag = (meta.get("pipeline_tag") or "").strip().lower()
    from_tags = {TAG_LANES[t] for t in
                 (str(x).strip().lower() for x in (meta.get("tags") or []))
                 if t in TAG_LANES}
    if tag in PIPELINE_LANES:
        lane = PIPELINE_LANES[tag]
        # REGISTRY OVER REGISTRY, BEFORE REGISTRY OVER PROSE. `text-generation`
        # is a supertype: OmniSVG1.1_8B declares it and is tagged SVG,
        # Image-to-SVG and Text-to-SVG, and was filed under `code`, where the
        # lane would hand it to mlx_lm.server. The publisher is not wrong; the
        # specific answer is simply elsewhere on the same card.
        #
        # ONE specific lane overrules, several do not: tags naming two lanes
        # is the ambiguity lanes.from_prose already refuses to resolve, and
        # the publisher's own answer is a better fallback than a coin flip.
        specific = from_tags - {lane}
        if tag == GENERIC_PIPELINE and len(specific) == 1:
            return specific.pop()
        return lane
    if len(from_tags) == 1:
        return from_tags.pop()
    if from_tags:
        return ""      # several lanes named and none of them the publisher's
    return lanes.from_prose(prose)

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


class Gone(InspectError):
    """The registry says this id does not exist, as opposed to could not be
    reached. The two need opposite handling: a 404 settles a candidate, an
    unreachable registry must settle nothing, or one bad afternoon marks a
    sweep's worth of real models as broken."""


@dataclass
class Fit:
    repo: str
    #: Which registry answered for this name, as memory_store spells it. A
    #: source clone and a model card are two different readings and the verdict
    #: rules differ between them, so the Fit says which one it is.
    registry: str = "github"
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
    #: Its weights are GGUF, which only llama.cpp loads. A format is not a
    #: runtime and this field is not one either: it is the evidence that the
    #: `llamacpp` runtime is required, resolved against the machine in
    #: decide(). Issue #228.
    gguf: bool = False
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
    try:
        proc = subprocess.run(argv, cwd=cwd, capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        # A TimeoutExpired is not an InspectError, so it escaped the caller's
        # per-candidate guard and took the whole batch down with it. Issue #181.
        raise InspectError(f"{argv[0]}: no answer in {timeout:.0f}s") from None
    if proc.returncode != 0:
        raise InspectError(f"{argv[0]}: {proc.stderr.strip()[:200]}")
    return proc.stdout


def cloned(dest: Path) -> bool:
    """Did a clone finish? A killed one leaves a .git and no worktree."""
    dest = Path(dest)
    return (dest / ".git").is_dir() and any(
        p.name != ".git" for p in dest.iterdir())


def clone(repo: str, dest: Path, run=_run) -> Path:
    """Shallow, single branch, no tags, no history. Source only."""
    dest = Path(dest)
    if dest.exists():
        # A HALF-CLONE READS AS ZERO FILES, and zero files is a verdict of no
        # mlx, no cuda, no weights -- confidently wrong rather than loud.
        if cloned(dest):
            return dest
        shutil.rmtree(dest, ignore_errors=True)
    try:
        run(["git", "clone", "--depth", "1", "--single-branch", "--no-tags",
             f"https://github.com/{repo}.git", str(dest)])
    except InspectError:
        shutil.rmtree(dest, ignore_errors=True)
        raise
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


#: What the registry is asked for one model. `blobs=true` is what carries the
#: per-file sizes; without it `siblings` is a list of names and nothing can be
#: weighed.
MODEL_URL = "https://huggingface.co/api/models/{model_id}?blobs=true"


def hf_model(model_id: str, fetch=None) -> dict:
    """One model's registry record, or a raised reason it has none.

    Gone and InspectError are kept apart on purpose: `gh api` taught this the
    same lesson (github.NotFound), and the cost of confusing them is a sweep
    that empties itself during an outage.
    """
    from harness import feeds
    if fetch is None:
        def fetch(url):
            return feeds.fetch(url, retries=SIZE_RETRIES, delay=SIZE_DELAY)
    try:
        return json.loads(fetch(MODEL_URL.format(model_id=model_id)))
    except Exception as exc:  # noqa: BLE001 - classified, then re-raised
        detail = str(exc)
        if "HTTP 404" in detail:
            raise Gone(f"{model_id}: the registry has no model by that name")
        # 401 is a GATED model, which exists. Settling it as missing would
        # throw away exactly the candidates that need a licence click.
        raise InspectError(f"{model_id}: {detail[:160]}") from exc


#: Tags that say nothing about whether a model is worth measuring. Licences,
#: regions and file formats are facts about paperwork and packaging.
_NOISE_TAGS = re.compile(
    r"^(license|licence|region|arxiv|doi|dataset|language|base_model:quantized|"
    r"autotrain|endpoints_compatible|text-generation-inference|"
    r"safetensors|gguf|pytorch|transformers|diffusers|onnx|tensorboard|"
    r"has_space|model[-_]index|conversational|en|zh|multilingual)")


#: How a repo says its weights are GGUF. The library and the tag are the
#: registry's own answer; a `.gguf` file in the listing is the thing itself.
#: The NAME is last and narrowest -- `-GGUF` or `_GGUF` as a suffix or segment
#: -- because a substring match calls `org/ggufmaker-tools` a GGUF repo.
_GGUF_NAME = re.compile(r"(^|[-_.])gguf($|[-_.])", re.I)

#: llama.cpp's quantisation labels: Q4_K_M, Q8_0, Q1_0, IQ2_XS. Distinctive
#: because MLX spells its quantisations `-4bit`, `-8bit`, `-bf16`, with no
#: underscore. Across the whole store this matches exactly one repo, and that
#: one is a GGUF whose name says so no other way.
#:
#: THE FREE-TEXT DESCRIPTION IS DELIBERATELY NOT READ. Matching `gguf` in
#: prose flags `unslothai/unsloth`, a training tool that merely SUPPORTS GGUF
#: export, and refusing a tool for the formats it can write is the opposite of
#: the question. Only the registry's structured fields and the name.
_GGUF_QUANT = re.compile(r"(^|[-_.])i?q[0-9]+_[0-9a-z]+", re.I)


def is_gguf(model_id: str, data: dict) -> bool:
    """Are this repo's weights GGUF, which only llama.cpp loads? #228."""
    if (data.get("library_name") or "").strip().lower() == "gguf":
        return True
    if any(str(t).strip().lower() == "gguf" for t in (data.get("tags") or [])):
        return True
    if any(str(f.get("rfilename", "")).lower().endswith(".gguf")
           for f in (data.get("siblings") or [])):
        return True
    name = model_id or ""
    return bool(_GGUF_NAME.search(name) or _GGUF_QUANT.search(name))


#: Kernel and format words that mean NVIDIA hardware, whatever the card's
#: `library_name` says. Deliberately NARROW: every one of these names a CUDA
#: kernel or an NVIDIA-only numeric format, so a match is a fact about the
#: weights rather than a guess.
#:
#: `awq` and `gptq` are DELIBERATELY ABSENT. They are quantisation formats with
#: CUDA kernels in practice and implementations elsewhere, so refusing them
#: would be predicting a failure rather than reading one.
_NEEDS_CUDA = re.compile(
    r"\b(cuda|tensorrt|gemlite|nvfp4|modelopt|marlin|exllama|bitsandbytes)\b",
    re.I)

#: The card's own `library_name`, which describes what SERVES the weights
#: rather than what they are. `served by vllm` needs a vLLM server; that is a
#: runtime a machine either has or does not, exactly like llamacpp.
_SERVED_BY = re.compile(r"served by (\w[\w.-]*)", re.I)

#: library_name -> the runtime this project probes for. Only the ones where
#: the serving framework IS the runtime; `transformers` is absent because
#: torch runs everywhere and the question it raises -- what conversion costs
#: on this machine -- is a different one (#245).
_SERVED_RUNTIME = {"vllm": "vllm"}


def runtime_needed(description: str) -> str:
    """The runtime this card says its weights need, or "".

    READ FROM THE REGISTRY'S OWN TAGS, the same source screen.is_attachment
    uses to answer "is this a model at all". This answers the next question:
    is it a model THIS machine can run.

    Returns a runtime name for machine.refuses() rather than a verdict, so the
    answer stays a fact about the machine asking. The same gemlite weights are
    perfectly runnable on a box with a card.
    """
    text = description or ""
    if _NEEDS_CUDA.search(text):
        return "cuda"
    served = _SERVED_BY.search(text)
    if served:
        return _SERVED_RUNTIME.get(served.group(1).lower(), "")
    return ""


def screen_words(tag: str) -> bool:
    """Does this tag say the thing ATTACHES to a model rather than being one?

    Asks screen.NOT_A_MODEL rather than keeping a second copy of it: two lists
    of what an adapter looks like is the defect this project has filed under
    other names six times.
    """
    from harness import screen
    return bool(screen.is_attachment(tag))


def card_description(data: dict) -> str:
    """What a model card says that bears on whether to measure this.

    THE JUDGE SCORES A DESCRIPTION, and with only a name to read it gave six
    models with known opposite outcomes the same score. Issue #175.

    Deliberately NOT downloads or likes. Popularity is anti-correlated with
    novelty -- the thing worth finding is by definition under-discussed at the
    moment it matters -- and a rubric handed a download count will rank the
    most-downloaded re-upload above anything new.

    WHAT IS KEPT is what a person would read to decide: the task, the runtime,
    what it was built from, and how big it is. `base_model:` lineage in
    particular separates a genuine model from a requantised copy of one, which
    is most of what a sweep finds.
    """
    tags = [str(t).strip() for t in (data.get("tags") or []) if str(t).strip()]
    lineage = [t for t in tags if t.lower().startswith("base_model")]
    plain = [t for t in tags
             if not _NOISE_TAGS.match(t.lower()) and t not in lineage]
    bits = []
    if data.get("pipeline_tag"):
        bits.append(f"task {data['pipeline_tag']}")
    if data.get("library_name"):
        bits.append(f"served by {data['library_name']}")
    if plain:
        # A TAG THAT DECIDES WHETHER THIS IS A MODEL AT ALL SURVIVES THE CAP.
        # `Xanthius/Ace-Step-1.5-XL-Concept-Sliders` is tagged music, audio,
        # sound, singing, concepts, slider, LORA -- and `lora` is the seventh,
        # so plain[:6] dropped the only word that mattered. It then ranked top
        # of the queue at +5.3 and would have spent a fetch and a screen on an
        # adapter no runner can load.
        decisive = [t for t in plain if screen_words(t)]
        rest = [t for t in plain if t not in decisive]
        bits.append("tagged " + ", ".join(decisive + rest[:6]))
    if lineage:
        # THE RELATION TYPE IS THE DISCRIMINATOR, so it travels with the id.
        # HF types these -- base_model:adapter:X, :finetune:X, :quantized:X --
        # and stripping the prefix threw away the one field that separates a
        # thing that runs from a thing that attaches to something that runs.
        parents = sorted({t.split(":")[-1] for t in lineage})
        kinds = {t.lower().split(":")[1] for t in lineage
                 if t.lower().count(":") >= 2}
        how = "adapter of" if "adapter" in kinds else "built from"
        bits.append(f"{how} " + ", ".join(parents[:3]))
    total = sum(s.get("size") or 0 for s in (data.get("siblings") or []))
    if total > 0:
        bits.append(f"{total / GIB:.1f} GiB of weights")
    return "; ".join(bits)[:300]


def inspect_model(model_id: str, *, data: dict | None = None, fetch=None,
                  ceiling: int | None = None, dead_days: int = DEAD_DAYS,
                  machine=None) -> Fit:
    """Read a HuggingFace model card and say whether it can run here.

    THE OTHER HALF OF THE TIER. `inspect()` clones a source tree from GitHub,
    which is the only thing it can do, and the sweep proposes HuggingFace ids
    exclusively -- so 227 of 235 candidates came back 404 from an API that was
    never going to have heard of them. Issue #167.

    Nothing is cloned and nothing is downloaded: one registry call, the same
    one hf_facts() already makes, read for size, lane and recency.
    """
    data = hf_model(model_id, fetch=fetch) if data is None else data
    fit = Fit(repo=model_id, registry=ms.HUGGINGFACE)
    sizes = [s.get("size") or 0 for s in (data.get("siblings") or [])]
    total = sum(sizes)
    if total > 0:
        fit.weights[model_id] = total
        fit.largest = fit.smallest = total
    else:
        # Not zero. A card that lists no blob sizes is unmeasured, and an
        # unknown size reported as zero reads as "small enough".
        fit.unsized.append(model_id)
    fit.headline = [model_id]
    lane = lane_for(data)
    if lane:
        fit.lanes[model_id] = lane
    tags = [str(t).lower() for t in (data.get("tags") or [])]
    fit.mlx = (data.get("library_name") or "").lower() == "mlx" or "mlx" in tags
    fit.gguf = is_gguf(model_id, data)
    fit.last_commit = (data.get("lastModified") or "").strip()
    fit.description = card_description(data)
    return decide(fit, ceiling=ceiling, dead_days=dead_days, machine=machine)


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
    if fit.gguf:
        offered.append("llamacpp")

    # A repo offering more than one runs wherever ONE of them lands. An MLX
    # import beside a CUDA pin is a project with two paths, and each machine
    # has one of them; refusing it on either was the old rule's mistake in the
    # one case it got right for the wrong reason.
    if offered and all(machine.refuses(r) for r in offered):
        # A repo offering SEVERAL runtimes, on a machine with none of them, has
        # no single missing runtime to name. Reporting the first one in the
        # list read as "needs-mlx" beside a reason listing CUDA packages, which
        # is a verdict recorded as terminal in the store and never revisited.
        fit.verdict = machine.refuses(offered[0])
        named = (", ".join(fit.cuda[:3]) if "cuda" in offered
                 else "GGUF weights" if offered[0] == "llamacpp" else "mlx")
        if len(offered) > 1:
            fit.why = (f"depends on {named}, and this machine has none of "
                       f"{', '.join(offered)}")
        else:
            fit.why = f"depends on {named}, and this machine has no {offered[0]}"
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
    # A WEIGHTS REPO HAS NOTHING TO CALL BY CONSTRUCTION, so this rule belongs
    # to source trees only. Applying it to a model card refuses every model in
    # the registry for not being a program.
    if not fit.entry_points and fit.registry != ms.HUGGINGFACE:
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
