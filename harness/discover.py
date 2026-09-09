"""What can this machine do that nobody has measured?

Every discovery pass in this project was a person or an agent grepping and
writing prose, which is stale the moment anything is installed and has already
missed things for months:

  * Qwen2.5 entered as a smoke test and served four lanes for four months, and
    nothing reported that the text aliases had never been compared to anything.
  * `local-small`, `q3-1.7b` and `q3-coder` were defined gateway aliases with
    ZERO measurements between them.
  * mflux ships around twenty generate entry points; the image lane has used
    two of them.
  * Kokoro exposes 54 voices; five are cached and three were compared.

A number in a document is wrong by the next commit. A command is right whenever
it is run.

`capabilities()` LOOKS INWARD, at what is installed here. `external()` looks
OUTWARD, because everything this project got wrong about candidate selection was
the world moving while nothing here noticed -- Qwen2.5 served four lanes for
four months while Qwen3 shipped, and ComfyUI was dismissed in one paragraph.

`external()` ASKS REGISTRIES, NOT A LANGUAGE MODEL. A model would produce
plausible names for repositories that do not exist, and this project has a rule
against asserting specifics from pretrained memory. An API answer is a fact with
a URL and a date attached, and it proposes CANDIDATES TO MEASURE rather than
conclusions: the eval decides, this only says what is worth putting in front of
it.

Everything degrades rather than raising: discovery runs on machines that do not
have everything, and a missing tool is a finding rather than a crash.
"""
from __future__ import annotations

import json
import os
import shutil

import httpx
from dataclasses import dataclass, field

from pathlib import Path

from harness import paths
from harness.stages import (ENTRY_POINT_STAGES, STAGE_ENTRY_POINTS,
                            stage_unavailable)

REPO = Path(__file__).resolve().parent.parent

#: Entry points that GENERATE. mflux also ships concept/capabilities/completions
#: helpers, and listing those would invite someone to evaluate a tool that
#: produces no artifact.
_GENERATE_PREFIX = "mflux-generate"
#: These generate, but not from a prompt alone: they need a control image, a
#: mask, or an existing picture. They are the WORKFLOW vocabulary -- the exact
#: thing ComfyUI is wanted for -- and mflux ships all of them natively in MLX.
#:
#: The first version of this module DROPPED them, because they cannot answer a
#: plain-prompt image case. That hid the most important gap in the project
#: behind a tidy list. They are reported as a distinct kind instead, so nobody
#: drops one into the image lane and gets a failure that means nothing.
_NEEDS_INPUT = ("controlnet", "depth", "fill", "redux", "edit", "upscal",
                "kontext", "inpaint", "img2img", "in-context", "concept",
                "refine", "inspire")


@dataclass
class Capability:
    kind: str          # model | engine | tool | method
    name: str
    lane: str
    source: str        # where it was found, so a claim can be checked
    #: The command that would measure it. A gap nobody knows how to close is a
    #: complaint rather than a finding.
    how: str
    measured: bool = False
    present: bool = True
    #: Installed but unusable, and why. Distinct from `present` and `measured`.
    blocked: str = ""
    #: How much this looks like it runs on Apple Silicon. See feeds.relevance().
    relevance: int = 0
    note: str = ""


def gateway_aliases(config: Path | None = None) -> list[Capability]:
    """Text candidates, read from the gateway config rather than guessed."""
    # THE CONFIG FOLLOWS THE MACHINE. Read unconditionally, this listed the
    # Mac's MLX aliases as a card's own text candidates: names that resolve to
    # weights it cannot load. serve-gateway.sh already takes $GATEWAY_CONFIG,
    # and reading the same variable keeps the two from disagreeing about which
    # models this machine serves.
    config = config or Path(os.environ.get(
        "GATEWAY_CONFIG", REPO / "gateway" / "config.yaml"))
    try:
        import yaml
        data = yaml.safe_load(Path(config).read_text(encoding="utf-8")) or {}
    except (OSError, ValueError):
        return []
    out = []
    for entry in data.get("model_list") or []:
        name = entry.get("model_name")
        if not name:
            continue
        upstream = (entry.get("litellm_params") or {}).get("model", "")
        out.append(Capability(
            # The path it was actually read from, since there is more than
            # one and a row claiming the wrong provenance is worse than none.
            "model", name, "text", f"{config.name} -> {upstream}",
            f"uv run python -m evals.run --modality extract --candidates {name}"))
    return out


def image_engines(bindir: Path | None = None) -> list[Capability]:
    """mflux generate entry points that take a prompt and nothing else."""
    if bindir is None:
        bindir = Path.home() / ".local/share/uv/tools/mflux/bin"
    try:
        names = sorted(p.name for p in Path(bindir).iterdir())
    except OSError:
        return []
    out = []
    for n in names:
        workflowish = any(k in n.lower() for k in _NEEDS_INPUT)
        if not (n.startswith(_GENERATE_PREFIX) or workflowish):
            continue
        if n in ("mflux-info", "mflux-capabilities", "mflux-completions",
                 "mflux-save", "mflux-train", "mflux-lora-library"):
            continue
        if workflowish:
            # Three of these have a runner; the rest are still gaps.
            stage = ENTRY_POINT_STAGES.get(n)
            if stage:
                broken = stage_unavailable(stage)
                out.append(Capability(
                    "workflow", n, "image", str(bindir),
                    f"uv run python -m evals.run --modality image "
                    f"--candidates {stage}:mflux:flux2-klein-4b",
                    blocked=broken,
                    note="runnable via ChainRunner: stage one generates, "
                         "this consumes the result"))
                continue
            out.append(Capability(
                "workflow", n, "image", str(bindir),
                "no runner yet -- needs one that supplies its input; see #24",
                note="needs an input image, mask or reference: this is the "
                     "ComfyUI-style workflow vocabulary, shipped natively"))
            continue
        spec = (n.replace(_GENERATE_PREFIX + "-", "mflux:")
                if "-" in n[len(_GENERATE_PREFIX):] else "mflux:dev")
        out.append(Capability(
            "engine", n, "image", str(bindir),
            f"uv run python -m evals.run --modality image --candidates {spec}"))
    return out


#: engine name -> (lane, the runtime it needs, an example model for the command).
#: mflux is absent on purpose: image_engines() reads its ~20 entry points,
#: including the workflow ones that need an input image, and reporting it here
#: as well would double every row on a Mac.
_ENGINE_RUNTIME = {
    "h3": ("video", "mlx", ""),
    "diffusers": ("image", "cuda", "stabilityai/sdxl-turbo"),
    "diffusers-video": ("video", "cuda", "Lightricks/LTX-Video"),
}


def generator_engines(machine=None) -> list[Capability]:
    """The engines harness/engines.py knows and this machine can invoke.

    image_engines() finds mflux by listing a uv tools directory, which is the
    right way to read a package that ships twenty entry points and the only way
    on the Mac. On a machine without mflux it correctly finds nothing, and
    discovery then reported no image or video engine at all while two of them
    sat in the checkout.

    Availability is asked as a runtime rather than a platform, so a Linux box
    with the same card answers identically.
    """
    from harness import engines

    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()

    out = []
    for name in sorted(_ENGINE_RUNTIME):
        lane, runtime, example = _ENGINE_RUNTIME[name]
        if name not in engines._BUILDERS:
            continue
        refused = machine.refuses(runtime)
        if refused:
            continue
        spec = f"{name}:{example}" if example else name
        out.append(Capability(
            "engine", name, lane, "harness/engines.py",
            f"uv run python -m evals.run --modality {lane} "
            f"--candidates {spec}",
            note=f"needs {runtime}"))
    return out


def external_tools() -> list[Capability]:
    """CLIs the harness can drive. Absence is a finding, not an error."""
    known = [
        ("whisperkit-cli", "stt", "CoreML Whisper",
         "--candidates stt:large-v3,backend=whisperkit,language=en"),
        ("vtracer", "svg", "raster to vector",
         "--candidates trace:mflux:flux2-klein-4b"),
        ("rsvg-convert", "svg", "SVG rasterizer for the ink check", "(used by the checker)"),
        ("ffmpeg", "video", "demux and frame extraction", "(used by the checker)"),
        ("rec", "stt", "sox recorder for `lh hear`", "(used by the CLI)"),
    ]
    out = []
    for name, lane, why, how in known:
        found = shutil.which(name)
        if name == "vtracer" and not found:
            # It is a python package here rather than a binary.
            try:
                import vtracer  # noqa: F401
                found = "python package"
            except ImportError:
                found = None
        out.append(Capability("tool", name, lane, why, how, present=bool(found),
                              note="" if found else "not installed"))
    return out


def methods() -> list[Capability]:
    """Workflows the harness implements, as opposed to models it can call."""
    out = [
        Capability("method", "llm", "svg", "harness/cli.py",
                   "--candidates local-large",
                   note="the baseline every workflow has to beat"),
        Capability("method", "trace", "svg", "harness/vector.py",
                   "--candidates trace:mflux:flux2-klein-4b",
                   note="raster then vectorize; beat five language models 4/4 "
                        "to 2/6 on this lane"),
        Capability("method", "repair", "code", "evals/runners/repair.py",
                   "--candidates repair:q3-4b --modality code",
                   note="generate, check with the checker that already exists, "
                        "repair; code 20/27 -> 24/27 at --repeat 3"),
        Capability("method", "repair", "svg", "evals/runners/repair.py",
                   "--candidates repair:local-large --modality svg",
                   note="svg 6/9 -> 9/9 at --repeat 3, mean ~1.4 attempts"),
    ]
    for stage in sorted(STAGE_ENTRY_POINTS):
        broken = stage_unavailable(stage)
        out.append(Capability(
            "method", stage, "image", "evals/runners/chain.py",
            f"--candidates {stage}:mflux:flux2-klein-4b --modality image",
            blocked=broken, note="two-stage image workflow"))
    return out


def cached_audio_models(root: Path | None = None) -> list[Capability]:
    """Speech models already on disk. Downloading is the slow part, so what is
    cached is what can be measured today."""
    root = root or Path(os.environ.get("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    try:
        dirs = sorted(p.name for p in Path(root).iterdir() if p.is_dir())
    except OSError:
        return []
    out = []
    for d in dirs:
        if not d.startswith("models--"):
            continue
        repo = d[len("models--"):].replace("--", "/", 1).replace("--", "-")
        low = repo.lower()
        if any(k in low for k in ("parakeet", "whisper", "canary", "voxtral")):
            lane, how = "stt", f"--candidates stt:{repo}"
        elif any(k in low for k in ("kokoro", "tts", "chatterbox")):
            lane, how = "tts", f"--candidates tts:{repo}"
        else:
            continue
        out.append(Capability("model", repo, lane, str(root), how))
    return out


def capabilities() -> list[Capability]:
    """Everything this machine can be asked to do, in one list."""
    return (gateway_aliases() + image_engines() + generator_engines()
            + cached_audio_models() + methods() + external_tools()
            + not_adopted())


def measured() -> set[str]:
    """Candidate names that appear in any run's results.json.

    Read from the receipts the runs already write, so this cannot drift from
    what was actually evaluated.
    """
    names: set[str] = set()
    try:
        # RECURSIVE. Runs nest -- an archived batch sits at
        # runs/legacy-logs/ev-extract/results.json, three levels down -- and
        # iterating only the top level found 12 names where 31 receipts
        # existed. Under-reporting sends someone to re-run finished work,
        # which is the dangerous direction for this tool to be wrong in.
        receipts = sorted(paths.runs().rglob("results.json"))
    except OSError:
        return names
    for f in receipts:
        if not f.is_file():
            continue
        try:
            data = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            # A corrupt or half-written run must not stop discovery.
            continue
        names.update((data.get("summary") or {}).keys())
    return names


def _segments(name: str) -> set[str]:
    """The parts of a candidate name that could identify a capability.

    A results row is `trace/mflux/flux2-klein-4b-q8` or
    `Kokoro-82M-bf16/ff_siwis`, so the identifying part is a SEGMENT rather
    than the whole string.
    """
    return {p for p in name.split("/") if p}


def _was_measured(name: str, done: set[str]) -> bool:
    """Exact segment match, never a substring.

    Substring matching marked a capability called `X` as measured because
    "X" appears inside "Chatterbox-Multilingual-MLX-v2-Q8". Wrongly marking
    something DONE hides work, which is worse than wrongly offering it twice.
    """
    tail = name.split("/")[-1]
    for m in done:
        segs = _segments(m)
        if name in segs or tail in segs:
            return True
    return False


def annotate(caps: list[Capability] | None = None) -> list[Capability]:
    """Mark each capability with whether anything has measured it."""
    caps = capabilities() if caps is None else caps
    done = measured()
    for c in caps:
        c.measured = _was_measured(c.name, done)
    return caps


def gaps(caps: list[Capability] | None = None) -> list[Capability]:
    """What exists here and has never been run."""
    return [c for c in annotate(caps) if not c.measured and c.present]


# ---------------------------------------------------------------------------
# Looking outward
# ---------------------------------------------------------------------------

#: What to ask a registry for, per lane and per RUNTIME. Deliberately narrow:
#: proposing a candidate that cannot run here wastes the reader's time rather
#: than informing them, and that reasoning cuts both ways. Every query used to
#: be prefixed mlx-community, which asked a Mac question from a machine with an
#: NVIDIA card and proposed weights it cannot load.
#:
#: Keyed by runtime rather than by platform, so a Linux box with the same card
#: asks the same questions and a new kind of machine is a row here.
_LANE_QUERIES = {
    "mlx": {
        "text": ["mlx-community/Qwen3", "mlx-community/Llama-3",
                 "mlx-community/gemma"],
        "stt": ["mlx-community/parakeet", "mlx-community/whisper",
                "mlx-community/canary"],
        "tts": ["mlx-community/Kokoro", "mlx-community/Chatterbox",
                "mlx-community/TTS"],
        "image": ["mlx-community/FLUX", "mlx-community/Qwen-Image",
                  "mlx-community/Z-Image"],
        "video": ["mlx-community/MiniMax", "mlx-community/Wan"],
    },
    # GGUF for the text lane because llama.cpp is what serves it here, and the
    # upstream repos for the rest because diffusers and faster-whisper load
    # those directly rather than a converted copy.
    "cuda": {
        "text": ["Qwen3 GGUF", "bartowski/Qwen3", "unsloth/Qwen3-GGUF"],
        "stt": ["Systran/faster-whisper", "distil-whisper", "nvidia/parakeet"],
        "tts": ["hexgrad/Kokoro", "ResembleAI/chatterbox", "coqui/XTTS"],
        "image": ["black-forest-labs/FLUX", "stabilityai/stable-diffusion",
                  "Qwen/Qwen-Image"],
        "video": ["Lightricks/LTX-Video", "Wan-AI/Wan2", "THUDM/CogVideoX"],
    },
    # UNTESTED, and here so that detecting rocm cannot leave a lane with
    # nowhere to look. ROCm runs the same PyTorch repos as CUDA in most cases.
    "rocm": {
        "text": ["Qwen3 GGUF", "bartowski/Qwen3"],
        "stt": ["Systran/faster-whisper", "distil-whisper"],
        "tts": ["hexgrad/Kokoro", "ResembleAI/chatterbox"],
        "image": ["black-forest-labs/FLUX", "stabilityai/stable-diffusion"],
        "video": ["Lightricks/LTX-Video", "THUDM/CogVideoX"],
    },
    # A cpu runs GGUF through llama.cpp and whisper through CTranslate2, both
    # slowly and both really. This entry exists because CI has no card: every
    # query lived under an accelerator runtime, so a machine with only a cpu
    # got an empty list and discovery reported no candidates rather than
    # saying it had nowhere to look.
    "cpu": {
        "text": ["Qwen3 GGUF", "bartowski/Qwen3"],
        "stt": ["Systran/faster-whisper", "distil-whisper"],
        "tts": ["hexgrad/Kokoro", "coqui/XTTS"],
        "image": ["stabilityai/stable-diffusion", "segmind/tiny-sd"],
        "video": ["THUDM/CogVideoX", "Lightricks/LTX-Video"],
    },
}

#: Lanes served by a tool rather than by a runtime-specific model, so every
#: machine looks in the same place. Without this they would fall through to an
#: empty list and propose nothing, silently.
_LANE_QUERIES_ANY = {
    "svg": ["starvector", "OmniSVG"],
}

#: Every lane discovery can search, whichever machine is asking.
LANES = sorted(set(_LANE_QUERIES_ANY) | {
    lane for per_lane in _LANE_QUERIES.values() for lane in per_lane})


def lane_queries(lane: str, machine=None) -> list[str]:
    """What to ask the registry for `lane` on this machine.

    A machine with more than one runtime asks for both, since either would run.
    """
    if lane not in LANES:
        raise ValueError(
            f"no external queries defined for lane {lane!r}; "
            f"known: {', '.join(LANES)}")
    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()

    out: list[str] = list(_LANE_QUERIES_ANY.get(lane, []))
    for runtime in sorted(machine.runtimes):
        out += _LANE_QUERIES.get(runtime, {}).get(lane, [])
    return out

_HOW = {
    "text": "--modality extract --candidates <alias for {id}>",
    "stt": "--modality stt --candidates stt:{id}",
    "tts": "--modality tts --candidates tts:{id}",
    "image": "--modality image --candidates mflux:{id}",
    "svg": "--modality svg --candidates <needs a runner: see issue #3>",
    "video": "--modality video --candidates <needs a runner>",
}


def _hf_models(query: str, limit: int) -> list[dict]:
    """Ask the HuggingFace registry. Returns [] on any network trouble."""
    try:
        r = httpx.get("https://huggingface.co/api/models",
                      params={"search": query, "limit": limit,
                              "sort": "downloads", "direction": -1,
                              # The list endpoint omits lastModified unless
                              # asked, and every proposal printed a blank date.
                              # An undated proposal is what this exists to
                              # avoid.
                              "full": "true"},
                      timeout=20)
        r.raise_for_status()
        return r.json()
    except Exception:  # noqa: BLE001
        # Offline, rate-limited, or the API changed shape. Discovery that
        # cannot reach the network reports nothing; it does not stop the
        # command that called it.
        return []


def external(lane: str, limit: int = 8) -> list[Capability]:
    """Candidates the world has that this machine has never measured.

    Proposals, not conclusions. Each carries the registry URL it came from and
    the date the registry last changed it, so a stale suggestion is visible as
    stale rather than quietly believed.
    """
    queries = lane_queries(lane)

    done = measured()
    seen: set[str] = set()
    out: list[Capability] = []
    for query in queries:
        for m in _hf_models(query, limit):
            repo = m.get("id") or ""
            if not repo or repo in seen:
                continue
            if _was_measured(repo, done):
                continue          # already measured here
            seen.add(repo)
            when = (m.get("lastModified") or m.get("createdAt") or "")[:10]
            out.append(Capability(
                "model", repo, lane,
                f"https://huggingface.co/{repo}",
                _HOW.get(lane, "--candidates {id}").format(id=repo),
                note=f"registry last modified {when or 'unknown'}; "
                     f"{m.get('downloads', 0):,} downloads"))
    return out[:limit]


def _hf_exists(name: str) -> str:
    """Resolve a name lifted from prose to a real repo id, or "" if it is not one."""
    for m in _hf_models(name, 3):
        repo = (m.get("id") or "")
        if repo.split("/")[-1].lower() == name.split("/")[-1].lower():
            return repo
    return ""


def _gh_exists(repo: str, client=None) -> bool:
    from harness import github
    try:
        return (client or github.Client()).exists(repo)
    except Exception:  # noqa: BLE001 - never let the check itself drop a sweep
        return True


def from_feeds(sources=None, reader=None, verify=True,
               limit: int = 25, min_relevance: int | None = None,
               store=None, gh=None, comments: int = 0,
               comment_reader=None, mentions=None,
               mention_comments: int = 8) -> list[Capability]:
    """Candidates the community is talking about that nothing here has measured.

    A LINKED repo is already an id. A name lifted from prose is a claim, and is
    resolved against the registry before it is offered -- the same bar a
    language model's suggestions are held to.
    """
    from harness import feeds

    from harness import memory_store as ms

    sources = feeds.load_sources() if sources is None else sources
    reader = reader or feeds.read
    done = measured()
    out: list[Capability] = []
    seen: set[str] = set()
    # A proposal with a terminal verdict has been answered. Re-offering it every
    # sweep is what makes an operator stop reading the output.
    settled = ms.settled(store) if store is not None else set()
    suppressed = 0

    for src in sources:
        if not src.enabled:
            continue
        # A source read some other way is not a broken feed. Reading it here
        # would report the crowd source as unreachable on every sweep.
        if src.kind not in ("atom", "releases"):
            continue
        try:
            entries = reader(src)
        except Exception as exc:  # noqa: BLE001
            out.append(Capability(
                "feed", src.name, src.lane, src.url,
                "check the URL, or the network", present=False,
                blocked=f"could not read: {str(exc)[:160]}"))
            continue

        if src.kind == "releases":
            # The signal here is drift, not the repo name. Proposing
            # `ml-explore/mlx` to a project built on MLX is noise.
            package = feeds.TRACKS.get(src.name, "")
            newest = feeds.newest_release(entries)
            have = feeds.installed_version(package) if package else ""
            if feeds.behind(newest, have):
                out.append(Capability(
                    "update", package, src.lane, src.url,
                    f"upgrade {package} {have} -> {newest}, then re-run the "
                    f"lane's eval to confirm nothing regressed",
                    note=f"running {have}, latest is {newest}"))
            continue

        # The comparative judgements are in the REPLIES, not the post. A post
        # title says "what do you use for local image generation"; the answer
        # names four tools across four lanes. Issue #78.
        # Comments are their OWN source, not more of the post's. Folding them
        # together would move the recap's measured extraction precision (0.45,
        # issue #49) without anyone changing the recap, and the baseline stops
        # meaning anything. They are a different kind of text with a different
        # hit rate, so they get a different name in the store.
        batches = [(src.name, list(entries))]
        if comments and src.kind == "atom":
            reader_c = comment_reader or feeds.comments
            replies: list = []
            for e in entries[:comments]:
                if not e.link:
                    continue
                try:
                    replies += reader_c(e.link)
                except Exception:  # noqa: BLE001 - one dead thread is not a sweep
                    continue
            if replies:
                batches.append((f"{src.name}-comments", replies))

        proposals: list = []
        for origin, batch in batches:
            proposals += feeds.candidates(batch, origin)
            if not (mentions and origin.endswith("-comments")):
                continue
            # A model call per comment, so only the longest few: a two-word
            # reply names nothing, and the thread that prompted this had 126
            # comments of which one carried the ranking worth having. #79.
            richest = sorted(batch, key=lambda e: -len(e.body or ""))
            for e in richest[:mention_comments]:
                for name, claim in mentions(e.body):
                    proposals.append(feeds.Proposal(
                        name, claim or f"named in: {e.title}"[:160], origin,
                        e.link, e.updated, "candidate",
                        feeds.relevance(f"{name} {claim}")))

        for p in proposals:
            # Every drop is RECORDED. Without the thrown-away names there is no
            # denominator, and a store holding only survivors can report that
            # 100% of proposals resolved, which is true and means nothing.
            # Issue #49.
            def drop(reason, name=None, origin=None):
                if store is not None:
                    ms.reject(store, name or p.name, origin or p.source, reason)

            if min_relevance is not None and p.relevance < min_relevance:
                drop("below-relevance")
                continue
            repo = p.name
            if p.kind == "tool":
                # A github repo is something to read, not an mflux candidate.
                if repo.lower() in seen:
                    drop("duplicate")
                    continue
                seen.add(repo.lower())
                # A repo name in prose is a claim, held to the same bar as a
                # model name. Fails OPEN: only a definite 404 drops it, so an
                # unreachable API cannot silently empty a sweep.
                if verify and not _gh_exists(repo, client=gh):
                    drop("not-a-repo")
                    continue
                out.append(Capability(
                    "proposal", repo, src.lane, p.url or src.url,
                    f"https://github.com/{repo}",
                    note=f"{p.why[:120]} [{src.name} {p.when}]"))
                continue
            if p.kind != "repo":
                if not verify:
                    continue
                repo = _hf_exists(p.name)
                if not repo:
                    drop("unresolvable")
                    continue      # prose that names nothing real
            if repo.lower() in seen:
                drop("duplicate", repo)
                continue
            if _was_measured(repo, done):
                drop("already-measured", repo)
                continue
            seen.add(repo.lower())
            if store is not None:
                ms.record(store, ms.Seen(
                    name=repo, source=p.source or src.name, url=p.url or src.url,
                    why=p.why[:160], relevance=p.relevance, kind=p.kind,
                    # `repo` is the linked id, or the id a prose name resolved
                    # to. Either way it is the verified thing, so keep it.
                    lane=src.lane, resolved=repo))
            if repo in settled:
                suppressed += 1
                drop("settled", repo)
                continue
            tag = f" [apple silicon +{p.relevance}]" if p.relevance > 0 else ""
            out.append(Capability(
                "proposal", repo, src.lane, p.url or src.url,
                _HOW.get(src.lane, "--candidates {id}").format(id=repo),
                measured=False,
                note=f"{p.why[:120]}{tag} [{src.name} {p.when}]"))
            out[-1].relevance = p.relevance
    # Most relevant to this machine first. A CUDA-only proposal is noise here
    # and was previously ranked identically to a native-MLX one.
    out.sort(key=lambda c: -getattr(c, "relevance", 0))
    out = out[:limit]
    if suppressed:
        out.append(Capability(
            "note", f"{suppressed} already answered", "all", "discovery.db",
            "lh discover --recurrence to see what is known",
            note=f"{suppressed} proposal(s) suppressed: already measured, "
                 f"declined or broken"))
    return out


def feed_sources(sources=None, reader=None) -> list[Capability]:
    """Sources the feeds point at that this harness does not read.

    Never auto-enabled. A source URL lifted from untrusted prose needs a human
    nod before the harness starts fetching it on a schedule.
    """
    from harness import feeds

    sources = feeds.load_sources() if sources is None else sources
    reader = reader or feeds.read
    entries = []
    for src in sources:
        if not src.enabled:
            continue
        try:
            entries.extend(reader(src))
        except Exception:  # noqa: BLE001
            continue
    return [Capability("source", p.name, "all", p.url,
                       "add it to discovery-sources.json to start reading it",
                       present=False, note=p.why)
            for p in feeds.candidate_sources(entries, sources)]


#: Things deliberately NOT adopted, so the decision surfaces where someone would
#: go looking rather than only in PLAN.md. Same job the BROKEN state does for a
#: stage: stop the question being rediscovered from scratch.
#: name, lane, why, issue, and THE RUNTIME THE ARGUMENT WAS MADE ON. A
#: decision is an argument rather than a fact, and this one rests on mflux
#: shipping the workflows natively: true where mflux runs and no argument at
#: all on a machine that has neither. Telling such a machine the question is
#: closed hides a candidate behind reasoning that does not apply to it.
NOT_ADOPTED = [
    ("ComfyUI", "image",
     "the reason to want it is workflows, and mflux ships 19 of them natively "
     "in MLX. It would add a torch/MPS runtime and a server with a UI, on a "
     "machine chosen for MLX and a project whose caller is an agent",
     20, "mlx"),
]


def not_adopted(machine=None) -> list[Capability]:
    """Recorded decisions against a tool, with the issue that argued it.

    Shown only to a machine the argument was made for. See NOT_ADOPTED.
    """
    if machine is None:
        from harness import machine as _machine
        machine = _machine.detect()
    return [Capability("decision", name, lane, f"issue #{issue}",
                       "reopen the issue if a workflow exists there and "
                       "nowhere in mflux",
                       present=False, blocked=f"not adopted: {why}. See #{issue}")
            for name, lane, why, issue, runtime in NOT_ADOPTED
            if machine.refuses(runtime) is None]
