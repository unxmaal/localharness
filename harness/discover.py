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
    #: Is it installed at all? False means absent from this machine.
    present: bool = True
    #: Installed, but it CANNOT RUN, and this says why. Distinct from
    #: `present` because "not installed" and "installed and broken" are
    #: different findings with different fixes, and distinct from `measured`
    #: because a thing can have been measured and be broken now -- which is
    #: exactly upscale-seedvr2, measured 0/3 precisely BECAUSE it is broken.
    #: A row that has been measured must never read as healthy on that basis.
    blocked: str = ""
    note: str = ""


def gateway_aliases(config: Path | None = None) -> list[Capability]:
    """Text candidates, read from the gateway config rather than guessed."""
    config = config or (REPO / "gateway" / "config.yaml")
    try:
        import yaml
        data = yaml.safe_load(Path(config).read_text()) or {}
    except (OSError, ValueError):
        return []
    out = []
    for entry in data.get("model_list") or []:
        name = entry.get("model_name")
        if not name:
            continue
        upstream = (entry.get("litellm_params") or {}).get("model", "")
        out.append(Capability(
            "model", name, "text", f"gateway/config.yaml -> {upstream}",
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
            # THREE OF THESE NOW HAVE A RUNNER. Reporting the whole set as
            # "no runner yet" is how this tool started lying: ChainRunner
            # landed and discover went on describing the state of the repo
            # before it. What has a runner is a candidate; what does not is
            # still a gap, and the two must not read the same.
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
    """Workflows the harness implements, as opposed to models it can call.

    It used to list two and cite #18 as the reason it was short. #18 is closed
    and there are now six, so the list is generated from the registries rather
    than retyped -- a hand-maintained inventory is stale by the next commit,
    which is the entire premise of this module.
    """
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
    return (gateway_aliases() + image_engines() + cached_audio_models()
            + methods() + external_tools())


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
            data = json.loads(f.read_text())
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

#: What to ask a registry for, per lane. Deliberately narrow queries against
#: mlx-community: this machine runs MLX, and proposing a candidate that cannot
#: run here wastes the reader's time rather than informing them.
_LANE_QUERIES = {
    "text": ["mlx-community/Qwen3", "mlx-community/Llama-3", "mlx-community/gemma"],
    "stt": ["mlx-community/parakeet", "mlx-community/whisper", "mlx-community/canary"],
    "tts": ["mlx-community/Kokoro", "mlx-community/Chatterbox", "mlx-community/TTS"],
    "image": ["mlx-community/FLUX", "mlx-community/Qwen-Image", "mlx-community/Z-Image"],
    "svg": ["starvector", "OmniSVG"],
    "video": ["mlx-community/MiniMax", "mlx-community/Wan"],
}

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
    if lane not in _LANE_QUERIES:
        raise ValueError(
            f"no external queries defined for lane {lane!r}; "
            f"known: {', '.join(sorted(_LANE_QUERIES))}")

    done = measured()
    seen: set[str] = set()
    out: list[Capability] = []
    for query in _LANE_QUERIES[lane]:
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
