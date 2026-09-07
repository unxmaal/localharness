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

THIS MODULE ONLY LOOKS INWARD -- at what is installed here. It cannot tell you
that the world has moved on, which is the larger half and the reason Qwen3
shipped unnoticed. That is issue #19.

Everything degrades rather than raising: discovery runs on machines that do not
have everything, and a missing tool is a finding rather than a crash.
"""
from __future__ import annotations

import json
import os
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from harness import paths

REPO = Path(__file__).resolve().parent.parent

#: Entry points that GENERATE. mflux also ships concept/capabilities/completions
#: helpers, and listing those would invite someone to evaluate a tool that
#: produces no artifact.
_GENERATE_PREFIX = "mflux-generate"
#: These generate, but not a plain image from a prompt: they need a control
#: image, a mask or an existing picture, so they cannot be dropped into the
#: image lane as-is.
_NEEDS_INPUT = ("controlnet", "depth", "fill", "redux", "edit", "upscal",
                "kontext", "inpaint", "img2img")


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
        if not n.startswith(_GENERATE_PREFIX):
            continue
        if any(k in n.lower() for k in _NEEDS_INPUT):
            continue
        spec = n.replace(_GENERATE_PREFIX + "-", "mflux:") if "-" in n[len(_GENERATE_PREFIX):] else "mflux:dev"
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

    Deliberately short, and that is the point: issue #18. The suite has spent
    its life comparing models, and the one workflow it has (`trace`) beat five
    language models on its lane.
    """
    return [
        Capability("method", "llm", "svg", "harness/cli.py",
                   "--candidates local-large"),
        Capability("method", "trace", "svg", "harness/vector.py",
                   "--candidates trace:mflux:flux2-klein-4b"),
    ]


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


def annotate(caps: list[Capability] | None = None) -> list[Capability]:
    """Mark each capability with whether anything has measured it."""
    caps = capabilities() if caps is None else caps
    done = measured()
    for c in caps:
        # A candidate name is not always the capability name: an engine spec
        # becomes `mflux/flux2-klein-4b-q8` in the table, and a repo id is
        # reported by its last segment.
        tail = c.name.split("/")[-1]
        c.measured = any(tail in m or c.name in m for m in done)
    return caps


def gaps(caps: list[Capability] | None = None) -> list[Capability]:
    """What exists here and has never been run."""
    return [c for c in annotate(caps) if not c.measured and c.present]
