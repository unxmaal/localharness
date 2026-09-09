"""Runner for the svg lane's THIRD method: a model that emits SVG directly.

The lane had two: five language models writing path data, and `trace` (generate
a raster, vectorize it). Both are general tools pointed at vectors. OmniSVG is
Qwen2.5-VL with an SVG tokenizer bolted on, trained to emit draw commands as
tokens, so it is the only candidate here that was built for the job.

It runs as a subprocess for two reasons that are not style. Its inference
script imports sibling modules by bare name and reads `./config.yaml`, so it
has to run from its own checkout; and it pins transformers 4.51.3 against a
torch this project does not otherwise install, so it gets a venv of its own.
`scripts/setup-omnisvg.sh` builds both.

Cost is not comparable to the LLM path and should not be read as though it
were: a model load plus a batched sample of several candidates, against a
gateway call of a few seconds. That is the same honest asymmetry `trace` has.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from harness import proc

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError

DEFAULT_HOME = Path.home() / "localharness" / "omnisvg"

#: Qwen half and OmniSVG half, per model size. The decoder weights are a
#: state_dict over a stock Qwen2.5-VL that OmniSVG does not redistribute, so
#: both have to be on disk before anything runs.
MODELS = {
    "4B": ("Qwen/Qwen2.5-VL-3B-Instruct", "OmniSVG/OmniSVG1.1_4B"),
    "8B": ("Qwen/Qwen2.5-VL-7B-Instruct", "OmniSVG/OmniSVG1.1_8B"),
}

#: Upstream samples `candidates + 4` sequences in one batch and keeps the first
#: that renders to a non-empty raster. The buffer is theirs, not ours; asking
#: for one still costs five.
DEFAULT_CANDIDATES = 1

#: A model load plus a batched sample on MPS. Measured at several minutes a
#: case, so the ceiling is generous rather than tight.
TIMEOUT = 1800.0


def home() -> Path:
    return Path(os.environ.get("OMNISVG_HOME") or DEFAULT_HOME)


def interpreter(root: Path | None = None) -> Path:
    override = os.environ.get("OMNISVG_PYTHON")
    if override:
        return Path(override)
    return (root or home()) / ".venv" / "bin" / "python"


def cached(repo_id: str, filename: str) -> Path | None:
    """The local directory holding `filename` for `repo_id`, or None.

    Resolved from the cache rather than downloaded: HF_HUB_OFFLINE is 1
    everywhere in this project on purpose, and an eval that quietly pulls
    16 GiB mid-run is not measuring what it claims to.
    """
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return None
    hit = try_to_load_from_cache(repo_id, filename)
    return Path(hit).parent if isinstance(hit, str) else None


class OmniSVGRunner(BaseRunner):
    def __init__(self, size: str = "4B", *, candidates: int = DEFAULT_CANDIDATES,
                 root: Path | str | None = None, timeout: float = TIMEOUT):
        if size not in MODELS:
            raise ValueError(
                f"unknown OmniSVG size {size!r}; known: "
                f"{', '.join(sorted(MODELS))}")
        if candidates < 1:
            raise ValueError(f"candidates must be at least 1, got {candidates}")
        self.size = size
        self.candidates = candidates
        self.root = Path(root) if root is not None else home()
        self.timeout = timeout
        self.candidate = f"omnisvg:{size}"

    def _paths(self) -> tuple[Path, Path]:
        qwen_repo, omni_repo = MODELS[self.size]
        qwen = cached(qwen_repo, "config.json")
        omni = cached(omni_repo, "pytorch_model.bin")
        missing = [r for r, p in ((qwen_repo, qwen), (omni_repo, omni)) if p is None]
        if missing:
            raise RunnerError(
                f"weights not cached: {', '.join(missing)}. "
                f"Run scripts/setup-omnisvg.sh")
        return qwen, omni

    def generate(self, case: Case):
        script = self.root / "inference.py"
        if not script.exists():
            raise RunnerError(
                f"no OmniSVG checkout at {self.root}. "
                f"Run scripts/setup-omnisvg.sh, or set OMNISVG_HOME")
        python = interpreter(self.root)
        if not python.exists():
            raise RunnerError(
                f"no interpreter at {python}. Run scripts/setup-omnisvg.sh")
        qwen, omni = self._paths()

        # Its own output directory per case: upstream names the file from the
        # first 50 characters of the prompt, so two cases can collide and a
        # stale file from a previous case would be read as this one's answer.
        with tempfile.TemporaryDirectory(prefix="omnisvg-") as tmp:
            work = Path(tmp)
            prompts = work / "prompt.txt"
            prompts.write_text(case.prompt.strip() + "\n", encoding="utf-8")
            out = work / "out"

            argv = [str(python), str(script),
                    "--task", "text-to-svg",
                    "--input", str(prompts),
                    "--output", str(out),
                    "--model-size", self.size,
                    "--model-path", str(qwen),
                    "--weight-path", str(omni),
                    "--num-candidates", str(self.candidates)]
            try:
                r = proc.run(argv, timeout=self.timeout, cwd=str(self.root))
            except subprocess.TimeoutExpired as exc:
                raise RunnerError(
                    f"omnisvg exceeded {self.timeout:.0f}s") from exc
            except FileNotFoundError as exc:
                raise RunnerError(f"{exc} is not installed") from exc
            except OSError as exc:
                raise RunnerError(f"could not launch omnisvg: {exc}") from exc

            if not r.ok:
                raise RunnerError(
                    f"omnisvg exited {r.returncode}: {r.stderr.strip()[-300:]}",
                    peak_kb=r.peak_kb)

            svgs = sorted(out.glob("*.svg")) if out.exists() else []
            if not svgs:
                # It exits 0 when every sampled candidate fails to render, so a
                # clean exit is not evidence that anything was produced.
                raise RunnerError(
                    "omnisvg exited 0 but produced no SVG "
                    "(every sampled candidate rendered empty)",
                    peak_kb=r.peak_kb)
            return svgs[0].read_text(encoding="utf-8"), r.peak_kb
