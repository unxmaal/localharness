"""Two-stage image workflows: generate, then do something to the result.

THIS IS THE VOCABULARY COMFYUI IS WANTED FOR, and mflux ships all of it
natively in MLX: controlnet, depth, fill, redux, kontext, in-context, two
upscalers, LoRA. Nineteen primitives, and until this runner existed none of
them could be a candidate, because every runner here assumed one command
produces the artifact. The gap was invisible for months while the argument was
about whether to adopt ComfyUI.

A STAGE is a function that builds the command line for the second step, given
the first step's output. Adding `controlnet` should be a stage, not another
runner.

BOTH STAGES COST. The row reports how many ran, because a two-stage workflow
against a one-stage candidate in the same table otherwise looks free.
"""
from __future__ import annotations

import shutil
from pathlib import Path

from harness import proc
from harness.engines import Engine

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError

MFLUX_BIN = Path.home() / ".local/share/uv/tools/mflux/bin"


def _mflux(name: str) -> str:
    """Absolute path: an eval launched by launchd has almost no PATH."""
    found = shutil.which(name)
    return found or str(MFLUX_BIN / name)


def upscale_seedvr2(src: Path, dst: Path, params: dict) -> list[str]:
    """SeedVR2 diffusion super-resolution.

    `--low-ram` and `--vae-tiling` are not optional on 32 GB: the VAE decode is
    the peak phase and tiling it is the difference between running and taking
    the machine down. This project has already done the latter once.
    """
    width = int(params.get("width") or 512)
    return [_mflux("mflux-upscale-seedvr2"),
            "--image-path", str(src),
            "--output", str(dst),
            "--resolution", str(width * 2),
            "--low-ram", "--vae-tiling",
            "--quantize", "4",
            "--no-metadata"]


#: name -> builds the second stage's command line.
STAGES = {"upscale-seedvr2": upscale_seedvr2}
#: How each stage changes the output resolution, so the checker can be told
#: what the workflow INTENDED rather than what the case asked to generate.
STAGE_SCALE = {"upscale-seedvr2": 2}


class ChainRunner(BaseRunner):
    def __init__(self, engine: Engine, stage: str, outdir: str | Path,
                 timeout: float = 1800.0):
        if stage not in STAGES:
            raise ValueError(
                f"unknown stage {stage!r}; known: {', '.join(sorted(STAGES))}")
        self.engine = engine
        self.stage = stage
        self.timeout = timeout
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.candidate = f"{stage}/{engine.name}"
        self.last_metrics: dict = {}
        #: Set by generate() once the stage says what it produced.
        self.expect_size: tuple | None = None

    def generate(self, case: Case):
        stem = self.candidate.replace("/", "_")
        # KEPT, not cleaned up: when the final image is wrong the first
        # question is whether stage one was already wrong, and deleting the
        # intermediate throws away the only way to answer.
        mid = (self.outdir / f"{stem}--{case.id}--stage1.png").resolve()
        out = (self.outdir / f"{stem}--{case.id}.png").resolve()
        for p in (mid, out):
            if p.exists():
                p.unlink()

        scale = STAGE_SCALE.get(self.stage, 1)
        w, h = case.params.get("width"), case.params.get("height")
        self.expect_size = (w * scale, h * scale) if (w and h) else None

        params = {"width": case.params.get("width"),
                  "height": case.params.get("height"),
                  "steps": None, "seed": case.params.get("seed")}
        try:
            argv = self.engine.argv(case.prompt, mid, params)
        except ValueError as exc:
            raise RunnerError(str(exc)) from exc

        first = self._run(argv, f"{self.engine.name} (stage 1)")
        if not mid.exists():
            raise RunnerError(f"{self.engine.name} exited 0 but wrote no image")
        self.last_metrics = {"stages": 1}

        second = self._run(STAGES[self.stage](mid, out, params),
                           f"{self.stage} (stage 2)")
        if not out.exists():
            raise RunnerError(f"{self.stage} exited 0 but wrote no image")
        self.last_metrics = {"stages": 2}
        # The peak of the WHOLE workflow, which is the number that decides
        # whether it fits on this machine.
        return out, max(first.peak_kb, second.peak_kb)

    def _run(self, argv: list[str], what: str):
        try:
            r = proc.run(argv, timeout=self.timeout, cwd=self.engine.cwd)
        except FileNotFoundError as exc:
            raise RunnerError(f"{what}: {exc} is not installed") from exc
        except OSError as exc:
            raise RunnerError(f"{what}: could not launch: {exc}") from exc
        if not r.ok:
            raise RunnerError(
                f"{what} exited {r.returncode}: {r.stderr.strip()[-200:]}")
        return r

    def score_kwargs(self) -> dict:
        # What this WORKFLOW meant to produce. Without it an upscaler fails
        # the size check for doing its job.
        return {"expect_size": self.expect_size} if self.expect_size else {}

    def extra_metrics(self) -> dict:
        return dict(self.last_metrics)
