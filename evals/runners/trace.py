"""Runner for the svg lane's OTHER method: draw it, then vectorize it.

The lane has five language models measured on it and all five produce valid
markup that is not the picture -- an LLM writes bezier coordinates it cannot
see. This candidate generates a raster with an image engine and vectorizes it,
and is scored by the SAME svg checker on the SAME cases, which is the only way
the two methods can be compared at all.

The comparison is not like-for-like on cost and should not be read as one: the
LLM path is seconds, this is a diffusion run plus 0.05s of tracing. That is a
real difference and it belongs in the table rather than in a footnote, which is
why the runner reports the engine's wall clock and peak memory honestly.
"""
from __future__ import annotations

from pathlib import Path

from harness import proc, vector
from harness.engines import Engine

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError

#: Appended to every prompt. A photographic subject vectorizes into thousands
#: of paths; flat shapes on white vectorize into an icon. Without this the
#: method is being measured with its worst foot forward.
TRACE_STYLE = ("flat vector illustration, simple clean shapes, bold outlines, "
               "solid colours, white background, no gradients, no texture")


class TraceRunner(BaseRunner):
    def __init__(self, engine: Engine, outdir: str | Path,
                 width: int = 512, height: int = 512,
                 preset: str = "illustration"):
        if preset not in vector.TRACE_PRESETS:
            raise ValueError(
                f"unknown trace preset {preset!r}; "
                f"known: {', '.join(sorted(vector.TRACE_PRESETS))}")
        self.engine = engine
        self.width = width
        self.height = height
        self.preset = preset
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        name = "trace" if preset == "illustration" else f"trace-{preset}"
        self.candidate = f"{name}/{engine.name}"

    def generate(self, case: Case):
        stem = self.candidate.replace("/", "_")
        png = (self.outdir / f"{stem}--{case.id}.png").resolve()
        if png.exists():
            png.unlink()

        params = {"width": self.width, "height": self.height,
                  "steps": None, "seed": case.params.get("seed")}
        try:
            argv = self.engine.argv(f"{case.prompt}, {TRACE_STYLE}", png, params)
        except ValueError as exc:
            raise RunnerError(str(exc)) from exc

        try:
            r = proc.run(argv, timeout=self.engine.timeout,
                         stream=self.engine.stream, cwd=self.engine.cwd)
        except FileNotFoundError as exc:
            raise RunnerError(f"{exc} is not installed") from exc
        except OSError as exc:
            raise RunnerError(f"could not launch {self.engine.name}: {exc}") from exc
        if not r.ok:
            raise RunnerError(
                f"{self.engine.name} exited {r.returncode}: "
                f"{r.stderr.strip()[-200:]}")
        if not png.exists():
            raise RunnerError(f"{self.engine.name} exited 0 but left no output")

        try:
            svg = vector.trace(png, preset=self.preset)
        except vector.VectorError as exc:
            raise RunnerError(str(exc)) from exc
        # The engine's peak, not the tracer's: tracing is 0.05s and a few MB,
        # and reporting it would hide where the cost actually is.
        return svg, r.peak_kb
