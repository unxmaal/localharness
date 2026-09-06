"""Runner for generators that are external processes.

Not image-specific: mflux, h3.c and any future audio engine are the same shape,
so they plug in as Engines rather than as new Runner classes.

There is no completion to parse here. What is measurable is the process itself:
exit status, wall time, peak memory, and whether the file it left behind is
real. All three come from harness.proc, which is also what the CLI uses, so the
eval measures the exact command the product runs.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from harness import proc
from harness.engines import Engine

from evals.core import Case
from evals.runners.base import BaseRunner, RunnerError


class ProcessRunner(BaseRunner):
    def __init__(self, engine: Engine, outdir: str | Path,
                 timeout: float | None = None, adherence: str | None = None):
        self.engine = engine
        # A run-level choice, so it travels with the runner rather than being
        # read from a global by the checker.
        self.adherence = adherence
        self.candidate = engine.name
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout if timeout is not None else engine.timeout

    def score_kwargs(self) -> dict:
        return {"adherence": self.adherence} if self.adherence else {}

    def generate(self, case: Case):
        # `/` in a model name would otherwise open a directory that does not
        # exist: mflux/z-image-turbo-q8 is one candidate, not a path.
        stem = self.candidate.replace("/", "_")
        # Absolute: an engine with its own working directory would otherwise
        # write a relative path inside that directory rather than here.
        out = (self.outdir / f"{stem}--{case.id}"
               f"{self.engine.output_suffix}").resolve()
        if out.exists():
            out.unlink()

        try:
            argv = self.engine.argv(case.prompt, out, case.params)
        except ValueError as exc:
            raise RunnerError(str(exc)) from exc

        try:
            r = proc.run(argv, timeout=self.timeout, stream=self.engine.stream,
                         cwd=self.engine.cwd)
        except subprocess.TimeoutExpired as exc:
            raise RunnerError(f"timed out after {self.timeout}s") from exc
        except FileNotFoundError as exc:
            raise RunnerError(f"{argv[0]} not installed or not on PATH") from exc
        except OSError as exc:
            raise RunnerError(f"could not launch: {exc}") from exc

        if not r.ok:
            detail = f"exit {r.returncode}"
            tail = (r.stderr or "").strip().splitlines()
            if tail:
                detail += f": {tail[-1]}"
            raise RunnerError(detail, peak_kb=r.peak_kb)

        return out, r.peak_kb
