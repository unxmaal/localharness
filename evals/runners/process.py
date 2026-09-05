"""Runner for generators that are external processes.

Not image-specific: mflux, h3.c and any audio engine are the same shape, so they
plug in as Engines rather than as new Runner classes.

The text runner parses a completion; there is nothing to parse here. What is
measurable is the process itself: exit status, wall time, peak resident memory,
and whether the file it left behind is a real image.

Peak RSS matters more than it looks on a 32GB machine. The h3 work established
that the binding constraint on this hardware is a single phase's peak, not the
model's size on disk, and the same is true of diffusion.
"""
from __future__ import annotations

import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from evals.checks import image as image_check
from evals.core import Case, Result


@dataclass(frozen=True)
class Engine:
    """How to invoke one image generator.

    `argv` builds the command from (case, output_path, assertions) so a new
    engine is a data change rather than a code change, which is the same reason
    the gateway exists for text.
    """
    name: str
    argv: Callable[[Case, Path, dict], list[str]]
    output_suffix: str = ".png"


def measure_peak_kb(argv, timeout=None, capture=True):
    """Run argv, returning (peak_footprint_kb, returncode).

    Uses `/usr/bin/time -l`, whose "peak memory footprint" is macOS's
    phys_footprint: it accounts for compressed and swapped pages and is the
    number Activity Monitor shows. This is the same metric the h3 runs were
    reported in, so image and video numbers are finally comparable.

    Not resource.getrusage(RUSAGE_CHILDREN).ru_maxrss: that is a monotone
    high-water mark across ALL waited children, so a before/after delta reads 0
    for every child after the largest, and plain RSS falls under memory
    pressure. Both errors were live, and a documented "2x the memory"
    conclusion rested on them.
    """
    import re
    import shutil
    import subprocess

    # /usr/bin/time always exists, so a missing target would otherwise surface
    # as time's own non-zero exit rather than as "not installed".
    if shutil.which(argv[0]) is None:
        raise FileNotFoundError(argv[0])

    proc = subprocess.run(["/usr/bin/time", "-l", *argv],
                          capture_output=capture, text=True, timeout=timeout)
    peak = 0
    m = re.search(r"(\d+)\s+peak memory footprint", proc.stderr or "")
    if m:
        peak = int(m.group(1)) // 1024  # bytes -> KB
    return peak, proc.returncode


class ProcessRunner:
    def __init__(self, engine: Engine, outdir: str | Path,
                 timeout: float = 900.0):
        self.engine = engine
        self.outdir = Path(outdir)
        self.outdir.mkdir(parents=True, exist_ok=True)
        self.timeout = timeout

    def run(self, case: Case) -> Result:
        out = self.outdir / f"{self.engine.name}--{case.id}{self.engine.output_suffix}"
        if out.exists():
            out.unlink()
        argv = self.engine.argv(case, out, case.assertions)

        started = time.monotonic()
        try:
            peak_kb, returncode = measure_peak_kb(argv, timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return self._fail(case, started,
                              f"timed out after {self.timeout}s")
        except FileNotFoundError:
            return self._fail(case, started,
                              f"{argv[0]} not installed or not on PATH")
        except OSError as exc:
            return self._fail(case, started, f"could not launch: {exc}")
        elapsed = time.monotonic() - started

        if returncode != 0:
            return self._fail(case, started, f"exit {returncode}", peak_kb)

        expect = None
        if case.assertions.get("width") and case.assertions.get("height"):
            expect = (case.assertions["width"], case.assertions["height"])
        checked = image_check.check(out, expect=expect)

        return Result(case.id, self.engine.name, checked.ok,
                      round(elapsed, 2), peak_kb, checked.reason,
                      artifact=str(out) if out.exists() else None,
                      warnings=checked.warnings)

    def _fail(self, case: Case, started: float, detail: str,
              peak_kb: int = 0) -> Result:
        return Result(case.id, self.engine.name, False,
                      round(time.monotonic() - started, 2), peak_kb, detail)


def mflux_engine(spec: str, quantize: int | None = 8,
                 steps: int | None = None) -> Engine:
    """mflux, the MLX-native image stack.

    `spec` is either a model that ships its own entry point:

        z-image-turbo        -> mflux-generate-z-image-turbo

    or `entrypoint/model` for variants selected by a flag:

        flux2/flux2-klein-4b -> mflux-generate-flux2 --model flux2-klein-4b

    The distinction matters: flux2-klein-4b has no binary of its own, and
    guessing one produces a "not installed" failure that reads as a missing
    dependency rather than a bad spec.
    """
    if "/" in spec:
        entrypoint, model = spec.split("/", 1)
    else:
        entrypoint, model = spec, None
    binary = f"mflux-generate-{entrypoint}"
    label = model or entrypoint

    def argv(case: Case, out: Path, a: dict) -> list[str]:
        cmd = [binary, "--prompt", case.prompt, "--output", str(out)]
        if model:
            cmd += ["--model", model]
        if a.get("width"):
            cmd += ["--width", str(a["width"])]
        if a.get("height"):
            cmd += ["--height", str(a["height"])]
        n = a.get("steps", steps)
        if n:
            cmd += ["--steps", str(n)]
        if a.get("seed") is not None:
            cmd += ["--seed", str(a["seed"])]
        if quantize:
            cmd += ["-q", str(quantize)]
        return cmd

    return Engine(name=f"mflux/{label}", argv=argv)
