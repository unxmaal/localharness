"""Runner for image generators that are external processes.

The text runner parses a completion; there is nothing to parse here. What is
measurable is the process itself: exit status, wall time, peak resident memory,
and whether the file it left behind is a real image.

Peak RSS matters more than it looks on a 32GB machine. The h3 work established
that the binding constraint on this hardware is a single phase's peak, not the
model's size on disk, and the same is true of diffusion.
"""
from __future__ import annotations

import resource
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


class ImageRunner:
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

        before = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        started = time.monotonic()
        try:
            proc = subprocess.run(argv, capture_output=True, text=True,
                                  timeout=self.timeout)
        except subprocess.TimeoutExpired:
            return self._fail(case, started,
                              f"timed out after {self.timeout}s")
        except FileNotFoundError:
            return self._fail(case, started,
                              f"{argv[0]} not installed or not on PATH")
        except OSError as exc:
            return self._fail(case, started, f"could not launch: {exc}")

        elapsed = time.monotonic() - started
        after = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
        # ru_maxrss is the high-water mark across ALL children, so it only
        # rises. The delta attributes the peak to this run when it is the
        # largest so far, and reports the standing mark otherwise.
        peak_kb = _to_kb(max(after - before, after if before == 0 else 0) or after)

        if proc.returncode != 0:
            tail = (proc.stderr or proc.stdout or "").strip().splitlines()
            hint = tail[-1][:160] if tail else ""
            return self._fail(case, started,
                              f"exit {proc.returncode}: {hint}", peak_kb)

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


def _to_kb(maxrss: int) -> int:
    """ru_maxrss is BYTES on macOS and KILOBYTES on Linux."""
    import sys
    return maxrss // 1024 if sys.platform == "darwin" else maxrss


def mflux_engine(model: str, quantize: int | None = 8,
                 steps: int | None = None) -> Engine:
    """mflux, the MLX-native image stack (mflux-generate-<model>)."""
    binary = f"mflux-generate-{model}"

    def argv(case: Case, out: Path, a: dict) -> list[str]:
        cmd = [binary, "--prompt", case.prompt, "--output", str(out)]
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

    return Engine(name=f"mflux/{model}", argv=argv)
