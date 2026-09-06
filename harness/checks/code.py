"""Does the generated code work?

Every other check in this suite inspects an artifact. This one RUNS it, because
a syntax check is nearly worthless for code: the interesting failures all parse.
An off-by-one, a reversed comparison, a forgotten edge case -- all valid Python.

    THIS EXECUTES MODEL-GENERATED CODE.

The bounds are a subprocess, a wall-clock timeout, and a scratch working
directory, and that is the whole of the sandbox. There is no seccomp, no
container, and no filesystem restriction beyond the cwd: code that calls
shutil.rmtree on an absolute path will do it. That is an acceptable trade for
running your own local models against your own cases on your own machine, and
it is not acceptable for running an eval suite someone else wrote. Read the
cases before you run them.

The result is a FRACTION, not a verdict: three assertions out of four is a
different model from zero out of four, and a binary pass throws that away.
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from harness.checks.base import extract

DEFAULT_TIMEOUT = 15.0

# Runs each check in its own try/except so one exception does not hide the
# results of the others, and reports as JSON on the last line so a print() in
# the generated code cannot be mistaken for the result.
_HARNESS = '''
import json as _json, sys as _sys

def raises(_fn, *_args, exc=Exception, **_kwargs):
    """True if calling _fn raises `exc`. Available to every check expression.

    "It raises ValueError on bad input" is a normal thing to assert and there
    is no readable way to write it as a bare expression. Note it re-raises
    anything that is NOT `exc`, so a NameError in the generated code cannot
    pass as correct error handling.
    """
    try:
        _fn(*_args, **_kwargs)
    except exc:
        return True
    return False

_results = []
for _i, _src in enumerate({checks!r}):
    try:
        _ok = bool(eval(_src))
        _results.append([_i, _ok, "" if _ok else "returned False"])
    except Exception as _e:
        _results.append([_i, False, f"{{type(_e).__name__}}: {{_e}}"])
_sys.stderr.write("\\n__RESULTS__" + _json.dumps(_results) + "\\n")
'''


@dataclass
class CodeResult:
    ok: bool
    reason: str = ""
    warnings: list[str] = field(default_factory=list)
    passed: int = 0
    total: int = 0

    @property
    def metrics(self) -> dict:
        return {"code_pass": round(self.passed / self.total, 4) if self.total else 0.0}


def recover(text: str) -> str:
    """The code, out of whatever prose the model wrapped it in."""
    return extract(text, ())


def check(source: str, checks: list[str],
          timeout: float = DEFAULT_TIMEOUT) -> CodeResult:
    """Run `source`, then evaluate each expression in `checks` against it."""
    if not checks:
        # A code case with nothing to run would pass every model, which is
        # worse than not having the case.
        raise ValueError("a code check needs at least one assertion to run")

    body = recover(source)
    if not body.strip():
        return CodeResult(False, "no code in the response", total=len(checks))

    program = body + "\n" + _HARNESS.format(checks=checks)
    with tempfile.TemporaryDirectory() as scratch:
        script = Path(scratch) / "candidate.py"
        script.write_text(program)
        try:
            proc = subprocess.run(
                [sys.executable, str(script)], capture_output=True, text=True,
                timeout=timeout,
                # Scratch cwd, so generated code that writes a file does not
                # write it into the repo.
                cwd=scratch)
        except subprocess.TimeoutExpired:
            return CodeResult(False, f"timed out after {timeout}s",
                              total=len(checks))

    marker = "__RESULTS__"
    line = next((ln for ln in (proc.stderr or "").splitlines()
                 if ln.startswith(marker)), None)
    if line is None:
        # The module never reached the harness: an import error, a syntax
        # error, or something raised at definition time.
        detail = (proc.stderr or "").strip().splitlines()
        summary = detail[-1] if detail else f"exit {proc.returncode}"
        kind = "syntax error" if "SyntaxError" in (proc.stderr or "") else "failed to run"
        return CodeResult(False, f"{kind}: {summary}", total=len(checks))

    results = json.loads(line[len(marker):])
    passed = sum(1 for _, ok, _ in results if ok)
    if passed == len(checks):
        return CodeResult(True, "", passed=passed, total=len(checks))

    first = next((r for r in results if not r[1]), None)
    detail = f"{checks[first[0]]} -> {first[2]}" if first else ""
    return CodeResult(False, f"{passed}/{len(checks)} checks passed; {detail}",
                      passed=passed, total=len(checks))
