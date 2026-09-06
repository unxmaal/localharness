"""Case loading, scoring and summarizing for the eval suite.

The suite exists to answer one question cheaply: given several candidates for a
job, which one should this machine use? So everything here is shaped around
producing comparable rows, not around any particular model or engine.

A case is data. A runner turns (case, candidate) into an artifact. Scoring is
mechanical and shared, so two candidates are always judged by the same ruler.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness.checks import html as html_check
from harness.checks import image as image_check
from harness.checks import svg as svg_check
from harness.checks.base import CheckResult


@dataclass(frozen=True)
class Case:
    id: str
    modality: str
    prompt: str
    #: Generation knobs handed to the engine (width, steps, seed...).
    params: dict = field(default_factory=dict)
    #: Checks applied to whatever came back.
    assertions: dict = field(default_factory=dict)
    source: Path | None = None


def _check_text(artifact, case: Case, checker) -> CheckResult:
    return checker(artifact)


def _check_image(artifact, case: Case) -> CheckResult:
    """Honouring the requested size IS the check, so it reads from params.

    Width and height used to live under `assert:` and do both jobs at once,
    which meant the engine was reading the assertion block.
    """
    p = case.params
    expect = (p["width"], p["height"]) if p.get("width") and p.get("height") else None
    r = image_check.check(artifact, expect=expect)
    return CheckResult(r.ok, r.reason, r.warnings)


# One entry point per modality, so two candidates are always judged by the same
# ruler. Images were scored inside their runner and therefore lost every shared
# assertion; that fork is what this dict closes.
CHECKERS = {
    "svg": lambda a, c: _check_text(a, c, svg_check.check),
    "web": lambda a, c: _check_text(a, c, html_check.check),
    "image": _check_image,
}
# Modalities that may appear in a case file. video/tts/stt are declarable but
# not yet judgeable; score() says so rather than passing them.
MODALITIES = set(CHECKERS) | {"video", "tts", "stt"}

# Generation knobs any engine might accept. Validated at load so `widht: 512`
# costs nothing instead of silently generating at the default size and passing.
PARAM_KEYS = {"width", "height", "steps", "seed", "guidance", "frames",
              "seconds", "voice", "speed", "layers", "reuse", "ssd_streaming"}
# Assertions that need text to search. Declaring one on an image case can only
# pass vacuously until the suite can OCR, so it is rejected rather than ignored.
TEXT_ASSERTIONS = {"min_shapes", "must_contain", "must_not_contain"}
ASSERTION_KEYS = {"svg": TEXT_ASSERTIONS, "web": TEXT_ASSERTIONS}


@dataclass
class Result:
    case_id: str
    candidate: str
    passed: bool
    seconds: float
    peak_kb: int
    detail: str
    artifact: str | None = None
    warnings: list[str] = field(default_factory=list)


def load_cases(directory: str | Path) -> list[Case]:
    """Load every *.yaml under `directory`, sorted by id for stable runs."""
    directory = Path(directory)
    cases: list[Case] = []
    for path in sorted(directory.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text()) or {}
        # Name the file in every error: a broken case in a 50-case run must not
        # fail anonymously.
        for required in ("id", "modality", "prompt"):
            if not raw.get(required):
                raise ValueError(f"{path.name}: missing '{required}'")
        modality = raw["modality"]
        if modality not in MODALITIES:
            raise ValueError(
                f"{path.name}: unknown modality '{modality}' "
                f"(known: {', '.join(sorted(MODALITIES))})")
        params = raw.get("params") or {}
        assertions = raw.get("assert") or {}
        _reject_unknown(path, "params", set(params), PARAM_KEYS)
        _reject_unknown(path, "assert", set(assertions),
                        ASSERTION_KEYS.get(modality, set()))
        cases.append(Case(id=raw["id"], modality=modality, prompt=raw["prompt"],
                          params=params, assertions=assertions, source=path))
    return cases


def _reject_unknown(path: Path, block: str, given: set, allowed: set) -> None:
    unknown = given - allowed
    if unknown:
        raise ValueError(
            f"{path.name}: unknown key(s) in '{block}': "
            f"{', '.join(sorted(unknown))}"
            + (f" (allowed: {', '.join(sorted(allowed))})" if allowed
               else f" ('{block}' takes nothing for modality {path.parent.name})"))


def score(case: Case, artifact) -> Result:
    """Judge an artifact against its case. Mechanical, so it is comparable.

    `artifact` is the completion text for a text modality and the path to the
    produced file for a binary one.
    """
    checker = CHECKERS.get(case.modality)
    if checker is None:
        # Not a pass. A modality with nothing measuring it would otherwise show
        # a 100% pass rate, which is the most misleading number the suite could
        # print.
        return Result(case.id, "", False, 0.0, 0,
                      f"no checker for modality '{case.modality}'")

    r = checker(artifact, case)
    if not r.ok:
        return Result(case.id, "", False, 0.0, 0, r.reason,
                      warnings=r.warnings)

    a = case.assertions
    if not a:
        return Result(case.id, "", True, 0.0, 0, "", warnings=r.warnings)

    min_shapes = a.get("min_shapes")
    if min_shapes is not None and r.shape_count < min_shapes:
        return Result(case.id, "", False, 0.0, 0,
                      f"expected at least {min_shapes} shapes, drew "
                      f"{r.shape_count}", warnings=r.warnings)

    for needle in a.get("must_contain") or []:
        if needle.lower() not in artifact.lower():
            return Result(case.id, "", False, 0.0, 0,
                          f"missing required content: {needle}",
                          warnings=r.warnings)

    for needle in a.get("must_not_contain") or []:
        if needle.lower() in artifact.lower():
            return Result(case.id, "", False, 0.0, 0,
                          f"contains forbidden content: {needle}",
                          warnings=r.warnings)

    return Result(case.id, "", True, 0.0, 0, "", warnings=r.warnings)


def summarize(results: list[Result]) -> dict:
    """Roll rows up per candidate into something you can rank on."""
    by: dict[str, list[Result]] = {}
    for r in results:
        by.setdefault(r.candidate, []).append(r)

    out: dict[str, dict] = {}
    for candidate, rows in by.items():
        times = [r.seconds for r in rows]
        out[candidate] = {
            "total": len(rows),
            "passed": sum(1 for r in rows if r.passed),
            "pass_rate": round(sum(1 for r in rows if r.passed) / len(rows), 3),
            # Median, not mean: one cold model load should not decide which
            # candidate looks fastest.
            "median_s": round(statistics.median(times), 3) if times else 0.0,
            "total_s": round(sum(times), 1),
            "peak_kb": max((r.peak_kb for r in rows), default=0),
            "warnings": sum(len(r.warnings) for r in rows),
            "failures": [f"{r.case_id}: {r.detail}" for r in rows
                         if not r.passed],
        }
    return out
