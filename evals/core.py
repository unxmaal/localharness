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
from harness.checks import speech as speech_check
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


def _check_image(artifact, case: Case) -> CheckResult:
    """Honouring the requested size IS the check, so it reads from params.

    Width and height used to live under `assert:` and do both jobs at once,
    which meant the engine was reading the assertion block.
    """
    p = case.params
    expect = (p["width"], p["height"]) if p.get("width") and p.get("height") else None
    r = image_check.check(artifact, expect=expect)
    return CheckResult(r.ok, r.reason, r.warnings)


def _check_tts(artifact, case: Case, transcriber=None) -> CheckResult:
    """The sentence that was asked for IS the reference transcript."""
    return speech_check.check(artifact, reference=case.prompt,
                              max_wer=case.assertions.get("max_wer"),
                              transcriber=transcriber)


# One entry point per modality, so two candidates are always judged by the same
# ruler. Images were scored inside their runner and therefore lost every shared
# assertion; that fork is what this dict closes.
CHECKERS = {
    "svg": lambda a, c, **kw: svg_check.check(a),
    "web": lambda a, c, **kw: html_check.check(a),
    "image": lambda a, c, **kw: _check_image(a, c),
    "tts": _check_tts,
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
ASSERTION_KEYS = {"svg": TEXT_ASSERTIONS, "web": TEXT_ASSERTIONS,
                  "tts": {"max_wer"}}


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
    #: Numbers a checker produced alongside its verdict (wer, ocr score...).
    #: A pass rate separates working from broken; these are what put two
    #: working candidates in an order.
    metrics: dict = field(default_factory=dict)


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
        _reject_unknown(path, modality, "params", set(params), PARAM_KEYS)
        _reject_unknown(path, modality, "assert", set(assertions),
                        ASSERTION_KEYS.get(modality, set()))
        cases.append(Case(id=raw["id"], modality=modality, prompt=raw["prompt"],
                          params=params, assertions=assertions, source=path))
    return cases


def _reject_unknown(path: Path, modality: str, block: str, given: set,
                    allowed: set) -> None:
    unknown = given - allowed
    if unknown:
        raise ValueError(
            f"{path.name}: unknown key(s) in '{block}': "
            f"{', '.join(sorted(unknown))}"
            + (f" (allowed for modality '{modality}': "
               f"{', '.join(sorted(allowed))})" if allowed
               else f" ('{block}' takes nothing for modality '{modality}')"))


def score(case: Case, artifact, **checker_kwargs) -> Result:
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

    r = checker(artifact, case, **checker_kwargs)
    metrics = getattr(r, "metrics", {})
    if not r.ok:
        return Result(case.id, "", False, 0.0, 0, r.reason,
                      warnings=r.warnings, metrics=metrics)

    a = case.assertions
    if not a:
        return Result(case.id, "", True, 0.0, 0, "", warnings=r.warnings,
                      metrics=metrics)

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

    return Result(case.id, "", True, 0.0, 0, "", warnings=r.warnings,
                  metrics=metrics)


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
            "metrics": _mean_metrics(rows),
            # Kept alongside the mean because an average of mostly-zeros hides
            # the one case that fell over, which is usually the interesting one.
            "metrics_worst": _worst_metrics(rows),
            "failures": [f"{r.case_id}: {r.detail}" for r in rows
                         if not r.passed],
        }
    return out


def _gather_metrics(rows: list[Result]) -> dict[str, list[float]]:
    gathered: dict[str, list[float]] = {}
    for r in rows:
        for name, value in (r.metrics or {}).items():
            if isinstance(value, (int, float)):
                gathered.setdefault(name, []).append(float(value))
    return gathered


def _mean_metrics(rows: list[Result]) -> dict:
    """Mean of each metric across the rows that reported it.

    MEAN, not median, and the difference matters. Latency takes a median so one
    cold model load cannot decide which candidate looks fastest. A quality
    metric is the opposite case: most cases score a clean 0.0 and the entire
    signal lives in the few that do not, so a median reports 0.0 for a
    candidate that mangled a whole case. Measured: two Kokoro voices both
    medianed 0.0 while one of them had a 0.15 word error rate on a case the
    other got perfect.

    A candidate that reported nothing gets an empty dict rather than zeros:
    zero is the best possible error rate and would rank an unmeasured candidate
    first.
    """
    return {name: round(statistics.fmean(vals), 4)
            for name, vals in _gather_metrics(rows).items()}


def _worst_metrics(rows: list[Result]) -> dict:
    """The worst value of each metric. Every metric so far is an error rate,
    where worst means highest; a metric where higher is better would need this
    to know its direction."""
    return {name: round(max(vals), 4)
            for name, vals in _gather_metrics(rows).items()}
