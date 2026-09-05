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

from evals.checks import html as html_check
from evals.checks import svg as svg_check

# Modalities the harness knows how to judge. Anything else is a typo, and a
# typo should cost nothing rather than showing up after a 40-minute run.
CHECKERS = {"svg": svg_check.check, "web": html_check.check}
MODALITIES = set(CHECKERS) | {"image", "video", "tts", "stt"}


@dataclass(frozen=True)
class Case:
    id: str
    modality: str
    prompt: str
    assertions: dict = field(default_factory=dict)
    source: Path | None = None


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
        cases.append(Case(id=raw["id"], modality=modality, prompt=raw["prompt"],
                          assertions=raw.get("assert") or {}, source=path))
    return cases


def score(case: Case, artifact: str) -> Result:
    """Judge an artifact against its case. Mechanical, so it is comparable."""
    checker = CHECKERS.get(case.modality)
    if checker is None:
        # Binary modalities (image/video/audio) are scored by their runner,
        # which knows how to look at the file it produced.
        return Result(case.id, "", True, 0.0, 0, "")

    r = checker(artifact)
    if not r.ok:
        return Result(case.id, "", False, 0.0, 0, r.reason,
                      warnings=r.warnings)

    a = case.assertions
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
