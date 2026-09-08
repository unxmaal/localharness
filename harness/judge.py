"""Score a proposal 1-10 with a cheap local model and a versioned rubric. #54.

The cheapest tier. It reads text and spends no GPU generation, so it can triage
the ~47 proposals a sweep produces before anything is run.

A JUDGE IS A METRIC, and a metric without a negative control is noise. This
project retracted a whole set of speaker-similarity numbers for skipping that
step. `control()` scores items whose real outcome is already known; run it and
check the separation before quoting any score.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import yaml

from harness import completion

RUBRIC_DIR = Path(__file__).resolve().parent / "rubrics"
DEFAULT_RUBRIC = "novelty"


class JudgeError(RuntimeError):
    """The rubric could not be loaded or the model gave no usable score."""


@dataclass
class Rubric:
    name: str
    version: int
    model: str
    low: int
    high: int
    prompt: str

    @property
    def identity(self) -> str:
        """What makes two judged runs comparable. See evals.core.comparable."""
        return f"{self.name}@{self.version}"


def load(name: str = DEFAULT_RUBRIC, directory: Path | None = None) -> Rubric:
    path = Path(directory or RUBRIC_DIR) / f"{name}.yaml"
    try:
        raw = yaml.safe_load(path.read_text()) or {}
    except OSError as exc:
        known = sorted(p.stem for p in Path(directory or RUBRIC_DIR).glob("*.yaml"))
        raise JudgeError(f"no rubric {name!r} at {path} "
                         f"(known: {', '.join(known) or 'none'})") from exc
    scale = raw.get("scale") or [1, 10]
    body = [raw.get("question", "").strip(), ""]
    for key, header in (("what_scores_high", "SCORES HIGH"),
                        ("what_scores_low", "SCORES LOW")):
        if raw.get(key):
            body.append(f"{header}:")
            body += [f"- {item.strip()}" for item in raw[key]]
            body.append("")
    body.append(raw.get("instructions", "").strip())
    return Rubric(name=raw.get("name", name), version=int(raw.get("version", 0)),
                  model=raw.get("model", "q3-4b"), low=int(scale[0]),
                  high=int(scale[1]), prompt="\n".join(body).strip())


_SCORE = re.compile(r"\b(10|[1-9])\b")


def parse_score(text: str, low: int = 1, high: int = 10) -> tuple[int, str]:
    """First integer in range, plus the rest as the reasoning."""
    for line in text.strip().splitlines():
        m = _SCORE.search(line)
        if m and low <= int(m.group(1)) <= high:
            score = int(m.group(1))
            why = line[m.end():].strip(" .:-")
            rest = text.strip().splitlines()[1:]
            return score, (why or " ".join(rest)).strip()[:300]
    raise JudgeError(f"no score in {low}-{high} found in: {text.strip()[:160]!r}")


def describe(name: str, why: str = "", source: str = "", times_seen: int = 0,
             relevance: int = 0, inspected: str = "", platform: str = "",
             weights: str = "") -> str:
    """What the model is shown.

    Provenance goes in alongside the prose deliberately: a judge given only a
    description is partly scoring copywriting, and the recap entries this reads
    are one-line marketing blurbs.

    THE INSPECT RESULT GOES IN TOO, and that fixed a real disagreement. The
    judge scored apple/coreai-models 3/10 off a generic description while the
    inspect tier had already cloned it and established it was MLX-native with
    weights that fit. The tier holding more evidence lost to the tier holding
    less. These fields are facts read from a source tree, not prose. Issue #69.
    """
    bits = [f"NAME: {name}"]
    if why:
        bits.append(f"DESCRIPTION: {why}")
    if source:
        bits.append(f"SEEN IN: {source}")
    if times_seen:
        bits.append(f"TIMES SEEN: {times_seen}")
    if relevance:
        bits.append(f"APPLE SILICON RELEVANCE: {relevance:+d}")
    if inspected:
        bits.append(f"READ FROM ITS SOURCE: {inspected}")
    if platform:
        bits.append(f"RUNTIME: {platform}")
    if weights:
        bits.append(f"WEIGHTS IT NAMES: {weights}")
    return "\n".join(bits)


def score(item: str, rubric: Rubric | None = None, *, gateway: str = "",
          complete=None) -> tuple[int, str]:
    """Score one described item. Returns (score, reasoning)."""
    rubric = rubric or load()
    complete = complete or completion.complete
    kw = {"model": rubric.model, "modality": "extract"}
    if gateway:
        kw["gateway"] = gateway
    text = complete(f"{rubric.prompt}\n\n---\n{item}\n---", **kw)
    return parse_score(text, rubric.low, rubric.high)


#: Items whose real outcome is already known here, for calibrating a rubric.
#: Descriptions only, as a feed would carry them: the judge must reach the right
#: order without being told the answer.
CONTROL = [
    ("trace",
     "Generate a raster image with a diffusion model, then vectorize it to SVG "
     "paths with vtracer. Composes two tools that are already installed.",
     "won"),
    ("repair",
     "Generate, run the existing checker on the output, then ask the same model "
     "again with the checker's complaint attached. No new model or download.",
     "won"),
    ("upscale-seedvr2",
     "SeedVR2 diffusion super-resolution as a second stage after generation.",
     "lost"),
    ("local-small",
     "A 0.5B instruct model, the smallest in its family, for writing SVG markup.",
     "lost"),
    ("q3-14b",
     "A 14B hybrid reasoning model that emits its thinking separately from its "
     "answer.",
     "lost"),
]


def control(rubric: Rubric | None = None, *, gateway: str = "",
            complete=None) -> dict:
    """Score known-good against known-bad and report whether they separate.

    Run this BEFORE quoting any figure from this judge. If the classes overlap,
    the rubric does not discriminate on this data and no ranking may be drawn
    from it, however sensible the scores look individually.
    """
    rubric = rubric or load()
    rows = []
    for name, why, outcome in CONTROL:
        s, reason = score(describe(name, why), rubric, gateway=gateway,
                          complete=complete)
        rows.append({"name": name, "outcome": outcome, "score": s,
                     "why": reason})
    won = [r["score"] for r in rows if r["outcome"] == "won"]
    lost = [r["score"] for r in rows if r["outcome"] == "lost"]
    return {"rubric": rubric.identity, "model": rubric.model, "rows": rows,
            "won_min": min(won), "lost_max": max(lost),
            "gap": min(won) - max(lost),
            "separates": min(won) > max(lost)}
