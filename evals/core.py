"""Case loading, scoring and summarizing for the eval suite.

The suite exists to answer one question cheaply: given several candidates for a
job, which one should this machine use? So everything here is shaped around
producing comparable rows, not around any particular model or engine.

A case is data. A runner turns (case, candidate) into an artifact. Scoring is
mechanical and shared, so two candidates are always judged by the same ruler.
"""
from __future__ import annotations

import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness.checks import adherence as adherence_check
from harness.checks import code as code_check
from harness.checks import html as html_check
from harness.checks import image as image_check
from harness.checks import ocr as ocr_check
from harness.checks import render as render_check
from harness.checks import video as video_check
from harness.checks import speech as speech_check
from harness.checks import svg as svg_check
from harness.checks.base import CheckResult, extract


@dataclass(frozen=True)
class Case:
    id: str
    modality: str
    prompt: str
    #: Material the model works ON, as opposed to the instruction. A log-triage
    #: case is one line of instruction and forty of material; keeping them
    #: apart leaves the prompt readable and the material swappable.
    context: str = ""
    #: Input audio for a transcription case, resolved beside the case file.
    audio: Path | None = None
    #: Which METHODS this case can fairly test, empty meaning all of them.
    #: chart-bars asserts must_contain ["text"], which a vectorizer cannot
    #: satisfy at any quality: tracing turns glyphs into outlines, so a traced
    #: SVG has no <text> element by construction. The case was written when an
    #: LLM was the only method and it encodes that assumption; handing it to
    #: another method measures the assumption rather than the candidate.
    methods: tuple = ()
    #: Which language the speech is in. Selection uses it: an English-only
    #: candidate handed a French case scores near 1.0 word error rate, which
    #: is a row about the wrong instrument rather than about the candidate.
    language: str = "en"
    #: Generation knobs handed to the engine (width, steps, seed...).
    params: dict = field(default_factory=dict)
    #: Checks applied to whatever came back.
    assertions: dict = field(default_factory=dict)
    source: Path | None = None


def _check_image(artifact, case: Case, adherence: str | None = None,
                 expect_size: tuple | None = None) -> CheckResult:
    """Honouring the requested size IS the check, so it reads from params.

    Width and height used to live under `assert:` and do both jobs at once,
    which meant the engine was reading the assertion block.
    """
    p = case.params
    # A WORKFLOW may legitimately change the resolution: an upscaler produces
    # a different size than the case asked to be generated, and failed every
    # case for doing exactly its job. The case says what to GENERATE; a runner
    # that scales says what it INTENDED.
    if expect_size:
        expect = tuple(expect_size)
    else:
        expect = ((p["width"], p["height"])
                  if p.get("width") and p.get("height") else None)
    r = image_check.check(artifact, expect=expect)
    # Pixels first, always. "no text found" is a true and useless diagnosis for
    # a uniform grey square, and it would send you looking at the wrong thing.
    if not r.ok:
        return CheckResult(r.ok, r.reason, r.warnings)

    metrics: dict = {}
    warnings_out = list(r.warnings)

    # Opt-in: it loads a multi-GB preference model, which is not a tax to put
    # on every image run.
    if adherence:
        a = adherence_check.check(artifact, case.prompt, backend=adherence)
        metrics.update(a.metrics)
        warnings_out += a.warnings

    expect_text = case.assertions.get("text")
    if not expect_text:
        out = CheckResult(True, "", warnings_out)
        out.metrics = metrics
        return out

    o = ocr_check.check(artifact, expect=expect_text,
                        max_cer=case.assertions.get("max_cer",
                                                    ocr_check.DEFAULT_MAX_CER))
    out = CheckResult(o.ok, o.reason, warnings_out + o.warnings)
    metrics.update(o.metrics)
    out.metrics = metrics
    return out


def _check_code(artifact, case: Case) -> CheckResult:
    """Runs the generated code. See harness/checks/code.py for what that means."""
    r = code_check.check(artifact, case.assertions.get("checks") or [])
    out = CheckResult(r.ok, r.reason, r.warnings)
    out.metrics = r.metrics
    return out


def _check_extract(artifact, case: Case) -> CheckResult:
    """The small, fast lane: read a blob, answer one narrow question.

    There is nothing structural to check -- the answer is prose -- so the whole
    verdict comes from the assertions, which score() applies. This exists so
    the modality is judged rather than silently passing.
    """
    return CheckResult(True, "")


def _check_tts(artifact, case: Case, transcriber=None,
               ref_audio=None) -> CheckResult:
    """The sentence that was asked for IS the reference transcript.

    `ref_audio` is the clip a cloning model was imitating. WER cannot see
    whether it succeeded: an intelligible clone in a completely different voice
    scores a perfect 0.000, so the property cloning exists to deliver went
    unmeasured until this.
    """
    out = speech_check.check(artifact, reference=case.prompt,
                             max_wer=case.assertions.get("max_wer"),
                             transcriber=transcriber,
                             language=case.language)
    if ref_audio:
        try:
            from harness.checks import similarity
            score = similarity.compare(ref_audio, artifact)
        except Exception as exc:  # noqa: BLE001
            # Never fail a tts row because the optional metrics group is
            # absent; the WER is still a real measurement without it.
            out.warnings.append(f"speaker similarity unavailable: {exc}")
        else:
            metrics = dict(getattr(out, "metrics", {}) or {})
            metrics["speaker_similarity"] = round(score, 4)
            out.metrics_extra = metrics
    return out


def _check_stt(artifact, case: Case) -> CheckResult:
    """The mirror of tts, and the reason it is a better measurement.

    Here the audio is the input and the prompt is a transcript a HUMAN wrote,
    so the word error rate is the STT model's alone. The tts round trip is
    joint with whatever reads it back and cannot separate the two.
    """
    rate = speech_check.wer(case.prompt, artifact, case.language)
    errors, words = speech_check.wer_counts(case.prompt, artifact,
                                            case.language)
    limit = case.assertions.get("max_wer")
    out = CheckResult(limit is None or rate <= limit, "")
    out.metrics = {"wer": round(rate, 4),
                   "wer_errors": errors, "wer_words": words}
    if not out.ok:
        out.reason = (f"word error rate {rate:.2f} over the limit of {limit}; "
                      f"heard {str(artifact).strip()!r}")
    return out


# One entry point per modality, so two candidates are always judged by the same
# ruler. Images were scored inside their runner and therefore lost every shared
# assertion; that fork is what this dict closes.
def _check_svg(artifact, case: Case) -> CheckResult:
    """Parse it, then draw it.

    Structural first, because unparseable markup should be reported as
    unparseable rather than as something that failed to render. Then rasterize,
    because well-formed SVG that draws nothing visible -- white on white, a
    shape outside the viewBox, everything behind an opaque rect -- passes every
    structural check there is.
    """
    r = svg_check.check(artifact)
    if not r.ok:
        return r

    drawn = render_check.check(extract(artifact, ("svg",)))
    out = CheckResult(drawn.ok, drawn.reason, r.warnings + drawn.warnings,
                      shape_count=r.shape_count, has_title=r.has_title)
    out.metrics = dict(drawn.metrics)
    # Issue #4: a traced icon is ~28KB where a hand-authored one is hundreds of
    # bytes, and nothing reported it until someone opened the file.
    out.metrics["svg_bytes"] = len(extract(artifact, ("svg",)).encode())
    return out


#: h3 takes --frames or --seconds and treats a second as 24 frames, so a case
#: may ask for either and the checker has to know they mean the same thing.
FPS = 24


def _check_video(artifact, case: Case) -> CheckResult:
    p = case.params
    expect = ((p["width"], p["height"])
              if p.get("width") and p.get("height") else None)
    frames = p.get("frames")
    if frames is None and p.get("seconds"):
        frames = int(p["seconds"]) * FPS
    r = video_check.check(artifact, expect=expect, frames=frames)
    out = CheckResult(r.ok, r.reason, r.warnings)
    out.metrics = r.metrics
    return out


def _check_web(artifact, case: Case) -> CheckResult:
    """Parse it, then render it in a browser.

    Structural first: prose where a page should be must be reported as prose,
    not as something that failed to paint. Then rasterize, because a page can
    parse, have a body, be perfectly self-contained, and show a white
    rectangle.
    """
    r = html_check.check(artifact)
    if not r.ok:
        return r

    drawn = render_check.check_html(extract(artifact, ("html", "!doctype")))
    out = CheckResult(drawn.ok, drawn.reason, r.warnings + drawn.warnings,
                      shape_count=r.shape_count, has_title=r.has_title)
    out.metrics = drawn.metrics
    return out


CHECKERS = {
    "svg": lambda a, c, **kw: _check_svg(a, c),
    "web": lambda a, c, **kw: _check_web(a, c),
    "image": lambda a, c, **kw: _check_image(a, c, kw.get("adherence"),
                                            kw.get("expect_size")),
    "video": lambda a, c, **kw: _check_video(a, c),
    "code": lambda a, c, **kw: _check_code(a, c),
    "extract": lambda a, c, **kw: _check_extract(a, c),
    "tts": _check_tts,
    "stt": lambda a, c, **kw: _check_stt(a, c),
}
# Modalities that may appear in a case file. video/tts/stt are declarable but
# not yet judgeable; score() says so rather than passing them.
MODALITIES = set(CHECKERS)

# Which way each metric runs. Not optional metadata: every metric was an error
# rate to begin with, so "lower is better" got baked into both the ranking and
# the worst-case column, and `ink` and `motion` silently inverted both the
# moment they arrived -- more ink IS better, and the "worst" ink in a run was
# being reported as the best-drawn case in it.
METRIC_DIRECTION = {
    "wer": "lower",        # word error rate
    "cer": "lower",        # character error rate of OCR'd text
    # NEUTRAL: reported, never ranked on. It exists as a FLOOR -- it catches
    # SVG that passes every structural check and renders as an empty
    # rectangle -- and the checker already fails those outright, so the
    # direction adds nothing above zero. As "higher" it crowned OmniSVG,
    # which had the worst pass rate in the svg lane and the best ink,
    # because drawing one big filled blob instead of the three shapes the
    # case asked for is what marking 85% of the canvas looks like.
    "ink": "neutral",      # fraction of an SVG canvas actually marked
    "motion": "higher",    # change between video frames
    "adherence": "higher",  # how well the picture matches the prompt
    # Does the clone sound like its reference? A WER cannot see this at
    # all: an intelligible clone in the wrong voice scores a perfect 0.
    "speaker_similarity": "higher",
    "tokens_per_s": "higher",
    # NEUTRAL: reported, never ranked on. More tokens is not better -- as
    # "higher" it would have put the most verbose candidate first. It is here
    # because it EXPLAINS a latency: q3-4b looked 4x slower than local-large
    # and was actually writing 2.7x as much, faster per token.
    "completion_tokens": "neutral",
    "code_pass": "higher",  # fraction of a code case's assertions that ran green
    # NEUTRAL: a traced illustration is legitimately large and a UI glyph is
    # legitimately small, so ranking on it would crown the blank document.
    "svg_bytes": "neutral",
}


def direction_of(metric: str) -> str:
    """"lower", "higher" or "neutral". Warns on an undeclared metric rather
    than guessing
    silently, since a wrong guess inverts a ranking with no visible symptom."""
    if metric not in METRIC_DIRECTION:
        warnings.warn(
            f"metric {metric!r} declares no direction in METRIC_DIRECTION; "
            f"assuming lower is better", UserWarning, stacklevel=2)
        return "lower"
    return METRIC_DIRECTION[metric]


# Generation knobs any engine might accept. Validated at load so `widht: 512`
# costs nothing instead of silently generating at the default size and passing.
PARAM_KEYS = {"width", "height", "steps", "seed", "guidance", "frames",
              "seconds", "voice", "speed", "layers", "reuse", "ssd_streaming"}
# Assertions that need text to search. Declaring one on an image case can only
# pass vacuously until the suite can OCR, so it is rejected rather than ignored.
TEXT_ASSERTIONS = {"min_shapes", "must_contain", "must_not_contain"}

_WORDY = re.compile(r"^\w+$")


def contains(artifact: str, needle: str) -> bool:
    """Is `needle` in `artifact`, without matching half a longer word?

    `needle in artifact.lower()` made "text" satisfied by "context" anywhere in
    the document -- a comment, a CSS class, an attribute value -- and
    chart-bars asserted exactly that needle. A bare word now needs word
    boundaries; anything carrying punctuation ("<table", 'type="password"')
    stays a raw substring, because that is how every author here already writes
    them and how they read.
    """
    if _WORDY.match(needle):
        return re.search(rf"\b{re.escape(needle)}\b", artifact, re.I) is not None
    return needle.lower() in artifact.lower()
#: `equals` is the narrowest and most useful shape for a delegated lookup: one
#: token out and nothing else, so the answer can be used without parsing.
EXTRACT_ASSERTIONS = {"must_contain", "must_not_contain", "equals"}
ASSERTION_KEYS = {"svg": TEXT_ASSERTIONS, "web": TEXT_ASSERTIONS,
                  "extract": EXTRACT_ASSERTIONS,
                  "code": {"checks"},
                  # `text` is OCR'd out of the produced image; `max_cer` is how
                  # wrong the rendering may be. Not must_contain: there is no
                  # text to search, and that could only ever pass vacuously.
                  "image": {"text", "max_cer"},
                  "tts": {"max_wer"},
                  "stt": {"max_wer"}}


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


# ---------------------------------------------------------------------------
# How a candidate failed, and whether two runs may be compared at all
# ---------------------------------------------------------------------------

#: Substrings that mean the candidate produced NOTHING, as opposed to producing
#: something wrong. Read off real failure details rather than invented.
_EMPTY_MARKS = ("empty completion", "no answer", "left no output",
                "no output", "wrote nothing", "returned no text")
#: Substrings that mean the INSTRUMENT broke. An outage is not a candidate
#: scoring badly, and recording it as one is how a dead server becomes a model
#: ranking -- whisper scored 0/40 that way and sorted above two working models.
_ERROR_MARKS = ("timed out", "timeout", "unreachable", "not installed",
                "could not launch", "connection", "http 5", "server error",
                # Metal says "GPU Timeout Error", never "timed out". The first
                # trace run hit one and it was scored as a wrong drawing.
                "[metal]", "out of memory", "traceback")


#: A subprocess that exits NON-ZERO crashed; it did not produce a bad
#: artifact. `exited 0 but left no output` is deliberately excluded by the
#: digit class, because that tool ran fine and simply wrote nothing.
_NONZERO_EXIT = re.compile(r"exited [1-9]")


def failure_kind(detail: str) -> str:
    """"" | "wrong" | "empty" | "error" for one failure's detail line.

    A pass rate says how OFTEN a candidate fails and never HOW. Qwen3-14B
    scored 7/9 on svg and looked like a winner; both failures were null
    content, because it is a thinking model that spent the whole token budget
    reasoning. Nothing in the table could show that.
    """
    if not detail:
        return ""
    low = detail.lower()
    # Order matters: an instrument failure often also produced nothing, and the
    # broken instrument is the more useful of the two readings.
    if _NONZERO_EXIT.search(low) or any(m in low for m in _ERROR_MARKS):
        return "error"
    if any(m in low for m in _EMPTY_MARKS):
        return "empty"
    return "wrong"


@dataclass(frozen=True)
class Receipt:
    """What a run WAS, so two runs can be told apart before being ranked.

    Borrowed from EnviousWispr's `model_registry.comparable()`. The axes are
    the ones that change what the exam asks, and each exclusion is argued in
    `comparable()` so it can be challenged rather than discovered.
    """
    modality: str
    case_ids: tuple
    repeat: int
    sampling: dict
    gateway: str
    adherence: str = ""
    #: Which cost tier produced these rows. A screen answers "did it run"; a
    #: measurement answers "is it better". Ranking one against the other
    #: compares two different exams.
    tier: str = "measure"

    def as_dict(self) -> dict:
        return {"modality": self.modality, "case_ids": list(self.case_ids),
                "repeat": self.repeat, "sampling": dict(self.sampling),
                "gateway": self.gateway, "adherence": self.adherence,
                "tier": self.tier}


def comparable(a: Receipt, b: Receipt) -> tuple[bool, str]:
    """May these two runs be ranked in one table?

    IN, because each changes what was asked:
      * `modality` and `case_ids` -- a different exam, or a different number of
        questions on it. The ids and not just the count: nine easy cases and
        nine hard ones are not one lane.
      * `sampling` -- adding a repetition penalty changed what the svg lane
        produces, so a run from before it is a different exam from one after.
      * `adherence` -- PickScore and HPSv2 are two graders.

    OUT, each for a stated reason:
      * TIMING AND MEMORY. They are outputs of the run, not properties of the
        exam. Two runs may be compared on quality while their latencies are not
        comparable at all, which is exactly the warm-versus-cold trap: an eval
        median amortises the model load across cases and a one-shot CLI call
        does not.
      * `gateway` HOST. The same aliases served from another machine answer the
        same questions. The alias NAMES are part of the candidate, not the run.
      * Wall-clock time and git sha. Recorded in the receipt for provenance,
        deliberately not compared: a commit that touches the README does not
        invalidate a measurement, and a commit that touches sampling is already
        caught by `sampling`.
    """
    if a.tier != b.tier:
        return False, (f"different tier: {a.tier} vs {b.tier}. A screen asks "
                       f"whether it ran; a measurement asks whether it is "
                       f"better")
    if a.modality != b.modality:
        return False, f"different modality: {a.modality} vs {b.modality}"
    if tuple(a.case_ids) != tuple(b.case_ids):
        return False, (f"different case set: {len(a.case_ids)} vs "
                       f"{len(b.case_ids)} cases")
    if a.repeat != b.repeat:
        return False, f"different repeat: {a.repeat} vs {b.repeat}"
    if dict(a.sampling) != dict(b.sampling):
        return False, f"different sampling: {a.sampling} vs {b.sampling}"
    if a.adherence != b.adherence:
        return False, (f"different adherence backend: {a.adherence!r} vs "
                       f"{b.adherence!r}")
    return True, "same exam"


def load_cases(directory: str | Path) -> list[Case]:
    """Load every *.yaml under `directory`, sorted by id for stable runs."""
    directory = Path(directory)
    cases: list[Case] = []
    for path in sorted(directory.rglob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
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
        context = _load_context(path, raw)
        audio = _load_audio(path, raw, modality)
        params = raw.get("params") or {}
        assertions = raw.get("assert") or {}
        if modality == "code" and not assertions.get("checks"):
            # A code case with nothing to run passes every model, which is
            # worse than not having the case at all.
            raise ValueError(f"{path.name}: a code case needs assert.checks")
        _reject_unknown(path, modality, "params", set(params), PARAM_KEYS)
        _reject_unknown(path, modality, "assert", set(assertions),
                        ASSERTION_KEYS.get(modality, set()))
        cases.append(Case(id=raw["id"], modality=modality, prompt=raw["prompt"],
                          context=context, audio=audio, params=params,
                          assertions=assertions, source=path,
                          language=raw.get("language") or "en",
                          methods=tuple(raw.get("methods") or ())))
    return cases


def _load_context(path: Path, raw: dict) -> str:
    """Inline `context:`, or `context_file:` resolved beside the case."""
    inline, filename = raw.get("context"), raw.get("context_file")
    if inline and filename:
        raise ValueError(
            f"{path.name}: set 'context' or 'context_file', not both")
    if filename:
        source = path.parent / filename
        if not source.exists():
            raise ValueError(f"{path.name}: context_file '{filename}' not found "
                             f"beside the case")
        return source.read_text(encoding="utf-8")
    return inline or ""


def _load_audio(path: Path, raw: dict, modality: str) -> Path | None:
    """`audio_file:`, absolute or resolved beside the case.

    Absolute is the normal shape for a corpus: LibriSpeech lives on a volume
    and its cases are generated per machine rather than shipped with the repo.
    """
    filename = raw.get("audio_file")
    if not filename:
        if modality == "stt":
            raise ValueError(f"{path.name}: an stt case needs audio_file")
        return None
    source = Path(filename)
    if not source.is_absolute():
        source = path.parent / filename
    if not source.exists():
        raise ValueError(f"{path.name}: audio_file '{filename}' not found")
    return source


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
    # metrics_extra lets a checker ADD to a dataclass property it does not
    # own; SpeechResult.metrics is computed, not stored.
    metrics = getattr(r, "metrics_extra", None) or getattr(r, "metrics", {})
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
        if not contains(artifact, needle):
            return Result(case.id, "", False, 0.0, 0,
                          f"missing required content: {needle}",
                          warnings=r.warnings)

    for needle in a.get("must_not_contain") or []:
        if needle.lower() in artifact.lower():
            return Result(case.id, "", False, 0.0, 0,
                          f"contains forbidden content: {needle}",
                          warnings=r.warnings)

    exact = a.get("equals")
    if exact is not None and str(artifact).strip().lower() != str(exact).strip().lower():
        return Result(case.id, "", False, 0.0, 0,
                      f"expected exactly {exact!r}, got {str(artifact).strip()!r}",
                      warnings=r.warnings, metrics=metrics)

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
            # The FIRST case, kept beside the median rather than replaced by
            # it. Measured 1.5x to 8.4x the median across real runs, and the
            # image lane has recorded 239s against a 43s warm steady state.
            # A one-shot caller -- `lh say`, `lh hear`, the MCP server -- meets
            # this number, never the median. Issue #89.
            #
            # It is only a COLD measurement for the first candidate in a run;
            # later ones may inherit a warm cache or pay a model swap instead,
            # so `first_is_cold` says which this is rather than implying it.
            "first_s": round(times[0], 3) if times else 0.0,
            "first_is_cold": candidate == next(iter(by)),
            "total_s": round(sum(times), 1),
            "peak_kb": max((r.peak_kb for r in rows), default=0),
            "warnings": sum(len(r.warnings) for r in rows),
            "metrics": _mean_metrics(rows),
            # Kept alongside the mean because an average of mostly-zeros hides
            # the one case that fell over, which is usually the interesting one.
            "metrics_worst": _worst_metrics(rows),
            "failures": [f"{r.case_id}: {r.detail}" for r in rows
                         if not r.passed],
            # HOW they failed, not just how many. See failure_kind().
            "wrong": sum(1 for r in rows
                         if not r.passed and failure_kind(r.detail) == "wrong"),
            "empty": sum(1 for r in rows
                         if not r.passed and failure_kind(r.detail) == "empty"),
            "errored": sum(1 for r in rows
                           if not r.passed and failure_kind(r.detail) == "error"),
            # How many rows each metric was actually computed over. A mean over
            # 2 of 9 cases printed beside a mean over 9 is not a comparison.
            "metric_n": {k: len(v) for k, v in _gather_metrics(rows).items()},
            # WHICH cases this candidate actually sat. Two candidates in one
            # run can get different sets -- a case may be unfair to a method,
            # or to a language -- and then their pass rates are not comparable.
            "case_ids": sorted({r.case_id.split("#")[0] for r in rows}),
        }
    return out


def _gather_metrics(rows: list[Result]) -> dict[str, list[float]]:
    gathered: dict[str, list[float]] = {}
    for r in rows:
        for name, value in (r.metrics or {}).items():
            if isinstance(value, (int, float)):
                gathered.setdefault(name, []).append(float(value))
    return gathered


#: Metrics that are RATIOS, with the numerator and denominator they are made
#: of. Aggregating these as a mean of per-case rates lets a two-word utterance
#: weigh as much as a forty-word one; the correct total is sum(errors) over
#: sum(words), which is what every ASR benchmark means by "WER".
RATIO_METRICS = {"wer": ("wer_errors", "wer_words")}
#: Bookkeeping that should not appear as a column of its own.
_COMPANIONS = {name for pair in RATIO_METRICS.values() for name in pair}


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
    gathered = _gather_metrics(rows)
    out = {}
    for name, vals in gathered.items():
        if name in _COMPANIONS:
            continue
        numerator, denominator = RATIO_METRICS.get(name, (None, None))
        if numerator in gathered and denominator in gathered:
            total = sum(gathered[denominator])
            out[name] = round(sum(gathered[numerator]) / total, 4) if total else 0.0
        else:
            out[name] = round(statistics.fmean(vals), 4)
    return out


def _worst_metrics(rows: list[Result]) -> dict:
    """The worst value of each metric, which way round depending on direction."""
    return {name: round(max(vals) if direction_of(name) == "lower" else min(vals), 4)
            for name, vals in _gather_metrics(rows).items()
            if name not in _COMPANIONS}
