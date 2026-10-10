"""Case loading, scoring and summarizing for the eval suite.

The suite exists to answer one question cheaply: given several candidates for a
job, which one should this machine use? So everything here is shaped around
producing comparable rows, not around any particular model or engine.

A case is data. A runner turns (case, candidate) into an artifact. Scoring is
mechanical and shared, so two candidates are always judged by the same ruler.
"""
from __future__ import annotations

import hashlib
import json
import re
import statistics
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from harness import reasons
from harness.checks import adherence as adherence_check
from harness.checks import claims as claims_check
from harness.checks import code as code_check
from harness.checks import decide as decide_check
from harness.checks import html as html_check
from harness.checks import image as image_check
from harness.checks import music as music_check
from harness.checks import ocr as ocr_check
from harness.checks import pii as pii_check
from harness.checks import render as render_check
from harness.checks import retrieval as retrieval_check
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
    #: A file the candidate reads, resolved beside the case: the ocr lane's image. #562.
    input_file: Path | None = None
    #: Loaded from the local-only tree: its text never leaves this machine. #654.
    private: bool = False
    #: Source run before a code answer, from preamble.py beside the case: an import source's harness. #657.
    preamble: str = ""


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
    out = CheckResult(o.ok, o.reason, warnings_out + o.warnings,
                      failure_class=o.failure_class)
    metrics.update(o.metrics)
    out.metrics = metrics
    return out


def _check_code(artifact, case: Case) -> CheckResult:
    """Runs the generated code. See harness/checks/code.py for what that means."""
    r = code_check.check(artifact, case.assertions.get("checks") or [],
                         preamble=getattr(case, "preamble", ""))
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


def _check_music(artifact, case: Case, transcriber=None, **_) -> CheckResult:
    """The lyrics that were asked for ARE the reference transcript.

    The tts round-trip applied to singing, and the same joint measurement: the
    rate covers the generator and the ear together.

    `expect_vocals: false` inverts it. An instrumental case scores whether
    anything was sung at all rather than whether the right words were, which
    makes it both a real capability check and the lane's own negative control
    -- the thing that makes a sung rate mean something instead of being a bare
    number.
    """
    a = case.assertions
    r = music_check.check(artifact,
                          lyrics=str(case.params.get("lyrics", "")),
                          max_wer=a.get("max_wer"),
                          duration_s=case.params.get("duration"),
                          expect_vocals=a.get("expect_vocals", True),
                          transcriber=transcriber,
                          language=case.language)
    out = CheckResult(r.ok, r.reason, list(r.warnings))
    out.metrics = r.metrics
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


def _check_decide(artifact, case: Case) -> CheckResult:
    if isinstance(artifact, Path):
        # A process engine wrote the canonical JSON to a file.
        if not artifact.exists():
            return CheckResult(False, f"engine left no output at {artifact.name}")
        artifact = artifact.read_text(encoding="utf-8")
    return decide_check.check(artifact, case.params["schema"],
                              case.assertions["answers"])


def _check_claims(artifact, case: Case) -> CheckResult:
    """Schema first, then emitted claims matched to the reviewed ones. #654."""
    if isinstance(artifact, Path):
        if not artifact.exists():
            return CheckResult(False, f"engine left no output at {artifact.name}")
        artifact = artifact.read_text(encoding="utf-8")
    a = case.assertions
    return claims_check.check(artifact, case.params["schema"], a.get("reviews") or [],
                              bool(a.get("expect_empty")))


def _check_pii(artifact, case: Case) -> CheckResult:
    """Token F1 of the marked spans against the labelled ones. #564."""
    if isinstance(artifact, Path):
        if not artifact.exists():
            return CheckResult(False, f"engine left no output at {artifact.name}")
        artifact = artifact.read_text(encoding="utf-8")
    return pii_check.check(str(artifact or ""), case.prompt, case.params["spans"])


def _check_retrieval(artifact, case: Case) -> CheckResult:
    """Recall@k decides; nDCG@10 orders. #563."""
    if isinstance(artifact, Path):
        if not artifact.exists():
            return CheckResult(False, f"engine left no output at {artifact.name}")
        artifact = artifact.read_text(encoding="utf-8")
    return retrieval_check.check(str(artifact or ""), case.assertions["relevant"],
                                 int(case.assertions.get("k", retrieval_check.DEFAULT_K)))


def _check_ocr(artifact, case: Case) -> CheckResult:
    """Character error rate of the transcription against the reference. #562."""
    if isinstance(artifact, Path):
        if not artifact.exists():
            return CheckResult(False, f"engine left no output at {artifact.name}")
        artifact = artifact.read_text(encoding="utf-8")
    errors, chars = ocr_check.text_cer(case.assertions["text"], str(artifact or ""))
    rate = min(1.0, errors / chars) if chars else 1.0
    limit = case.assertions.get("max_cer", ocr_check.DEFAULT_MAX_CER)
    out = CheckResult(rate <= limit, "")
    out.metrics = {"cer": round(rate, 4), "cer_errors": errors, "cer_chars": chars}
    if not out.ok:
        out.reason = (f"character error rate {rate:.2f} over the limit of {limit}; "
                      f"read {str(artifact).strip()[:80]!r}")
    return out


def _check_agent(artifact, case: Case) -> CheckResult:
    """The runner graded the sandbox before deleting it; this reads its verdict. #474."""
    try:
        record = json.loads(artifact) if isinstance(artifact, str) else {}
    except ValueError:
        return CheckResult(False, "agent transcript is not JSON")
    grade = record.get("grade") or {}
    metrics = dict(record.get("metrics") or {})
    if grade.get("passed") and not metrics.get("agent_valid_calls"):
        grade = {"passed": False, "detail": "no valid tool call: the task needs tools"}
    out = CheckResult(bool(grade.get("passed")), grade.get("detail") or "")
    out.metrics = metrics
    return out


CHECKERS = {
    "agent": lambda a, c, **kw: _check_agent(a, c),
    "decide": lambda a, c, **kw: _check_decide(a, c),
    "claims": lambda a, c, **kw: _check_claims(a, c),
    "ocr": lambda a, c, **kw: _check_ocr(a, c),
    "retrieval": lambda a, c, **kw: _check_retrieval(a, c),
    "pii": lambda a, c, **kw: _check_pii(a, c),
    "svg": lambda a, c, **kw: _check_svg(a, c),
    "music": _check_music,
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
    # How far a track missed the length it was asked for. The only music axis
    # with a direction: it is a request the artifact either honoured or did
    # not, and unlike a tempo estimate it has no prior to be dragged by.
    "duration_error_s": "lower",
    # NEUTRAL: reported, never ranked on. A 30s track is not better or worse
    # than a 10s one -- the CASE decides the length -- so this is here to
    # explain `duration_error_s` rather than to order anything.
    "seconds": "neutral",
    # NEUTRAL: reported, never ranked on. More tokens is not better -- as
    # "higher" it would have put the most verbose candidate first. It is here
    # because it EXPLAINS a latency: q3-4b looked 4x slower than local-large
    # and was actually writing 2.7x as much, faster per token.
    "completion_tokens": "neutral",
    "code_pass": "higher",  # fraction of a code case's assertions that ran green
    # NEUTRAL: a traced illustration is legitimately large and a UI glyph is
    # legitimately small, so ranking on it would crown the blank document.
    "svg_bytes": "neutral",
    # decide (#423): pooled over fields; Brier and ECE score the probabilities.
    "decide_accuracy": "higher",
    "decide_brier": "lower",
    "decide_ece": "lower",
    # NEUTRAL: 1 when every field came with probabilities, 0 when one-hot.
    "calibrated": "neutral",
    # NEUTRAL: 0 when the probabilities came from one score per call, not per option (#312).
    "per_option": "neutral",
    # agent, needle (#312): calls the engine withheld and the loop confirmed; its own peak.
    "needle_withheld_calls": "neutral",
    "needle_peak_mb": "lower",
    # agent (#474): completion decides; valid calls break ties; latency is reported beside.
    "agent_valid_call_rate": "higher",
    "agent_steps": "neutral",
    "agent_tool_calls": "neutral",
    "agent_valid_calls": "neutral",
    "agent_ttft_sum_s": "neutral",
    "agent_model_s": "neutral",
    "agent_tool_s": "neutral",
    "agent_wall_s": "neutral",
    "agent_no_tool_steps": "neutral",
    # The per-slot context the candidate was served at. #498.
    "agent_ctx": "neutral",
    "prompt_tokens": "neutral",
    # methods (#576): what a method paid for its row; cost is reported, never ranked on.
    "method_calls": "neutral",
    "method_samples": "neutral",
    "method_chosen": "neutral",
    # retrieval (#563): every relevant document in the top k, and how high.
    "retrieval_recall": "higher",
    "retrieval_ndcg": "higher",
    # pii (#564): token F1 over marked spans, with its two halves beside it.
    "pii_f1": "higher",
    "pii_precision": "higher",
    "pii_recall": "higher",
    # claims (#654): matched against human reviews; expect_empty cases report apart.
    # Reviewed-case metrics are per review interface, never pooled (#661).
    "claims_schema_valid": "higher",
    "claims_empty_rate": "higher",
    "claims_emitted": "neutral",
    "claims_judge_calibrated": "neutral",
    "claims_matcher_version": "neutral",
    **{claims_check.metric(name, i): way for i in claims_check.INTERFACES
       for name, way in claims_check.PER_INTERFACE.items()},
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
              "seconds", "voice", "speed", "layers", "reuse", "ssd_streaming",
              # music: what the model is ASKED for, which is also what the
              # checker holds it to. `duration` is a request here and a
              # measurement in the receipt, and the gap between them is the
              # adherence axis.
              "lyrics", "bpm", "duration", "keyscale", "timesignature",
              "instrumental",
              # music, style transfer: `task` selects cover over text2music,
              # `ref` names the audio whose style is being taken, and
              # `cover_strength` says how much of it to keep. The reference
              # belongs to the CASE rather than the engine spec because it is
              # what the case is asking about. #275.
              "task", "ref", "cover_strength",
              # decide: the flat schema of enum and boolean fields. #423.
              "schema",
              # claims: the system prompt the case is asked with. #654.
              "system",
              # agent: the case bundle, the pinned tools, the caps, the padding. #474.
              "repo", "tools", "max_steps", "pad_tokens", "timeout_s"}
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
                  "stt": {"max_wer"},
                  # `expect_vocals: false` inverts the question from "were the
                  # right words sung" to "was anything sung at all", which is
                  # both a capability check and the lane's own ceiling control.
                  "music": {"max_wer", "expect_vocals", "duration_s"},
                  "decide": {"answers"},
                  "agent": {"answer", "hidden"},
                  "ocr": {"text", "max_cer"},
                  "retrieval": {"relevant", "k"},
                  "pii": {"pii"},
                  "claims": {"reviews", "expect_empty", "basis", "origin"}}


#: Lanes whose runner returns text rather than a file.
TEXT_MODALITIES = {"svg", "web", "code", "extract", "decide", "claims"}
#: Lanes whose runner returns the path of a file it made.
MEDIA_MODALITIES = {"image", "video", "tts", "music"}
#: The suffix a text lane's output is written to the run dir under.
TEXT_SUFFIX = {"svg": ".svg", "web": ".html", "code": ".py", "decide": ".json", "claims": ".json",
               "agent": ".json"}
#: Lanes whose case hands the candidate a file to read through input_file. #562.
INPUT_MODALITIES = {"ocr", "retrieval"}
#: Lanes whose runner drives a tool loop over a sandboxed repo. #474.
AGENT_MODALITIES = {"agent"}


def artifact_name(candidate: str, case_id: str, suffix: str) -> str:
    """The one file name a case's artifact is written under. #429."""
    return f"{candidate.replace('/', '_')}--{case_id}{suffix}"


@dataclass
class Result:
    case_id: str
    candidate: str
    passed: bool
    seconds: float
    peak_kb: int
    detail: str
    #: The text a text runner returned; None when the runner made a file. #463.
    output: str | None = None
    #: The file written for this row under artifact_name; None if none was. #463.
    artifact_path: str | None = None
    warnings: list[str] = field(default_factory=list)
    #: Numbers a checker produced alongside its verdict (wer, ocr score...).
    #: A pass rate separates working from broken; these are what put two
    #: working candidates in an order.
    metrics: dict = field(default_factory=dict)
    #: Why it failed, set by the runner where it failed: a reasons class. #408.
    failure_class: str = ""
    #: The harness limit it hit, as a `limit:` predicate body. #406.
    limit: str = ""
    #: The candidates row it ran as, set where the run is stored. #429.
    candidate_id: int | None = None
    #: Seconds from request to first content / reasoning token; None if unseen. #468.
    ttft_s: float | None = None
    first_reasoning_s: float | None = None
    #: The server's own prompt processing time, where it reports one. #468.
    prefill_s: float | None = None
    #: First request after a load with no warm-up; None where unknown. #468.
    cold: bool | None = None
    #: Where an out-of-process engine says it ran: {"device", "attn"}; {} if unsaid. #604.
    runtime: dict = field(default_factory=dict)
    #: One {max_tokens, tokens, seconds, passed, failure_class} per budget-ladder rung tried; [] unladdered. #668.
    attempts: list = field(default_factory=list)


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
    #: What the weights loaded into, as "<kind>:<name>" -- "unified:arm64",
    #: "discrete:NVIDIA GeForce RTX 4070". Every lane now has an implementation
    #: on both machines and a different tool behind each, so two results.json
    #: files from one lane may have been produced by different programs on
    #: different silicon. Empty for runs written before this existed.
    accelerator: str = ""
    #: What MEASURED the numbers, as name -> implementation. The card is not
    #: the instrument: the same RTX 4070 under Windows and under Linux gives
    #: the identical `accelerator` string, while peak memory comes from a job
    #: object on one and from ru_maxrss on the other, and CER comes from two
    #: different OCR engines. Those are two graders by exactly the argument
    #: that already disqualifies PickScore against HPSv2.
    instruments: dict = field(default_factory=dict)
    engines: dict = field(default_factory=dict)
    #: WHERE THE WORK EXECUTED: in-pod, gpu-node, or a host outside the cluster
    #: (harness.machine.WHERE). A pod that dispatches to a host is a different
    #: exam from a pod that runs the work, and without this a result reads as
    #: "the cluster measured it" when the cluster only asked -- this project's
    #: own part's-cost-reported-as-the-whole's. Empty for runs written before
    #: this existed. Issue #170.
    where: str = ""
    #: SWAP IN USE WHEN THE RUN STARTED, in MB. Not a comparability axis --
    #: timing is already excluded from comparable() as an output of the run
    #: rather than a property of the exam -- but a timing taken on a machine at
    #: 19.9 GB of swap is not the same number as one taken on a quiet machine,
    #: and #142 spent two issues finding that out. Recorded so a figure cannot
    #: be quoted without the condition that produced it.
    swap_used_mb: int = 0
    #: Current pressure at run start. Not a comparability axis. #283.
    pressure: dict = field(default_factory=dict)
    #: What the cases SAID, not what they were called. `case_ids` are names,
    #: and a name survives every edit to the thing it names: change
    #: fox-snow.yaml from 512 to 1024, or rewrite its prompt, and the id, the
    #: count and the receipt are all unchanged while the exam is not. That is
    #: the same defect that produced #141 and #142, sitting inside the
    #: machinery built to refuse it. Empty for runs written before this
    #: existed.
    cases_digest: str = ""
    #: Which side of the holdout split ran, and the assignment version. #479.
    split: str = ""
    split_version: str = ""
    #: receipt key -> the method and its base, for each method candidate. Not an axis. #576.
    methods: dict = field(default_factory=dict)
    #: receipt key -> "device:attention" an out-of-process engine answered on. An axis. #604.
    devices: dict = field(default_factory=dict)
    #: receipt key -> how its server was launched (ds4: streaming, expert cache, ctx). An axis. #611.
    launch: dict = field(default_factory=dict)
    #: The reply budget every text request ran at; 0 where the lane generates no text. An axis. #628.
    max_tokens: int = 0
    #: Exam knob settings by knob name (harness.knobs.settings); an axis. #636.
    knobs: dict = field(default_factory=dict)
    #: receipt key -> each time the router evicted its model mid-run; any entry refuses ranking. #649.
    router_swaps: dict = field(default_factory=dict)
    #: The budget ladder a cut-off reply was retried up, max_tokens its top rung; () for one budget. An axis. #668.
    budget_ladder: tuple = ()
    #: receipt key -> the model id each text candidate resolved to at run time. Not an axis. #670.
    resolved: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {"modality": self.modality, "case_ids": list(self.case_ids),
                "repeat": self.repeat, "sampling": dict(self.sampling),
                "gateway": self.gateway, "adherence": self.adherence,
                "tier": self.tier, "accelerator": self.accelerator,
                "instruments": dict(self.instruments),
                "engines": dict(self.engines),
                "where": self.where,
                "swap_used_mb": self.swap_used_mb,
                "pressure": dict(self.pressure),
                "cases_digest": self.cases_digest,
                "split": self.split, "split_version": self.split_version,
                "methods": dict(self.methods),
                "devices": dict(self.devices),
                "launch": dict(self.launch),
                "max_tokens": self.max_tokens,
                "knobs": dict(self.knobs),
                "router_swaps": dict(self.router_swaps),
                "budget_ladder": list(self.budget_ladder),
                "resolved": dict(self.resolved)}

    @classmethod
    def from_dict(cls, raw: dict) -> "Receipt":
        """A receipt read back from results.json or the store; a missing budget is the constant it ran at."""
        budget = raw.get("max_tokens")
        if budget is None:
            budget = legacy_budget(raw.get("modality") or "")
        return cls(modality=raw["modality"], case_ids=tuple(raw.get("case_ids") or ()),
                   repeat=int(raw.get("repeat") or 1), sampling=raw.get("sampling") or {},
                   gateway=raw.get("gateway") or "", adherence=raw.get("adherence") or "",
                   tier=raw.get("tier") or "measure", accelerator=raw.get("accelerator") or "",
                   instruments=raw.get("instruments") or {}, engines=raw.get("engines") or {},
                   where=raw.get("where") or "", swap_used_mb=raw.get("swap_used_mb") or 0,
                   pressure=raw.get("pressure") or {},
                   cases_digest=raw.get("cases_digest") or "", split=raw.get("split") or "",
                   split_version=raw.get("split_version") or "",
                   methods=raw.get("methods") or {}, devices=raw.get("devices") or {},
                   launch=raw.get("launch") or {}, max_tokens=int(budget),
                   knobs=legacy_knobs(raw),
                   router_swaps=raw.get("router_swaps") or {},
                   budget_ladder=tuple(int(n) for n in raw.get("budget_ladder") or ()),
                   resolved=raw.get("resolved") or {})


def legacy_knobs(raw: dict) -> dict:
    """The exam knob settings a receipt records; one from before #636 ran at the constants of the time."""
    from harness import knobs
    if "knobs" in raw:
        return dict(raw.get("knobs") or {})
    return knobs.settings(raw.get("modality") or "")


def legacy_budget(modality: str) -> int:
    """The reply budget a run from before #628 used: the hard constants of the time."""
    if modality in AGENT_MODALITIES:
        return 8000
    return 4000 if modality in TEXT_MODALITIES else 0


def cases_digest(cases) -> str:
    """Identity by content, for the parts of a case that change the question.

    `prompt`, `context` and `params` are what is asked; `assertions` is what
    counts as a right answer. All four change the exam while leaving the id
    alone, which is why the id cannot stand in for them.

    `source` and `methods` are excluded: a moved file and a case declared
    unfair to one method ask the same question of the candidates that do run.
    """
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.id):
        _digest_parts(h, c)
    # A matcher change rescores the same claims cases, so it is a different exam; case_digest stays put for holdout. #662.
    if any(c.modality == "claims" for c in cases):
        h.update(f"claims_matcher\x00{claims_check.MATCHER_VERSION}".encode("utf-8"))
    return h.hexdigest()[:16]


def _digest_parts(h, c) -> None:
    for part in (c.id, c.prompt, getattr(c, "context", ""),
                 json.dumps(dict(c.params), sort_keys=True, default=str),
                 json.dumps(dict(c.assertions), sort_keys=True, default=str)):
        h.update(str(part).encode("utf-8"))
        h.update(b"\x00")
    # The input's bytes, not its path: a moved image asks the same question. #562.
    source = getattr(c, "input_file", None)
    if source is not None and Path(source).is_file():
        h.update(hashlib.sha256(Path(source).read_bytes()).digest())
    preamble = getattr(c, "preamble", "")
    if preamble:
        h.update(b"preamble\x00" + preamble.encode("utf-8"))


def case_digest(case) -> str:
    """One case's content identity, over the same parts as cases_digest. #479."""
    h = hashlib.sha256()
    _digest_parts(h, case)
    return h.hexdigest()


def contaminated(r: Receipt) -> str:
    """Why a run cannot be ranked at all, "" when nothing says so. #649."""
    for key, swaps in sorted((r.router_swaps or {}).items()):
        if swaps:
            first = swaps[0]
            return (f"router swap during the run: {key} lost its model {len(swaps)} time(s), "
                    f"first at case {first.get('case')} ({first.get('at')}), to "
                    f"{', '.join(first.get('resident') or []) or 'nothing resident'}. Its timings "
                    f"and timeouts measured a reload")
    return ""


def comparable(a: Receipt, b: Receipt) -> tuple[bool, str]:
    """May these two runs be ranked in one table?

    IN, because each changes what was asked:
      * `modality` and `case_ids` -- a different exam, or a different number of
        questions on it. The ids and not just the count: nine easy cases and
        nine hard ones are not one lane.
      * `cases_digest` -- and the ids are not enough either, because a name
        survives every edit to the thing it names. Nine cases REWRITTEN IN
        PLACE keep their ids. Compared only where both runs carry one.
      * `sampling` -- adding a repetition penalty changed what the svg lane
        produces, so a run from before it is a different exam from one after.
      * `adherence` -- PickScore and HPSv2 are two graders.
      * `instruments` -- so is an OCR engine, and so is a peak-memory method.
        Compared only where both runs name the same instrument.
      * `where` -- a pod that dispatches to a host measured a host, and a pod
        that ran the work measured a pod. Same argument as the accelerator,
        one level out: the scheduler, the memory ceiling and the queue are all
        different, and only one of the two has a card it can see.

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

    REFUSED OUTRIGHT: a run whose model the llama-server router evicted mid-run
    (`router_swaps`, #649). Its timings and timeouts measured a reload.
    """
    for r in (a, b):
        why = contaminated(r)
        if why:
            return False, why
    if a.tier != b.tier:
        return False, (f"different tier: {a.tier} vs {b.tier}. A screen asks "
                       f"whether it ran; a measurement asks whether it is "
                       f"better")
    if a.modality != b.modality:
        return False, f"different modality: {a.modality} vs {b.modality}"
    if tuple(a.case_ids) != tuple(b.case_ids):
        return False, (f"different case set: {len(a.case_ids)} vs "
                       f"{len(b.case_ids)} cases")
    if a.cases_digest and b.cases_digest and a.cases_digest != b.cases_digest:
        return False, (f"same case ids, different cases: {a.cases_digest} vs "
                       f"{b.cases_digest}. A prompt, a param or an assertion "
                       f"was edited in place")
    if a.split and b.split and (a.split, a.split_version) != (b.split, b.split_version):
        return False, (f"different split: {a.split} v{a.split_version} vs "
                       f"{b.split} v{b.split_version}")
    if a.repeat != b.repeat:
        return False, f"different repeat: {a.repeat} vs {b.repeat}"
    if tuple(a.budget_ladder) != tuple(b.budget_ladder):
        def said(r):
            return ",".join(map(str, r.budget_ladder)) or f"none (one budget, {r.max_tokens})"
        return False, (f"different budget ladder: {said(a)} vs {said(b)}. A retry at a larger "
                       f"budget turns a cut-off reply into an answer")
    if a.max_tokens != b.max_tokens:
        return False, (f"different reply budget: {a.max_tokens} vs {b.max_tokens} "
                       f"tokens. A reasoning model cut off at the smaller one fails "
                       f"on the budget, not on the answer")
    if dict(a.knobs) != dict(b.knobs):
        moved = [f"{k} {a.knobs.get(k)} vs {b.knobs.get(k)}"
                 for k in sorted(set(a.knobs) | set(b.knobs)) if a.knobs.get(k) != b.knobs.get(k)]
        return False, (f"different knob settings: {'; '.join(moved)}. A limit the harness chose "
                       f"decides which replies count")
    if dict(a.sampling) != dict(b.sampling):
        return False, f"different sampling: {a.sampling} vs {b.sampling}"
    if a.adherence != b.adherence:
        return False, (f"different adherence backend: {a.adherence!r} vs "
                       f"{b.adherence!r}")
    # LAST, and only when both runs say. A different accelerator means
    # different arithmetic, a different memory ceiling and -- since every lane
    # now has an implementation per machine -- frequently a different program
    # producing the artifact. That is a larger difference than the adherence
    # grader, which is already disqualifying.
    #
    # An EMPTY field is not a mismatch. Runs written before this existed carry
    # none, and refusing those would invalidate every measurement this project
    # has, including the STT corpus and the svg three-way, which are good.
    if a.accelerator and b.accelerator and a.accelerator != b.accelerator:
        return False, (f"different accelerator: {a.accelerator} vs "
                       f"{b.accelerator}. Two machines, and each lane has its "
                       f"own tool on each")
    if a.where and b.where and a.where != b.where:
        return False, (f"different place: {a.where} vs {b.where}. One of these "
                       f"ran the work and the other asked somebody else to")
    # ONLY THE INSTRUMENTS BOTH RUNS NAME. A run that recorded no OCR engine is
    # not thereby different from one that did, and adding a fourth instrument
    # later must not retroactively invalidate every result on disk. Same rule
    # as the empty accelerator above, applied per key.
    # A CPU answer and an MPS answer are two programs; compared only where both runs say. #604.
    for key in sorted(set(a.devices) & set(b.devices)):
        if a.devices[key] != b.devices[key]:
            return False, (f"different device for {key}: {a.devices[key]} vs "
                           f"{b.devices[key]}")
    # One model served two ways is two exams: SSD streaming changes memory and speed. #611.
    for key in sorted(set(a.launch) & set(b.launch)):
        if a.launch[key] != b.launch[key]:
            return False, (f"different launch for {key}: {a.launch[key]} vs "
                           f"{b.launch[key]}")
    for name in sorted(set(a.instruments) & set(b.instruments)):
        mine, theirs = a.instruments[name], b.instruments[name]
        if mine and theirs and mine != theirs:
            return False, (f"different {name}: {mine} vs {theirs}. The hardware "
                           f"is not the tooling: the same card with a different "
                           f"grader, or a different server, is a different exam")
    return True, "same exam"


#: libyaml when the wheel has it: several hundred imported cases load per test run. #603.
_LOADER = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


#: The cases a clone ships.
SHIPPED = Path(__file__).resolve().parent / "cases"


def local_root() -> Path:
    """Cases that never enter git, such as verbatim chat transcripts. #654."""
    from harness import paths
    return paths.home() / "cases"


def load_suite(root: str | Path | None = None, local: str | Path | None = None) -> list[Case]:
    """The shipped cases plus this machine's local-only ones, marked private. #654."""
    local = Path(local) if local is not None else local_root()
    shipped = load_cases(root if root is not None else SHIPPED)
    return shipped + (load_cases(local, private=True) if local.is_dir() else [])


def load_cases(directory: str | Path, private: bool = False) -> list[Case]:
    """Load every *.yaml under `directory`, sorted by id for stable runs."""
    return [load_case(path, private) for path in sorted(Path(directory).rglob("*.yaml"))]


def load_case(path: Path, private: bool = False) -> Case:
    """One case file, validated."""
    path = Path(path)
    raw = yaml.load(path.read_text(encoding="utf-8"), Loader=_LOADER) or {}
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
    input_file = _load_input(path, raw, modality)
    params = raw.get("params") or {}
    assertions = raw.get("assert") or {}
    if modality == "code" and not assertions.get("checks"):
        # A code case with nothing to run passes every model, which is
        # worse than not having the case at all.
        raise ValueError(f"{path.name}: a code case needs assert.checks")
    _reject_unknown(path, modality, "params", set(params), PARAM_KEYS)
    _reject_unknown(path, modality, "assert", set(assertions),
                    ASSERTION_KEYS.get(modality, set()))
    prompt = raw["prompt"]
    if modality == "decide":
        try:
            decide_check.validate(params.get("schema"), assertions.get("answers"))
        except decide_check.SchemaError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
        # Every text runner then asks the same question; nimble reads the schema.
        prompt = f"{prompt.rstrip()}\n\n{decide_check.render(params['schema'])}"
    if modality == "claims":
        try:
            claims_check.validate_case(params.get("system"), params.get("schema"),
                                       assertions.get("reviews"), assertions.get("expect_empty"))
        except ValueError as exc:
            raise ValueError(f"{path.name}: {exc}") from exc
    if modality == "agent":
        params = {**params, **_agent_params(path, params, assertions)}
    if modality == "ocr" and not str(assertions.get("text") or "").strip():
        raise ValueError(f"{path.name}: an ocr case needs assert.text, the reference")
    if modality == "retrieval":
        _check_relevant(path, input_file, assertions.get("relevant"))
    if modality == "pii":
        params = {**params, "spans": _pii_spans(path, prompt, assertions.get("pii"))}
    return Case(id=raw["id"], modality=modality, prompt=prompt,
                context=context, audio=audio, params=params,
                assertions=assertions, source=path, input_file=input_file,
                language=raw.get("language") or "en",
                methods=tuple(raw.get("methods") or ()), private=private,
                preamble=_load_preamble(path, modality))


def _load_preamble(path: Path, modality: str) -> str:
    """preamble.py beside a code case, shared by every case of its import source. #657."""
    beside = path.parent / "preamble.py"
    return beside.read_text(encoding="utf-8") if modality == "code" and beside.is_file() else ""


def _agent_params(path: Path, params: dict, assertions: dict) -> dict:
    """Validate an agent case's bundle and pin its content into the digest. #474."""
    from evals import sandbox
    bundle = path.parent / str(params.get("repo") or "")
    if not params.get("repo") or not (bundle / "repo").is_dir():
        raise ValueError(f"{path.name}: params.repo must name a bundle with a repo/ dir")
    tools = params.get("tools") or []
    unknown = set(tools) - set(sandbox.TOOLS)
    if not tools or unknown:
        raise ValueError(f"{path.name}: params.tools must pin tools from "
                         f"{', '.join(sandbox.TOOLS)}; got {sorted(unknown) or 'none'}")
    if not (assertions.get("answer") or assertions.get("hidden")):
        raise ValueError(f"{path.name}: an agent case needs assert.answer or assert.hidden")
    if assertions.get("hidden") and not (bundle / "hidden").is_dir():
        raise ValueError(f"{path.name}: assert.hidden needs {params['repo']}/hidden/")
    h = hashlib.sha256()
    for f in sorted(p for p in bundle.rglob("*") if p.is_file()
                    and "__pycache__" not in p.parts
                    and p.relative_to(bundle).parts[0] in ("repo", "hidden")):
        h.update(f.relative_to(bundle).as_posix().encode("utf-8") + b"\x00")
        h.update(f.read_bytes().replace(b"\r\n", b"\n") + b"\x00")
    return {"repo_digest": h.hexdigest()[:16]}


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


def _pii_spans(path: Path, text: str, labels) -> list[list[int]]:
    """Each labelled string's [start, end] in the sentence; it must occur exactly once. #564."""
    if not isinstance(labels, list):
        raise ValueError(f"{path.name}: a pii case needs assert.pii, a list (empty for none)")
    spans = []
    for label in map(str, labels):
        at = text.find(label)
        if at < 0:
            raise ValueError(f"{path.name}: {label!r} is not in the sentence")
        if text.find(label, at + 1) >= 0:
            raise ValueError(f"{path.name}: {label!r} occurs more than once; label a longer string")
        spans.append([at, at + len(label)])
    return spans


def _check_relevant(path: Path, corpus: Path, relevant) -> None:
    """A retrieval case's labels name documents its corpus holds. #563."""
    if not relevant or not isinstance(relevant, list):
        raise ValueError(f"{path.name}: a retrieval case needs assert.relevant, a list of ids")
    ids = {json.loads(line)["id"] for line in corpus.read_text(encoding="utf-8").splitlines()
           if line.strip()}
    missing = sorted(set(map(str, relevant)) - ids)
    if missing:
        raise ValueError(f"{path.name}: relevant ids {missing} are not in {corpus.name}")


def _load_input(path: Path, raw: dict, modality: str) -> Path | None:
    """`input_file:`, resolved beside the case; required where the lane reads one. #562."""
    filename = raw.get("input_file")
    if not filename:
        if modality in INPUT_MODALITIES:
            raise ValueError(f"{path.name}: an {modality} case needs input_file")
        return None
    source = path.parent / filename
    if not source.is_file():
        raise ValueError(f"{path.name}: input_file '{filename}' not found beside the case")
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
        limit = getattr(r, "limit", "")
        return Result(case.id, "", False, 0.0, 0, r.reason,
                      warnings=r.warnings, metrics=metrics,
                      failure_class=getattr(r, "failure_class", ""),
                      limit=limit if isinstance(limit, str) else "")

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
        # A list is alternatives: any one satisfies it. #365.
        options = needle if isinstance(needle, list) else [needle]
        if not any(contains(artifact, n) for n in options):
            return Result(case.id, "", False, 0.0, 0,
                          f"missing required content: {' or '.join(options)}",
                          warnings=r.warnings)

    for needle in a.get("must_not_contain") or []:
        if needle.lower() in artifact.lower():
            return Result(case.id, "", False, 0.0, 0,
                          f"contains forbidden content: {needle}",
                          warnings=r.warnings)

    exact = a.get("equals")
    # A list is alternatives, any one exact match passes. #603.
    options = exact if isinstance(exact, list) else [exact]
    if exact is not None and str(artifact).strip().lower() not in {
            str(o).strip().lower() for o in options}:
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
        pooled = _decide_calibration(rows)
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
            "metrics": {**_mean_metrics(rows), **pooled},
            # Kept alongside the mean because an average of mostly-zeros hides
            # the one case that fell over, which is usually the interesting one.
            "metrics_worst": _worst_metrics(rows),
            "failures": [f"{r.case_id}: {r.detail}" for r in rows
                         if not r.passed],
            # The classes the runners set, which a tier decides from. #408.
            "failure_classes": _count(r.failure_class for r in rows
                                      if not r.passed and r.failure_class),
            "limits": sorted({r.limit for r in rows if not r.passed and r.limit}),
            # HOW they failed, not just how many. See failure_kind(); a budget
            # row is our limit, counted apart from a wrong answer. #628.
            "budget": sum(1 for r in rows if _over_budget(r)),
            "wrong": sum(1 for r in rows if not _over_budget(r)
                         and not r.passed and failure_kind(r.detail) == "wrong"),
            "empty": sum(1 for r in rows if not _over_budget(r)
                         and not r.passed and failure_kind(r.detail) == "empty"),
            "errored": sum(1 for r in rows if not _over_budget(r)
                           and not r.passed and failure_kind(r.detail) == "error"),
            # How many rows each metric was actually computed over. A mean over
            # 2 of 9 cases printed beside a mean over 9 is not a comparison.
            "metric_n": {**{k: len(v) for k, v in _gather_metrics(rows).items()},
                         **({"decide_ece": _decision_count(rows)} if pooled else {})},
            # WHICH cases this candidate actually sat. Two candidates in one
            # run can get different sets -- a case may be unfair to a method,
            # or to a language -- and then their pass rates are not comparable.
            "case_ids": sorted({r.case_id.split("#")[0] for r in rows}),
            **first_token(rows),
            **agent_summary(rows),
            **by_rung(rows),
        }
    return out


def by_rung(rows) -> dict:
    """Per budget-ladder rung: rows that tried it, passed at it, and the seconds spent there; {} unladdered. #668."""
    out: dict = {}
    for r in rows:
        for a in r.attempts or ():
            got = out.setdefault(str(a["max_tokens"]), {"tried": 0, "passed": 0, "seconds": 0.0})
            got["tried"] += 1
            got["passed"] += 1 if a.get("passed") else 0
            got["seconds"] = round(got["seconds"] + float(a.get("seconds") or 0.0), 3)
    return {"by_rung": dict(sorted(out.items(), key=lambda kv: int(kv[0])))} if out else {}


def _over_budget(r) -> bool:
    return not r.passed and r.failure_class == reasons.TOKEN_BUDGET_EXHAUSTED


def agent_summary(rows) -> dict:
    """The agent lane's per-candidate roll-up, beside completion; {} elsewhere. #474."""
    got = [r.metrics for r in rows if "agent_steps" in (r.metrics or {})]
    ctx = max((r.metrics or {}).get("agent_ctx") or 0 for r in rows) if rows else 0
    if not got:
        return {"agent": {"ctx": ctx}} if ctx else {}
    calls = sum(m.get("agent_tool_calls", 0) for m in got)
    return {"agent": {
        "completed": sum(1 for r in rows if r.passed and "agent_steps" in (r.metrics or {})),
        "cases": len(got),
        "valid_call_rate": round(sum(m.get("agent_valid_calls", 0) for m in got) / calls, 4)
        if calls else 0.0,
        "steps_median": statistics.median(m["agent_steps"] for m in got),
        "total_s": round(sum(m.get("agent_wall_s", 0) for m in got), 1),
        "ttft_sum_s": round(sum(m.get("agent_ttft_sum_s", 0) for m in got), 1),
        "model_s": round(sum(m.get("agent_model_s", 0) for m in got), 1),
        **({"ctx": ctx} if ctx else {})}}


def p95(values: list[float]) -> float | None:
    """Nearest-rank 95th percentile, as throughput.sweep takes it."""
    v = sorted(values)
    return v[min(len(v) - 1, int(len(v) * 0.95))] if v else None


def first_token(rows) -> dict:
    """TTFT over warm rows only: a cold one carries the load. #468.

    Reported beside latency, never ranked on: no lane decides by it.
    """
    def warm(name):
        return [float(getattr(r, name)) for r in rows
                if getattr(r, name, None) is not None and not getattr(r, "cold", None)]

    def med(v):
        return round(statistics.median(v), 3) if v else None
    ttft = warm("ttft_s")
    cold = [float(r.ttft_s) for r in rows
            if getattr(r, "cold", None) and r.ttft_s is not None]
    top = p95(ttft)
    return {"ttft_median_s": med(ttft),
            "ttft_p95_s": None if top is None else round(top, 3),
            "ttft_n": len(ttft),
            "ttft_cold_s": round(cold[0], 3) if cold else None,
            "first_reasoning_median_s": med(warm("first_reasoning_s")),
            "prefill_median_s": med(warm("prefill_s"))}


def _decisions(rows: list[Result]) -> list:
    return [pair for r in rows for pair in (r.metrics or {}).get("decisions") or []]


def _decision_count(rows: list[Result]) -> int:
    return len(_decisions(rows))


def _decide_calibration(rows: list[Result]) -> dict:
    """Pooled ECE over every field decision, once there are enough to bin."""
    pairs = _decisions(rows)
    if len(pairs) < decide_check.MIN_ECE_N:
        return {}
    return {"decide_ece": round(decide_check.ece(pairs), 4)}


def _count(items) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        out[item] = out.get(item, 0) + 1
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
RATIO_METRICS = {"wer": ("wer_errors", "wer_words"),
                 "agent_valid_call_rate": ("agent_valid_calls", "agent_tool_calls"),
                 "decide_accuracy": ("decide_correct", "decide_fields"),
                 "decide_brier": ("decide_brier_sum", "decide_fields"),
                 "cer": ("cer_errors", "cer_chars"),
                 "pii_f1": ("pii_2tp", "pii_f1_den"),
                 "pii_precision": ("pii_tp", "pii_pred"),
                 "pii_recall": ("pii_tp", "pii_gold"),
                 "claims_empty_rate": ("claims_empty_kept", "claims_empty_cases"),
                 **{claims_check.metric(name, i): (claims_check.metric(n, i), claims_check.metric(d, i))
                    for i in claims_check.INTERFACES
                    for name, (n, d) in claims_check.PER_INTERFACE_RATIOS.items()}}
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
