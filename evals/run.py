"""Run the eval suite and print a comparison.

    uv run python -m evals.run --modality svg --candidates local-mid,local-small
    uv run python -m evals.run --modality image \
        --candidates mflux:flux2-klein-4b,mflux:z-image-turbo

Artifacts and results.json land in $LOCALHARNESS_HOME/runs/<stamp>-<modality>/
unless --out says otherwise.

A candidate is either a gateway alias (text modalities) or an engine spec
(anything that runs as a process). Which one it is decides the runner, and a
candidate is only handed the cases it can actually run: giving mflux an SVG
case produces a failure row that says nothing about mflux.

Sequenced by candidate, never interleaved: mlx_lm.server hot-swaps models per
request, so alternating between two aliases pays a model load on every case and
measures disk speed instead of the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from harness import audio, completion, paths
from harness.engines import Engine, parse_options, resolve

from dataclasses import replace

from evals.core import (MODALITIES, Case, Receipt, cases_digest, comparable,
                        direction_of, load_cases, summarize)
from evals.environment import capture
from evals.runners.process import ProcessRunner
from evals.runners.chain import ChainRunner
from evals.runners.repair import RepairRunner
from evals.runners.speech import SpeechRunner
from evals.runners.omnisvg import DEFAULT_CANDIDATES, OmniSVGRunner
from evals.runners.trace import TraceRunner
from evals.runners.text import CompletionRunner
from evals.runners.transcription import TranscriptionRunner

ROOT = Path(__file__).resolve().parent
TEXT_MODALITIES = {"svg", "web", "code", "extract"}
ALL_MODALITIES = sorted(MODALITIES)


PROCESS_ENGINES = ("mflux", "h3")
#: The svg lane's second METHOD: draw a raster, then vectorize it. Written as
#: `trace:<engine spec>` so the engine underneath stays the ordinary spec.
TRACE_PREFIX = "trace"
# candidate prefix -> vector.TRACE_PRESETS key
TRACE_PREFIXES = {"trace": "illustration", "trace-icon": "icon"}
#: Generate, check, repair. A WORKFLOW rather than a model: `local-large` and
#: `repair:local-large` are different products. See evals/runners/repair.py.
REPAIR_PREFIX = "repair"
REPAIR_OPTIONS = {"attempts"}
#: The svg lane's THIRD method: a model that emits draw commands as tokens.
#: `omnisvg:4B` -- a size, not an engine spec, because the weights it needs
#: are two fixed repos rather than anything the caller chooses.
OMNISVG_PREFIX = "omnisvg"
OMNISVG_OPTIONS = {"candidates"}
#: Two-stage image workflows, named for their second stage. See
#: evals/runners/chain.py -- this is the ComfyUI vocabulary mflux already ships.
from evals.runners.chain import STAGES as CHAIN_STAGES  # noqa: E402
TTS_OPTIONS = {"voice", "ref_audio", "lang_code", "ear"}
STT_OPTIONS = {"backend", "language"}


def kind_of(candidate: str) -> str:
    """process, tts, or gateway. The prefix decides, so a typo in the rest of
    the spec is reported as a bad spec rather than silently becoming a model
    name the gateway has never heard of."""
    head = candidate.split(":", 1)[0].split(",", 1)[0].strip()
    if head in PROCESS_ENGINES:
        return "process"
    if head in ("tts", "stt"):
        return head
    if head in TRACE_PREFIXES:
        return head
    if head == OMNISVG_PREFIX:
        return OMNISVG_PREFIX
    if head == REPAIR_PREFIX:
        return REPAIR_PREFIX
    if head in CHAIN_STAGES:
        return "chain"
    return "gateway"


def modality_of(candidate: str) -> str | None:
    """The one modality this candidate can run, or None for text candidates,
    which can run all of them."""
    kind = kind_of(candidate)
    if kind in ("tts", "stt"):
        return kind
    if kind in TRACE_PREFIXES:
        # It answers svg cases; the engine underneath makes images, which is
        # the whole point and would be the wrong modality to select on.
        return "svg"
    if kind == OMNISVG_PREFIX:
        return "svg"
    if kind == REPAIR_PREFIX:
        # A text candidate wearing a loop: it runs every text lane, same as
        # the model it wraps.
        return None
    if kind == "chain":
        return "image"
    if kind == "process":
        engine = engine_for(candidate)
        return engine.modality if engine else None
    return None


def language_of(candidate: str) -> str | None:
    """The language a speech candidate can be measured in, or None for the
    lanes that have no language at all.

    For tts it is the EAR's language, not the model's: a French sentence can
    only be scored by a transcriber that speaks French, so the two are the same
    constraint rather than two settings that happen to agree.
    """
    kind = kind_of(candidate)
    if kind not in ("tts", "stt"):
        return None
    options = _options_of(candidate)
    if kind == "stt":
        return options.get("language") or "en"
    _, _, language = options.get("ear", "server").partition(":")
    return language.strip() or "en"


def _options_of(candidate: str) -> dict:
    """The key=value tail of a candidate spec, or {} if it does not parse.

    Selection must not raise: a bad spec is reported by build_runner, with the
    full message, rather than here as a filtering accident.
    """
    _, _, rest = candidate.partition(":")
    _, _, optstr = rest.partition(",")
    try:
        return parse_options(optstr, candidate)
    except ValueError:
        return {}


def engine_for(candidate: str) -> Engine | None:
    try:
        return resolve(candidate)
    except ValueError:
        return None


def _speech_runner(candidate: str, outdir: Path | None) -> SpeechRunner:
    _, _, rest = candidate.partition(":")
    model, _, optstr = rest.partition(",")
    try:
        options = parse_options(optstr, candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = set(options) - TTS_OPTIONS
    if unknown:
        raise SystemExit(f"unknown tts option(s) {', '.join(sorted(unknown))}; "
                         f"allowed: {', '.join(sorted(TTS_OPTIONS))}")
    if not model:
        raise SystemExit("a tts candidate needs a model, e.g. "
                         "tts:mlx-community/Kokoro-82M-bf16,voice=am_adam")
    if outdir is None:
        raise SystemExit(f"{candidate} writes audio; pass --out")
    ref = options.get("ref_audio", "")
    if ref and not Path(ref).exists():
        # Cloning is the slow lane. Finding the typo forty cases in is the
        # expensive way to find it.
        raise SystemExit(f"{candidate}: no reference audio at {ref}")
    try:
        # Only Kokoro has a voice table; a candidate may legitimately name none.
        return SpeechRunner(model=model, outdir=outdir,
                            voice=options.get("voice", ""),
                            ref_audio=ref or None,
                            lang_code=options.get("lang_code", ""),
                            ear=options.get("ear", "server"))
    except ValueError as exc:
        # A bad ear is a bad candidate string, not a traceback.
        raise SystemExit(f"{candidate}: {exc}") from exc


def _transcription_runner(candidate: str) -> TranscriptionRunner:
    _, _, rest = candidate.partition(":")
    model, _, optstr = rest.partition(",")
    model = model.strip()
    try:
        options = parse_options(optstr, candidate)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    unknown = set(options) - STT_OPTIONS
    if unknown:
        raise SystemExit(f"unknown stt option(s) {', '.join(sorted(unknown))}; "
                         f"allowed: {', '.join(sorted(STT_OPTIONS))}")
    if not model:
        raise SystemExit("an stt candidate needs a model, e.g. "
                         "stt:mlx-community/parakeet-tdt-0.6b-v2")
    try:
        return TranscriptionRunner(
            model=model,
            backend=options.get("backend", "server"),
            language=options.get("language", ""))
    except ValueError as exc:
        # A bad backend is a bad candidate string, and gets the same treatment
        # as a bad engine spec: named before the corpus runs, not a traceback.
        raise SystemExit(f"{candidate}: {exc}") from exc


def build_runner(candidate: str, gateway: str, outdir: Path | None,
                 adherence: str | None = None):
    kind = kind_of(candidate)
    if kind == "gateway":
        return CompletionRunner(gateway, candidate)
    if kind == "tts":
        return _speech_runner(candidate, outdir)
    if kind == "stt":
        return _transcription_runner(candidate)
    if kind == REPAIR_PREFIX:
        _, _, rest = candidate.partition(":")
        model, _, optstr = rest.partition(",")
        try:
            options = parse_options(optstr, candidate)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        unknown = set(options) - REPAIR_OPTIONS
        if unknown:
            raise SystemExit(
                f"unknown repair option(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(REPAIR_OPTIONS))}")
        if not model.strip():
            raise SystemExit("a repair candidate needs a model, e.g. "
                             "repair:local-large")
        return RepairRunner(gateway, model.strip(),
                            attempts=int(options.get("attempts", 3)))
    if kind == "chain":
        stage, _, spec = candidate.partition(":")
        if not spec.strip():
            raise SystemExit(f"{candidate} needs a base engine, e.g. "
                             f"{stage}:mflux:flux2-klein-4b")
        try:
            engine = resolve(spec.strip())
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if outdir is None:
            raise SystemExit(f"{candidate} writes images; pass --out")
        return ChainRunner(engine, stage, outdir)
    if kind == OMNISVG_PREFIX:
        _, _, rest = candidate.partition(":")
        size, _, optstr = rest.partition(",")
        try:
            options = parse_options(optstr, candidate)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        unknown = set(options) - OMNISVG_OPTIONS
        if unknown:
            raise SystemExit(
                f"unknown omnisvg option(s) {', '.join(sorted(unknown))}; "
                f"allowed: {', '.join(sorted(OMNISVG_OPTIONS))}")
        try:
            return OmniSVGRunner(
                size.strip() or "4B",
                candidates=int(options.get("candidates", DEFAULT_CANDIDATES)))
        except ValueError as exc:
            raise SystemExit(f"{candidate}: {exc}") from exc
    if kind in TRACE_PREFIXES:
        spec = candidate.partition(":")[2].strip()
        if not spec:
            raise SystemExit(f"a {kind} candidate needs an engine, e.g. "
                             f"{kind}:mflux:flux2-klein-4b")
        try:
            engine = resolve(spec)
        except ValueError as exc:
            raise SystemExit(str(exc)) from exc
        if outdir is None:
            raise SystemExit(f"{candidate} writes images; pass --out")
        return TraceRunner(engine, outdir, preset=TRACE_PREFIXES[kind])
    try:
        engine = resolve(candidate)
    except ValueError as exc:
        # Before anything runs: a typo must not cost a forty-minute generation.
        raise SystemExit(str(exc)) from exc
    if outdir is None:
        raise SystemExit(
            f"{candidate} writes files; pass --out to say where they go")
    return ProcessRunner(engine, outdir, adherence=adherence)


def method_of(candidate: str) -> str:
    """Which METHOD this candidate is, for cases that declare what they can
    fairly test.

    Not a taxonomy of models. The only distinction any case has needed is
    whether a method can put a <text> element in the document at all, and two
    of the three here cannot: `trace` turns glyphs into outlines, and OmniSVG's
    tokenizer emits move/line/curve/arc/close and nothing else. Calling
    everything that is not `trace` an LLM handed chart-bars to OmniSVG, which
    then failed it for missing <text> -- measuring the method, which is the
    exact thing Case.methods exists to prevent.
    """
    kind = kind_of(candidate)
    if kind in TRACE_PREFIXES:
        return TRACE_PREFIX
    if kind == OMNISVG_PREFIX:
        return OMNISVG_PREFIX
    return "llm"


def cases_for(candidate: str, cases: list[Case]) -> list[Case]:
    """The cases this candidate can actually run."""
    modality = modality_of(candidate)
    if modality is None:
        return [c for c in cases if c.modality in TEXT_MODALITIES]
    picked = [c for c in cases if c.modality == modality]
    # A case may declare which methods it can fairly test. See Case.methods.
    picked = [c for c in picked
              if not c.methods or method_of(candidate) in c.methods]
    language = language_of(candidate)
    if language is not None:
        picked = [c for c in picked if c.language == language]
    return picked


# Modalities whose output varies run to run. Diffusion varies enormously with
# the seed and a language model at temperature 0.2 is not deterministic either;
# Kokoro at a fixed voice and speed produces the same bytes every time, so
# repeating it burns time averaging three identical numbers. `extract` is
# excluded for a different reason: the answer is one token and the whole point
# of the lane is that it is cheap, so three samples of "137" buys nothing.
STOCHASTIC_MODALITIES = {"image", "video", "svg", "web", "code"}


#: Screen settings: the smallest thing that still proves the pipeline ran.
SCREEN_PARAMS = {"width": 256, "height": 256, "steps": 2, "seconds": None,
                 "frames": None}


def screen_cases(cases: list[Case]) -> list[Case]:
    """One case per modality, shrunk. Cheap enough to be wrong about."""
    import dataclasses
    picked: dict[str, Case] = {}
    for c in sorted(cases, key=lambda c: c.id):
        picked.setdefault(c.modality, c)
    out = []
    for c in picked.values():
        params = dict(c.params)
        for k, v in SCREEN_PARAMS.items():
            if k in params and v is not None:
                params[k] = min(params[k], v) if isinstance(params[k], int) else v
        out.append(dataclasses.replace(c, params=params))
    return out


def expand_cases(cases: list[Case], repeat: int) -> list[Case]:
    """Turn each stochastic case into `repeat` cases with different seeds.

    One sample per prompt ranks noise. A single generation decides a comparison
    on luck, and the whole point of the suite is that the comparison means
    something.
    """
    if repeat < 1:
        raise SystemExit(f"--repeat must be at least 1, got {repeat}")
    if repeat == 1:
        return cases

    out: list[Case] = []
    for case in cases:
        if case.modality not in STOCHASTIC_MODALITIES:
            out.append(case)
            continue
        base = case.params.get("seed", 0)
        for i in range(repeat):
            out.append(replace(case, id=f"{case.id}#{i + 1}",
                               # A copy, never the same dict: Case is frozen but
                               # its params are not, and mutating them would
                               # rewrite the case the caller still holds.
                               params={**case.params, "seed": base + i}))
    return out


def unrun_summary(cases: list[Case], candidates: list[str]) -> str:
    """Which cases no candidate in this run can execute.

    A case that silently never runs is invisible, and the summary below it
    looks complete. That is the same class of problem as a silent pass: an
    entire lane goes unmeasured and nothing says so.
    """
    covered = {c.id for candidate in candidates
               for c in cases_for(candidate, cases)}
    missed = [c for c in cases if c.id not in covered]
    if not missed:
        return ""
    by_modality: dict[str, list[str]] = {}
    for case in missed:
        by_modality.setdefault(case.modality, []).append(case.id)
    return "; ".join(f"{modality}: {', '.join(ids)}"
                     for modality, ids in sorted(by_modality.items()))


def select_cases(cases: list[Case], modality: str) -> list[Case]:
    if modality != "all" and modality not in MODALITIES:
        raise SystemExit(f"unknown modality '{modality}'; known: "
                         f"{', '.join(ALL_MODALITIES)}")
    chosen = cases if modality == "all" else [c for c in cases
                                              if c.modality == modality]
    if not chosen:
        raise SystemExit(f"no cases for modality '{modality}'")
    return chosen


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.run")
    ap.add_argument("--compare", nargs="+", metavar="RESULTS.JSON",
                    help="put finished runs in one table, or refuse if their "
                         "receipts say they are not comparable")
    ap.add_argument("--modality", required=False,
                    help=f"one of {', '.join(ALL_MODALITIES)}, or 'all'")
    ap.add_argument("--candidates", required=False,
                    help="comma-separated gateway aliases and/or engine specs")
    ap.add_argument("--gateway", default="http://127.0.0.1:4000")
    ap.add_argument("--cases", default=str(ROOT / "cases"))
    ap.add_argument("--out", default=None,
                    help="write artifacts and results.json here "
                         "(default: a new directory under "
                         "$LOCALHARNESS_HOME/runs)")
    ap.add_argument("--adherence", choices=("pickscore", "hpsv2"), default=None,
                    help="score how well each image matches its prompt. Loads a "
                         "multi-GB preference model, so it is opt-in and needs "
                         "`uv sync --group metrics`. Scores from the two "
                         "backends are NOT comparable to each other")
    ap.add_argument("--screen", action="store_true",
                   help="cheapest tier: one case per candidate at minimal "
                        "settings, answering only whether it runs. Never a "
                        "ranking")
    ap.add_argument("--repeat", type=int, default=1,
                    help="samples per case, each with a different seed. One "
                         "sample per prompt ranks noise; 3 is the usual "
                         "minimum for an image comparison you would act on")
    args = ap.parse_args(argv)
    if args.compare:
        return compare_runs(args.compare)
    # Required for a RUN, not for a comparison. Left off `required=True` so
    # `--compare` can stand alone; enforced here so a normal run still fails
    # loudly rather than halfway through.
    missing = [f"--{n}" for n in ("modality", "candidates")
               if not getattr(args, n)]
    if missing:
        raise SystemExit(f"{' and '.join(missing)} required (or use --compare)")

    if args.screen:
        # A screen is allowed to be statistically worthless. Its job is to
        # reject what does not run at all, which is how most things here have
        # failed: seedvr2 crashed 0/3, local-small never closed a tag 0/9.
        args.repeat = 1
        args.adherence = ""
    cases = expand_cases(select_cases(load_cases(args.cases), args.modality),
                         args.repeat)
    if args.screen:
        cases = screen_cases(cases)
    # An engine spec contains commas, which are also the candidate separator.
    # Split on commas that start a new candidate, i.e. those followed by a
    # known engine prefix or by something with no '=' in it.
    candidates = split_candidates(args.candidates)
    outdir = resolve_outdir(args.out, args.modality)
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    skipped = unrun_summary(cases, candidates)
    if skipped:
        print(f"\nnot run, no candidate for them -- {skipped}", file=sys.stderr)

    results = []
    for candidate in candidates:
        mine = cases_for(candidate, cases)
        if not mine:
            print(f"\n── {candidate}: no cases of a modality it can run, skipped",
                  file=sys.stderr)
            continue
        runner = build_runner(candidate, args.gateway, outdir,
                              adherence=args.adherence)
        print(f"\n── {runner.candidate}", flush=True)
        for case in mine:
            r = runner.run(case)
            results.append(r)
            mark = "pass" if r.passed else "FAIL"
            note = "" if r.passed else f"  {r.detail}"
            warn = f"  ({len(r.warnings)} warn)" if r.warnings else ""
            print(f"  {mark}  {r.seconds:6.2f}s  {case.id}{warn}{note}",
                  flush=True)
            if outdir and r.artifact and case.modality in TEXT_MODALITIES:
                ext = {"svg": "svg", "web": "html", "code": "py"}.get(
                    case.modality, "txt")
                (outdir / f"{runner.candidate.replace('/', '_')}--{case.id}.{ext}"
                 ).write_text(r.artifact, encoding="utf-8")

    if not results:
        raise SystemExit("nothing ran: no candidate matched any case")

    report(summarize(results))
    if outdir:
        # The RECEIPT: what this run was, so a later run can be told apart
        # from it before anyone ranks the two together. See core.comparable().
        receipt = Receipt(
            modality=args.modality,
            case_ids=tuple(sorted({c.id for c in cases})),
            repeat=args.repeat,
            sampling={m: dict(v) for m, v in sorted(completion.SAMPLING.items())},
            gateway=args.gateway,
            adherence=getattr(args, "adherence", "") or "",
            tier="screen" if getattr(args, "screen", False) else "measure",
            accelerator=accelerator_id(),
            instruments=instruments(),
            cases_digest=cases_digest(cases))
        (outdir / "results.json").write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "environment": capture(),
             "receipt": receipt.as_dict(),
             "summary": summarize(results),
             "rows": [vars(r) for r in results]}, indent=2),
            encoding="utf-8")
        print(f"\nartifacts + results.json in {outdir}")
    return 0


def instruments() -> dict:
    """What measured this run, for the receipt. See core.Receipt.instruments.

    Best effort by design: a receipt must not fail a finished run, and an
    instrument nobody could name is recorded as absent rather than as a
    mismatch -- comparable() compares only the keys both runs carry.
    """
    found = {}
    try:
        from harness import proc
        found["peak"] = proc.PEAK_METHOD
    except Exception:  # noqa: BLE001
        pass
    try:
        from harness.checks import ocr
        found["ocr"] = ocr.available_backend() or ""
    except Exception:  # noqa: BLE001
        pass
    return {k: v for k, v in found.items() if v}


def accelerator_id() -> str:
    """The receipt's accelerator field. `<kind>:<name>`, or empty if the
    machine cannot be asked -- an empty field is treated as unknown by
    comparable() rather than as a mismatch."""
    try:
        from harness import machine
        acc = machine.detect().accelerator
    except Exception:  # noqa: BLE001 - a receipt must not fail a finished run
        return ""
    return f"{acc.kind}:{acc.name}" if acc.kind else ""


def compare_runs(files: list[str]) -> int:
    """Put two or more finished runs in one table -- or refuse to.

    `comparable()` has existed and been tested since the receipts were added,
    and nothing called it. A guard that nothing invokes is a function.

    Refusing is the feature. Rows have been ranked here across runs with
    different sampling and different candidate sets, and the reader had no way
    to know.
    """
    import json

    loaded = []
    for f in files:
        try:
            data = json.loads(Path(f).read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            print(f"cannot read {f}: {exc}")
            return 1
        raw = data.get("receipt")
        if not raw:
            print(f"{f} carries no receipt, so it cannot be compared with "
                  f"anything. Runs written before receipts existed are in this "
                  f"state; re-run to get one.")
            return 1
        loaded.append((f, Receipt(modality=raw["modality"],
                                  case_ids=tuple(raw["case_ids"]),
                                  repeat=raw["repeat"],
                                  sampling=raw["sampling"],
                                  gateway=raw["gateway"],
                                  adherence=raw.get("adherence", ""),
                                  tier=raw.get("tier", "measure"),
                                  accelerator=raw.get("accelerator", ""),
                                  instruments=raw.get("instruments") or {},
                                  cases_digest=raw.get("cases_digest", "")),
                       data.get("summary") or {}))

    first_file, first, _ = loaded[0]
    for f, receipt, _ in loaded[1:]:
        ok, why = comparable(first, receipt)
        if not ok:
            print(f"REFUSED: {first_file} and {f} are not comparable -- {why}.")
            print("Ranking them in one table would compare two different exams.")
            return 1

    merged: dict = {}
    for f, _, summary in loaded:
        for name, row in summary.items():
            # Same candidate in two comparable runs: keep them apart by file,
            # since two samples of one candidate is a repeat, not a duplicate.
            key = name if name not in merged else f"{name} ({Path(f).parent.name})"
            merged[key] = row
    report(merged)
    return 0


def resolve_outdir(out: str | None, modality: str) -> Path:
    """Where this run's artifacts and results.json go.

    Always somewhere, and always the same shape. --out used to be REQUIRED for
    any lane that writes a file, so every invocation in this repo's history
    picked a directory by hand -- .logs/img, .logs/voices, .logs/ev-svg-fair --
    and none of them agreed.
    """
    if out:
        return Path(out)
    return paths.new_run(modality)


def split_candidates(raw: str) -> list[str]:
    """Split a candidate list on commas, keeping engine options attached.

    `mflux:z-image-turbo,quantize=4,local-mid` is two candidates, not three:
    a chunk containing '=' belongs to the candidate before it.
    """
    out: list[str] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" in chunk and out:
            out[-1] += f",{chunk}"
        else:
            out.append(chunk)
    return out


def _note_ranking_disagreements(summary: dict, metric_names: list) -> None:
    """Say so when the pass rate and a metric order the candidates differently.

    Seen live on the code lane: one model passed 50% of cases to another's 33%
    while writing 52.8% correct code to the other's 75.6%. A case with six
    strict checks fails outright on a single miss, so the binary view punishes
    the better model. Both numbers are true and neither is the answer on its
    own, so the reader has to be told they point different ways rather than
    reading the top line as a verdict.
    """
    if len(summary) < 2:
        return
    by_rate = [n for n, _ in sorted(summary.items(),
                                    key=lambda kv: -kv[1]["pass_rate"])]
    for metric in metric_names:
        # A TIE on pass rate cannot disagree with anything. Three parakeets all
        # scored 40/40 and the note still announced that one "passes more
        # cases", which is not a disagreement, it is the note describing sort
        # order noise.
        rates = {s.get("pass_rate", 0) for s in summary.values()}
        if len(rates) < 2:
            continue

        # A metric computed over a DIFFERENT NUMBER OF CASES per candidate is
        # not a ranking, and this note was making exactly that comparison: on
        # its first live run it announced that local-mid scored better on ink,
        # from the one case it passed, against a rival scored over two.
        partial = {name for name, s in summary.items()
                   if s.get("metric_n", {}).get(metric, s["total"]) < s["total"]}
        if partial:
            continue

        # Only candidates that actually reported this metric. A model that
        # failed every case has no opinion about it, and reading its absence as
        # a score is how a broken candidate wins.
        scored = {n: s for n, s in summary.items() if metric in s["metrics"]}
        if len(scored) < 2:
            continue
        by_metric = [n for n, _ in sorted(
            scored.items(), key=lambda kv: kv[1]["metrics"][metric],
            reverse=direction_of(metric) == "higher")]
        by_rate_scored = [n for n in by_rate if n in scored]
        if by_metric[0] != by_rate_scored[0]:
            print(f"\n  Note: pass rate and {metric} DISAGREE. {by_rate_scored[0]} "
                  f"passes more cases; {by_metric[0]} scores better on "
                  f"{metric}. A case fails outright on one missed check, so "
                  f"the pass rate punishes a strong model that slips once.")


def report(summary: dict) -> None:
    """Print the comparison, ordered by what actually distinguishes candidates.

    Pass rate first, then the quality metrics, then latency. Sorting on latency
    before quality is how a faster-but-worse candidate ends up on the top line.
    """
    metric_names = sorted({name for s in summary.values()
                           for name in s.get("metrics", {})})

    def rank(item):
        _, s = item
        # Negated for a higher-is-better metric so one ascending sort handles
        # both. Ranking every metric as an error rate put the candidate that
        # drew LEAST on the top line.
        scores = []
        for name in metric_names:
            # A NEUTRAL metric is reported and never ranked on. completion
            # tokens is the case: more is not better, and as higher-is-better
            # it would have put the most verbose candidate first.
            if direction_of(name) == "neutral":
                continue
            value = (s.get("metrics") or {}).get(name)
            if value is None:
                # A candidate that reported nothing must sort LAST, not as a
                # perfect score. Defaulting a lower-is-better metric to 0.0 put
                # a model that failed all 40 cases at the top of the ranking.
                scores.append(float("inf"))
            else:
                scores.append(value if direction_of(name) == "lower" else -value)
        return (-s.get("pass_rate", 0), scores, s.get("median_s", 0))

    arrows = {n: {"lower": "v", "higher": "^"}.get(direction_of(n), "-")
              for n in metric_names}
    header = (f"{'candidate':30} {'pass':>7} {'rate':>6} {'median':>8} "
              f"{'first':>8} {'peak':>9}")
    for name in metric_names:
        header += f" {name + ' ' + arrows[name]:>9} {name + '.worst':>13}"
    print("\n" + "=" * len(header))
    print(header)

    for name, s in sorted(summary.items(), key=rank):
        # .get throughout: report() now also renders summaries READ FROM DISK
        # for --compare, and a run written by an older version will not have
        # every key this one expects. Degrading is right; a KeyError on a
        # historical result is not.
        peak_kb = s.get("peak_kb") or 0
        peak = f"{peak_kb / 1024 / 1024:.1f}GiB" if peak_kb else "-"
        line = (f"{name:30} {s.get('passed', 0):>3}/{s.get('total', 0):<3} "
                f"{s.get('pass_rate', 0):>6.0%} "
                f"{s.get('median_s', 0):>7.2f}s "
                f"{s.get('first_s', 0):>7.2f}s {peak:>9}")
        for metric in metric_names:
            value = (s.get("metrics") or {}).get(metric)
            worst = s.get("metrics_worst", {}).get(metric)
            line += (f" {value:>9.3f}" if value is not None else f" {'-':>9}")
            line += (f" {worst:>13.3f}" if worst is not None else f" {'-':>13}")
        print(line)

    cold = [n for n, s2 in summary.items() if s2.get("first_is_cold")]
    if cold:
        print(f"\n`first` is the FIRST case, not a warm number. For {cold[0]} it "
              f"is a cold start;\nfor the others the runtime was already up. A "
              f"one-shot caller meets `first`.")

    if metric_names:
        low = [n for n in metric_names if direction_of(n) == "lower"]
        high = [n for n in metric_names if direction_of(n) == "higher"]
        flat = [n for n in metric_names if direction_of(n) == "neutral"]
        legend = []
        if low:
            legend.append(f"{', '.join(low)}: lower is better (v)")
        if high:
            legend.append(f"{', '.join(high)}: higher is better (^)")
        if flat:
            legend.append(f"{', '.join(flat)}: reported, not ranked on (-)")
        print("  " + "; ".join(legend))

    # HOW they failed, not just how often. Qwen3-14B scored 7/9 on svg and
    # looked like a winner; both failures were null content from a thinking
    # model that spent the budget reasoning. The pass rate could not say so.
    for name, s in sorted(summary.items(), key=rank):
        kinds = [(k, s.get(k, 0)) for k in ("wrong", "empty", "errored")]
        shown = [f"{n} {k}" for k, n in kinds if n]
        if len(shown) > 0 and s.get("passed", 0) < s.get("total", 0):
            print(f"  {name}: {', '.join(shown)}")

    # Two candidates in ONE run can sit different exams: a case may be unfair to
    # a method (chart-bars asserts <text>, which tracing cannot emit) or to a
    # language. Their pass rates are then not comparable, and 4/4 against 2/6
    # reads exactly as though they were.
    sat = {name: set(s.get("case_ids", [])) for name, s in summary.items()}
    if len({frozenset(v) for v in sat.values()}) > 1:
        shared = set.intersection(*sat.values()) if sat else set()
        print("  Note: candidates sat DIFFERENT cases, so the pass rates are "
              "not directly comparable.")
        for name in sorted(sat):
            extra = sorted(sat[name] - shared)
            if extra:
                print(f"    {name}: also ran {', '.join(extra)}")
        print(f"    shared by all: {', '.join(sorted(shared)) or 'none'}")

    # A mean over two of nine cases is not comparable with a mean over nine.
    # local-mid's svg ink of 0.564 was exactly that, printed beside 0.229.
    for name, s in sorted(summary.items(), key=rank):
        thin = {m: n for m, n in (s.get("metric_n") or {}).items()
                if m in metric_names and n < s.get("total", 0)}
        if thin:
            parts = ", ".join(f"{m} over {n}/{s.get('total', 0)}"
                              for m, n in sorted(thin.items()))
            print(f"  {name}: PARTIAL -- {parts}. Not comparable with a row "
                  f"scored over all {s.get('total', 0)}.")

    for name, s in summary.items():
        for f in s.get("failures") or []:
            print(f"  {name}: {f}")

    _note_ranking_disagreements(summary, metric_names)

    # Only worth saying when nothing separates the candidates. With a quality
    # metric in hand the suite CAN rank them, and repeating the disclaimer
    # would train the reader to skip it.
    all_pass = [s for s in summary.values() if s["pass_rate"] == 1.0]
    distinct = len({tuple(sorted(s["metrics"].items()))
                    for s in summary.values()}) > 1
    if len(all_pass) > 1 and not distinct:
        print("\nNote: every candidate passed everything and no metric "
              "separates them. These checks are a competence gate, not a "
              "quality ranking.")


if __name__ == "__main__":
    raise SystemExit(main())
