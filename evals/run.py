"""Run the eval suite and print a comparison.

    uv run python -m evals.run --modality svg --candidates local-mid,local-small
    uv run python -m evals.run --modality image --out .logs/img \
        --candidates mflux:flux2-klein-4b,mflux:z-image-turbo

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

from harness import audio, completion
from harness.engines import Engine, parse_options, resolve

from dataclasses import replace

from evals.core import (MODALITIES, Case, Receipt, direction_of,
                        load_cases, summarize)
from evals.environment import capture
from evals.runners.process import ProcessRunner
from evals.runners.speech import SpeechRunner
from evals.runners.text import CompletionRunner
from evals.runners.transcription import TranscriptionRunner

ROOT = Path(__file__).resolve().parent
TEXT_MODALITIES = {"svg", "web", "code", "extract"}
ALL_MODALITIES = sorted(MODALITIES)


PROCESS_ENGINES = ("mflux", "h3")
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
    return "gateway"


def modality_of(candidate: str) -> str | None:
    """The one modality this candidate can run, or None for text candidates,
    which can run all of them."""
    kind = kind_of(candidate)
    if kind in ("tts", "stt"):
        return kind
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
    try:
        engine = resolve(candidate)
    except ValueError as exc:
        # Before anything runs: a typo must not cost a forty-minute generation.
        raise SystemExit(str(exc)) from exc
    if outdir is None:
        raise SystemExit(
            f"{candidate} writes files; pass --out to say where they go")
    return ProcessRunner(engine, outdir, adherence=adherence)


def cases_for(candidate: str, cases: list[Case]) -> list[Case]:
    """The cases this candidate can actually run."""
    modality = modality_of(candidate)
    if modality is None:
        return [c for c in cases if c.modality in TEXT_MODALITIES]
    picked = [c for c in cases if c.modality == modality]
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
    ap.add_argument("--modality", required=True,
                    help=f"one of {', '.join(ALL_MODALITIES)}, or 'all'")
    ap.add_argument("--candidates", required=True,
                    help="comma-separated gateway aliases and/or engine specs")
    ap.add_argument("--gateway", default="http://127.0.0.1:4000")
    ap.add_argument("--cases", default=str(ROOT / "cases"))
    ap.add_argument("--out", default=None,
                    help="write artifacts and results.json here")
    ap.add_argument("--adherence", choices=("pickscore", "hpsv2"), default=None,
                    help="score how well each image matches its prompt. Loads a "
                         "multi-GB preference model, so it is opt-in and needs "
                         "`uv sync --group metrics`. Scores from the two "
                         "backends are NOT comparable to each other")
    ap.add_argument("--repeat", type=int, default=1,
                    help="samples per case, each with a different seed. One "
                         "sample per prompt ranks noise; 3 is the usual "
                         "minimum for an image comparison you would act on")
    args = ap.parse_args(argv)

    cases = expand_cases(select_cases(load_cases(args.cases), args.modality),
                         args.repeat)
    # An engine spec contains commas, which are also the candidate separator.
    # Split on commas that start a new candidate, i.e. those followed by a
    # known engine prefix or by something with no '=' in it.
    candidates = split_candidates(args.candidates)
    outdir = Path(args.out) if args.out else None
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
                 ).write_text(r.artifact)

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
            adherence=getattr(args, "adherence", "") or "")
        (outdir / "results.json").write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "environment": capture(),
             "receipt": receipt.as_dict(),
             "summary": summarize(results),
             "rows": [vars(r) for r in results]}, indent=2))
        print(f"\nartifacts + results.json in {outdir}")
    return 0


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
            value = s["metrics"].get(name)
            if value is None:
                # A candidate that reported nothing must sort LAST, not as a
                # perfect score. Defaulting a lower-is-better metric to 0.0 put
                # a model that failed all 40 cases at the top of the ranking.
                scores.append(float("inf"))
            else:
                scores.append(value if direction_of(name) == "lower" else -value)
        return (-s["pass_rate"], scores, s["median_s"])

    arrows = {n: "v" if direction_of(n) == "lower" else "^" for n in metric_names}
    header = f"{'candidate':30} {'pass':>7} {'rate':>6} {'median':>8} {'peak':>9}"
    for name in metric_names:
        header += f" {name + ' ' + arrows[name]:>9} {name + '.worst':>13}"
    print("\n" + "=" * len(header))
    print(header)

    for name, s in sorted(summary.items(), key=rank):
        peak = f"{s['peak_kb'] / 1024 / 1024:.1f}GiB" if s["peak_kb"] else "-"
        line = (f"{name:30} {s['passed']:>3}/{s['total']:<3} "
                f"{s['pass_rate']:>6.0%} {s['median_s']:>7.2f}s {peak:>9}")
        for metric in metric_names:
            value = s["metrics"].get(metric)
            worst = s.get("metrics_worst", {}).get(metric)
            line += (f" {value:>9.3f}" if value is not None else f" {'-':>9}")
            line += (f" {worst:>13.3f}" if worst is not None else f" {'-':>13}")
        print(line)

    if metric_names:
        low = [n for n in metric_names if direction_of(n) == "lower"]
        high = [n for n in metric_names if direction_of(n) == "higher"]
        legend = []
        if low:
            legend.append(f"{', '.join(low)}: lower is better (v)")
        if high:
            legend.append(f"{', '.join(high)}: higher is better (^)")
        print("  " + "; ".join(legend))

    # HOW they failed, not just how often. Qwen3-14B scored 7/9 on svg and
    # looked like a winner; both failures were null content from a thinking
    # model that spent the budget reasoning. The pass rate could not say so.
    for name, s in sorted(summary.items(), key=rank):
        kinds = [(k, s.get(k, 0)) for k in ("wrong", "empty", "errored")]
        shown = [f"{n} {k}" for k, n in kinds if n]
        if len(shown) > 0 and s["passed"] < s["total"]:
            print(f"  {name}: {', '.join(shown)}")

    # A mean over two of nine cases is not comparable with a mean over nine.
    # local-mid's svg ink of 0.564 was exactly that, printed beside 0.229.
    for name, s in sorted(summary.items(), key=rank):
        thin = {m: n for m, n in s.get("metric_n", {}).items()
                if m in metric_names and n < s["total"]}
        if thin:
            parts = ", ".join(f"{m} over {n}/{s['total']}"
                              for m, n in sorted(thin.items()))
            print(f"  {name}: PARTIAL -- {parts}. Not comparable with a row "
                  f"scored over all {s['total']}.")

    for name, s in summary.items():
        for f in s["failures"]:
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
