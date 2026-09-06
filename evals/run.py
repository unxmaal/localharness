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

from harness import audio
from harness.engines import Engine, parse_options, resolve

from dataclasses import replace

from evals.core import MODALITIES, Case, load_cases, summarize
from evals.environment import capture
from evals.runners.process import ProcessRunner
from evals.runners.speech import SpeechRunner
from evals.runners.text import CompletionRunner

ROOT = Path(__file__).resolve().parent
TEXT_MODALITIES = {"svg", "web"}
ALL_MODALITIES = sorted(MODALITIES)


PROCESS_ENGINES = ("mflux", "h3")
TTS_OPTIONS = {"voice"}


def kind_of(candidate: str) -> str:
    """process, tts, or gateway. The prefix decides, so a typo in the rest of
    the spec is reported as a bad spec rather than silently becoming a model
    name the gateway has never heard of."""
    head = candidate.split(":", 1)[0].split(",", 1)[0].strip()
    if head in PROCESS_ENGINES:
        return "process"
    if head == "tts":
        return "tts"
    return "gateway"


def modality_of(candidate: str) -> str | None:
    """The one modality this candidate can run, or None for text candidates,
    which can run all of them."""
    kind = kind_of(candidate)
    if kind == "tts":
        return "tts"
    if kind == "process":
        engine = engine_for(candidate)
        return engine.modality if engine else None
    return None


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
    return SpeechRunner(model=model, outdir=outdir,
                        voice=options.get("voice", audio.DEFAULT_VOICE))


def build_runner(candidate: str, gateway: str, outdir: Path | None):
    kind = kind_of(candidate)
    if kind == "gateway":
        return CompletionRunner(gateway, candidate)
    if kind == "tts":
        return _speech_runner(candidate, outdir)
    try:
        engine = resolve(candidate)
    except ValueError as exc:
        # Before anything runs: a typo must not cost a forty-minute generation.
        raise SystemExit(str(exc)) from exc
    if outdir is None:
        raise SystemExit(
            f"{candidate} writes files; pass --out to say where they go")
    return ProcessRunner(engine, outdir)


def cases_for(candidate: str, cases: list[Case]) -> list[Case]:
    """The cases this candidate can actually run."""
    modality = modality_of(candidate)
    if modality is None:
        return [c for c in cases if c.modality in TEXT_MODALITIES]
    return [c for c in cases if c.modality == modality]


# Modalities whose output varies run to run. Diffusion varies enormously with
# the seed and a language model at temperature 0.2 is not deterministic either;
# Kokoro at a fixed voice and speed produces the same bytes every time, so
# repeating it burns time averaging three identical numbers.
STOCHASTIC_MODALITIES = {"image", "video", "svg", "web"}


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

    results = []
    for candidate in candidates:
        mine = cases_for(candidate, cases)
        if not mine:
            print(f"\n── {candidate}: no cases of a modality it can run, skipped",
                  file=sys.stderr)
            continue
        runner = build_runner(candidate, args.gateway, outdir)
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
                ext = "svg" if case.modality == "svg" else "html"
                (outdir / f"{runner.candidate.replace('/', '_')}--{case.id}.{ext}"
                 ).write_text(r.artifact)

    if not results:
        raise SystemExit("nothing ran: no candidate matched any case")

    report(summarize(results))
    if outdir:
        (outdir / "results.json").write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "environment": capture(),
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


def report(summary: dict) -> None:
    """Print the comparison, ordered by what actually distinguishes candidates.

    Pass rate first, then the quality metrics, then latency. Sorting on latency
    before quality is how a faster-but-worse candidate ends up on the top line.
    """
    metric_names = sorted({name for s in summary.values()
                           for name in s.get("metrics", {})})

    def rank(item):
        _, s = item
        # Every metric so far is an error rate, so lower is better. A metric
        # where higher is better would need a direction declared with it.
        return (-s["pass_rate"],
                [s["metrics"].get(n, 0.0) for n in metric_names],
                s["median_s"])

    header = f"{'candidate':30} {'pass':>7} {'rate':>6} {'median':>8} {'peak':>9}"
    for name in metric_names:
        header += f" {name:>7} {name + '.max':>11}"
    print("\n" + "=" * len(header))
    print(header)

    for name, s in sorted(summary.items(), key=rank):
        peak = f"{s['peak_kb'] / 1024 / 1024:.1f}GiB" if s["peak_kb"] else "-"
        line = (f"{name:30} {s['passed']:>3}/{s['total']:<3} "
                f"{s['pass_rate']:>6.0%} {s['median_s']:>7.2f}s {peak:>9}")
        for metric in metric_names:
            value = s["metrics"].get(metric)
            worst = s.get("metrics_worst", {}).get(metric)
            line += (f" {value:>7.3f}" if value is not None else f" {'-':>7}")
            line += (f" {worst:>11.3f}" if worst is not None else f" {'-':>11}")
        print(line)

    for name, s in summary.items():
        for f in s["failures"]:
            print(f"  {name}: {f}")

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
