"""Run the eval suite and print a comparison.

    uv run python -m evals.run --modality svg --candidates local-mid,local-summarize

Sequenced by candidate, never interleaved: mlx_lm.server hot-swaps models per
request, so alternating between two aliases pays a model load on every single
case and measures disk speed instead of the model.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from evals.core import load_cases, summarize
from evals.runners.text import TextRunner

ROOT = Path(__file__).resolve().parent
TEXT_MODALITIES = {"svg", "web"}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="evals.run")
    ap.add_argument("--modality", required=True,
                    help="svg, web, or 'all' for every text modality")
    ap.add_argument("--candidates", required=True,
                    help="comma-separated gateway aliases")
    ap.add_argument("--gateway", default="http://127.0.0.1:4000")
    ap.add_argument("--cases", default=str(ROOT / "cases"))
    ap.add_argument("--out", default=None,
                    help="write artifacts and results.json here")
    args = ap.parse_args(argv)

    wanted = TEXT_MODALITIES if args.modality == "all" else {args.modality}
    cases = [c for c in load_cases(args.cases) if c.modality in wanted]
    if not cases:
        print(f"no cases for modality {args.modality}", file=sys.stderr)
        return 2

    candidates = [c.strip() for c in args.candidates.split(",") if c.strip()]
    outdir = Path(args.out) if args.out else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    results = []
    for candidate in candidates:
        runner = TextRunner(args.gateway, candidate)
        print(f"\n── {candidate}", flush=True)
        for case in cases:
            r = runner.run(case)
            results.append(r)
            mark = "pass" if r.passed else "FAIL"
            note = "" if r.passed else f"  {r.detail}"
            warn = f"  ({len(r.warnings)} warn)" if r.warnings else ""
            print(f"  {mark}  {r.seconds:6.2f}s  {case.id}{warn}{note}",
                  flush=True)
            if outdir and r.artifact:
                ext = "svg" if case.modality == "svg" else "html"
                (outdir / f"{candidate.replace('/', '_')}--{case.id}.{ext}"
                 ).write_text(r.artifact)

    summary = summarize(results)
    print("\n" + "=" * 66)
    print(f"{'candidate':22} {'pass':>7} {'rate':>6} {'median':>8} {'total':>8}")
    for name, s in sorted(summary.items(),
                          key=lambda kv: (-kv[1]["pass_rate"],
                                          kv[1]["median_s"])):
        print(f"{name:22} {s['passed']:>3}/{s['total']:<3} "
              f"{s['pass_rate']:>6.0%} {s['median_s']:>7.2f}s "
              f"{s['total_s']:>7.1f}s")
    for name, s in summary.items():
        for f in s["failures"]:
            print(f"  {name}: {f}")

    if outdir:
        (outdir / "results.json").write_text(json.dumps(
            {"generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
             "summary": summary,
             "rows": [vars(r) for r in results]}, indent=2))
        print(f"\nartifacts + results.json in {outdir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
