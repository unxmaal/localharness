"""Is the difference between two candidates real? Issues #8 and #57.

    uv run python -m evals.compare <run-dir> --baseline parakeet-tdt-0.6b-v2

A summary table ranks candidates. It cannot say whether the gap between two of
them is a finding or noise, and reading a ranking as though it could is how
this project once published a 2x claim that was 1.43x at four times the sample
size, and called a significant loss a tie in the same run.

PAIRED, over CASES. Resample the case list, recompute each candidate's metric
on that same list, take the difference. Pairing removes case difficulty, which
is the dominant source of variance here: every speech model shares the same
worst clip. Comparing two independently-resampled runs would drown the effect
in exactly the variance pairing removes.

Corpus rates, never the mean of per-case rates: a rate is a ratio, and
averaging ratios weights a two-word clip like a forty-word one (RULE #186).
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from evals.core import direction_of

DEFAULT_RESAMPLES = 3000
#: Metrics that are a ratio of two counts, where the corpus rate is the sum of
#: numerators over the sum of denominators rather than a mean of the rates.
RATIOS = {"wer": ("wer_errors", "wer_words")}


def load(run: str | Path) -> dict:
    path = Path(run)
    if path.is_dir():
        path = path / "results.json"
    return json.loads(path.read_text())


def paired(rows: list[dict], metric: str) -> tuple[dict, list[str]]:
    """{candidate: {case_id: (numerator, denominator)}}, and the shared cases."""
    num, den = RATIOS.get(metric, (metric, ""))
    by: dict[str, dict[str, tuple[float, float]]] = {}
    for r in rows:
        m = r.get("metrics") or {}
        if num not in m:
            continue
        by.setdefault(r["candidate"], {})[r["case_id"]] = (
            float(m[num]), float(m.get(den, 1.0)) if den else 1.0)
    if not by:
        return {}, []
    shared = sorted(set.intersection(*(set(v) for v in by.values())))
    return by, shared


def rate(by: dict, candidate: str, cases: list[str]) -> float:
    num = sum(by[candidate][c][0] for c in cases)
    den = sum(by[candidate][c][1] for c in cases)
    return num / den if den else 0.0


def interval(by: dict, cases: list[str], candidate: str, baseline: str, *,
             resamples: int = DEFAULT_RESAMPLES, seed: int = 1,
             confidence: float = 0.95) -> dict:
    """Difference in corpus rate, with a bootstrap confidence interval.

    An interval that spans zero means the two are NOT separable on this data,
    however different the point estimates look.
    """
    rng = random.Random(seed)
    point = rate(by, candidate, cases) - rate(by, baseline, cases)
    diffs = []
    for _ in range(resamples):
        sample = [cases[rng.randrange(len(cases))] for _ in cases]
        diffs.append(rate(by, candidate, sample) - rate(by, baseline, sample))
    diffs.sort()
    tail = (1.0 - confidence) / 2.0
    lo = diffs[int(tail * resamples)]
    hi = diffs[min(int((1.0 - tail) * resamples), resamples - 1)]
    return {"candidate": candidate, "baseline": baseline, "difference": point,
            "low": lo, "high": hi, "cases": len(cases),
            "separable": lo > 0 or hi < 0}


def verdict(row: dict, metric: str) -> str:
    """Which way a separable difference points, in the metric's own direction.

    direction_of returns "lower"/"higher"/"neutral", and a NEUTRAL metric is
    reported and never ranked on, so a difference in one is not a verdict.
    """
    if not row["separable"]:
        return "not separable"
    way = direction_of(metric)
    if way == "neutral":
        return "not ranked on"
    worse = row["difference"] > 0
    if way == "higher":
        worse = not worse
    return "worse" if worse else "BETTER"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("run", help="a run directory or a results.json")
    p.add_argument("--baseline", help="what to compare against "
                                      "(default: the first candidate)")
    p.add_argument("--metric", default="wer")
    p.add_argument("--resamples", type=int, default=DEFAULT_RESAMPLES)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--json", action="store_true")
    a = p.parse_args(argv)

    data = load(a.run)
    by, cases = paired(data.get("rows", []), a.metric)
    if not cases:
        print(f"no paired cases carrying {a.metric!r} in {a.run}")
        return 1
    names = sorted(by)
    baseline = a.baseline or names[0]
    if baseline not in by:
        print(f"no candidate {baseline!r}. Have: {', '.join(names)}")
        return 1
    rows = [interval(by, cases, n, baseline, resamples=a.resamples, seed=a.seed)
            for n in names if n != baseline]
    if a.json:
        print(json.dumps({"baseline": baseline, "metric": a.metric,
                          "cases": len(cases), "rows": rows}, indent=2))
        return 0
    print(f"\n{len(cases)} paired cases, {a.resamples} resamples, "
          f"metric {a.metric}")
    print(f"baseline {baseline}: {rate(by, baseline, cases):.4f}\n")
    for r in rows:
        print(f"  {r['candidate']:30.30s} {rate(by, r['candidate'], cases):.4f}"
              f"  {r['difference']:+.4f}  95% CI "
              f"[{r['low']:+.4f}, {r['high']:+.4f}]  {verdict(r, a.metric)}")
    print("\nAn interval spanning zero means these are not separable on this "
          "data,\nhowever different the point estimates look.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
