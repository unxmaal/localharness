"""The measure tier, verify and throughput: run a candidate's cases and settle what they say."""
from __future__ import annotations

import time

from harness import lanes, paths, reasons
from harness.commands.common import emit, err, note
from harness.commands import jobs as jobs_cmd


def cmd_verify(a) -> int:
    """Establish that each lane works, by running its own default. #234."""
    import subprocess

    from harness import report, verify

    state = report.state()
    tasks = verify.plan(state["lanes"], only=getattr(a, "lane", ""),
                        force=getattr(a, "all", False))
    if not tasks:
        # NOT "every lane has a receipt". A parked lane has none and is
        # excluded from the plan by design, so saying so here would report a
        # decision as a measurement -- the exact confusion #244 removed from
        # the report a few lines away.
        parked = [l for l in state["lanes"] if l.get("parked")]
        note("every lane that should run here has a recent receipt")
        for l in parked:
            note(f"  {l['lane']} is parked and was not run: {l['parked']}")
        emit(planned=[], parked={l["lane"]: l["parked"] for l in parked})
        return 0

    runnable = [t for t in tasks if not t.skip]
    budget = sum(t.cost_s for t in runnable)
    note(f"\n{len(runnable)} lane(s) to verify, roughly {budget // 60}m "
         f"{budget % 60}s in total:\n")
    for t in tasks:
        note(f"  {t.lane:8} {t.candidate[:40]:40} "
             f"{t.skip or f'~{t.cost_s}s'}")
        note(f"           {t.why}")
    if not getattr(a, "run", False):
        note("\n--run to spend it. Nothing is downloaded either way.")
        emit(planned=[jobs_cmd._task_row(t) for t in tasks], ran=False)
        return 0

    rc = 0
    results = []
    for t in runnable:
        # Its own directory, read back from the store by that name. #410.
        out = paths.new_run(t.lane)
        argv = t.argv + ["--out", str(out)]
        note(f"\n=== {t.lane} ===\n    {' '.join(argv)}", flush=True)
        proc = subprocess.run(argv, capture_output=True, text=True)
        data = _receipt_at(out) or {}
        got = verify.verdict(data) if proc.returncode == 0 else "broken"
        line = (verify.summarise(t.lane, data) if proc.returncode == 0
                else (proc.stderr.strip().splitlines() or ["no stderr"])[-1])
        results.append((t.lane, got, line))
        note(f"    {got.upper()}: {line[:150]}")
        if got == "broken":
            rc = 1
    note("\n=== what the lanes do ===")
    for lane, got, _ in results:
        note(f"  {got:8} {lane}")
    emit(ok=rc == 0, planned=[jobs_cmd._task_row(t) for t in tasks], ran=True,
         results=[{"lane": l, "verdict": g, "summary": s_}
                  for l, g, s_ in results])
    return rc


def _ttft_pair(p50, p95) -> str:
    from harness.report import ttft_text
    return ttft_text(p50, p95)


def _load_pair(pair) -> str:
    return "/".join("-" if x is None else f"{x:.1f}" for x in (pair or [None, None]))


def cmd_throughput(a) -> int:
    """How much faster a text spec goes with several requests in flight. #310."""
    import contextlib
    import json as _json
    from harness import exclusive, serving, throughput, vllm
    if not a.texts and not a.claims:
        return err("throughput needs --texts or --claims")
    texts: list = []
    for path in [a.texts] if a.texts else a.claims:
        try:
            rows = [_json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]
        except (OSError, ValueError) as exc:
            return err(f"cannot read {path}: {exc}")
        texts += (throughput.claims_requests(rows) if not a.texts
                  else [str(r.get(a.field) or "") for r in rows])
    texts = texts[: a.n]
    levels = tuple(int(x) for x in a.levels.split(",") if x.strip())
    serve = getattr(a, "serve", None)
    try:
        where = serving.route(a.model, a.gateway)
        engine = serve or serving.engine_for(a.model)
    except ValueError as exc:
        return err(str(exc))
    if serve and not a.model.startswith(serving.VLLM_PREFIX):
        return err(f"--serve starts a vLLM server; --model must be "
                   f"{serving.VLLM_PREFIX}<repo id>, not {a.model}")
    from harness import models
    models.warn(a.model)
    note(f"{models.display(a.model)} on {engine}: {len(texts)} texts at {levels} in flight, "
         f"max_tokens {a.max_tokens}", flush=True)
    pid = getattr(a, "server_pid", None)
    with exclusive.held("eval"), contextlib.ExitStack() as stack:
        if serve:
            port = int(where.base.rsplit(":", 1)[1])
            note(f"  starting {serve} on port {port}", flush=True)
            srv = stack.enter_context(vllm.served(
                vllm.argv(serve, where.model, port), port,
                log=paths.home() / "logs" / f"{serve}-{port}.log"))
            pid = srv.proc.pid
        got = throughput.sweep(
            where.model, texts, levels=levels, max_tokens=a.max_tokens,
            gateway=where.base,
            footprint=(lambda: throughput.footprint_tree(pid)) if pid else None)
    if got and "warmup_s" in got[0]:
        note(f"  warm-up {got[0]['warmup_s']:.2f}s, not counted"
             f"{'' if got[0]['warmup_ok'] else ' (FAILED)'}", flush=True)
    base = got[0]["per_hour"] or 1
    for r in got:
        peak = r.get("peak_bytes")
        note(f"  {r['concurrency']:2d} in flight  {r['per_hour']:7.1f}/h  "
             f"x{r['per_hour'] / base:.2f}  {r.get('tokens_per_s', 0):7.1f} tok/s  "
             f"p50 {r['p50_s']:6.2f}s  p95 {r['p95_s']:6.2f}s  "
             f"ttft {_ttft_pair(r.get('ttft_p50_s'), r.get('ttft_p95_s'))}  "
             f"peak {'-' if peak is None else f'{peak / 1024 ** 3:.1f} GiB'}  "
             f"errors {r['errors']}  "
             f"schema {r.get('schema_valid', 0)}/{r.get('schema_checked', 0)}  "
             f"load {_load_pair(r.get('load_avg'))}  "
             f"tokens {r['completion_tokens']}", flush=True)
        for why, n in (r.get("error_kinds") or {}).items():
            note(f"      {n} x {why}", flush=True)
    emit(model=a.model, engines={a.model: engine}, max_tokens=a.max_tokens, levels=got)
    return 0


def measurable(store, top: int, want: str = "") -> list[dict]:
    """Survivors in the scoped lane, filtered before the limit. #386."""
    from harness import memory_store as ms
    rows = ms.survivors(store, limit=1_000_000)
    if want:
        rows = [r for r in rows if lanes.serves(r.get("lane"), want)]
    return rows[:top]


def _measure_and_adopt(a, row: dict) -> int:
    """_measure, then unload the challenger it loaded. #444."""
    from harness import router
    loaded: list[str] = []
    try:
        return _measure(a, row, loaded)
    finally:
        for spec in loaded:
            router.release_spec(spec, f"measured {row['name']}")


def _measure(a, row: dict, loaded: list) -> int:
    """One challenger against the lane's incumbent, then the verdict.

    PAIRED AND IN ONE RUN. Both candidates see the same cases, the same repeat
    count and the same machine, so the only axis that moved is the candidate.
    Two separate runs would be two receipts that comparable() would refuse, and
    rightly.
    """
    import subprocess

    from harness import adopt, candidates, screen, winners
    from harness import memory_store as ms

    name = row["name"]
    # A text candidate is filed under `code` and can be measured in web, svg
    # and extract too. When the caller scoped the loop to one of those, that
    # is the lane to measure in: the incumbent, the cases and the metric all
    # belong to the lane being asked about, not the one the row was filed
    # under. #207.
    want = (getattr(a, "lane", "") or "").strip().lower()
    lane = (want if want and lanes.serves(row.get("lane"), want)
            else lanes.canonical(row.get("lane")))
    store = ms.connect()
    try:
        spec = candidates.for_proposal(store, lane, name,
                                       row.get("attaches_to") or "")
    finally:
        store.close()
    if not spec:
        return err(f"{name}: screened in the {lane} lane and no candidate "
                   f"spec can be built for it")
    incumbent = adopt.default_for(lane, winners.typed().get(lane, ""))
    if not incumbent:
        print(f"  {name}: the {lane} lane has no incumbent to beat, so there "
              f"is nothing to compare against. Measure it on its own first.")
        return 0
    inc_spec = screen.candidate_for(lane, incumbent) or incumbent
    if spec == inc_spec:
        # An adopted winner is the incumbent; measuring it against itself
        # wrote `declined` for the lane's own default. #393.
        print(f"  {name}: already the {lane} lane's default ({spec})")
        _settle(name, "measured", f"{lane}: already the lane's default")
        return 0
    # THE PAIR MUST REACH THE SAME SERVER. One --gateway serves the whole run,
    # so when the challenger's repo id sends it to mlx_lm.server the incumbent
    # cannot travel as a LiteLLM alias: :8081 has never heard of `q3-4b`, the
    # control scored 0 of 27, and the run had nothing to compare against. The
    # config maps every alias to the upstream behind it. #223.
    if screen.routed_gateway(name):
        upstream = screen.upstream_of(incumbent)
        if upstream:
            inc_spec = screen.candidate_for(lane, upstream) or upstream
    # NAME THE DIRECTORY, DO NOT GUESS AT IT AFTERWARDS. A newest-receipt read
    # whichever directory sorted highest, and `legacy-ev-small-code` outranks
    # every timestamp because `l` sorts above `2`. The loop measured two
    # candidates and then read a receipt from a different experiment. #222.
    out = (paths.home() / "runs"
           / f"{time.strftime('%Y%m%d-%H%M%S')}-adopt-{lane}")
    split, plan = _plan(a, lane, inc_spec)
    argv = ["uv", "run", "python", "-m", "evals.run", "--modality", lane,
            "--repeat", str(plan.repeat),
            "--out", str(out),
            "--candidates", f"{inc_spec},{spec}"]
    # ROUTE IT THE WAY THE SCREEN DOES. LiteLLM validates `model` against its
    # alias table and a discovered candidate is always a repo id, so the
    # measure sent every request to a server that was never going to accept
    # the name: 0/27 at 11ms a case, reported as "does not beat the incumbent
    # on the lane's metric". #223, which is #206 at the tier its fix did not
    # reach.
    route = screen.routed_gateway(name)
    if route:
        argv += ["--gateway", route]
    print(f"\n  {lane}: {name} against {incumbent}")
    print(f"    {' '.join(argv)}", flush=True)
    loaded.append(spec)
    proc = subprocess.run(argv, capture_output=True, text=True)
    if proc.returncode != 0:
        err(proc.stderr.strip()[-400:] or "no stderr")
        return 1
    data = _receipt_at(out)
    if not data:
        return err(f"{name}: the run stored no receipt for {out}, so nothing "
                   f"can be adopted from it")
    summary = data.get("summary") or {}
    rows = data.get("rows") or []
    # The run's own specs map is the mapping; store it, then read keys back.
    store = ms.connect()
    try:
        candidates.from_receipt(store, data.get("specs") or {}, lane=lane,
                                proposals={spec: name})
        inc_key = candidates.key_for(store, inc_spec)
        ch_key = candidates.key_for(store, spec)
    finally:
        store.close()
    inc_row = _summary_row(summary, inc_key)
    ch_row = _summary_row(summary, ch_key)
    # ASSERT THE RUN IS THE ONE THAT WAS ASKED FOR. Naming the directory stops
    # the loop reading a stranger's receipt; this stops it reading a receipt
    # that is its own and yet describes a different exam, which a crashed or
    # partially-skipped candidate produces. #222.
    if len(summary) and not (inc_row or ch_row):
        return err(f"{name}: the receipt at {out} names {sorted(summary)!r} "
                   f"and neither candidate this run asked for, so it does not "
                   f"describe the run that was just made")
    if not inc_row:
        # THE INCUMBENT IS THE CONTROL. A candidate measured beside a control
        # that did not run says nothing about the candidate, which is the
        # lesson the tts lane already paid for (#194/#195). Name it as the
        # control rather than as a missing summary key: a 15-minute run that
        # ends in "the summary names [...]" makes the reader go looking in the
        # receipt for a spelling problem. Issue #214.
        return err(f"{name}: the incumbent {incumbent} contributed no rows, so "
                   f"this run has no control and nothing can be concluded from "
                   f"it. The summary names {sorted(summary)!r}")
    if not ch_row:
        return err(f"{name}: the challenger contributed no rows. The summary "
                   f"names {sorted(summary)!r}")
    # A CANDIDATE THAT NEVER RAN IS NOT A CANDIDATE THAT LOST. Every row a
    # harness refusal means the request never reached a model, and handing
    # that to adopt.decide dresses a routing failure as a quality result.
    # The rows carry the runner's failure class. #223, #408.
    # THE CONTROL MUST HAVE RUN. This is the general form of the refusal check
    # below, and it catches every variant of "the request never reached a
    # model" without anyone having to classify it first: a 404 from a
    # doubled /v1 got past the phrase list, both candidates scored 0/27, and
    # the loop reported "does not beat the incumbent on the lane's metric".
    # A candidate measured beside a control that passed nothing says nothing
    # about the candidate. #223, and the lesson the tts lane paid for in #194.
    if not int(inc_row.get("passed") or 0):
        return err(f"{name}: the incumbent {incumbent} passed "
                   f"0 of {inc_row.get('total') or '?'}, so this run has no "
                   f"working control and nothing can be concluded from it. "
                   f"Fix the lane before reading the challenger.")
    refused = _all_refused(rows, ch_row.get("candidate") or name)
    if refused:
        store = ms.connect()
        try:
            ms.decide(store, name, "queued", tier=ms.SCREEN,
                      detail=f"not measured: {refused}",
                      run_id=data.get("run_id"), reason=reasons.HARNESS)
        except (KeyError, ms.IllegalTransition):
            pass      # measured by hand, never proposed; the report still stands
        finally:
            store.close()
        return err(f"{name}: every case was refused before it reached a model "
                   f"({refused}). The incumbent passed, so this says nothing "
                   f"about the candidate and it stays queued.")
    verdict = adopt.decide(lane, inc_row, ch_row, rows, split=split, plan=plan, spec=spec)
    print(f"    {'ADOPTED' if verdict.adopt else 'kept the incumbent'}: "
          f"{verdict.why}")
    store = ms.connect()
    try:
        # On the proposal the candidate maps to, so it leaves survivors. #393.
        adopt.record(store, verdict, spec=spec, run_id=data.get("run_id"))
    except ms.IllegalTransition as exc:
        print(f"  skipped {exc}", flush=True)
    finally:
        store.close()
    return 0


def _plan(a, lane: str, inc_spec: str):
    """The lane's split and the repeat the paired gate needs on its holdout, within the time budget. #479, #591."""
    from evals.run import STOCHASTIC_MODALITIES
    from harness import candidates, holdout, power
    from harness import memory_store as ms

    split = holdout.for_lane(lane)
    store = ms.connect()
    try:
        draws, median_s = power.incumbent_profile(
            store, lane, candidates.key_of(inc_spec) or inc_spec)
    finally:
        store.close()
    rates, seen = power.rates_from(draws, split.holdout)
    effect = getattr(a, "effect", None) or power.effect_for(lane)
    explicit = getattr(a, "repeat", None)
    plan = power.plan([rates[c] for c in split.holdout], effect,
                      stochastic=lane in STOCHASTIC_MODALITIES,
                      repeat=explicit, observed=seen,
                      rho=power.icc([list(draws.values())]))
    budget = float(getattr(a, "power_budget_min", None) or power.BUDGET_MIN) * 60
    plan = power.within_budget(
        plan, [rates[c] for c in split.holdout],
        cases_run=len(set(split.dev) | set(split.holdout)), median_s=median_s,
        budget_s=budget, explicit=bool(explicit))
    print(f"    power: repeat {plan.repeat} over {len(split.holdout)} holdout "
          f"case(s) gives {plan.power:.2f} to detect +{plan.effect:.2f} at "
          f"alpha {plan.alpha}{'; ' + plan.why if plan.why else ''}"
          f"{'; ' + split.caveat if split.caveat else ''}", flush=True)
    missing = power.unmeasured(draws, split.holdout)
    if missing:
        print(f"    projected: {len(missing)} of {len(split.holdout)} holdout "
              f"unmeasured, drawn from the incumbent's measured per-case rates",
              flush=True)
    return split, plan


def _settle(name: str, outcome: str, detail: str) -> None:
    from harness import memory_store as ms
    store = ms.connect()
    try:
        ms.decide(store, name, outcome, tier=ms.MEASURE, detail=detail[:200],
                  reason=reasons.CANDIDATE)
    except (KeyError, ms.IllegalTransition):
        pass
    finally:
        store.close()


def _summary_row(summary: dict, key: str) -> dict | None:
    """The summary entry under the receipt key the candidates table holds. #407."""
    if not key or key not in summary:
        return None
    return {**(summary[key] or {}), "candidate": key}


#: `_all_refused` found no rows under the key it was given while the receipt
#: held rows under others. Reported rather than returned as "": the lookup is
#: the thing that failed, and saying "it ran fine" would be a guess.
NO_ROWS_FOR_CANDIDATE = "no rows under that name in the receipt"


def _all_refused(rows, candidate: str) -> str:
    """The failure class, when EVERY row for `candidate` never reached a model.

    Empty when any row actually reached a model, because then the candidate
    really was measured and a low score is its own.

    MATCHED ON THE RECEIPT KEY, which is what the rows carry. A bare repo id
    matches nothing for an engine lane, where the key is `mflux/<id>-q8`, and
    "matched nothing" used to return "" -- the same answer as "it ran" -- so a
    mis-keyed lookup silently disabled the guard instead of failing. A safety
    check whose lookup miss looks like a pass is worse than no check.
    """
    mine = [r for r in rows or [] if r.get("candidate") == candidate]
    if not mine:
        # No rows at all is a different fact and the caller handles it; no
        # rows for THIS candidate when others have some is a key mismatch.
        return "" if not rows else NO_ROWS_FOR_CANDIDATE
    seen = {r.get("failure_class") or "" for r in mine}
    return sorted(seen)[0] if seen <= set(reasons.NEVER_RAN) else ""


def _receipt_at(out, conn=None) -> dict | None:
    """The stored run for the directory this invocation named. #222, #410.

    The only way to know which run a receipt describes is to have named it.
    """
    from harness import runs
    with runs.store(conn) as c:
        return runs.receipt_at(c, out)
