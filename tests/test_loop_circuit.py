"""The whole loop through `soh discover --loop --run`, faked only at the network and process edges. #492 F(2)."""
import contextlib
import functools
import io
import json
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

import httpx
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import fakes  # noqa: E402

from harness import (adopt, cli, completion, downloads, gateway, gateway_switch as gs,  # noqa: E402
                     inspect as ins, machine as _machine, memory)
from harness import memory_store as ms  # noqa: E402
from harness.memory import Accelerator  # noqa: E402

CHALLENGER = "openbmb/MiniCPM5-1B"
INCUMBENT_UPSTREAM = "mlx-community/Qwen3-4B-Instruct-2507-4bit"
FEED = "https://feeds.invalid/sohot/atom"
SERVER = "http://127.0.0.1:9"
REAL_RUN = subprocess.run
CASES = Path(__file__).resolve().parents[1] / "evals" / "cases"


def _hand_written(cases):
    """Imported cases (#603) are data the importer tests cover; the circuit runs the hand-written lane."""
    from evals import importers
    names = {s.name for s in importers.REGISTRY}
    return [c for c in cases if c.source is None or c.source.parent.name not in names]


@functools.lru_cache(maxsize=None)
def _cases(lane):
    from evals.core import load_cases
    from evals.run import select_cases
    return select_cases(_hand_written(load_cases(CASES)), lane)


def _reference(case_id):
    case = next(c for c in _cases("code") if c.id == case_id)
    return (case.source.parent / "reference" / f"{case_id}.py").read_text(encoding="utf-8")


class World:
    """The internet, the model server and the child processes, with one knob per fault."""

    def __init__(self):
        self.hf_429_after, self.hf_answered = None, 0
        self.snapshot_files = {"config.json": "{}", "model.safetensors": "x",
                               "tokenizer.json": "{}"}
        self.server = "ok"
        self.decide_answer = None
        self.unexpected, self.evals, self.real, self.checked = [], [], [], {}
        from harness import holdout
        split = holdout.for_lane("code")
        self.prompts = {c.id: c.prompt.strip() for c in _cases("code")}
        # The control passes one holdout case and two dev cases; the challenger passes them all.
        self.incumbent_passes = {split.holdout[0], *split.dev[:2]}

    def urlopen(self, req, timeout=None):
        url = req.full_url if hasattr(req, "full_url") else str(req)
        if url == FEED:
            body = (f'<feed xmlns="http://www.w3.org/2005/Atom"><entry><title>new small model'
                    f'</title><link href="https://huggingface.co/{CHALLENGER}"/>'
                    f'<updated>2026-10-01T00:00:00Z</updated><content>Try '
                    f'https://huggingface.co/{CHALLENGER} for code</content></entry></feed>')
        elif url.startswith("https://huggingface.co/api/models/"):
            if self.hf_429_after is not None and self.hf_answered >= self.hf_429_after:
                raise urllib.error.HTTPError(url, 429, "Too Many Requests", {}, None)
            self.hf_answered += 1
            body = fakes.registry()(url)
        else:
            self.unexpected.append(url)
            raise urllib.error.URLError("no network in tests")
        return contextlib.closing(io.BytesIO(body.encode("utf-8")))

    def snapshot(self, repo_id, **kw):
        where = downloads.hub_dir(repo_id) / "snapshots" / "0"
        where.mkdir(parents=True, exist_ok=True)
        for name, text in self.snapshot_files.items():
            (where / name).write_text(text, encoding="utf-8")
        return str(where)

    def run(self, argv, *a, **kw):
        argv = [str(x) for x in argv]
        if argv[:2] == ["gh", "api"]:
            return subprocess.CompletedProcess(argv, 0, "[]", "")
        if argv[:2] == ["git", "clone"]:
            return subprocess.CompletedProcess(argv, 128, "", "fatal: no network in tests")
        if "evals.run" in argv:
            return self._evals(argv[argv.index("evals.run") + 1:])
        self.real.append(argv[:2])
        if argv[0] == sys.executable and argv[-1].endswith("candidate.py"):
            # The code checker's child is deterministic in its program, so a repeat reuses it.
            program = Path(argv[-1]).read_text(encoding="utf-8")
            if program not in self.checked:
                self.checked[program] = REAL_RUN(argv, *a, **kw)
            return self.checked[program]
        return REAL_RUN(argv, *a, **kw)

    def _evals(self, args):
        from evals import run as er
        self.evals.append(args)
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                rc = er.main(args) or 0
            except SystemExit as exc:
                rc = exc.code if isinstance(exc.code, int) else 1
                err.write(str(exc.code))
        return subprocess.CompletedProcess(args, rc, out.getvalue(), err.getvalue())

    def post(self, base, payload, timeout):
        request = httpx.Request("POST", f"{base}/v1/chat/completions")
        if self.server == "dead":
            return httpx.Response(404, text='{"error": "generation thread died"}', request=request)
        if self.server == "cold-timeout":
            raise httpx.ReadTimeout("timed out", request=request)
        text, tokens = self._answer(payload)
        choice = {"index": 0, "message": {"role": "assistant", "content": text},
                  "finish_reason": "stop",
                  "logprobs": {"content": tokens} if tokens else None}
        body = {"model": payload["model"], "choices": [choice],
                "usage": {"prompt_tokens": 10, "completion_tokens": 10}}
        if payload.get("stream"):
            # LiteLLM drops logprobs from a streamed reply. RULE #426.
            choice["logprobs"] = None
            return completion.Streamed(body, {"ttft_s": 0.01})
        return httpx.Response(200, json=body, request=request)

    def _answer(self, payload):
        user = payload["messages"][-1]["content"]
        if "Reply with the word ok." in user:
            return "ok", []
        if self.decide_answer is not None:
            return self.decide_answer
        for case_id, prompt in self.prompts.items():
            if prompt in user:
                good = (payload["model"] == CHALLENGER
                        or case_id in self.incumbent_passes)
                return (_reference(case_id) if good else ""), []
        return "", []


def mac():
    return _machine.Machine(frozenset({"mlx", "cpu"}),
                            Accelerator("unified", 32.0, 22.0, "Mac14,12"))


@pytest.fixture
def world(monkeypatch, tmp_path, _home):
    from evals import core
    from harness import holdout
    real_load = core.load_cases
    monkeypatch.setattr(core, "load_cases", lambda d: _hand_written(real_load(d)))
    holdout._splits.cache_clear()
    w = World()
    (Path(_home) / "discovery-sources.json").write_text(json.dumps({"sources": [
        {"name": "fake-feed", "url": FEED, "kind": "atom", "lane": "all"}]}),
        encoding="utf-8")
    config = tmp_path / "config.yaml"
    config.write_text(yaml.safe_dump({"model_list": [
        {"model_name": INCUMBENT_UPSTREAM, "litellm_params": {
            "model": f"openai/{INCUMBENT_UPSTREAM}", "api_base": f"{SERVER}/v1",
            "api_key": "not-needed"}}]}), encoding="utf-8")
    monkeypatch.setenv(gateway.ENV_VAR, str(config))
    from evals import environment
    monkeypatch.setattr(environment, "capture", lambda: {
        "hw_model": "Mac14,12", "os": "macOS-test", "arch": "arm64", "versions": {}})
    monkeypatch.setattr(ms.machines, "_THIS_MACHINE", None)
    gateway.write_served(config)
    monkeypatch.setattr(urllib.request, "urlopen", w.urlopen)
    monkeypatch.setattr(subprocess, "run", w.run)
    monkeypatch.setattr(completion, "_post", w.post)
    import huggingface_hub
    monkeypatch.setattr(huggingface_hub, "snapshot_download", w.snapshot)
    monkeypatch.setattr(ins, "SIZE_DELAY", 0.0)
    _machine.detect.cache_clear()
    monkeypatch.setattr(_machine, "detect", mac)
    monkeypatch.setattr(memory, "check_model", lambda *a, **k: (True, ""))
    switched = []
    live = {}

    # The gateway serves what the served config said when it last restarted (#522).
    def restart():
        live.clear()
        live.update({e["model_name"]: (e.get("litellm_params") or {}).get("model")
                     for e in gateway.load(gateway.served_path()).get("model_list") or []})

    def not_serving(base, want):
        return "; ".join(f"{a} serves {live.get(a)!r}" for a, m in want.items() if live.get(a) != m)
    monkeypatch.setattr(gs, "in_flight", lambda base: 0)
    monkeypatch.setattr(gs, "not_serving", not_serving)
    monkeypatch.setattr(gateway, "refresh_gateway", lambda: switched.extend(gs.switch(
        base=SERVER, restart=restart, idle_s=0, max_wait_s=0, poll_s=0)))
    w.switched = switched
    w.config = config
    yield w
    holdout._splits.cache_clear()


def latest(name):
    conn = ms.connect()
    try:
        row = conn.execute(
            "SELECT p.state AS outcome, v.tier, v.reason, v.until, v.detail FROM proposals p "
            "JOIN verdicts v ON v.id = p.state_verdict_id WHERE p.name = ?", (name,)).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


def loop(*extra):
    return cli.main(["discover", "--loop", "--run", "--lane", "code", "--top", "1",
                     "--budget-gib", "12", "--repeat", "4", *extra])


def adopted():
    conn = ms.connect()
    try:
        return {lane: row["spec"] for lane, row in adopt.current(conn).items()}
    finally:
        conn.close()


def challenger_evals(world):
    """Eval runs that named the challenger; the loop's method crossings run without it. #576."""
    return [e for e in world.evals if any(CHALLENGER in str(a) for a in e)]


def assert_a_harness_fact(row):
    """Not a terminal verdict about the candidate: a waypoint, or a reason that is not the candidate's with a way back."""
    assert row is not None
    if row["outcome"] not in ms.TERMINAL:
        return
    assert row["reason"] != "candidate", row
    assert (row["until"] or "").strip(), f"terminal with no condition that reopens it: {row}"


NETWORK_TOOLS = {"git", "gh", "curl", "uv", "launchctl", "launchd.sh"}
GIT_ONLINE = {"clone", "fetch", "pull", "push", "ls-remote"}


def test_the_loop_ends_with_the_lane_alias_serving_the_challenger(world):
    loop()
    assert world.unexpected == [], world.unexpected
    online = [a for a in world.real if Path(a[0]).name in NETWORK_TOOLS
              and (Path(a[0]).name != "git" or a[1:] and a[1] in GIT_ONLINE)]
    assert online == [], online
    assert adopted() == {"code": CHALLENGER}
    served = yaml.safe_load(gateway.served_path(world.config).read_text(encoding="utf-8"))
    alias = {e["model_name"]: e["litellm_params"] for e in served["model_list"]}
    assert alias["sohot-code"]["model"] == f"openai/{CHALLENGER}"
    assert [(r["lane"], r["new_spec"]) for r in world.switched] == [("code", CHALLENGER)]


# --- the faults that bit this project: each one a fact about the harness, never a verdict on the model


def test_a_method_crossed_with_the_incumbent_climbs_the_same_ladder(world):
    """Nobody typed `plan:<incumbent>`: the loop crossed it, screened it and measured it
    against the incumbent with the paired gate, and its verdict carries its cost. #576."""
    loop("--top", "3")
    assert adopted() == {"code": CHALLENGER}
    for spec in (f"plan:{INCUMBENT_UPSTREAM}", f"best-of:3:{INCUMBENT_UPSTREAM}"):
        conn = ms.connect()
        try:
            tiers = [r["tier"] for r in conn.execute(
                "SELECT v.tier FROM verdicts v JOIN proposals p ON p.id = v.proposal_id "
                "WHERE p.name = ? ORDER BY v.id", (spec,))]
            final = latest(spec)
        finally:
            conn.close()
        assert tiers[:2] == ["inspect", "screen"], (spec, tiers)
        assert final["tier"] == "adopt", (spec, final)
        conn = ms.connect()
        try:
            power = conn.execute(
                "SELECT v.power FROM verdicts v JOIN proposals p ON p.state_verdict_id = v.id "
                "WHERE p.name = ?", (spec,)).fetchone()["power"]
        finally:
            conn.close()
        cost = json.loads(power)["cost"]
        assert cost["base"] == INCUMBENT_UPSTREAM and cost["calls"] == (2.0 if spec[0] == "p" else 3.0)


def test_a_model_server_that_dies_mid_screen_is_the_harnesss_fault(world):
    world.server = "dead"
    assert loop() == 1
    row = latest(CHALLENGER)
    assert row["tier"] == "screen" and row["reason"] == "harness", row
    assert_a_harness_fact(row)
    assert adopted() == {}


def test_a_rate_limited_registry_at_inspect_settles_nothing(world):
    world.hf_429_after = 0
    loop()
    assert world.hf_answered == 0 and challenger_evals(world) == []
    row = latest(CHALLENGER)
    assert row is None or row["outcome"] not in ms.TERMINAL, row


def test_a_rate_limited_registry_at_fetch_settles_nothing(world):
    world.hf_429_after = 1
    loop()
    assert world.hf_answered == 1 and challenger_evals(world) == []
    row = latest(CHALLENGER)
    assert row["tier"] == "inspect" and row["outcome"] == "queued", row


def test_a_disk_below_the_floor_at_fetch_settles_nothing(world, monkeypatch):
    from harness import fetching
    monkeypatch.setattr(fetching, "free_bytes", lambda path=None: 10 * fetching.GIB)
    loop()
    assert challenger_evals(world) == []
    row = latest(CHALLENGER)
    assert row["tier"] == "fetch" and "floor" in row["detail"], row
    assert row["outcome"] not in ms.TERMINAL, row


def test_a_readme_only_snapshot_is_never_screened_or_settled(world):
    world.snapshot_files = {"README.md": "# a card", "LICENSE": "apache-2.0"}
    loop()
    assert challenger_evals(world) == []
    assert latest(CHALLENGER)["outcome"] not in ms.TERMINAL, latest(CHALLENGER)


def test_a_timeout_on_a_cold_load_is_a_limit_with_a_way_back(world):
    world.server = "cold-timeout"
    loop()
    row = latest(CHALLENGER)
    assert row["tier"] == "screen" and row["reason"] == "limit", row
    assert row["until"].startswith("limit:load_timeout_s"), row
    assert_a_harness_fact(row)


def _decide_reply(case):
    from harness.checks import decide
    gold = case.assertions["answers"]
    letters = {}
    for name, spec in case.params["schema"].items():
        codes = {v: k for k, v in decide.code_map(spec).items()}
        letters[name] = codes[decide.key(gold[name])]
    tokens = [{"token": "{", "logprob": 0.0, "top_logprobs": []}]
    for letter in letters.values():
        other = "B" if letter == "A" else "A"
        tokens.append({"token": letter, "logprob": -0.1,
                       "top_logprobs": [{"token": letter, "logprob": -0.1},
                                        {"token": other, "logprob": -2.4}]})
    return json.dumps(letters), tokens


def test_logprobs_survive_a_streaming_transport_into_the_stored_score(world):
    from evals.run import screen_cases
    world.decide_answer = _decide_reply(screen_cases(_cases("decide"))[0])
    conn = ms.connect()
    try:
        fakes.seeded_store(conn, [(CHALLENGER, "decide", 2.0, 2)])
    finally:
        conn.close()
    assert cli.main(["fetch", "--run", "--limit", "1"]) == 0
    assert cli.main(["discover", "--screen", "--run", "--lane", "decide", "--top", "1"]) == 0
    assert latest(CHALLENGER)["outcome"] == "screened"
    conn = ms.connect()
    try:
        metrics = [json.loads(r["metrics"] or "{}") for r in conn.execute(
            "SELECT metrics FROM results WHERE candidate = ?", (CHALLENGER,))]
    finally:
        conn.close()
    assert metrics and all(m.get("calibrated") == 1.0 for m in metrics), metrics
