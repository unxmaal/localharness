"""local_complete and local_decide: a text subtask on the lane's adopted model. #475."""
import json
import math
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from harness import completion, context, delegate, exclusive, serving

ANSWER_AT = 0.3
SERVED = "upstream/Real-Model-4bit"


class Fake(ThreadingHTTPServer):
    """An OpenAI-compatible chat server that streams its answer after ANSWER_AT."""
    daemon_threads = True
    reply = "def add(a, b):\n    return a + b\n"
    logprobs = None
    stall_s = 0.0
    #: What LiteLLM in front of mlx_lm.server did, measured 2026-10-06.
    drop_streamed_logprobs = True
    #: A thinking model that spends the whole budget before answering. #590.
    reasoning_only = False

    def __init__(self):
        super().__init__(("127.0.0.1", 0), Handler)
        self.seen = []

    @property
    def url(self):
        return f"http://127.0.0.1:{self.server_address[1]}"


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def do_POST(self):
        s = self.server
        s.seen.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
        if s.stall_s:
            time.sleep(s.stall_s)
        if not s.seen[-1].get("stream"):
            choice = {"message": {"role": "assistant", "content": s.reply},
                      "finish_reason": "stop"}
            if s.logprobs:
                choice["logprobs"] = {"content": s.logprobs}
            data = json.dumps({"model": SERVED, "choices": [choice],
                               "usage": {"prompt_tokens": 9, "completion_tokens": 7}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
            return
        if s.drop_streamed_logprobs:
            s.logprobs = None
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(ANSWER_AT)
        choice = {"index": 0, "delta": {"content": s.reply}}
        usage = {"prompt_tokens": 9, "completion_tokens": 7}
        if s.reasoning_only:
            choice = {"index": 0, "delta": {"reasoning_content": "Let me think. " * 50}}
            usage = {"prompt_tokens": 9, "completion_tokens": s.seen[-1]["max_tokens"],
                     "completion_tokens_details": {
                         "reasoning_tokens": s.seen[-1]["max_tokens"]}}
        if s.logprobs:
            choice["logprobs"] = {"content": s.logprobs}
        for chunk in ({"model": SERVED, "choices": [choice]},
                      {"model": SERVED, "choices": [{"index": 0, "delta": {},
                                                     "finish_reason": "stop"}]},
                      {"choices": [], "usage": usage}):
            self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
            self.wfile.flush()
        self.wfile.write(b"data: [DONE]\n\n")


@pytest.fixture
def fake(monkeypatch):
    srv = Fake()
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    routed = []

    def route(lane, chosen=None, via=""):
        routed.append(lane)
        return f"sohot-{lane}", serving.Route(srv.url, f"sohot-{lane}", {})
    monkeypatch.setattr(delegate, "route", route)
    monkeypatch.setattr(context, "served_ctx", lambda spec: None)
    srv.routed = routed
    yield srv
    srv.shutdown()
    srv.server_close()


def test_a_completion_carries_text_the_model_that_answered_and_its_timing(fake):
    got = delegate.complete("code", "write add")
    assert got["text"] == Fake.reply
    assert got["model"] == SERVED and got["spec"] == "sohot-code"
    assert ANSWER_AT <= got["ttft_s"] <= got["seconds"] < ANSWER_AT + 5
    assert (got["prompt_tokens"], got["completion_tokens"]) == (9, 7)
    sent = fake.seen[0]
    assert sent["stream"] is True and sent["model"] == "sohot-code"
    assert sent["messages"][0]["content"] == completion.SYSTEM["code"]


def test_the_callers_system_prompt_and_knobs_reach_the_server(fake):
    delegate.complete("extract", "summarise", system="Be terse.",
                      max_tokens=64, temperature=0.7)
    sent = fake.seen[0]
    assert sent["messages"][0] == {"role": "system", "content": "Be terse."}
    assert sent["max_tokens"] == 64 and sent["temperature"] == 0.7


def test_the_lane_resolves_through_the_route_the_lane_commands_use(monkeypatch):
    """No stub between the tool and serving.route: the adopted spec is what is sent."""
    monkeypatch.setattr(delegate, "lane_model", lambda lane, chosen=None: "q3-coder")
    spec, where = delegate.route("code")
    assert spec == "q3-coder" and where == serving.route("q3-coder")
    from harness import cli
    assert cli._text_route(type("A", (), {})(), "code") == (spec, where)


@pytest.mark.parametrize("lane", ["image", "video", "say", "", "../code"])
def test_only_text_lanes_are_served(fake, lane):
    with pytest.raises(delegate.Refused, match="not a text lane"):
        delegate.complete(lane, "x")
    assert fake.seen == [] and fake.routed == []


def test_every_text_lane_is_allowed(fake):
    for lane in ("code", "web", "svg", "extract", "decide"):
        delegate.complete(lane, "x")
    assert fake.routed == ["code", "web", "svg", "extract", "decide"]


def test_a_prompt_over_the_cap_is_refused_before_anything_is_sent(fake, monkeypatch):
    monkeypatch.setattr(delegate, "MAX_PROMPT_CHARS", 100)
    with pytest.raises(delegate.Refused, match="over the 100"):
        delegate.complete("code", "x" * 60, system="y" * 60)
    with pytest.raises(delegate.Refused, match="max_tokens"):
        delegate.complete("code", "x", max_tokens=delegate.MAX_TOKENS + 1)
    assert fake.seen == []


def test_thinking_false_turns_reasoning_off_through_the_chat_template(fake):
    delegate.complete("code", "x", thinking=False)
    assert fake.seen[-1]["chat_template_kwargs"] == {"enable_thinking": False}


def test_thinking_left_unsaid_serves_the_lane_as_it_was_measured(fake):
    delegate.complete("code", "x")
    assert "chat_template_kwargs" not in fake.seen[-1]
    delegate.complete("decide", "x")
    assert fake.seen[-1]["chat_template_kwargs"] == completion.TEMPLATE["decide"]
    delegate.complete("decide", "x", thinking=True)
    assert fake.seen[-1]["chat_template_kwargs"] == {"enable_thinking": True}


def test_max_tokens_may_rise_to_what_the_served_context_leaves(fake, monkeypatch):
    monkeypatch.setattr(context, "served_ctx", lambda spec: 262144)
    delegate.complete("code", "x", max_tokens=20000)
    assert fake.seen[-1]["max_tokens"] == 20000
    with pytest.raises(delegate.Refused, match="262144"):
        delegate.complete("code", "x" * 3000, max_tokens=262144 - 500)
    assert len(fake.seen) == 1


def test_an_unknown_served_context_keeps_the_fixed_cap(fake):
    with pytest.raises(delegate.Refused, match=str(delegate.MAX_TOKENS)):
        delegate.complete("code", "x", max_tokens=delegate.MAX_TOKENS + 1)
    assert fake.seen == []


def test_a_budget_spent_on_reasoning_says_how_much_and_offers_thinking_off(fake):
    fake.reasoning_only = True
    with pytest.raises(completion.CompletionError) as e:
        delegate.complete("code", "x", max_tokens=300)
    msg = str(e.value)
    assert "300 tokens" in msg and "reasoning" in msg and "thinking=false" in msg
    assert e.value.failure_class == "token_budget_exhausted"


def test_a_stalled_server_times_out_rather_than_hanging(fake):
    fake.stall_s = 3.0
    started = time.perf_counter()
    with pytest.raises(completion.CompletionError, match="timed out"):
        delegate.complete("code", "x", timeout=0.5)
    assert time.perf_counter() - started < 2.5


def test_a_held_machine_lock_refuses_at_once_and_names_the_holder(fake):
    """A batch run holds the lock for hours; a delegated call must not wait behind it."""
    import subprocess
    import sys
    code = ("from harness import exclusive; import time\n"
            "with exclusive.held('eval'):\n print('held', flush=True); time.sleep(30)\n")
    child = subprocess.Popen([sys.executable, "-c", code], stdout=subprocess.PIPE,
                             text=True, env=dict(os.environ))
    try:
        assert child.stdout.readline().strip() == "held"
        started = time.perf_counter()
        with pytest.raises(delegate.Refused, match="busy with eval"):
            delegate.complete("code", "x")
        # Refused while the holder still holds; a wall-clock bound under 1s flaked on Windows CI (#623).
        assert child.poll() is None and time.perf_counter() - started < 15
        assert fake.seen == []
    finally:
        child.kill()
        child.wait()
    assert delegate.busy() == ""


SCHEMA = {"kind": {"type": "enum", "choices": ["bug", "feature", "question"],
                   "description": "what the issue is"},
          "urgent": {"type": "boolean", "description": "needs action today"}}


def test_decide_returns_answers_and_probabilities_from_the_logprobs(fake):
    fake.reply = '{"kind": "A", "urgent": "B"}'
    fake.logprobs = [
        {"token": "{", "logprob": 0.0, "top_logprobs": []},
        {"token": "A", "logprob": math.log(0.8),
         "top_logprobs": [{"token": "A", "logprob": math.log(0.8)},
                          {"token": "C", "logprob": math.log(0.2)}]},
        {"token": "B", "logprob": math.log(0.9),
         "top_logprobs": [{"token": "B", "logprob": math.log(0.9)},
                          {"token": "A", "logprob": math.log(0.1)}]}]
    got = delegate.decide("triage this", SCHEMA, context="it crashes on start")
    assert got["answers"] == {"kind": "bug", "urgent": "true"}
    assert got["probabilities"]["kind"] == pytest.approx(
        {"bug": 0.8, "feature": 0.0, "question": 0.2})
    assert got["probabilities"]["urgent"] == pytest.approx({"false": 0.1, "true": 0.9})
    assert got["model"] == SERVED and got["seconds"] > 0
    assert fake.routed == ["decide"]
    sent = fake.seen[0]
    assert sent["logprobs"] is True and sent["temperature"] == 0.0
    assert "it crashes on start" in sent["messages"][-1]["content"]


@pytest.mark.parametrize("schema,match", [
    ({}, "must map"), ({"kind": {"type": "enum", "choices": ["a"],
                                  "description": "d"}}, "2-26 distinct"),
    ({"kind": {"type": "boolean"}}, "needs a description"), ({"kind": "bug"}, "object")])
def test_a_bad_schema_is_refused_before_anything_is_sent(fake, schema, match):
    with pytest.raises(ValueError, match=match):
        delegate.decide("q", schema)
    assert fake.seen == []


def test_an_unusable_decision_is_an_error_not_an_empty_answer(fake):
    fake.reply = "I think it is a bug."
    with pytest.raises(ValueError, match="no usable answer"):
        delegate.decide("triage", SCHEMA)


# ---- the MCP surface ---------------------------------------------------------

@pytest.fixture
def mcp_server():
    return pytest.importorskip(
        "harness.mcp_server", reason="needs the `mcp` group: uv run --group mcp pytest")


def _call(server, name, **args):
    import anyio
    return anyio.run(lambda: server.SERVER.call_tool(name, args))


def test_both_tools_are_registered_with_their_signatures(mcp_server):
    tools = {t.name: t for t in mcp_server.SERVER._tool_manager.list_tools()}
    props = tools["local_complete"].parameters["properties"]
    assert set(props) == {"lane", "prompt", "system", "max_tokens", "temperature",
                          "thinking"}
    assert props["thinking"]["default"] is None
    assert props["lane"]["default"] == "code" and props["max_tokens"]["default"] == 2048
    assert set(tools["local_decide"].parameters["properties"]) == {
        "question", "schema", "context"}


def test_the_tool_passes_thinking_and_a_large_budget_through(fake, mcp_server, monkeypatch):
    monkeypatch.setattr(context, "served_ctx", lambda spec: 262144)
    mcp_server.local_complete(prompt="x", thinking=False, max_tokens=20000)
    assert fake.seen[-1]["chat_template_kwargs"] == {"enable_thinking": False}
    assert fake.seen[-1]["max_tokens"] == 20000


def test_the_tool_returns_fields_not_json_in_a_string(fake, mcp_server):
    got = mcp_server.local_complete(prompt="write add", lane="code")
    assert got.text == Fake.reply and got.model == SERVED
    assert got.ttft_s >= ANSWER_AT and got.seconds >= got.ttft_s


def test_a_refusal_reaches_the_caller_as_an_mcp_error_with_its_reason(fake, mcp_server):
    from mcp.server.mcpserver.exceptions import ToolError, UnexpectedToolError
    with pytest.raises(ToolError, match="not a text lane") as got:
        _call(mcp_server, "local_complete", prompt="x", lane="image")
    assert not isinstance(got.value, UnexpectedToolError)


def test_decide_over_mcp_has_the_soh_decide_shape(fake, mcp_server):
    fake.reply = '{"kind": "B", "urgent": "A"}'
    got = mcp_server.local_decide(question="triage", schema=SCHEMA)
    assert got.answers == {"kind": "feature", "urgent": "false"}
    assert got.probabilities == {"kind": {"feature": 1.0}, "urgent": {"false": 1.0}}


def test_a_failed_soh_reaches_the_caller_with_its_message(monkeypatch, mcp_server):
    """#477: a RuntimeError reached the client as only 'Error executing tool svg'."""
    import subprocess
    from mcp.server.mcpserver.exceptions import UnexpectedToolError
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: subprocess.CompletedProcess(
        a, 1, "", "gateway unreachable at the gateway"))
    with pytest.raises(Exception, match="gateway unreachable") as got:
        _call(mcp_server, "svg", prompt="a gear")
    assert not isinstance(got.value, UnexpectedToolError)


def test_the_lock_probe_never_takes_the_lock_away_from_a_waiter():
    """busy() takes and drops the lock; a holder afterwards still gets it."""
    assert delegate.busy() == ""
    with exclusive.held("eval") as waited:
        assert waited is False


def test_mcp_artifacts_follow_the_home_in_force_when_called(mcp_server, tmp_path, monkeypatch):
    """OUTDIR was fixed at import, so a test importing it at collection wrote to the real home."""
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path / "elsewhere"))
    assert str(mcp_server._out("svg", ".svg")).startswith(str(tmp_path / "elsewhere"))


def test_a_reply_that_echoes_an_alias_is_reported_by_the_real_id(fake, monkeypatch):
    """An old gateway echoes the alias in `model`; the caller still learns what answered. #670."""
    import sys
    from harness import gateway
    monkeypatch.setenv("GATEWAY_CONFIG", str(gateway.REPO / "gateway" / "config.yaml"))
    monkeypatch.setattr(sys.modules[__name__], "SERVED", "eval-7b")
    assert delegate.complete("code", "write add")["model"] == "Qwen2.5-7B-Instruct-Q4_K_M"
