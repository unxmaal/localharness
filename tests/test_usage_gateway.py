"""#481 end to end: a real LiteLLM proxy loads gateway/usage_log.py and its rows reach the store.

Needs litellm, which the suite does not install: `make test-litellm`.
"""
import json
import os
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

pytest.importorskip("litellm", reason="needs litellm: make test-litellm")
httpx = pytest.importorskip("httpx")

from harness import memory_store as ms  # noqa: E402
from harness import paths  # noqa: E402

pytestmark = pytest.mark.litellm
REPO = Path(__file__).resolve().parents[1]
SECRET = "SECRET-PROMPT-TEXT"
KEY = "sk-test-usage"
CALL = {"id": "c1", "type": "function", "function": {"name": "read", "arguments": '{"path": "a"}'}}


class Upstream(BaseHTTPRequestHandler):
    """An OpenAI-compatible server that answers at once: a tool call, a stream, or a 500."""

    def log_message(self, *a):
        pass

    def _json(self, code, body):
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        if body["model"] == "boom":
            return self._json(500, {"error": {"message": "upstream fell over"}})
        if body.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for delta, finish in (({"role": "assistant", "content": "hel"}, None),
                                  ({"content": "lo"}, "stop")):
                chunk = {"id": "x", "object": "chat.completion.chunk", "created": 1,
                         "model": "up", "choices": [{"index": 0, "delta": delta,
                                                     "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(chunk)}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"data: [DONE]\n\n")
            return
        self._json(200, {"id": "x", "object": "chat.completion", "created": 1, "model": "up",
                         "choices": [{"index": 0, "finish_reason": "tool_calls",
                                      "message": {"role": "assistant", "content": None,
                                                  "tool_calls": [CALL]}}],
                         "usage": {"prompt_tokens": 9, "completion_tokens": 4,
                                   "total_tokens": 13}})


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


@pytest.fixture
def gateway(tmp_path):
    up = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    threading.Thread(target=up.serve_forever, daemon=True).start()
    base_up = f"http://127.0.0.1:{up.server_address[1]}/v1"
    # The served config sits beside gateway/; here shims beside the tmp config import its modules.
    for mod, name in (("usage_log", "proxy_handler_instance"), ("key_auth", "user_api_key_auth"),
                      ("served_model", "proxy_handler_instance")):
        (tmp_path / f"{mod}.py").write_text(
            f"import sys\nsys.path.insert(0, {str(REPO)!r})\n"
            f"from gateway.{mod} import {name}  # noqa: F401\n", encoding="utf-8")
    cfg = tmp_path / "config.yaml"
    cfg.write_text(f"""model_list:
  - model_name: sohot-code
    litellm_params: {{model: openai/up, api_base: "{base_up}", api_key: not-needed}}
  - model_name: boom
    litellm_params: {{model: openai/boom, api_base: "{base_up}", api_key: not-needed}}
litellm_settings:
  callbacks: ["usage_log.proxy_handler_instance", "served_model.proxy_handler_instance"]
  num_retries: 0
  use_chat_completions_url_for_anthropic_messages: true
general_settings:
  master_key: os.environ/LITELLM_MASTER_KEY
  custom_auth: key_auth.user_api_key_auth
""", encoding="utf-8")
    port = _free_port()
    log = open(tmp_path / "proxy.log", "w", encoding="utf-8")
    exe = Path(sys.executable).parent / "litellm"
    proc = subprocess.Popen(
        [str(exe), "--config", str(cfg), "--host", "127.0.0.1", "--port", str(port)],
        env={**os.environ, "LITELLM_MASTER_KEY": KEY, "GATEWAY_CONFIG": str(cfg),
             "LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES": "1"},
        stdout=log, stderr=subprocess.STDOUT)
    base = f"http://127.0.0.1:{port}"
    deadline = time.time() + 90
    while time.time() < deadline:
        try:
            if httpx.get(f"{base}/v1/models", headers={"Authorization": f"Bearer {KEY}"},
                         timeout=2).status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        proc.terminate()
        pytest.fail((tmp_path / "proxy.log").read_text(encoding="utf-8"))
    yield base
    proc.terminate()
    proc.wait(10)
    up.shutdown()


def _wait_rows(n, timeout=20.0):
    deadline = time.time() + timeout
    while True:
        conn = ms.connect()
        try:
            rows = [dict(r) for r in conn.execute("SELECT * FROM gateway_requests ORDER BY id")]
        finally:
            conn.close()
        if len(rows) >= n or time.time() > deadline:
            return rows
        time.sleep(0.25)


def test_real_requests_through_a_real_proxy_become_rows_without_text(gateway):
    h = {"Authorization": f"Bearer {KEY}", "User-Agent": "opencode/9.9 test"}
    msgs = [{"role": "user", "content": SECRET}]
    tools = [{"type": "function", "function": {"name": "read", "parameters": {"type": "object"}}}]
    assert httpx.post(f"{gateway}/v1/chat/completions", timeout=30, json={
        "model": "sohot-code", "messages": msgs}).status_code == 401
    assert httpx.post(f"{gateway}/v1/chat/completions", headers=h, timeout=30, json={
        "model": "sohot-code", "messages": msgs, "tools": tools}).status_code == 200
    with httpx.stream("POST", f"{gateway}/v1/chat/completions", headers=h, timeout=30,
                      json={"model": "sohot-code", "messages": msgs, "stream": True}) as r:
        for _ in r.iter_lines():
            pass
    assert httpx.post(f"{gateway}/v1/chat/completions", headers=h, timeout=30, json={
        "model": "boom", "messages": msgs}).status_code >= 500
    assert httpx.post(f"{gateway}/v1/messages", timeout=30, headers={
        "x-api-key": KEY, "anthropic-version": "2023-06-01", "User-Agent": "claude-cli/2.0"},
        json={"model": "sohot-code", "max_tokens": 20, "messages": msgs,
              "tools": [{"name": "read", "input_schema": {"type": "object"}}]}
    ).status_code == 200

    rows = _wait_rows(4)
    by = {(r["alias"], r["stream"], r["call_type"]): r for r in rows}
    plain = by[("sohot-code", 0, "acompletion")]
    assert plain["lane"] == "code" and plain["client"] == "opencode/9.9"
    assert (plain["prompt_tokens"], plain["completion_tokens"]) == (9, 4)
    assert (plain["tool_calls"], plain["tool_calls_valid"]) == (1, 1)
    assert plain["ttft_s"] is None and plain["total_s"] > 0
    streamed = by[("sohot-code", 1, "acompletion")]
    assert streamed["ttft_s"] is not None and 0 < streamed["ttft_s"] <= streamed["total_s"]
    failed = by[("boom", 0, "acompletion")]
    assert failed["error_class"] and failed["error_code"] == "500"
    anth = by[("sohot-code", 0, "anthropic_messages")]
    assert anth["client"] == "claude-cli/2.0" and anth["tool_calls_valid"] == 1
    for f in paths.home().glob("discovery.db*"):
        assert SECRET.encode() not in f.read_bytes(), f.name


def test_the_reply_names_the_model_that_answered_not_the_alias(gateway):
    """#670 inside a real LiteLLM proxy: two headers carry the upstream id. LiteLLM 1.100.0
    restamps the body's model to the name the client asked for after every hook."""
    h = {"Authorization": f"Bearer {KEY}"}
    msgs = [{"role": "user", "content": "hi"}]
    r = httpx.post(f"{gateway}/v1/chat/completions", headers=h, timeout=30,
                   json={"model": "sohot-code", "messages": msgs})
    assert r.status_code == 200
    assert r.json()["model"] == "sohot-code"
    assert r.headers.get("x-sohot-model") == "up"
    assert r.headers.get("x-sohot-served-as") == "sohot-code"
    with httpx.stream("POST", f"{gateway}/v1/chat/completions", headers=h, timeout=30,
                      json={"model": "sohot-code", "messages": msgs, "stream": True}) as s:
        assert s.headers.get("x-sohot-model") == "up"
        for _ in s.iter_lines():
            pass
