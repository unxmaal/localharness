"""The gateway's master key: one per machine, outside the repo, and every in-repo
client sends it. #482."""
import json
import os
import stat
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from harness import gateway_key as gk

REPO = Path(__file__).resolve().parents[1]


class Memory:
    """A secret store held in a dict, so no test touches the real Keychain."""

    def __init__(self, value=""):
        self.value = value
        self.writes = 0

    def get(self) -> str:
        return self.value

    def set(self, value: str) -> None:
        self.value = value
        self.writes += 1


def test_the_environment_overrides_the_store():
    assert gk.key({"SOHOT_GATEWAY_KEY": "sk-env"}, Memory("sk-stored")) == "sk-env"


def test_the_store_answers_when_the_environment_is_silent():
    assert gk.key({}, Memory("sk-stored")) == "sk-stored"


def test_no_key_anywhere_is_empty_not_an_error():
    assert gk.key({}, Memory()) == ""


def test_ensure_generates_once_and_then_keeps_it():
    store = Memory()
    first = gk.ensure(store)
    assert first.startswith("sk-") and len(first) > 30
    assert gk.ensure(store) == first
    assert store.writes == 1


def test_rotate_replaces_the_key():
    store = Memory("sk-old")
    new = gk.rotate(store)
    assert new != "sk-old" and new.startswith("sk-")
    assert store.get() == new


def test_headers_carry_the_key_as_a_bearer_token():
    assert gk.headers({}, Memory("sk-abc")) == {"Authorization": "Bearer sk-abc"}


def test_headers_are_empty_without_a_key():
    assert gk.headers({}, Memory()) == {}


def test_the_suite_never_reaches_the_real_keychain():
    """conftest swaps the default store; a test that forgets cannot leak."""
    assert not isinstance(gk.default_store(), gk.Keychain)


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX modes")
def test_the_file_store_is_private_to_its_owner(tmp_path):
    store = gk.FileStore(tmp_path / "sub" / "gateway.key")
    store.set("sk-secret")
    assert store.get() == "sk-secret"
    mode = stat.S_IMODE((tmp_path / "sub" / "gateway.key").stat().st_mode)
    assert mode == 0o600


def test_the_file_store_is_empty_when_there_is_no_file(tmp_path):
    assert gk.FileStore(tmp_path / "gateway.key").get() == ""


def test_off_the_mac_the_key_file_lives_under_the_home(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALHARNESS_HOME", str(tmp_path))
    store = gk.platform_store(platform="linux", which=lambda name: None)
    assert isinstance(store, gk.FileStore)
    assert store.path == tmp_path.resolve() / "gateway.key"


def test_on_the_mac_the_key_is_in_the_keychain():
    store = gk.platform_store(platform="darwin", which=lambda name: "/usr/bin/security")
    assert isinstance(store, gk.Keychain)


def test_the_keychain_never_puts_the_key_on_a_command_line():
    """argv is readable by every process on the machine; stdin is not."""
    calls = []

    def run(argv, **kw):
        calls.append((argv, kw.get("input")))
        return subprocess.CompletedProcess(argv, 0, "", "")

    gk.Keychain(run=run).set("sk-very-secret")
    argv, given = calls[0]
    assert all("sk-very-secret" not in a for a in argv)
    assert "sk-very-secret" in given
    assert gk.SERVICE in given


def test_the_keychain_reads_its_service(monkeypatch):
    seen = []

    def run(argv, **kw):
        seen.append(argv)
        return subprocess.CompletedProcess(argv, 0, "sk-from-keychain\n", "")

    assert gk.Keychain(run=run).get() == "sk-from-keychain"
    assert gk.SERVICE in seen[0] and "-w" in seen[0]


def test_a_keychain_with_no_item_is_empty():
    def run(argv, **kw):
        return subprocess.CompletedProcess(argv, 44, "", "could not be found")

    assert gk.Keychain(run=run).get() == ""


def test_the_module_prints_and_creates_the_key_for_the_launcher(tmp_path):
    """serve-gateway.sh runs `python -m harness.gateway_key ensure`."""
    env = {**os.environ, "LOCALHARNESS_HOME": str(tmp_path),
           "SOHOT_GATEWAY_STORE": "file", "PYTHONUTF8": "1"}
    env.pop("SOHOT_GATEWAY_KEY", None)
    first = subprocess.run([sys.executable, "-m", "harness.gateway_key", "ensure"],
                           cwd=REPO, env=env, capture_output=True, text=True,
                           encoding="utf-8", check=True).stdout.strip()
    again = subprocess.run([sys.executable, "-m", "harness.gateway_key", "ensure"],
                           cwd=REPO, env=env, capture_output=True, text=True,
                           encoding="utf-8", check=True).stdout.strip()
    assert first.startswith("sk-") and first == again
    assert (tmp_path / "gateway.key").read_text(encoding="utf-8").strip() == first


# --- clients -----------------------------------------------------------------

class Gateway(ThreadingHTTPServer):
    """A fake gateway that demands one key, as LiteLLM with a master key does."""

    def __init__(self, want: str, refuse_schema: bool = False):
        self.want, self.seen, self.anthropic, self.bodies = want, [], [], []
        self.refuse_schema = refuse_schema
        server = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                try:
                    sent = json.loads(raw or b"{}")
                except ValueError:
                    sent = {}
                server.bodies.append(sent)
                got = self.headers.get("Authorization", "")
                server.seen.append(got)
                server.anthropic.append(self.headers.get("x-api-key", ""))
                if got != f"Bearer {server.want}":
                    body = b'{"error":{"message":"Authentication Error"}}'
                    self.send_response(401)
                elif server.refuse_schema and sent.get("response_format"):
                    body = b'{"error":{"message":"served by mlx_lm.server"}}'
                    self.send_response(400)
                elif sent.get("response_format"):
                    body = json.dumps({"choices": [{"message": {
                        "content": '{"answer": "A"}'}}]}).encode()
                    self.send_response(200)
                else:
                    body = json.dumps({"choices": [{"message": {
                        "content": "ok"}}]}).encode()
                    self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        super().__init__(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.server_address[1]}"


@pytest.fixture
def gateway():
    g = Gateway("sk-right")
    yield g
    g.shutdown()


def test_a_completion_sends_the_key(gateway, monkeypatch):
    from harness import completion
    monkeypatch.setenv("SOHOT_GATEWAY_KEY", "sk-right")
    assert completion.complete("hi", model="m", gateway=gateway.url) == "ok"
    assert gateway.seen == ["Bearer sk-right"]


def test_a_streamed_completion_sends_the_key(gateway, monkeypatch):
    from harness import completion
    monkeypatch.setenv("SOHOT_GATEWAY_KEY", "sk-right")
    completion._post(gateway.url, {"model": "m", "stream": True}, 10)
    assert gateway.seen == ["Bearer sk-right"]


def test_a_refused_key_names_the_command_that_prints_it(gateway, monkeypatch):
    from harness import completion, reasons
    monkeypatch.setenv("SOHOT_GATEWAY_KEY", "sk-wrong")
    with pytest.raises(completion.CompletionError) as e:
        completion.complete("hi", model="m", gateway=gateway.url)
    assert "soh gateway key" in str(e.value)
    # Our configuration, not the candidate's fault. RULE #270.
    assert e.value.failure_class == reasons.HARNESS_ERROR


def test_no_key_at_all_names_the_command_too(gateway):
    from harness import completion
    with pytest.raises(completion.CompletionError) as e:
        completion.complete("hi", model="m", gateway=gateway.url)
    assert "soh gateway key" in str(e.value)
    assert gateway.seen == [""]


def test_the_rubric_eval_sends_the_key(monkeypatch):
    from harness import rubric_eval
    monkeypatch.setenv("SOHOT_GATEWAY_KEY", "sk-right")
    sent = {}

    def post(url, json, timeout, headers):
        sent.update(headers)
        raise rubric_eval.httpx.ConnectError("stop here")

    rubric = type("R", (), {"instructions": "", "name": "n", "schema": {},
                            "label": "l"})()
    rubric_eval.evaluate(rubric, "text", "m", "http://gw", post=post)
    assert sent == {"Authorization": "Bearer sk-right"}


def test_no_client_in_the_repo_sends_a_made_up_key():
    """sk-local and sk-x were what clients sent when the gateway took anything."""
    roots = ["harness", "evals", "scripts", "tools"]
    hits = []
    for root in roots:
        for path in (REPO / root).rglob("*"):
            if path.suffix in (".py", ".sh") and path.is_file():
                text = path.read_text(encoding="utf-8", errors="replace")
                for bad in ("sk-local", "Bearer sk-x", "x-api-key: sk-x"):
                    if bad in text:
                        hits.append(f"{path.relative_to(REPO)}: {bad}")
    assert hits == []


# --- the CLI -----------------------------------------------------------------

def test_soh_gateway_key_prints_the_key_creating_it_if_missing(capsys, monkeypatch):
    from harness import cli
    store = Memory()
    monkeypatch.setattr(gk, "default_store", lambda: store)
    assert cli.main(["gateway", "key"]) == 0
    printed = capsys.readouterr().out.strip()
    assert printed == store.value and printed.startswith("sk-")
    assert cli.main(["gateway", "key"]) == 0
    assert capsys.readouterr().out.strip() == printed


def test_soh_gateway_key_rotate_replaces_it(capsys, monkeypatch):
    from harness import cli
    store = Memory("sk-old")
    monkeypatch.setattr(gk, "default_store", lambda: store)
    assert cli.main(["gateway", "key", "--rotate"]) == 0
    out = capsys.readouterr()
    assert out.out.strip() == store.value != "sk-old"
    # The running gateway still holds the old key until it restarts.
    assert "restart" in out.err


# --- smoke.sh ----------------------------------------------------------------

def _smoke(env_extra: dict, tmp_path):
    from tests.shells import BASH
    import shutil
    if not BASH or not shutil.which("curl"):
        pytest.skip("needs bash and curl")
    env = {**os.environ, "LOCALHARNESS_HOME": str(tmp_path),
           "SOHOT_GATEWAY_STORE": "file", "READY_TIMEOUT": "2",
           "MLX_PORT": "9", "TTS_PORT": "9", **env_extra}
    if "SOHOT_GATEWAY_KEY" not in env_extra:
        env.pop("SOHOT_GATEWAY_KEY", None)
    return subprocess.run([BASH, "scripts/smoke.sh"], cwd=REPO, env=env,
                           capture_output=True, text=True, encoding="utf-8",
                           timeout=120)


def test_smoke_sends_the_key_on_both_routes(gateway, tmp_path):
    _smoke({"SOHOT_GATEWAY_KEY": "sk-right",
            "GATEWAY_PORT": str(gateway.server_address[1])}, tmp_path)
    assert gateway.seen and set(gateway.seen) <= {"Bearer sk-right", ""}
    assert "Bearer sk-right" in gateway.seen
    assert "sk-right" in gateway.anthropic


def test_smoke_asks_the_decide_alias_for_a_schema(gateway, tmp_path):
    """A decide alias on an engine that ignores response_format must not pass smoke. #572."""
    r = _smoke({"SOHOT_GATEWAY_KEY": "sk-right",
                "GATEWAY_PORT": str(gateway.server_address[1])}, tmp_path)
    decide = [b for b in gateway.bodies if b.get("model") == "sohot-decide"]
    assert decide and all(b["response_format"]["type"] == "json_schema" for b in decide)
    assert "PASS  sohot-decide" in r.stdout


def test_smoke_fails_when_the_decide_alias_refuses_a_schema(tmp_path):
    g = Gateway("sk-right", refuse_schema=True)
    try:
        r = _smoke({"SOHOT_GATEWAY_KEY": "sk-right",
                    "GATEWAY_PORT": str(g.server_address[1])}, tmp_path)
    finally:
        g.shutdown()
    assert r.returncode != 0
    assert "FAIL  sohot-decide" in r.stdout


def test_smoke_without_a_key_says_where_to_get_one(tmp_path):
    r = _smoke({"GATEWAY_PORT": "9"}, tmp_path)
    assert r.returncode != 0
    assert "soh gateway key" in r.stdout + r.stderr


#: What LiteLLM 1.100.0 sent before #501: 500 to no key, 400 to a key it took for a virtual one.
OLD_REFUSALS = [(500, '{"error":{"message":"Internal server error"}}', ""),
                (400, '{"error":{"message":"No connected db.","type":"no_db_connection"}}', "sk-x"),
                (401, '{"error":{"message":"Authentication Error"}}', "sk-x"),
                (403, '{"error":{"message":"forbidden"}}', "sk-x")]


@pytest.mark.parametrize("status,body,sent", OLD_REFUSALS,
                         ids=["500-no-key", "400-no-db", "401", "403"])
def test_every_shape_of_refusal_is_a_refusal(status, body, sent):
    assert gk.refused(status, body, sent)


@pytest.mark.parametrize("status,body,sent", [
    (500, '{"error":{"message":"upstream crashed"}}', "sk-x"),
    (400, '{"error":{"message":"bad request"}}', "sk-x"),
    (404, "not found", ""), (200, "", "")], ids=["500-with-key", "400", "404", "200"])
def test_other_errors_are_not_refusals(status, body, sent):
    assert not gk.refused(status, body, sent)


class Refusing(ThreadingHTTPServer):
    """A gateway that refuses every request one fixed way."""

    def __init__(self, status: int, body: str):
        payload = body.encode()

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

        super().__init__(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.server_address[1]}"


@pytest.mark.parametrize("status,body,sent", OLD_REFUSALS[:2], ids=["500-no-key", "400-no-db"])
def test_a_completion_names_the_command_whatever_status_refuses_it(monkeypatch, status, body,
                                                                   sent):
    from harness import completion, reasons
    if sent:
        monkeypatch.setenv("SOHOT_GATEWAY_KEY", sent)
    g = Refusing(status, body)
    try:
        with pytest.raises(completion.CompletionError) as e:
            completion.complete("hi", model="m", gateway=g.url)
    finally:
        g.shutdown()
    assert "soh gateway key" in str(e.value)
    assert e.value.failure_class == reasons.HARNESS_ERROR


def test_a_refused_rubric_eval_names_the_command(monkeypatch):
    from harness import rubric_eval
    monkeypatch.setenv("SOHOT_GATEWAY_KEY", "sk-wrong")
    g = Refusing(*OLD_REFUSALS[1][:2])
    rubric = type("R", (), {"instructions": "", "name": "n", "schema": {}, "label": "l"})()
    try:
        v = rubric_eval.evaluate(rubric, "text", "m", g.url)
    finally:
        g.shutdown()
    assert not v.ok and "soh gateway key" in v.error


@pytest.mark.parametrize("path", ["gateway/config.yaml", "gateway/config.cuda.yaml"])
def test_both_configs_answer_a_bad_key_with_401(path):
    """Through the hook beside the config, never by letting the request through."""
    import yaml
    body = yaml.safe_load((REPO / path).read_text(encoding="utf-8"))
    hook = body["general_settings"]["custom_auth"]
    assert (REPO / "gateway" / (hook.rsplit(".", 1)[0] + ".py")).exists()
    assert "allow_requests_on_db_unavailable" not in str(body)


def test_smoke_asks_by_model_id_never_by_a_deprecated_nickname(gateway, tmp_path):
    """#670: the old nicknames go next release, and smoke must not be what breaks then."""
    from harness import models
    _smoke({"SOHOT_GATEWAY_KEY": "sk-right",
            "GATEWAY_PORT": str(gateway.server_address[1])}, tmp_path)
    sent = {b.get("model") for b in gateway.bodies}
    assert sent and not sent & set(models.deprecated()), sent
