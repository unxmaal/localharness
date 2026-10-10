"""The structured-output text server on Apple Silicon. Issue #286.

mlx_lm.server ignores `response_format`, so a client that needs schema-valid
JSON is routed to llama-server, which enforces it (RULE #326).
"""
import re
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
SERVE = REPO / "scripts" / "serve-eval.sh"
LAUNCHD = REPO / "scripts" / "launchd.sh"
MAC = REPO / "gateway" / "config.yaml"
FETCH = REPO / "scripts" / "fetch-gguf.sh"


def _entries():
    body = yaml.safe_load(MAC.read_text(encoding="utf-8"))
    return [(e["model_name"], e["litellm_params"]) for e in body["model_list"]]


def _own_entries():
    """Entries that name a model of their own; a deprecated alias shares its target's params."""
    body = yaml.safe_load(MAC.read_text(encoding="utf-8"))
    return [e for e in body["model_list"] if not e.get("deprecated_for")]


def _default_port() -> str:
    m = re.search(r'LLAMACPP_PORT="\$\{LLAMACPP_PORT:-(\d+)\}"',
                  SERVE.read_text(encoding="utf-8"))
    assert m, "serve-eval.sh must default LLAMACPP_PORT"
    return m.group(1)


def test_the_eval_server_is_a_launchd_service():
    services = re.search(r'^SERVICES="([^"]+)"',
                         LAUNCHD.read_text(encoding="utf-8"), re.M).group(1)
    assert "eval" in services.split()


def test_the_eval_server_is_llama_server_and_not_a_second_copy_of_it():
    text = SERVE.read_text(encoding="utf-8")
    assert "exec scripts/serve-llamacpp.sh" in text


def test_the_eval_port_is_not_the_mlx_port():
    assert _default_port() != "8081", "mlx_lm.server holds 8081 on this machine"


def test_every_gguf_entry_reaches_the_eval_server():
    """Gauntlet #2: the port is written in the script and in the config, so
    the two copies are read and compared rather than trusted. A GGUF entry is
    one with a source_file; its name says nothing about where it is served. #670."""
    port = _default_port()
    ggufs = [e for e in _own_entries() if str(e.get("source_file", "")).endswith(".gguf")]
    assert ggufs, "no GGUF entry in gateway/config.yaml"
    for e in ggufs:
        assert e["litellm_params"]["api_base"] == f"http://127.0.0.1:{port}/v1", e["model_name"]


def test_only_gguf_entries_reach_the_eval_server():
    """An MLX entry on this port would silently lose the MLX engine; the router serves a
    file by its stem, so the upstream id must be the GGUF's stem."""
    port = _default_port()
    for e in _own_entries():
        params = e["litellm_params"]
        if f":{port}/" not in str(params.get("api_base", "")):
            continue
        source = str(e.get("source_file", ""))
        assert source.endswith(".gguf"), e["model_name"]
        assert params["model"].removeprefix("openai/") == Path(source).stem, e["model_name"]


def test_the_eval_server_unloads_when_idle():
    """Two resident servers on a 32 GB machine is the RULE #193 crash shape."""
    text = (REPO / "scripts" / "serve-llamacpp.sh").read_text(encoding="utf-8")
    assert "--sleep-idle-seconds" in text
    assert "LLAMACPP_SLEEP_IDLE=-1" not in SERVE.read_text(encoding="utf-8")


def test_weights_are_fetched_into_the_flat_directory_the_router_reads(
        monkeypatch):
    """The script goes through harness.gguf, which records the row. #411."""
    from harness import gguf
    text = FETCH.read_text(encoding="utf-8")
    assert "python -m harness.gguf" in text
    monkeypatch.setenv("HF_HOME", "/hf")
    monkeypatch.delenv("LLAMACPP_MODELS_DIR", raising=False)
    assert gguf.models_dir() == Path("/hf") / "gguf"


def test_the_context_size_is_always_explicit():
    """With none given, llama-server sized a 4B model's cache to 151,808
    tokens per slot, about 21 GB, and nearly wedged a 32 GB machine."""
    text = (REPO / "scripts" / "serve-llamacpp.sh").read_text(encoding="utf-8")
    m = re.search(r'--ctx-size "\$\{LLAMACPP_CTX:-(\d+)\}"', text)
    assert m, "serve-llamacpp.sh must pass --ctx-size"
    assert int(m.group(1)) <= 32768


def test_the_fetch_script_is_allowed_online():
    """env.sh sets HF_HUB_OFFLINE=1 for everything else; the one script whose
    job is to download inherited it and could fetch nothing. #319."""
    text = FETCH.read_text(encoding="utf-8")
    assert "HF_HUB_OFFLINE=0" in text


def test_a_fetched_gguf_restarts_the_router_so_it_can_be_served():
    """llama-server's router reads --models-dir at startup only (build 10809):
    a file fetched later answered `model ... not found`. #319."""
    from harness import gguf
    assert "python -m harness.gguf" in FETCH.read_text(encoding="utf-8")
    import inspect
    assert '"restart", "eval"' in (REPO / "harness" / "gguf.py").read_text(
        encoding="utf-8")
    assert "refresh_router" in inspect.getsource(gguf.download)


def test_restart_runs_under_the_machine_lock():
    text = LAUNCHD.read_text(encoding="utf-8")
    assert "restart)" in text
    block = text[text.index("restart)"):]
    assert "with-gpu-lock" in block.split(";;")[0]
