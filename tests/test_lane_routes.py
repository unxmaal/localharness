"""#297: lane commands serve what the lane adopted, on the server that serves it."""
import json

import httpx
import pytest
import respx
import yaml

from harness.commands import lanes as lanes_cmd
from harness import adopt, cli, gateway, router, serving, winners
from harness.proc import Outcome
from harness import memory_store as ms

GW = "http://127.0.0.1:4000"
MLX = "http://127.0.0.1:8081"
STEM = "Ornith-1.5-35B-Q4_K_M"

CONFIG = {"model_list": [
    {"model_name": "mlx-community/Qwen3-4B-Instruct-2507-4bit", "litellm_params": {
        "model": "openai/mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "api_base": f"{MLX}/v1", "api_key": "not-needed"}},
    {"model_name": "q3-4b", "litellm_params": {
        "model": "openai/mlx-community/Qwen3-4B-Instruct-2507-4bit",
        "api_base": f"{MLX}/v1", "api_key": "not-needed"}},
    {"model_name": "local-large", "litellm_params": {
        "model": "openai/mlx-community/Qwen2.5-7B-Instruct-4bit",
        "api_base": f"{MLX}/v1", "api_key": "not-needed"}},
]}


@pytest.fixture(autouse=True)
def config(tmp_path, monkeypatch):
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(CONFIG), encoding="utf-8")
    monkeypatch.setenv(gateway.ENV_VAR, str(path))
    monkeypatch.delenv("LLAMACPP_PORT", raising=False)
    return path


def adopt_for(lane: str, spec: str) -> None:
    conn = ms.connect()
    try:
        adopt.record(conn, adopt.Verdict(lane, "q3-4b", spec, True, "won",
                                         adopt.BY_HAND))
    finally:
        conn.close()


def answered(text="def f(): pass"):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


# ---- the resolver ------------------------------------------------------------

def test_a_llamacpp_spec_goes_to_the_router_by_its_stem():
    got = serving.route(f"llamacpp:{STEM}")
    assert (got.base, got.model) == (router.url(), STEM)


def test_the_router_port_follows_llamacpp_port(monkeypatch):
    monkeypatch.setenv("LLAMACPP_PORT", "8099")
    assert serving.route(f"llamacpp:{STEM}").base == "http://127.0.0.1:8099"


def test_a_gateway_alias_stays_on_the_gateway():
    got = serving.route("q3-4b")
    assert (got.base, got.model) == (GW, "q3-4b")


def test_an_mlx_repo_id_goes_to_the_upstream_that_hot_swaps():
    got = serving.route("mlx-community/Qwen3-8B-4bit")
    assert (got.base, got.model) == (MLX, "mlx-community/Qwen3-8B-4bit")


def test_an_explicit_gateway_is_used_verbatim():
    got = serving.route(STEM, "http://127.0.0.1:8082/")
    assert (got.base, got.model) == ("http://127.0.0.1:8082", STEM)


def test_sampling_options_travel_with_the_route():
    assert serving.route("q3-4b,temperature=0").sampling == {"temperature": 0.0}
    with pytest.raises(ValueError):
        serving.route("q3-4b,colour=red")


def test_a_non_text_engine_spec_is_refused():
    with pytest.raises(ValueError):
        serving.route("nimble:org/adapter")


def test_the_eval_runner_uses_the_same_resolver():
    from evals import run
    r = run.build_runner("mlx-community/Qwen3-8B-4bit", GW, None)
    assert (r.gateway, r.model) == (GW, "mlx-community/Qwen3-8B-4bit")
    r = run.build_runner(f"llamacpp:{STEM}", GW, None)
    assert (r.gateway, r.model) == (router.url(), STEM)


# ---- lane commands ------------------------------------------------------------

@respx.mock
def test_code_uses_the_typed_constant_with_no_adoption(capsys):
    hit = respx.post(f"{GW}/v1/chat/completions").mock(return_value=answered())
    assert cli.main(["code", "x"]) == 0
    assert json.loads(hit.calls.last.request.content)["model"] == cli.DEFAULT_CODE_MODEL


@respx.mock
def test_code_follows_an_adopted_llamacpp_winner_to_the_router(capsys):
    adopt_for("code", f"llamacpp:{STEM}")
    hit = respx.post(f"{router.url()}/v1/chat/completions").mock(return_value=answered())
    assert cli.main(["code", "x"]) == 0
    assert json.loads(hit.calls.last.request.content)["model"] == STEM


@respx.mock
def test_minus_m_still_overrides_the_adoption(capsys):
    adopt_for("code", f"llamacpp:{STEM}")
    hit = respx.post(f"{GW}/v1/chat/completions").mock(return_value=answered())
    assert cli.main(["code", "x", "-m", "local-large"]) == 0
    assert json.loads(hit.calls.last.request.content)["model"] == "local-large"


@pytest.mark.parametrize("lane,spec", [
    ("web", "mlx-community/Qwen3-8B-4bit"), ("svg", "mlx-community/Qwen3-8B-4bit"),
    ("extract", "mlx-community/Qwen3-8B-4bit"),
    # decide adopts only what llama-server serves. #572.
    ("decide", f"llamacpp:{STEM}")])
def test_every_text_lane_resolves_its_adoption_at_call_time(lane, spec):
    assert cli.lane_model(lane) == winners.typed()[lane]
    adopt_for(lane, spec)
    assert cli.lane_model(lane) == spec


def test_image_and_video_resolve_their_adoption_at_call_time(monkeypatch):
    built = []
    monkeypatch.setattr(lanes_cmd, "_generate", lambda spec, *a: built.append(spec) or 0)
    assert cli.main(["image", "fox"]) == 0
    assert cli.main(["video", "fox"]) == 0
    assert built == [cli.DEFAULT_IMAGE_ENGINE, cli.DEFAULT_VIDEO_ENGINE]
    adopt_for("image", "mflux:z-image-turbo")
    assert cli.main(["image", "fox"]) == 0
    assert built[-1] == "mflux:z-image-turbo"


def test_hear_and_say_resolve_stt_and_tts_at_call_time(monkeypatch, tmp_path):
    from harness import audio
    adopt_for("stt", "mlx-community/new-stt")
    seen = {}

    def post(url, files=None, data=None, timeout=None):
        seen.update(data)
        return httpx.Response(200, json={"text": "hi"},
                              request=httpx.Request("POST", url))
    monkeypatch.setattr(audio.httpx, "post", post)
    clip = tmp_path / "c.wav"
    clip.write_bytes(b"RIFF")
    assert audio.transcribe(clip) == "hi"
    assert seen["model"] == "mlx-community/new-stt"
    adopt_for("tts", "mlx-community/new-tts")
    assert audio.resolve_voice("bm_george").model == "mlx-community/new-tts"


@respx.mock
def test_decide_answers_with_probabilities(capsys, monkeypatch):
    import io
    monkeypatch.setattr("sys.stdin", io.StringIO("disk full on db1"))
    schema = {"urgent": {"type": "boolean", "description": "is it urgent"}}
    body = {"choices": [{"message": {"content": '{"urgent": "B"}'},
                         "logprobs": {"content": [
                             {"token": "{", "logprob": 0.0},
                             {"token": "B", "logprob": -0.1,
                              "top_logprobs": [{"token": "B", "logprob": -0.1},
                                               {"token": "A", "logprob": -2.4}]},
                             {"token": "}", "logprob": 0.0}]}}]}
    hit = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json=body))
    assert cli.main(["decide", "the server is down", "--schema",
                     json.dumps(schema), "--json"]) == 0
    out = json.loads(capsys.readouterr().out)["body"]
    assert out["answers"] == {"urgent": "true"}
    assert out["probabilities"]["urgent"]["true"] > 0.85
    sent = json.loads(hit.calls.last.request.content)
    assert sent["model"] == cli.DEFAULT_DECIDE_MODEL and sent["logprobs"] is True


# ---- the gateway's lane aliases ------------------------------------------------

def aliases(cfg) -> dict:
    return {e["model_name"]: e["litellm_params"] for e in cfg["model_list"]}


def test_a_lane_alias_points_at_the_typed_default_with_no_adoption(config):
    got = aliases(gateway.served(config))
    assert got["sohot-code"] == got[cli.DEFAULT_CODE_MODEL]
    assert got["sohot-extract"] == got["local-large"]
    assert got["local-large"]["model"] == aliases(CONFIG)["local-large"]["model"]


def test_an_adoption_change_repoints_the_lane_alias(config):
    before = aliases(gateway.served(config))["sohot-code"]
    adopt_for("code", f"llamacpp:{STEM}")
    got = aliases(gateway.served(config))
    assert got["sohot-code"] != before
    assert got["sohot-code"] == {"model": f"openai/{STEM}",
                                 "api_base": f"{router.url()}/v1",
                                 "api_key": "not-needed"}
    assert got[STEM] == got["sohot-code"]


def test_the_served_file_keeps_the_base_config_untouched(config):
    before = config.read_text(encoding="utf-8")
    out = gateway.write_served(config, defaults={"code": f"llamacpp:{STEM}"})
    assert out != config and out.parent == config.parent
    assert config.read_text(encoding="utf-8") == before
    assert "sohot-code" in aliases(yaml.safe_load(out.read_text(encoding="utf-8")))


def test_an_adoption_restarts_the_gateway_for_text_lanes_only(_no_real_services):
    adopt_for("tts", "mlx-community/new-tts")
    assert "gateway" not in _no_real_services
    adopt_for("code", f"llamacpp:{STEM}")
    assert "gateway" in _no_real_services


def test_mcp_text_tools_leave_the_model_to_the_lane(monkeypatch):
    mcp_server = pytest.importorskip("harness.mcp_server")
    argvs = []
    monkeypatch.setattr(mcp_server, "run_lh", lambda argv, timeout=0: argvs.append(argv))
    from mcp.server.mcpserver.exceptions import ToolError
    with pytest.raises(ToolError):
        mcp_server.code("x")
    assert "-m" not in argvs[-1][argvs[-1].index("code"):]
