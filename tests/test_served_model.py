"""The gateway's reply names the model that answered, not the alias asked for. #670."""
import asyncio
import importlib.util
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("served_model", REPO / "gateway" / "served_model.py")
served = importlib.util.module_from_spec(spec)
spec.loader.exec_module(served)


def test_the_headers_name_the_model_and_the_alias_even_when_streamed():
    got = asyncio.run(served.proxy_handler_instance.async_post_call_response_headers_hook(
        data={"model": "eval-7b", "stream": True}, user_api_key_dict=None, response=None))
    assert got == {"x-sohot-model": "Qwen2.5-7B-Instruct-Q4_K_M", "x-sohot-served-as": "eval-7b"}


def test_a_real_id_gets_no_served_as_header():
    got = asyncio.run(served.proxy_handler_instance.async_post_call_response_headers_hook(
        data={"model": "Qwen2.5-7B-Instruct-Q4_K_M"}, user_api_key_dict=None, response=None))
    assert got == {"x-sohot-model": "Qwen2.5-7B-Instruct-Q4_K_M"}


def test_both_gateway_configs_load_the_callback():
    for name in ("config.yaml", "config.cuda.yaml"):
        body = yaml.safe_load((REPO / "gateway" / name).read_text(encoding="utf-8"))
        assert "served_model.proxy_handler_instance" in body["litellm_settings"]["callbacks"], name


def test_a_name_nothing_serves_gets_no_header():
    got = asyncio.run(served.proxy_handler_instance.async_post_call_response_headers_hook(
        data={"model": "sohot-nothing"}, user_api_key_dict=None, response=None))
    assert got is None
