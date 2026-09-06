"""Text generation through the gateway, and recovering the artifact from it.

The system prompts live here rather than in the eval suite on purpose: if the
eval steers the model differently from the CLI, it is measuring a product that
does not ship.
"""
import httpx
import pytest
import respx

from harness import completion as comp

GW = "http://127.0.0.1:4000"


def reply(text):
    return httpx.Response(200, json={"choices": [{"message": {"content": text}}]})


@respx.mock
def test_complete_returns_the_message_content():
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("hello"))
    assert comp.complete("hi", model="local-mid", gateway=GW) == "hello"


@respx.mock
def test_the_modality_system_prompt_is_sent():
    import json
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("<svg/>"))
    comp.complete("a gear", model="local-mid", gateway=GW, modality="svg")
    sent = json.loads(route.calls[0].request.read())
    assert sent["messages"][0]["role"] == "system"
    assert "svg" in sent["messages"][0]["content"].lower()
    assert sent["messages"][1]["content"] == "a gear"
    assert sent["model"] == "local-mid"


@respx.mock
def test_an_unknown_modality_still_works_with_a_neutral_system_prompt():
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("ok"))
    assert comp.complete("hi", model="m", gateway=GW, modality="haiku") == "ok"


@respx.mock
def test_gateway_down_names_the_script_that_starts_it():
    respx.post(f"{GW}/v1/chat/completions").mock(side_effect=httpx.ConnectError("x"))
    with pytest.raises(comp.CompletionError) as e:
        comp.complete("hi", model="m", gateway=GW)
    assert "serve-gateway.sh" in str(e.value)


@respx.mock
def test_http_error_carries_the_body_because_that_is_where_litellm_explains():
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(400, text="model not in config"))
    with pytest.raises(comp.CompletionError) as e:
        comp.complete("hi", model="m", gateway=GW)
    assert "model not in config" in str(e.value)


@respx.mock
def test_an_empty_completion_is_an_error_not_an_empty_file():
    respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("   "))
    with pytest.raises(comp.CompletionError):
        comp.complete("hi", model="m", gateway=GW)


@respx.mock
def test_a_response_with_no_choices_is_reported_clearly():
    respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={"object": "error"}))
    with pytest.raises(comp.CompletionError):
        comp.complete("hi", model="m", gateway=GW)


@respx.mock
def test_timeout_is_reported_with_the_limit():
    respx.post(f"{GW}/v1/chat/completions").mock(
        side_effect=httpx.TimeoutException("t"))
    with pytest.raises(comp.CompletionError) as e:
        comp.complete("hi", model="m", gateway=GW, timeout=7.0)
    assert "7" in str(e.value)


# ---- artifact recovery -----------------------------------------------------

def test_extract_pulls_the_artifact_out_of_a_fence():
    got = comp.extract("here you go:\n```svg\n<svg><rect/></svg>\n```\nenjoy",
                       ("svg",))
    assert got.startswith("<svg") and got.endswith("</svg>")


def test_extract_finds_the_root_tag_without_a_fence():
    got = comp.extract("chatter <svg><rect/></svg> more chatter", ("svg",))
    assert got == "<svg><rect/></svg>"


def test_extract_keeps_the_doctype_for_html():
    got = comp.extract("<!doctype html><html><body>x</body></html>",
                       ("html", "!doctype"))
    assert got.lower().startswith("<html")


def test_extract_returns_stripped_text_when_no_root_tag_is_present():
    assert comp.extract("  just words  ", ("svg",)) == "just words"


def test_the_svg_and_web_system_prompts_forbid_external_references():
    assert "raster" in comp.SYSTEM["svg"].lower()
    assert "external" in comp.SYSTEM["web"].lower()
