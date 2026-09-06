"""Text generation through the gateway, and recovering the artifact from it.

The system prompts live here rather than in the eval suite on purpose: if the
eval steers the model differently from the CLI, it is measuring a product that
does not ship.
"""
import json

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


# ---- context and the two new lanes -----------------------------------------

@respx.mock
def test_context_is_sent_as_material_separate_from_the_instruction():
    import json
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("137"))
    comp.complete("What is the exit code?", model="m", gateway=GW,
                  modality="extract", context="process exited with 137")
    msgs = json.loads(route.calls[0].request.read())["messages"]
    body = msgs[-1]["content"]
    assert "process exited with 137" in body
    assert "What is the exit code?" in body


@respx.mock
def test_the_instruction_comes_after_the_material():
    """A small model that reads a long log and then a question does better than
    one that reads a question, forgets it, and then reads a log."""
    import json
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("x"))
    comp.complete("QUESTION", model="m", gateway=GW, modality="extract",
                  context="MATERIAL")
    body = json.loads(route.calls[0].request.read())["messages"][-1]["content"]
    assert body.index("MATERIAL") < body.index("QUESTION")


def test_the_extract_system_prompt_asks_for_the_answer_and_nothing_else():
    """The whole value of the lane is a short exact answer that can be used
    without parsing."""
    text = comp.SYSTEM["extract"].lower()
    assert "nothing else" in text or "only" in text
    assert "explain" in text or "prose" in text or "preamble" in text


def test_the_code_system_prompt_asks_for_code_only():
    text = comp.SYSTEM["code"].lower()
    assert "code" in text
    assert "fence" in text or "prose" in text or "explanation" in text


@respx.mock
def test_a_case_with_no_context_sends_the_prompt_unchanged():
    import json
    route = respx.post(f"{GW}/v1/chat/completions").mock(return_value=reply("x"))
    comp.complete("draw a gear", model="m", gateway=GW, modality="svg")
    assert json.loads(route.calls[0].request.read())["messages"][-1]["content"] \
        == "draw a gear"


def test_code_is_recovered_from_a_fence_but_svg_markup_is_not_mangled():
    assert "def f" in comp.artifact("```python\ndef f():\n    pass\n```", "code")
    assert comp.artifact("<svg><rect/></svg>", "svg").startswith("<svg")


# ---- sampling, per modality ------------------------------------------------
# Degenerate repetition is the SVG lane's loudest failure: the model emits a
# plausible <path>, and the highest-probability continuation is another one
# just like it, until the token budget runs out mid-attribute and the document
# is unclosed. A repetition penalty is the standard lever.
#
# It is per-modality because the lanes want opposite things. `extract` pulls
# one fact out of a log and should be as close to deterministic as the sampler
# allows; making it stochastic to fix a drawing problem would be a plain
# downgrade.

def test_svg_and_web_ask_for_a_repetition_penalty():
    for modality in ("svg", "web"):
        assert comp.SAMPLING[modality].get("repetition_penalty", 1.0) > 1.0


def test_extract_and_code_stay_deterministic():
    """A fact extractor that samples is a worse fact extractor."""
    for modality in ("extract", "code"):
        assert "repetition_penalty" not in comp.SAMPLING.get(modality, {})
        assert comp.SAMPLING.get(modality, {}).get(
            "temperature", comp.DEFAULT_TEMPERATURE) <= 0.2


@respx.mock
def test_the_penalty_reaches_the_gateway(tmp_path):
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "<svg/>"}}]}))
    comp.complete("a gear", model="local-large", modality="svg")
    sent = json.loads(route.calls[0].request.read())
    assert sent["repetition_penalty"] > 1.0


@respx.mock
def test_an_unsteered_modality_sends_no_extra_sampling(tmp_path):
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "hi"}}]}))
    comp.complete("hello", model="local-mid")
    sent = json.loads(route.calls[0].request.read())
    assert "repetition_penalty" not in sent
    assert sent["temperature"] == comp.DEFAULT_TEMPERATURE


@respx.mock
def test_a_caller_can_override_the_sampling_to_measure_it(tmp_path):
    """The eval has to be able to run the same case with and without the
    penalty, or its value stays an assertion rather than a measurement."""
    route = respx.post(f"{GW}/v1/chat/completions").mock(
        return_value=httpx.Response(200, json={
            "choices": [{"message": {"content": "<svg/>"}}]}))
    comp.complete("a gear", model="local-large", modality="svg",
                        sampling={"repetition_penalty": 1.0, "temperature": 0.5})
    sent = json.loads(route.calls[0].request.read())
    assert sent["repetition_penalty"] == 1.0
    assert sent["temperature"] == 0.5
