"""What the gateway's aliases point at.

An alias is how the eval suite names a candidate. Two aliases resolving to the
same weights is therefore not a configuration slip: it is a model compared
against itself, reported as a tie, with nothing in the output saying so. That
is the invariant both machines have to hold, and it is asserted against both.

The mechanism is the same on each, which was not obvious. mlx_lm.server
hot-swaps by the `model` field a request carries. llama-server does too, in its
router mode, so the CUDA config has the same shape rather than one port per
model. What differs is the ceiling: the router will hold several models at once
if allowed to, and one 12 GB card cannot.
"""
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
MAC = REPO / "gateway" / "config.yaml"
CUDA = REPO / "gateway" / "config.cuda.yaml"


def _entries(path):
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    return [(e["model_name"], e["litellm_params"]) for e in body["model_list"]]


def test_the_cuda_config_exists():
    assert CUDA.exists(), (
        "a machine with an NVIDIA card needs a gateway config of its own: the "
        "Mac one names mlx-community weights, which do not load under CUDA")


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_no_two_aliases_name_the_same_weights(path):
    """A sweep over both would measure one model twice and call it a tie."""
    seen = {}
    for name, params in _entries(path):
        model = params["model"]
        assert model not in seen, (
            f"{name} and {seen[model]} both resolve to {model}; a sweep over "
            f"the two would compare it against itself")
        seen[model] = name


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_every_alias_reaches_one_hot_swapping_server(path):
    """Both upstreams select the model from the request rather than from the
    port, so one api_base is right on either machine. One port per model would
    need every candidate in a sweep resident at once, which is the thing a
    12 GB card cannot do."""
    bases = {params["api_base"] for _, params in _entries(path)}
    assert len(bases) == 1, f"{path.name} spreads its aliases over {bases}"


def test_the_cuda_server_holds_one_model_at_a_time():
    """llama-server's router defaults to four resident models. Four at once is
    an out-of-memory on a 12 GB card, and the eval suite's own memory guard
    budgets for one resident model plus what a swap has not yet freed."""
    text = (REPO / "scripts" / "serve-llamacpp.sh").read_text(encoding="utf-8")
    assert "--models-max" in text, (
        "serve-llamacpp.sh should cap resident models; the default is 4")
    assert "${LLAMACPP_MAX_MODELS:-1}" in text, (
        "the cap should default to 1 and stay overridable on a larger card")


def test_the_gateway_takes_the_config_as_a_parameter():
    """One gateway script, two machines. A hardcoded path means the CUDA box
    needs an edited copy of the launcher."""
    text = (REPO / "scripts" / "serve-gateway.sh").read_text(encoding="utf-8")
    assert "${GATEWAY_CONFIG:-" in text, (
        "serve-gateway.sh should read its config path from $GATEWAY_CONFIG")


@pytest.mark.parametrize("path", [MAC, CUDA], ids=["mac", "cuda"])
def test_both_configs_keep_the_anthropic_routing_workaround(path):
    """LiteLLM routes /v1/messages through its Responses adapter without this,
    and neither upstream implements /v1/responses."""
    body = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert body.get("litellm_settings", {}).get(
        "use_chat_completions_url_for_anthropic_messages") is True
