"""What discovery asks a registry for depends on what the machine can run.

Every query was prefixed `mlx-community/`, which is right on a Mac and asks a
Mac question everywhere else: a sweep from a box with an NVIDIA card searched
HuggingFace for weights that cannot load on it, found them, and proposed them.

Keyed by RUNTIME rather than by platform, so a Linux box with the same card
asks the same questions and a new kind of machine is a row.
"""
import pytest

from harness import discover
from harness import machine as mach
from harness.memory import Accelerator

APPLE = mach.Machine(frozenset({"mlx", "cpu"}),
                     Accelerator("unified", 32.0, 25.0, "Mac16,1"))
CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                    Accelerator("discrete", 12.0, 10.5, "RTX 4070"))
LINUX_CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                          Accelerator("discrete", 12.0, 10.5, "RTX 4070"))

LANES = ("text", "stt", "tts", "image", "video")


@pytest.mark.parametrize("lane", LANES)
def test_a_card_is_not_sent_looking_for_mlx_weights(lane):
    """THE POINT OF THIS FILE. Proposing a candidate that cannot run here
    wastes the reader's time, which is the reasoning the mlx-community prefix
    was chosen for. It holds in both directions."""
    for query in discover.lane_queries(lane, CARD):
        assert "mlx-community" not in query.lower(), (
            f"{lane} asks a card for {query!r}")


@pytest.mark.parametrize("lane", LANES)
def test_a_mac_is_still_sent_looking_for_mlx_weights(lane):
    joined = " ".join(discover.lane_queries(lane, APPLE)).lower()
    assert "mlx" in joined, f"{lane} stopped asking a Mac for MLX weights"


@pytest.mark.parametrize("lane", LANES)
def test_both_machines_get_somewhere_to_look(lane):
    for m in (APPLE, CARD):
        assert discover.lane_queries(lane, m), f"{lane} has no query for {m.describe()}"


def test_linux_and_windows_with_the_same_card_ask_the_same_questions():
    for lane in LANES:
        assert discover.lane_queries(lane, CARD) == discover.lane_queries(lane, LINUX_CARD)


def test_a_lane_with_no_runtime_specific_answer_still_has_one():
    """svg is served by tools rather than by a runtime-specific model, so both
    machines look in the same place. A lane like that must not fall through to
    an empty list and silently propose nothing."""
    assert discover.lane_queries("svg", CARD) == discover.lane_queries("svg", APPLE)
    assert discover.lane_queries("svg", CARD)


def test_an_unknown_lane_says_which_ones_exist():
    with pytest.raises(ValueError) as exc:
        discover.lane_queries("telepathy", CARD)
    assert "text" in str(exc.value)


# ---- the engines this machine can actually invoke -------------------------

def test_the_cuda_generators_are_reported_on_a_card():
    """image_engines() enumerates mflux entry points from a uv tools path. On a
    machine without mflux it correctly finds nothing, and discovery then said
    this box has no image or video engine while two of them sat in the
    checkout."""
    names = {c.name for c in discover.generator_engines(CARD)}
    assert "diffusers" in names
    assert "diffusers-video" in names


def test_mlx_only_engines_are_not_offered_to_a_card():
    """h3 is Metal shaders. Offering it here proposes a run that cannot start."""
    names = {c.name for c in discover.generator_engines(CARD)}
    assert "h3" not in names


def test_mflux_is_left_to_the_enumerator_that_knows_its_entry_points():
    """It ships around twenty of them, including workflow ones that need an
    input image. Reporting it here as well would double every row on a Mac."""
    for m in (APPLE, CARD):
        assert "mflux" not in {c.name for c in discover.generator_engines(m)}


def test_each_reported_engine_carries_the_command_that_measures_it():
    for c in discover.generator_engines(CARD):
        assert "evals.run" in c.how, f"{c.name} has no command"


def test_linux_and_windows_with_the_same_card_report_the_same_engines():
    assert ([c.name for c in discover.generator_engines(CARD)]
            == [c.name for c in discover.generator_engines(LINUX_CARD)])


def test_capabilities_includes_the_engines_this_machine_can_invoke():
    """generator_engines() existed and nothing called it, so a discovery pass
    on a card still reported no image or video engine. A function nobody calls
    is the same as one nobody wrote."""
    names = {c.name for c in discover.capabilities()}
    for c in discover.generator_engines():
        assert c.name in names, f"{c.name} is reported by neither list"


def test_the_gateway_config_follows_the_machine():
    """gateway_aliases() read gateway/config.yaml unconditionally, so a card
    listed the Mac's MLX aliases as its own text candidates. serve-gateway.sh
    already takes $GATEWAY_CONFIG; this reads the same one."""
    import os
    from pathlib import Path
    cuda = Path("gateway/config.cuda.yaml").resolve()
    os.environ["GATEWAY_CONFIG"] = str(cuda)
    try:
        upstreams = {c.source for c in discover.gateway_aliases()}
        assert any("config.cuda" in u for u in upstreams), upstreams
    finally:
        del os.environ["GATEWAY_CONFIG"]


def test_a_decision_argued_on_one_machine_does_not_bind_another():
    """The ComfyUI decision reads "mflux ships 19 of them natively in MLX ... a
    machine chosen for MLX". That is a good argument on the Mac and no argument
    at all on a box with no mflux, where the workflows it names do not exist.
    A decision carries the runtime it was argued for, so a machine that does
    not share the premise is not told the question is closed."""
    on_card = {c.name for c in discover.not_adopted(CARD)}
    on_mac = {c.name for c in discover.not_adopted(APPLE)}
    assert "ComfyUI" in on_mac
    assert "ComfyUI" not in on_card
