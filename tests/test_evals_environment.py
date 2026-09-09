"""Every result set records the machine and versions that produced it.

The suite's stated purpose includes "re-run on the Studio and compare". That is
impossible without knowing what the first run ran on. It is also how the Discord
incident had to be recorded: by hand, in prose, after the fact.
"""
import sys

from evals.environment import capture


def test_captures_hardware_and_os():
    e = capture()
    assert e["hw_model"]
    assert e["os"], "every result set names the OS it ran on"
    assert e["memory_gb"] > 0
    # `macos` is the macOS version where there is one and empty elsewhere,
    # rather than a key named macos holding a Windows string.
    if sys.platform == "darwin":
        assert e["macos"]


def test_captures_tool_versions():
    e = capture()
    assert "mlx" in e["versions"]
    assert "python" in e["versions"]


def test_captures_memory_pressure_at_run_time():
    """A run under swap pressure produces pessimistic numbers. Record it rather
    than discovering it in a chat log afterwards."""
    e = capture()
    assert "swap_used_mb" in e
    assert e["swap_used_mb"] >= 0


def test_captures_the_git_sha_of_the_harness():
    e = capture()
    assert len(e["git_sha"]) >= 7


def test_is_json_serializable():
    import json
    json.dumps(capture())


def test_records_the_accelerator_that_produced_the_result():
    """A score sheet without its accelerator is not comparable to anything.

    `hw_model` identifies the GPU on Apple Silicon, where it is the same
    part as the CPU. On a PC it says nothing about it: an RTX 4070 result and
    an M2 Pro result are otherwise indistinguishable in a results.json, and
    the stated purpose of this module is comparing one machine against
    another.
    """
    a = capture()["accelerator"]
    assert a["kind"] in ("unified", "discrete")
    assert a["name"], "the accelerator is named"
    assert a["total_gb"] > 0
