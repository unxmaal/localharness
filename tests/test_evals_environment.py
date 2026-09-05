"""Every result set records the machine and versions that produced it.

The suite's stated purpose includes "re-run on the Studio and compare". That is
impossible without knowing what the first run ran on. It is also how the Discord
incident had to be recorded: by hand, in prose, after the fact.
"""
from evals.environment import capture


def test_captures_hardware_and_os():
    e = capture()
    assert e["hw_model"]
    assert e["macos"]
    assert e["memory_gb"] > 0


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
