"""A known-broken stage refuses early, and stops refusing once fixed."""
from pathlib import Path

import pytest

from harness import stages
from harness.stages import BROKEN_STAGES, stage_unavailable, tool_versions

BROKEN = {"mflux": "0.19.1", "mlx": "0.32.2"}


def test_the_measured_broken_versions_are_refused():
    why = stage_unavailable("upscale-seedvr2", BROKEN)
    assert why
    assert "#27" in why
    assert "mx.repeat" in why


def test_a_newer_mflux_is_allowed_through_to_be_tried():
    """The point of pinning. mflux ships a fix; the stage runs again with no
    code change here, and the eval reports what actually happened."""
    assert stage_unavailable("upscale-seedvr2",
                             {"mflux": "0.20.0", "mlx": "0.32.2"}) == ""
    assert stage_unavailable("upscale-seedvr2",
                             {"mflux": "0.19.1", "mlx": "0.33.0"}) == ""


def test_an_unknown_version_is_not_a_broken_one():
    """mflux missing entirely must produce "not installed" from the runner, not
    a confident bug report about a version nobody read."""
    assert stage_unavailable("upscale-seedvr2", {}) == ""


def test_a_healthy_stage_is_never_refused():
    assert stage_unavailable("controlnet", BROKEN) == ""
    assert stage_unavailable("upscale-controlnet", BROKEN) == ""


def test_every_broken_entry_names_an_issue_and_a_reason():
    for stage, (versions, why, issue) in BROKEN_STAGES.items():
        assert versions, f"{stage}: an unpinned refusal never expires"
        assert len(why) > 40, f"{stage}: the reason must be readable alone"
        assert isinstance(issue, int)


def test_tool_versions_reads_the_uv_tool_venv(tmp_path):
    site = tmp_path / "lib/python3.12/site-packages"
    site.mkdir(parents=True)
    for name in ("mflux-0.19.1.dist-info", "mlx-0.32.2.dist-info",
                 "mlx_metal-0.32.2.dist-info", "mflux", "notadistinfo"):
        (site / name).mkdir()
    tool_versions.cache_clear()
    try:
        got = tool_versions(tmp_path)
    finally:
        tool_versions.cache_clear()
    assert got["mflux"] == "0.19.1"
    assert got["mlx"] == "0.32.2"
    assert got["mlx-metal"] == "0.32.2"


def test_tool_versions_on_a_missing_root_is_empty_not_an_error():
    tool_versions.cache_clear()
    try:
        assert tool_versions(Path("/nonexistent/uv/tools/mflux")) == {}
    finally:
        tool_versions.cache_clear()


def test_the_pin_matches_what_is_actually_installed_here():
    """Red-proofs the whole guard against reality.

    If mflux is upgraded on this machine and #27 is still open, this fails and
    says so -- which is the moment to re-run the stage rather than to edit the
    pin.
    """
    tool_versions.cache_clear()
    have = tool_versions()
    if not have.get("mflux"):
        pytest.skip("mflux is not installed here")
    pinned = BROKEN_STAGES["upscale-seedvr2"][0]
    if have.get("mflux") != pinned["mflux"] or have.get("mlx") != pinned["mlx"]:
        pytest.fail(
            f"mflux/mlx moved to {have.get('mflux')}/{have.get('mlx')} from the "
            f"pinned {pinned}. Re-run upscale-seedvr2 and either close #27 or "
            f"re-pin against the versions it now fails on.")
