"""The services must start the same way twice.

`uv run --with litellm[proxy]` resolves the latest release on every launch. A
restart therefore silently changes 107 packages, which is the opposite of what
you want from a machine whose whole purpose is producing comparable
measurements: a number from last week and a number from today would have been
produced by different software.

It is also slow. The gateway restart in this session downloaded 22.5 MiB and
reinstalled 107 packages before it could serve.

These are grep-level assertions on the scripts rather than behavioural tests,
for the same reason `make lint` is: the alternative is starting real services
in CI, and the failure being guarded against is a missing pin, which is visible
in the text.
"""
import re
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
VERSIONS = REPO / "scripts" / "versions.sh"
PINNED = ("scripts/serve-gateway.sh", "scripts/serve-tts.sh")


def test_there_is_one_place_that_records_the_versions():
    assert VERSIONS.exists(), "scripts/versions.sh should hold every pin"


@pytest.mark.parametrize("script", PINNED)
def test_no_service_installs_an_unpinned_package(script):
    """`--with litellm[proxy]` takes whatever released this morning."""
    text = (REPO / script).read_text(encoding="utf-8")
    for spec in re.findall(r"--with\s+'?([^'\s\\]+)'?", text):
        if spec.startswith("$") or spec.startswith("\"$"):
            continue  # a variable, resolved from versions.sh
        assert re.search(r"[=<>~]", spec), (
            f"{script} installs {spec!r} unpinned; a restart would change it")


@pytest.mark.parametrize("script", PINNED)
def test_every_pinned_service_sources_the_shared_versions(script):
    assert "versions.sh" in (REPO / script).read_text(encoding="utf-8"), (
        f"{script} should read its pins from scripts/versions.sh")


def test_pins_are_exact_unless_the_range_is_argued_for():
    """A range is a pin that moves, and the point is that two runs a month
    apart used the same software. One range is deliberate -- setuptools, where
    any 70-80 works and nothing above 81 does -- so the rule is that a range
    must carry a comment saying it is on purpose, immediately above it."""
    lines = VERSIONS.read_text(encoding="utf-8").splitlines()
    pins = [(i, m.group(1)) for i, line in enumerate(lines)
            if (m := re.match(r'^[A-Z_]+_PIN="([^"]+)"', line))]
    assert pins, "versions.sh defines no pins"
    for index, pin in pins:
        if "==" in pin:
            continue
        preceding = " ".join(lines[max(0, index - 6):index]).lower()
        assert "on purpose" in preceding, (
            f"{pin!r} is a range with no argument for why; either pin it "
            f"exactly or say above it why a range is right")


def test_the_versions_file_says_how_to_bump_a_pin():
    """A pin nobody knows how to move becomes a pin nobody moves."""
    text = VERSIONS.read_text(encoding="utf-8").lower()
    assert "smoke" in text, "bumping a pin should point at the smoke test"


def test_setuptools_stays_below_81():
    """webrtcvad still imports pkg_resources, uv does not install setuptools
    into venvs on 3.12+, and setuptools >= 81 removed pkg_resources outright.
    Unpinned resolves to 84.x and mlx_audio dies on import."""
    assert re.search(r"setuptools[^\"']*<81", VERSIONS.read_text(encoding="utf-8"))
