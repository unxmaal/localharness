"""launchd units for the services.

The point is that the machine comes back up serving after a reboot or a crash
without anyone remembering four script names. The units are generated from one
template rather than hand-written, because near-identical plists drift:
the last time this repo had three near-identical things, the copies disagreed
about which port they used.
"""
import plistlib
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
GEN = REPO / "scripts" / "launchd.sh"
SERVICES = ("gateway", "mlx", "tts", "mcp")

#: launchd IS macOS. There is no Windows equivalent to generate plists for, and
#: Windows service supervision (Task Scheduler) is not implemented -- so these
#: skip rather than fail, which keeps "not built yet" distinct from "broken".
pytestmark = pytest.mark.skipif(
    sys.platform != "darwin",
    reason="launchd is macOS-only; Windows service supervision is unimplemented")


@pytest.fixture(scope="module")
def plists(tmp_path_factory):
    """Generate the units into a scratch directory and parse them."""
    out = tmp_path_factory.mktemp("launchagents")
    proc = subprocess.run(["bash", str(GEN), "generate", str(out)],
                          capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr
    return {p.stem: plistlib.loads(p.read_bytes()) for p in out.glob("*.plist")}


def test_one_unit_per_service(plists):
    labels = {v["Label"] for v in plists.values()}
    for service in SERVICES:
        assert any(service in label for label in labels), f"no unit for {service}"


@pytest.mark.parametrize("service", SERVICES)
def test_each_unit_runs_the_repo_script(plists, service):
    unit = next(v for k, v in plists.items() if service in k)
    argv = unit["ProgramArguments"]
    assert argv[0] == "/bin/bash", argv
    assert argv[1].endswith(f"serve-{service}.sh"), argv
    assert Path(argv[1]).is_absolute(), "launchd has no working directory"


@pytest.mark.parametrize("service", SERVICES)
def test_each_unit_restarts_on_crash(plists, service):
    unit = next(v for k, v in plists.items() if service in k)
    assert unit.get("KeepAlive") is True


@pytest.mark.parametrize("service", SERVICES)
def test_each_unit_logs_somewhere_you_can_read(plists, service):
    unit = next(v for k, v in plists.items() if service in k)
    for key in ("StandardOutPath", "StandardErrorPath"):
        assert unit.get(key), f"{service} has no {key}"
        assert Path(unit[key]).is_absolute()


@pytest.mark.parametrize("service", SERVICES)
def test_each_unit_carries_a_path_that_includes_homebrew(plists, service):
    """launchd starts jobs with PATH=/usr/bin:/bin:/usr/sbin:/sbin. uv, ffmpeg,
    rsvg-convert and rec all live in /opt/homebrew/bin, so without this every
    service dies on 'command not found' -- the same trap a GUI-spawned wezterm
    set for the voice scripts."""
    unit = next(v for k, v in plists.items() if service in k)
    assert "/opt/homebrew/bin" in unit["EnvironmentVariables"]["PATH"]


def test_labels_are_reverse_dns_and_distinct(plists):
    labels = [v["Label"] for v in plists.values()]
    assert len(set(labels)) == len(labels)
    for label in labels:
        assert label.count(".") >= 2, f"{label} is not a reverse-DNS label"


def test_the_script_refuses_an_unknown_subcommand():
    proc = subprocess.run(["bash", str(GEN), "frobnicate"],
                          capture_output=True, text=True)
    assert proc.returncode != 0
    assert "generate" in proc.stdout + proc.stderr


def test_generate_is_idempotent(tmp_path):
    """Re-running must overwrite rather than accumulate or fail."""
    for _ in range(2):
        proc = subprocess.run(["bash", str(GEN), "generate", str(tmp_path)],
                              capture_output=True, text=True)
        assert proc.returncode == 0, proc.stderr
    assert len(list(tmp_path.glob("*.plist"))) == len(SERVICES)


def test_install_is_a_separate_step_from_generate():
    """Writing into ~/Library/LaunchAgents and loading jobs is not something
    `generate` should do as a side effect."""
    text = GEN.read_text(encoding="utf-8")
    assert "install)" in text and "generate)" in text
    assert "uninstall)" in text, "anything that loads jobs must unload them"


def test_install_checks_the_weights_volume_is_readable_first():
    """macOS TCC blocks a launchd agent from reading /Volumes even though the
    volume stats fine, and mlx_lm turns that into a silent hang. Installing
    units that will wedge on first use is worse than refusing."""
    text = GEN.read_text(encoding="utf-8")
    assert "Full Disk Access" in text
    assert "preflight" in text.lower() or "readable" in text.lower()


def test_the_preflight_runs_in_the_launchd_domain_not_the_shell():
    """Checking readability from the installing terminal proves nothing: that
    terminal already has the access the agent lacks. It has to be tested from
    inside launchd."""
    text = GEN.read_text(encoding="utf-8")
    assert "launchctl" in text and ("probe" in text.lower() or "preflight" in text.lower())


def test_the_mcp_unit_can_find_lh(plists):
    """The MCP server shells out to `lh`, which uv installs into
    ~/.local/bin. launchd starts jobs with PATH=/usr/bin:/bin:/usr/sbin:/sbin
    and nothing else, so without this the unit loads, listens, and fails every
    single tool call with "lh: command not found"."""
    unit = next(v for k, v in plists.items() if "mcp" in k)
    assert ".local/bin" in unit["EnvironmentVariables"]["PATH"]


def test_probe_is_available_without_installing_anything():
    """The TCC verdict is the whole question, and finding it out should not
    require committing to an install first."""
    proc = subprocess.run(["bash", str(GEN), "probe"],
                          capture_output=True, text=True)
    # Either verdict is fine here; what matters is that it ran and said so.
    assert "ok" in proc.stdout.lower() or "denied" in (proc.stdout + proc.stderr).lower()
