"""Behaviour of scripts/preflight.sh, the host-level check for a new machine.

The script's whole value is telling the truth about what a machine has, so the
failure that matters is a WRONG answer, not a crash. It shipped one on its
first run: `fc-list | grep -qi dejavu` under `set -o pipefail` reported the
fonts missing on a machine that had both the fonts and fc-list, because grep -q
exits at the first match and the producer then dies of SIGPIPE. shellcheck
cannot see that, and a green shellcheck is what made it look finished.

NOTHING HERE DEPENDS ON WHAT THE RUNNER HAS INSTALLED. Every test builds a
PATH containing only stubs it created, so "ffmpeg is missing" means the test
said so and not that the runner lacks ffmpeg. That is the same reasoning as
test_env_sh.py: a test whose result depends on the machine is a test CI cannot
defend.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PREFLIGHT = REPO / "scripts" / "preflight.sh"

#: Real tools preflight.sh itself runs. Symlinked in so PATH can be replaced
#: wholesale and still leave the script able to execute.
#: env.sh runs inside preflight and needs its own tools to measure free space.
_ESSENTIAL = ("bash", "sh", "dirname", "uname", "grep", "find", "head",
              "readlink", "git", "sed", "cat", "ls", "env", "true", "sort",
              "df", "awk", "mkdir", "stat", "tr", "cut", "touch", "rm")

#: Everything preflight looks for that a stub can stand in for.
_CHECKED = ("git", "curl", "make", "shellcheck", "ffmpeg", "rsvg-convert",
            "zsh", "sox", "uv", "lh", "google-chrome", "aplay")

pytestmark = pytest.mark.skipif(
    sys.platform == "win32", reason="preflight.sh is for the unix machines"
)


def _stub(path: Path, body: str = "exit 0") -> None:
    path.write_text(f"#!/bin/sh\n{body}\n", encoding="utf-8")
    path.chmod(0o755)


def _bin(tmp_path, present=_CHECKED, **extra):
    """A directory that is the whole PATH: essentials, plus the named stubs."""
    d = tmp_path / "bin"
    d.mkdir(exist_ok=True)
    for tool in _ESSENTIAL:
        real = shutil.which(tool)
        if real and not (d / tool).exists():
            (d / tool).symlink_to(real)
    for tool in present:
        if tool not in _ESSENTIAL:
            _stub(d / tool)
    for name, body in extra.items():
        _stub(d / name.replace("_", "-"), body)
    return d


def _time_bin(tmp_path, reports=True):
    """A stand-in for /usr/bin/time that does or does not report peak memory."""
    p = tmp_path / "time-stub"
    _stub(p, 'echo "Maximum resident set size (kbytes): 1256" >&2' if reports
             else 'echo "0.00 real" >&2')
    return p


def run_preflight(tmp_path, bindir, time_reports=True, fonts=True, env=None):
    """Run preflight.sh with a synthetic PATH; return (code, stdout)."""
    fontdir = tmp_path / "fonts"
    fontdir.mkdir(exist_ok=True)
    if fonts:
        (fontdir / "DejaVuSans.ttf").write_bytes(b"")
    # env.sh checks writability BEFORE it creates anything, so a weights root
    # that does not exist yet is correctly refused. Make it first.
    hf = tmp_path / "hf"
    hf.mkdir(exist_ok=True)
    e = {
        "PATH": str(bindir),
        "HOME": str(tmp_path),
        "HF_ROOT": str(hf),
        "HF_MIN_FREE_GB": "0",
        "LH_PREFLIGHT_TIME_BIN": str(_time_bin(tmp_path, time_reports)),
        "LH_PREFLIGHT_FONT_DIRS": str(fontdir),
    }
    e.update(env or {})
    proc = subprocess.run(
        ["bash", str(PREFLIGHT)], cwd=REPO, env=e,
        capture_output=True, text=True,
    )
    return proc.returncode, proc.stdout


def test_a_machine_with_everything_passes(tmp_path):
    """The regression the pipefail bug broke: present things read as present."""
    code, out = run_preflight(tmp_path, _bin(tmp_path))
    assert "preflight: ok" in out, out
    assert code == 0, out


def test_fonts_that_exist_are_not_reported_missing(tmp_path):
    """Directly pins the SIGPIPE/pipefail defect this script shipped with."""
    _, out = run_preflight(tmp_path, _bin(tmp_path), fonts=True)
    assert "ok       DejaVu fonts" in out, out


def test_absent_fonts_are_reported(tmp_path):
    code, out = run_preflight(tmp_path, _bin(tmp_path), fonts=False)
    assert "MISSING  DejaVu fonts" in out, out
    assert code == 1


def test_a_missing_package_is_named_and_fails(tmp_path):
    present = [t for t in _CHECKED if t != "sox"]
    code, out = run_preflight(tmp_path, _bin(tmp_path, present))
    assert "MISSING  sox" in out, out
    assert "preflight: ok" not in out
    assert code == 1


def test_every_failure_is_reported_not_just_the_first(tmp_path):
    """One apt round-trip for a new machine, not one per missing package."""
    present = [t for t in _CHECKED if t not in ("sox", "ffmpeg", "zsh")]
    _, out = run_preflight(tmp_path, _bin(tmp_path, present))
    for missing in ("sox", "ffmpeg", "zsh"):
        assert f"MISSING  {missing}" in out, out


def test_time_that_cannot_report_peak_memory_fails(tmp_path):
    """proc.py raises rather than reporting zero, so presence is not enough."""
    code, out = run_preflight(tmp_path, _bin(tmp_path), time_reports=False)
    assert "cannot report peak memory" in out, out
    assert code == 1


def test_a_cpu_only_llama_build_is_caught(tmp_path):
    """The failure worth catching: it exits 0 and serves, with the card idle."""
    d = _bin(tmp_path)
    _stub(d / "llama-server")
    _stub(d / "llama-cli", 'echo "Available devices:"; echo "  (none)"')
    code, out = run_preflight(tmp_path, d)
    assert "reports no accelerator" in out, out
    assert code == 1


def test_an_accelerated_llama_build_passes(tmp_path):
    d = _bin(tmp_path)
    _stub(d / "llama-server")
    _stub(d / "llama-cli",
          'echo "Available devices:"; echo "  CUDA0: NVIDIA GeForce RTX 4070"')
    code, out = run_preflight(tmp_path, d)
    assert "backend: CUDA0" in out, out
    assert code == 0, out


def test_no_llama_server_is_a_note_not_a_failure(tmp_path):
    """A machine can be fine without the text lane; absence is not a defect."""
    code, out = run_preflight(tmp_path, _bin(tmp_path))
    assert "no llama-server" in out, out
    assert code == 0, out


def test_lh_absent_is_a_failure(tmp_path):
    present = [t for t in _CHECKED if t != "lh"]
    code, out = run_preflight(tmp_path, _bin(tmp_path, present))
    assert "MISSING  lh on PATH" in out, out
    assert code == 1
