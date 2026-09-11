"""Starting and stopping the desktop's servers.

This script was verified by hand, once, with a stub launcher, and the Windows
branch by use. Issue #131. A check that exists only as something a person did
is not a check, which is the standard this project holds everything else to.

No seam was needed. services.sh resolves its launchers relative to its own
location, so each test copies it into a temporary tree beside fake serve-*.sh
scripts and drives the real verbs against them.
"""
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest

import shells

REPO = Path(__file__).resolve().parent.parent
BASH = shells.resolve("bash")

pytestmark = pytest.mark.skipif(
    BASH is None, reason="no usable bash; services.sh is a bash script")


def tree(tmp_path, gateway: str) -> Path:
    """A copy of services.sh whose gateway launcher is `gateway`."""
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    shutil.copy(REPO / "scripts" / "services.sh", scripts / "services.sh")
    for name in ("serve-gateway.sh", "serve-llamacpp.sh", "serve-audio-cuda.sh"):
        body = gateway if name == "serve-gateway.sh" else "exit 0\n"
        path = scripts / name
        path.write_text("#!/usr/bin/env bash\n" + body, encoding="utf-8")
        path.chmod(0o755)
    return scripts / "services.sh"


def run(script: Path, *args) -> subprocess.CompletedProcess:
    home = script.parent.parent / "home"
    env = {**os.environ, "LOCALHARNESS_HOME": str(home)}
    return subprocess.run([BASH, str(script), *args], capture_output=True,
                          text=True, env=env, timeout=120)


#: A launcher that stays up. `exec` so the pid recorded is the thing to kill.
ALIVE = 'exec sleep 120\n'
#: One that dies on its first line, which is the case this file exists for.
DEAD = 'echo "boom" >&2\nexit 1\n'


def test_a_service_that_starts_is_reported_started_and_is_running(tmp_path):
    script = tree(tmp_path, ALIVE)
    started = run(script, "start", "gateway")
    assert started.returncode == 0, started.stderr
    assert "started" in started.stdout, started.stdout
    try:
        assert "up" in run(script, "status").stdout
    finally:
        run(script, "stop", "gateway")


def test_a_launcher_that_dies_immediately_is_not_reported_as_up(tmp_path):
    """THE PROPERTY THIS FILE EXISTS FOR. Start-Process returns a pid for a
    process that has already exited, so without the second look a dead service
    reads as running and the pid file backs the claim up."""
    script = tree(tmp_path, DEAD)
    got = run(script, "start", "gateway")
    assert got.returncode != 0, got.stdout
    assert "started" not in got.stdout, got.stdout
    assert "down" in run(script, "status").stdout
    # And it leaves no pid file behind to be believed later.
    assert not (tmp_path / "home" / "run" / "gateway.pid").exists()


def test_stop_ends_it_and_status_agrees(tmp_path):
    script = tree(tmp_path, ALIVE)
    run(script, "start", "gateway")
    pidfile = tmp_path / "home" / "run" / "gateway.pid"
    pid = int(pidfile.read_text().strip())
    stopped = run(script, "stop", "gateway")
    assert "stopped" in stopped.stdout, stopped.stdout
    assert not pidfile.exists()
    assert "down" in run(script, "status").stdout
    assert not _alive(pid), "the process outlived its own stop verb"


def test_starting_twice_does_not_start_twice(tmp_path):
    script = tree(tmp_path, ALIVE)
    first = run(script, "start", "gateway")
    try:
        second = run(script, "start", "gateway")
        assert "already up" in second.stdout, second.stdout
        assert _pid(first.stdout) == _pid(second.stdout)
    finally:
        run(script, "stop", "gateway")


def test_stopping_something_already_down_is_not_an_error(tmp_path):
    script = tree(tmp_path, ALIVE)
    got = run(script, "stop", "gateway")
    assert got.returncode == 0
    assert "not running" in got.stdout, got.stdout


def test_an_unknown_service_is_refused(tmp_path):
    script = tree(tmp_path, ALIVE)
    got = run(script, "start", "nosuch")
    assert "unknown service" in got.stderr, got.stderr


def _pid(text: str) -> str:
    return text.split("pid")[-1].strip(" )\n")


def _alive(pid: int) -> bool:
    time.sleep(0.5)
    if sys.platform == "win32":
        out = subprocess.run(["tasklist", "/FI", f"PID eq {pid}", "/NH"],
                             capture_output=True, text=True).stdout
        return str(pid) in out
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True
