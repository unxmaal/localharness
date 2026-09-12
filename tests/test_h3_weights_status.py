"""Behaviour of scripts/h3-weights-status.sh.

Its exit code is the contract: 0 complete, 1 running, 2 stopped short. Anything
scripting around a 134GB download depends on that being right, and on it not
mistaking an unrelated process for the downloader.
"""
import os
import subprocess
import time
from pathlib import Path

import pytest

import shells

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "h3-weights-status.sh"


def run_status(dest, extra_env=None):
    env = dict(os.environ, H3_MODEL_DIR=str(dest))
    env.pop("H3_DOWNLOAD_PIDFILE", None)
    if extra_env:
        env.update(extra_env)
    p = subprocess.run([shells.BASH, str(SCRIPT)], capture_output=True, text=True,
                       env=env)
    return p.returncode, p.stdout


def test_empty_dest_reports_stopped_short(tmp_path):
    code, out = run_status(tmp_path)
    assert code == 2
    assert "STOPPED SHORT" in out


def test_unrelated_process_is_not_mistaken_for_the_downloader(tmp_path):
    """A stray process whose command line mentions the model must not read as
    'downloading'.

    `pgrep -f MiniMax-H3` matches anything with that path in its arguments: an
    editor, a du, a grep, this very test. Reporting RUNNING then makes the
    script wait forever on a download that already stopped.
    """
    # python3 -c keeps the marker in argv. `bash -c "sleep 20" MARKER` does
    # NOT: bash execs the single command directly and the marker vanishes from
    # the command line, which made an earlier version of this test pass
    # vacuously against code that had the bug.
    decoy = subprocess.Popen(
        ["python3", "-c", "import time; time.sleep(20)",
         "/Volumes/FAST/MiniMax-H3"])
    try:
        time.sleep(0.5)
        code, out = run_status(tmp_path)
        assert "RUNNING" not in out, "a decoy process was read as the downloader"
        assert code == 2
    finally:
        decoy.kill()
        decoy.wait()


def test_live_downloader_reports_running(tmp_path):
    """A real download in flight, identified by its pidfile, reports RUNNING."""
    pidfile = tmp_path / "pid"
    # The sleeper writes its OWN pid, from the shell, which is what
    # fetch-h3-weights.sh does. A pid captured on the Python side is a Windows
    # pid, and the `kill -0` in the status script runs under MSYS bash, which
    # numbers processes differently -- so a live download read as stopped.
    proc = subprocess.Popen(
        [shells.BASH, "-c", f'echo $$ > "{pidfile.as_posix()}"; exec sleep 20'])
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text(encoding="utf-8").strip():
            break
        time.sleep(0.05)
    (tmp_path / "chunk").write_bytes(b"x" * 1024)
    try:
        code, out = run_status(tmp_path, {"H3_DOWNLOAD_PIDFILE": str(pidfile)})
        assert "RUNNING" in out
        assert code == 1
    finally:
        proc.kill()
        proc.wait()


def test_stale_pidfile_is_not_running(tmp_path):
    """A pidfile left behind by a crashed download must not read as RUNNING."""
    pidfile = tmp_path / "pid"
    proc = subprocess.Popen(["sleep", "0.1"])
    proc.wait()
    pidfile.write_text(str(proc.pid), encoding="utf-8")
    code, out = run_status(tmp_path, {"H3_DOWNLOAD_PIDFILE": str(pidfile)})
    assert "RUNNING" not in out
    assert code == 2
