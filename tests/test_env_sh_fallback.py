"""env.sh must not quietly relocate the weights cache.

Separate from test_env_sh.py because that module skips itself unless the
Models volume is mounted -- and a missing Models volume is exactly the
situation these tests are about.
"""
import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]

# ---- the drive that went away ---------------------------------------------
# A reboot brought the machine back without the weights volume. env.sh walked
# its candidate list exactly as designed, found the next location with room,
# and started the services against an EMPTY CACHE. They listened, served
# nothing, and said nothing: the only trace was one differing line in a log
# nobody reads until something is wrong.
#
# Falling back is right for a fresh machine and wrong for a machine that has
# 93GB of weights somewhere else. The difference is whether we have been here
# before, so env.sh now remembers where it landed last time.

def run_env_raw(hf_root=None, candidates=None, allow_move=None,
                min_free_gb="0", state=None):
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_MIN_FREE_GB": min_free_gb}
    if hf_root is not None:
        env["HF_ROOT"] = hf_root
    if candidates is not None:
        env["HF_CANDIDATES"] = candidates
    if allow_move is not None:
        env["HF_ALLOW_MOVE"] = allow_move
    if state is not None:
        env["HF_STATE_FILE"] = state
    return subprocess.run(
        ["bash", "-c", f'source "{REPO}/scripts/env.sh" && echo "HF_HOME=$HF_HOME"'],
        capture_output=True, text=True, env=env)


@pytest.fixture
def state(tmp_path):
    return str(tmp_path / "root-state")


def test_the_chosen_root_is_remembered(tmp_path, state):
    a = tmp_path / "a"
    a.mkdir()
    run_env_raw(candidates=str(a), state=state)
    assert Path(state).read_text().strip() == str(a)


def test_an_explicit_hf_root_is_never_blocked(tmp_path, state):
    """Setting HF_ROOT IS the caller saying where the weights are, which is the
    same statement HF_ALLOW_MOVE makes. Policing it would make the guard fire
    on every deliberate override."""
    a = tmp_path / "elsewhere"
    a.mkdir()
    Path(state).write_text(str(tmp_path / "recorded") + "\n")
    proc = run_env_raw(hf_root=str(a), state=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_falling_back_to_a_different_root_is_fatal(tmp_path, state):
    """The case that actually happened: Models is gone, T7 has room, and
    starting on T7 means serving from a cache with no models in it.

    The absent root is seeded into the state file rather than simulated by
    deleting a directory: env.sh rejected the real /Volumes/Models on FREE
    SPACE (unmounted, its mountpoint became the 91%-full root disk), not on
    the directory being missing, and a deleted tmp dir reproduces neither."""
    b = tmp_path / "fallback"
    b.mkdir()
    Path(state).write_text(str(tmp_path / "primary") + "\n")
    proc = run_env_raw(candidates=str(b), state=state)
    assert proc.returncode != 0, proc.stdout + proc.stderr
    msg = proc.stdout + proc.stderr
    assert "primary" in msg and "fallback" in msg


def test_the_fallback_refusal_says_how_to_override(tmp_path, state):
    b = tmp_path / "b"
    b.mkdir()
    Path(state).write_text(str(tmp_path / "a") + "\n")
    proc = run_env_raw(candidates=str(b), state=state)
    assert "HF_ALLOW_MOVE" in (proc.stdout + proc.stderr)


def test_the_refusal_says_the_old_root_is_missing_and_what_to_check(tmp_path, state):
    """The actionable half: an unplugged drive is a cable problem, and saying
    so beats reporting a cache path that means nothing on its own."""
    b = tmp_path / "b"
    b.mkdir()
    Path(state).write_text(str(tmp_path / "gone") + "\n")

    proc = run_env_raw(candidates=str(b), state=state)
    out = proc.stdout + proc.stderr
    assert "NOT PRESENT" in out
    assert "diskutil" in out


def test_an_explicit_move_is_allowed(tmp_path, state):
    b = tmp_path / "b"
    b.mkdir()
    Path(state).write_text(str(tmp_path / "a") + "\n")
    proc = run_env_raw(candidates=str(b), allow_move="1", state=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert str(b) in proc.stdout + proc.stderr


def test_returning_to_the_same_root_is_not_a_move(tmp_path, state):
    a = tmp_path / "a"
    a.mkdir()
    run_env_raw(hf_root=str(a), state=state)
    proc = run_env_raw(candidates=str(a), state=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr


def test_a_first_run_with_no_history_just_picks(tmp_path, state):
    a = tmp_path / "a"
    a.mkdir()
    proc = run_env_raw(candidates=str(a), state=state)
    assert proc.returncode == 0, proc.stdout + proc.stderr
