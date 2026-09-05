"""Behaviour of scripts/env.sh, the guard on where model weights land.

This script exists to stop huggingface_hub silently recreating its cache on the
internal disk, which is at 91% and cannot hold a 134GB checkpoint. Every case
here is a way that guard could fail open.
"""
import os
import subprocess
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MOUNTED = Path("/Volumes/Models")

pytestmark = pytest.mark.skipif(
    not MOUNTED.is_dir(),
    reason="needs the Models volume mounted to exercise the real mount check")


def run_env(hf_root=None, candidates=None):
    """Source env.sh in a clean shell; return (exit_code, HF_HOME, stderr)."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    if hf_root is not None:
        env["HF_ROOT"] = hf_root
    if candidates is not None:
        env["HF_CANDIDATES"] = candidates
    p = subprocess.run(
        ["bash", "-c", f'source "{REPO}/scripts/env.sh" && echo "HF_HOME=$HF_HOME"'],
        capture_output=True, text=True, env=env)
    home = ""
    for line in p.stdout.splitlines():
        if line.startswith("HF_HOME="):
            home = line.split("=", 1)[1]
    return p.returncode, home, p.stderr


@pytest.fixture
def scratch():
    d = MOUNTED / f".test-{uuid.uuid4().hex[:8]}"
    yield d
    if d.exists():
        subprocess.run(["chmod", "-R", "u+w", str(d)], capture_output=True)
        subprocess.run(["rm", "-rf", str(d)], capture_output=True)


# ---- explicit HF_ROOT ------------------------------------------------------

def test_explicit_usable_path_is_accepted(scratch):
    code, home, _ = run_env(hf_root=str(scratch / "hf"))
    assert code == 0
    assert home == str(scratch / "hf")


def test_explicit_unmounted_path_is_fatal():
    """An absent volume is refused.

    Note this is caught by the internal-disk check rather than by mount
    resolution: /Volumes/NoSuchVolume/hf walks up to /Volumes, which lives on
    "/". Mutation testing showed the mountpoint guard alone is redundant here.
    The behaviour is what matters, so both guards stay.
    """
    code, home, err = run_env(hf_root="/Volumes/NoSuchVolume/hf")
    assert code == 1
    assert home == ""
    assert "FATAL" in err


def test_explicit_path_not_on_a_mount_is_fatal(tmp_path):
    """A path on the internal disk must be refused even though it is writable.

    This is the actual failure being guarded: a plausible-looking local path
    that would quietly absorb 134GB the volume does not have.
    """
    code, _, err = run_env(hf_root=str(tmp_path / "hf"))
    assert code == 1
    assert "FATAL" in err


def test_explicit_unwritable_path_is_fatal(scratch):
    scratch.mkdir(parents=True)
    (scratch / "hf").mkdir()
    subprocess.run(["chmod", "500", str(scratch / "hf")], check=True)
    code, _, err = run_env(hf_root=str(scratch / "hf"))
    assert code == 1
    assert "FATAL" in err


# ---- auto-pick -------------------------------------------------------------

def test_autopick_takes_the_first_usable_candidate(scratch):
    good = scratch / "hf"
    code, home, _ = run_env(
        candidates=f"/Volumes/NoSuchVolume/hf:{good}")
    assert code == 0
    assert home == str(good)


def test_autopick_with_no_usable_candidate_is_fatal():
    code, _, err = run_env(candidates="/Volumes/NoSuchVolume/hf:/Volumes/AlsoNo/hf")
    assert code == 1
    assert "FATAL" in err


def test_autopick_does_not_create_dirs_on_rejected_candidates(scratch, tmp_path):
    """A predicate must not have side effects.

    The rejected candidate is FIRST and on the internal disk, so it is actually
    probed and actually refused. An earlier version of this test put the winner
    first, which meant the loser was never probed and the assertion could not
    fail; mutation testing caught that.
    """
    rejected = tmp_path / "internal" / "hf"
    winner = scratch / "hf"
    code, home, _ = run_env(candidates=f"{rejected}:{winner}")
    assert code == 0
    assert home == str(winner)
    assert not rejected.exists(), "probing created a directory on a rejected path"


# ---- exported, not merely set ---------------------------------------------

def test_hf_home_is_exported_to_children(scratch):
    """Child processes must see it; huggingface_hub reads it from the env."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_ROOT": str(scratch / "hf")}
    p = subprocess.run(
        ["bash", "-c",
         f'source "{REPO}/scripts/env.sh" && python3 -c '
         f'"import os; print(os.environ[\'HF_HOME\'])"'],
        capture_output=True, text=True, env=env)
    assert p.returncode == 0
    assert str(scratch / "hf") in p.stdout
