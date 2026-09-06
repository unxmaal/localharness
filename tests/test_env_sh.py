"""Behaviour of scripts/env.sh, the guard on where model weights land.

This script exists to stop huggingface_hub silently recreating its cache on the
internal disk, which is at 91% and cannot hold a 134GB checkpoint. Every case
here is a way that guard could fail open.
"""
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
MOUNTED = Path("/Volumes/Models")

pytestmark = pytest.mark.skipif(
    not MOUNTED.is_dir(),
    reason="needs the Models volume mounted to exercise the real mount check")


def run_env(hf_root=None, candidates=None, shell="bash", min_free_gb=None):
    """Source env.sh in a clean shell; return (exit_code, HF_HOME, stderr)."""
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"]}
    if hf_root is not None:
        env["HF_ROOT"] = hf_root
    if candidates is not None:
        env["HF_CANDIDATES"] = candidates
    if min_free_gb is not None:
        env["HF_MIN_FREE_GB"] = min_free_gb
    p = subprocess.run(
        [shell, "-c", f'source "{REPO}/scripts/env.sh" && echo "HF_HOME=$HF_HOME"'],
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


# ---- shell portability -----------------------------------------------------

@pytest.mark.parametrize("shell", ["bash", "zsh", "sh"])
def test_candidate_splitting_works_in_every_shell(scratch, shell):
    """env.sh is SOURCED, so it runs in whatever shell the user has.

    zsh does not word-split unquoted parameters. An IFS-based `for x in $VAR`
    loop silently yields ONE item there, and HF_HOME ends up set to the entire
    colon-joined candidate list. That is a plausible-looking path that does not
    exist, so weights would land somewhere useless.
    """
    if not shutil.which(shell):
        pytest.skip(f"{shell} not installed")
    code, home, _ = run_env(
        candidates=f"/Volumes/NoSuchVolume/hf:{scratch}/hf", shell=shell)
    assert code == 0
    assert home == f"{scratch}/hf", f"{shell} mis-split the candidate list"
    assert ":" not in home


# ---- free space, not "is it the internal disk" -----------------------------
#
# The rule was "refuse / and /System/Volumes/*", which encodes THIS machine:
# an M2 mini whose internal disk is at 91%. The M5 Ultra Studio arriving in
# November has a large internal SSD and may have no external volume attached at
# all, and the old rule would refuse it categorically.
#
# The real requirement was never "external". It was "enough room for a 134GB
# checkpoint", and a threshold says that directly.

def free_gb(path):
    st = os.statvfs(path)
    return st.f_bavail * st.f_frsize / 1e9


def test_the_default_threshold_holds_the_largest_checkpoint():
    """MiniMax-H3 FL2VA alone is 134GiB; fetch-h3-weights.sh wants 160GB of
    headroom. A threshold below that would let the guard pass and the download
    fail."""
    text = (REPO / "scripts" / "env.sh").read_text()
    import re
    m = re.search(r"HF_MIN_FREE_GB:-(\d+)", text)
    assert m, "env.sh has no default free-space threshold"
    assert int(m.group(1)) >= 160


def test_a_volume_without_room_is_refused_and_the_message_says_how_much(scratch):
    """The Models volume has hundreds of GB free, so demanding an absurd amount
    is the only way to exercise the threshold without filling a disk."""
    code, _, err = run_env(hf_root=str(scratch / "hf"),
                           min_free_gb="999999")
    assert code == 1
    assert "999999" in err
    assert "free" in err.lower()


def test_the_internal_disk_is_allowed_when_it_actually_has_room(tmp_path):
    """The Studio case. Nothing about being internal is disqualifying; running
    out of space is."""
    code, home, err = run_env(hf_root=str(tmp_path / "hf"), min_free_gb="0")
    assert code == 0, err
    assert home == str(tmp_path / "hf")


def test_the_internal_disk_is_still_refused_on_this_machine_by_default(tmp_path):
    """This mini's internal disk is at 91%, so the same threshold that lets the
    Studio through still stops the failure this guard was written for."""
    if free_gb("/") > 160:
        pytest.skip("internal disk now has room; the guard would correctly allow it")
    code, _, err = run_env(hf_root=str(tmp_path / "hf"))
    assert code == 1
    assert "FATAL" in err


def test_external_storage_is_preferred_over_a_roomy_internal_disk(scratch, tmp_path):
    """Order still matters: the external NVMe reads at 959 MB/s and load time
    scales with it. The threshold decides what is ALLOWED, not what is chosen.
    """
    internal = tmp_path / "hf"
    code, home, _ = run_env(candidates=f"{scratch}/hf:{internal}",
                            min_free_gb="0")
    assert code == 0
    assert home == f"{scratch}/hf"


def test_a_roomy_internal_disk_is_the_last_resort_not_the_first(scratch):
    """The shipped candidate list must end somewhere that works on a machine
    with no external volume, or the Studio needs hand configuration on day
    one."""
    text = (REPO / "scripts" / "env.sh").read_text()
    import re
    m = re.search(r'HF_CANDIDATES:-([^"]+)"', text)
    assert m, "env.sh has no default candidate list"
    candidates = m.group(1).split(":")
    assert candidates[0].startswith("/Volumes/"), candidates
    assert not candidates[-1].startswith("/Volumes/"), (
        f"the last candidate is still external, so a machine with no external "
        f"volume has no fallback: {candidates}")


def test_helper_functions_print_nothing_but_their_answer(scratch):
    """zsh's `local name`, with no assignment, PRINTS the parameter when it
    already has a value. Re-declaring inside a loop therefore emits
    "parent=/Volumes/..." on every iteration after the first, and since these
    helpers are read through command substitution the noise lands in the
    answer. bash and sh are silent, so this is invisible until someone sources
    env.sh from a zsh login shell -- which is eric's shell.
    """
    if not shutil.which("zsh"):
        pytest.skip("zsh not installed")
    probe = (f'source "{REPO}/scripts/env.sh" >/dev/null 2>&1; '
             f'echo "MP:[$(_hf_mountpoint /Volumes/NoSuchVolume/deep/path)]"; '
             f'echo "GB:[$(_hf_free_gb {scratch}/deep/path)]"')
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_ROOT": str(MOUNTED / "hf")}
    p = subprocess.run(["zsh", "-c", probe], capture_output=True, text=True,
                       env=env)
    for line in p.stdout.splitlines():
        assert "=" not in line.split(":", 1)[1], (
            f"a helper leaked a variable declaration into its output: {line}")


# ---- readability, not just existence ---------------------------------------
#
# A launchd agent gets "Operation not permitted" on /Volumes/Models: macOS TCC
# protects removable volumes and a background job has no way to ask for
# consent. The volume stats fine and appears in /Volumes, so every check this
# script had said it was usable -- and mlx_lm then hung forever inside
# os.listdir rather than reporting an error. A wedged server with no message is
# the worst possible way to learn about a permission.

def test_a_directory_that_cannot_be_listed_is_fatal(scratch, monkeypatch):
    """Existence and writability are not enough; the guard must actually read
    it."""
    text = (REPO / "scripts" / "env.sh").read_text()
    assert "_hf_readable" in text or "ls " in text or "listing" in text.lower(), (
        "env.sh never tries to READ the directory it selects")


def test_the_fatal_message_explains_the_tcc_case():
    """Whoever hits this needs to be told what to click, not just that it
    failed."""
    text = (REPO / "scripts" / "env.sh").read_text()
    assert "Full Disk Access" in text
    assert "launchd" in text.lower() or "background" in text.lower()


def test_a_readable_volume_still_passes(scratch):
    code, home, err = run_env(hf_root=str(scratch / "hf"))
    assert code == 0, err
    assert home == str(scratch / "hf")
