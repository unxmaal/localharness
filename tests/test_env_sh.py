"""Behaviour of scripts/env.sh, the guard on where model weights land.

This script exists to stop huggingface_hub silently recreating its cache
somewhere with no room, re-downloading tens of GB with no error.

NOTHING HERE NEEDS A MOUNTED VOLUME. It used to: every test was skipped unless
/Volumes/FAST was attached, so all 29 skipped on both CI runners and ran only
on one Mac. Ten of them then failed there for a week while CI stayed green,
because a skip and a pass look the same at a glance. A test CI cannot execute
is a test CI cannot defend, so these use tmp_path and the free-space threshold
is passed in rather than depending on what any particular disk has today.
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import shells

REPO = Path(__file__).resolve().parents[1]


def same_dir(a: str, b) -> bool:
    """Whether two spellings name the same directory.

    Git Bash reports /d/a/... where Python reports D:\\a\\..., so a string
    compare fails on Windows for two paths that are the same place. env.sh
    converts with cygpath before exporting, precisely because huggingface_hub
    is native Python and cannot read the MSYS form -- but a test that compared
    strings would still be asserting the spelling rather than the location.
    """
    if not a:
        return False
    try:
        return Path(a).resolve() == Path(b).resolve()
    except OSError:
        return False


def run_env(hf_root=None, shell="bash", min_free_gb="0", cwd=None):
    """Source env.sh in a clean shell; return (exit_code, HF_HOME, stderr)."""
    exe = shells.resolve(shell)
    if exe is None:
        pytest.skip(f"no {shell} on this machine")
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_MIN_FREE_GB": str(min_free_gb)}
    if hf_root is not None:
        env["HF_ROOT"] = hf_root
    p = subprocess.run(
        # `.` and not `source`: /bin/sh on Ubuntu is dash, which has no such
        # builtin and exits 127. macOS ships bash as sh and accepts both, which
        # is why this passed on two runners and not the third.
        [exe, "-c", f'. "{REPO}/scripts/env.sh" && echo "HF_HOME=$HF_HOME"'],
        capture_output=True, text=True, env=env, cwd=cwd)
    home = ""
    for line in p.stdout.splitlines():
        if line.startswith("HF_HOME="):
            home = line.split("=", 1)[1]
    return p.returncode, home, p.stderr


# ---- the default -----------------------------------------------------------

def test_the_default_is_in_the_checkout(tmp_path):
    """No candidate list, no mount, no drive letter. The same answer on the
    mini, the Windows box and a CI runner."""
    code, home, err = run_env()
    assert code == 0, err
    assert same_dir(home, REPO / "hf_root")


def test_the_default_does_not_follow_the_working_directory(tmp_path):
    """A relative default would scatter caches into whatever directory `lh`
    happened to be run from, which is the bug harness/paths.py remembers."""
    code, home, err = run_env(cwd=str(tmp_path))
    assert code == 0, err
    assert same_dir(home, REPO / "hf_root")


@pytest.mark.parametrize("shell", ["bash", "zsh", "sh"])
def test_the_default_is_the_same_in_every_shell(tmp_path, shell):
    """env.sh is SOURCED, so it runs in whatever shell the caller has. $0 is
    the shell's own name under dash and gives no path at all, BASH_SOURCE is
    bash-only and %x is zsh-only, so finding this file needs all three routes
    and a fallback."""
    code, home, err = run_env(shell=shell)
    assert code == 0, err
    assert same_dir(home, REPO / "hf_root")


# ---- an explicit location --------------------------------------------------

def test_an_explicit_root_is_accepted(tmp_path):
    code, home, err = run_env(hf_root=str(tmp_path / "hf"))
    assert code == 0, err
    assert same_dir(home, tmp_path / "hf")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="chmod 500 does not remove write access on Windows, where the "
           "equivalent is an ACL")
def test_an_explicit_root_is_still_checked(tmp_path):
    """"The caller says where" must not become "and stop looking". Skipping the
    checks for an explicit HF_ROOT was a real bug: an unwritable and a
    nonexistent path both passed with exit 0."""
    unwritable = tmp_path / "locked"
    unwritable.mkdir()
    subprocess.run(["chmod", "500", str(unwritable)], check=True)
    try:
        code, _, err = run_env(hf_root=str(unwritable / "hf"))
        assert code == 1
        assert "FATAL" in err
    finally:
        subprocess.run(["chmod", "700", str(unwritable)], check=True)


def test_a_root_without_room_is_refused_and_the_message_says_how_much(tmp_path):
    code, _, err = run_env(hf_root=str(tmp_path / "hf"), min_free_gb="999999")
    assert code == 1
    assert "999999" in err
    assert "free" in err.lower()


def test_the_message_names_the_knob_and_this_machine(tmp_path):
    """Whoever hits this needs to be told what to set, not just that it
    failed."""
    _, _, err = run_env(hf_root=str(tmp_path / "hf"), min_free_gb="999999")
    assert "HF_ROOT" in err
    assert "hf_root" in err


def test_nothing_is_created_until_a_location_is_chosen(tmp_path):
    rejected = tmp_path / "rejected"
    run_env(hf_root=str(rejected / "hf"), min_free_gb="999999")
    assert not rejected.exists(), "a refused location was created anyway"


def test_hf_home_is_exported_to_children(tmp_path):
    """`lh` spawns mflux, which reads HF_HOME from its own environment. Set but
    not exported is the same as unset one process down."""
    exe = shells.resolve("bash")
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_MIN_FREE_GB": "0", "HF_ROOT": str(tmp_path / "hf"),
           "HF_STATE_FILE": str(tmp_path / "state")}
    # `env` rather than a nested shell: interpolating the interpreter's path
    # into a command string ran "C:Program" on Windows, where bash lives under
    # Program Files. What is being tested is that the variable crosses a
    # process boundary at all, and any child will do.
    p = subprocess.run(
        [exe, "-c",
         f'. "{REPO}/scripts/env.sh" >/dev/null 2>&1 && '
         f'env | grep "^HF_HOME="'],
        capture_output=True, text=True, env=env)
    exported = ""
    for line in p.stdout.splitlines():
        if line.startswith("HF_HOME="):
            exported = line.split("=", 1)[1]
    assert same_dir(exported, tmp_path / "hf"), p.stdout


# ---- the threshold ---------------------------------------------------------

def test_the_default_threshold_does_not_veto_an_ordinary_machine():
    """It was 160GB, for MiniMax-H3's one 134GiB checkpoint, which made every
    ordinary disk unusable -- the in-tree default and every CI runner included
    -- to guard a download that guards itself."""
    text = (REPO / "scripts" / "env.sh").read_text(encoding="utf-8")
    import re
    m = re.search(r"HF_MIN_FREE_GB:-(\d+)", text)
    assert m, "env.sh has no default free-space threshold"
    assert 0 < int(m.group(1)) <= 50, (
        f"{m.group(1)}GB is high enough to refuse a normal machine")


def test_the_big_checkpoint_still_guards_its_own_headroom():
    """The 160GB number moved to the script that knows it needs it."""
    text = (REPO / "scripts" / "fetch-h3-weights.sh").read_text(encoding="utf-8")
    assert "160" in text


def test_the_python_and_shell_defaults_agree():
    """Two answers that disagree would put the CLI's weights somewhere the
    services do not look, and the download would be silent."""
    from harness import env as pyenv
    code, home, err = run_env()
    assert code == 0, err
    assert same_dir(home, pyenv.default_root())


# ---- readability, not just existence ---------------------------------------
#
# A launchd agent gets "Operation not permitted" on /Volumes/FAST: macOS TCC
# protects removable volumes and a background job has no way to ask for
# consent. The volume stats fine and appears in /Volumes, so every check this
# script had said it was usable -- and mlx_lm then hung forever inside
# os.listdir rather than reporting an error. See RULE #184 and #192.

def test_a_directory_that_cannot_be_listed_is_fatal():
    text = (REPO / "scripts" / "env.sh").read_text(encoding="utf-8")
    assert "_hf_readable" in text or "listing" in text.lower(), (
        "env.sh never tries to READ the directory it selects")


def test_the_fatal_message_explains_the_tcc_case():
    text = (REPO / "scripts" / "env.sh").read_text(encoding="utf-8")
    assert "Full Disk Access" in text
    assert "launchd" in text.lower() or "background" in text.lower()


def test_helper_functions_print_nothing_but_their_answer(tmp_path):
    """zsh's `local name`, with no assignment, PRINTS the parameter when it
    already has a value. These helpers are read through command substitution,
    so that noise becomes the answer. bash and sh are silent, which is why it
    survived until someone sourced env.sh from a zsh login shell."""
    if not shutil.which("zsh"):
        pytest.skip("zsh not installed")
    probe = (f'. "{REPO}/scripts/env.sh" >/dev/null 2>&1; '
             f'echo "MP:[$(_hf_mountpoint {tmp_path}/deep/path)]"; '
             f'echo "GB:[$(_hf_free_gb {tmp_path}/deep/path)]"')
    env = {"PATH": os.environ["PATH"], "HOME": os.environ["HOME"],
           "HF_MIN_FREE_GB": "0", "HF_ROOT": str(tmp_path / "hf"),
           "HF_STATE_FILE": str(tmp_path / "state")}
    p = subprocess.run(["zsh", "-c", probe], capture_output=True, text=True,
                       env=env)
    for line in p.stdout.splitlines():
        assert "=" not in line.split(":", 1)[1], (
            f"a helper leaked a variable declaration into its output: {line}")


# ---- what replaced the cache-moved guard -----------------------------------
#
# A reboot once brought this machine back without the weights volume. The
# candidate loop skipped the missing /Volumes/FAST, found /Volumes/PORTABLE with
# room, and the services started against an EMPTY CACHE, silently. A state file
# then remembered where we landed and refused to move without HF_ALLOW_MOVE=1.
#
# That guard was aimed at the SEARCH. With one configured location the failure
# is structural instead: an unreachable HF_ROOT is fatal, so there is nowhere
# to fall back TO. These are the tests for that property.

def test_an_unreachable_root_is_fatal_rather_than_falling_back(tmp_path):
    code, home, err = run_env(hf_root="/Volumes/NO_SUCH_VOLUME/hf")
    assert code == 1
    assert home == ""
    assert "FATAL" in err


def test_it_does_not_quietly_use_the_default_when_hf_root_is_bad(tmp_path):
    """The whole failure was starting somewhere else without saying so."""
    _, home, _ = run_env(hf_root="/Volumes/NO_SUCH_VOLUME/hf")
    assert not same_dir(home, REPO / "hf_root")
