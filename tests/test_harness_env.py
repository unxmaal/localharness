"""Where the weights live, resolved by the CLI itself.

`scripts/env.sh` has always done this for the SERVICES, but it is sourced by a
shell. Once `lh` is on PATH it runs from any directory, in any shell, with
nothing sourced -- and `lh image` spawns mflux, which reads HF_HOME. Unset, it
falls back to ~/.cache/huggingface and starts re-downloading weights that are
already on the volume.

ONE LOCATION, CHECKED. There is no candidate list any more: a search is how the
wrong disk gets chosen quietly, and the list was one person's Mac written into
the repo, forked once for Windows here and again in shell.
"""
import os
import sys

import pytest

from harness import env


def big(tmp_path, name):
    """A directory that passes every check."""
    d = tmp_path / name
    d.mkdir()
    return str(d)


# ---- the default -----------------------------------------------------------

def test_the_default_is_in_the_checkout():
    assert env.default_root().endswith(env.DEFAULT_ROOT)
    assert os.path.isabs(env.default_root())


def test_the_default_does_not_follow_the_working_directory(tmp_path, monkeypatch):
    """Relative would scatter caches into whatever directory `lh` ran in."""
    monkeypatch.chdir(tmp_path)
    assert env.default_root() == env.default_root()
    assert not env.default_root().startswith(str(tmp_path))


def test_the_environment_overrides_the_default(tmp_path, monkeypatch):
    monkeypatch.setenv(env.ROOT_VAR, str(tmp_path / "elsewhere"))
    assert env.configured() == str(tmp_path / "elsewhere")


# ---- what apply() does -----------------------------------------------------

def test_an_explicit_hf_home_is_respected(tmp_path):
    environ = {"HF_HOME": "/somewhere/deliberate"}
    assert env.apply(environ, root=big(tmp_path, "a")) == "/somewhere/deliberate"
    assert environ["HF_HOME"] == "/somewhere/deliberate"


def test_the_configured_root_is_used(tmp_path):
    good = big(tmp_path, "good")
    environ = {}
    assert env.apply(environ, root=good, min_free_gb=0) == good
    assert environ["HF_HOME"] == good


def test_hf_root_in_the_environment_is_read(tmp_path):
    good = big(tmp_path, "good")
    environ = {env.ROOT_VAR: good}
    assert env.apply(environ, min_free_gb=0) == good


def test_a_root_without_room_returns_none_rather_than_guessing(tmp_path):
    environ = {}
    assert env.apply(environ, root=big(tmp_path, "small"),
                     min_free_gb=10_000_000) is None
    assert "HF_HOME" not in environ


def test_a_root_that_does_not_exist_yet_is_fine_if_its_volume_has_room(tmp_path):
    """env.sh creates the directory once a location is chosen, so the target
    not being there yet is the normal first run rather than a failure."""
    assert env.apply({}, root=str(tmp_path / "not-yet"), min_free_gb=0) \
        == str(tmp_path / "not-yet")


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="models macOS TCC: chmod 000 does not remove read access on "
           "Windows, where the equivalent is an ACL")
def test_an_unreadable_root_is_refused(tmp_path):
    """macOS TCC: a volume can stat fine, report free space and appear in
    /Volumes while listing it raises. mlx_lm then hangs forever inside
    os.listdir with no log and 0% CPU. See RULE #184 and #192."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o000)
    try:
        assert env.apply({}, root=str(blocked), min_free_gb=0) is None
    finally:
        os.chmod(blocked, 0o755)


def test_offline_is_the_default_because_a_wifi_blip_reads_as_a_bad_model(tmp_path):
    """mlx_lm.server issues a HEAD to huggingface.co on every model switch even
    for local files."""
    environ = {}
    env.apply(environ, root=big(tmp_path, "a"), min_free_gb=0)
    assert environ["HF_HUB_OFFLINE"] == "1"


def test_an_explicit_offline_setting_is_not_overridden(tmp_path):
    environ = {"HF_HUB_OFFLINE": "0"}
    env.apply(environ, root=big(tmp_path, "a"), min_free_gb=0)
    assert environ["HF_HUB_OFFLINE"] == "0"


# ---- the two implementations must not drift --------------------------------

def test_the_threshold_matches_the_one_env_sh_uses():
    """Two answers that disagree would put the CLI's weights somewhere the
    services do not look, and the download would be silent."""
    text = open("scripts/env.sh", encoding="utf-8").read()
    assert f"HF_MIN_FREE_GB:-{env.HF_MIN_FREE_GB}" in text


def test_the_default_directory_name_matches_env_sh():
    text = open("scripts/env.sh", encoding="utf-8").read()
    assert f"/{env.DEFAULT_ROOT}" in text
