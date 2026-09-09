"""Where the weights live, resolved by the CLI itself.

`scripts/env.sh` has always done this for the SERVICES, but it is sourced by a
shell. Once `lh` is on PATH it runs from any directory, in any shell, with
nothing sourced -- and `lh image` spawns mflux, which reads HF_HOME. Unset, it
falls back to ~/.cache/huggingface and starts re-downloading weights that are
already on the volume.

The rule mirrors env.sh deliberately: free space, not "is this external", so it
refuses this mini's 91%-full internal disk for the reason that actually matters
and will not refuse the Studio's.
"""
import sys
import os

import pytest

from harness import env


def big(tmp_path, name):
    """A candidate directory that passes every check."""
    d = tmp_path / name
    d.mkdir()
    return str(d)


def test_an_explicit_hf_home_is_respected(tmp_path):
    environ = {"HF_HOME": "/somewhere/deliberate"}
    assert env.apply(environ, candidates=[big(tmp_path, "a")]) == "/somewhere/deliberate"
    assert environ["HF_HOME"] == "/somewhere/deliberate"


def test_the_first_usable_candidate_wins(tmp_path):
    first, second = big(tmp_path, "first"), big(tmp_path, "second")
    environ = {}
    assert env.apply(environ, candidates=[first, second], min_free_gb=0) == first
    assert environ["HF_HOME"] == first


def test_a_candidate_that_does_not_exist_is_skipped(tmp_path):
    good = big(tmp_path, "good")
    got = env.apply({}, candidates=["/no/such/volume/hf", good], min_free_gb=0)
    assert got == good


def test_a_candidate_without_room_is_skipped(tmp_path):
    """The 134 GiB video checkpoint is why this rule exists at all."""
    got = env.apply({}, candidates=[big(tmp_path, "small")],
                    min_free_gb=10_000_000)
    assert got is None


@pytest.mark.skipif(
    sys.platform == "win32",
    reason="models macOS TCC: chmod 000 does not remove read access on "
           "Windows, where the equivalent is an ACL")
def test_an_unreadable_candidate_is_skipped(tmp_path):
    """macOS TCC: a volume can stat fine, report free space and appear in
    /Volumes while listing it raises. mlx_lm then hangs forever inside
    os.listdir with no log and 0% CPU."""
    blocked = tmp_path / "blocked"
    blocked.mkdir()
    os.chmod(blocked, 0o000)
    try:
        got = env.apply({}, candidates=[str(blocked), big(tmp_path, "ok")],
                        min_free_gb=0)
        assert got == str(tmp_path / "ok")
    finally:
        os.chmod(blocked, 0o755)


def test_nothing_usable_returns_none_rather_than_guessing(tmp_path):
    environ = {}
    assert env.apply(environ, candidates=["/no/such/a", "/no/such/b"]) is None
    assert "HF_HOME" not in environ


def test_offline_is_the_default_because_a_wifi_blip_reads_as_a_bad_model(tmp_path):
    """mlx_lm.server issues a HEAD to huggingface.co on every model switch even
    for local files."""
    environ = {}
    env.apply(environ, candidates=[big(tmp_path, "a")], min_free_gb=0)
    assert environ["HF_HUB_OFFLINE"] == "1"


def test_an_explicit_offline_setting_is_not_overridden(tmp_path):
    environ = {"HF_HUB_OFFLINE": "0"}
    env.apply(environ, candidates=[big(tmp_path, "a")], min_free_gb=0)
    assert environ["HF_HUB_OFFLINE"] == "0"


def test_the_shipped_candidates_match_the_ones_env_sh_searches():
    """Two lists that disagree would put the CLI's weights somewhere the
    services do not look, and the download would be silent."""
    text = open("scripts/env.sh", encoding="utf-8").read()
    for cand in env.HF_CANDIDATES[:-1]:
        assert cand in text, f"{cand} is not in env.sh's search order"
    assert str(env.HF_MIN_FREE_GB) in text
