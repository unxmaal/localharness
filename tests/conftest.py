"""Suite-wide isolation.

A test that writes to the developer's real LOCALHARNESS_HOME is invisible on
CI, where that directory does not exist, and visible only as junk accumulating
on the one machine nobody re-clones. Issue #188 found two such keys, `s` and
`blocked`, sitting in a real discovery-state.json for ten days.
"""
import pytest

from harness import paths


@pytest.fixture(autouse=True)
def _home(tmp_path_factory, monkeypatch):
    """Every test gets its own LOCALHARNESS_HOME.

    Autouse and unconditional: an opt-in guard protects the tests that
    remembered, which are never the ones that leak.
    """
    home = tmp_path_factory.mktemp("lh-home")
    monkeypatch.setenv(paths.ENV_VAR, str(home))
    return home
