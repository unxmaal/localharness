"""One question must not have two answers.

THE VIOLATION: a policy decided once and written down twice, with nothing
holding the copies together. Both are right the day they are written and they
drift the first time only one is edited.

It has happened six times here, which is why this file exists as a CENSUS
rather than a seventh bespoke test. Nobody could see the pattern until it had
already happened five times, each fix filed as its own unrelated bug.

The registry below is the artifact. A seventh instance costs a row.
"""
import ast
import re
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parent.parent
GATEWAYS = ("gateway/config.yaml", "gateway/config.cuda.yaml")


#: what the policy is | where it lives twice | issue | the test that holds them
REGISTRY = [
    ("which audio player to try, and in what order",
     ("harness/audio.py", "scripts/preflight.sh"), "#153",
     "tests/test_preflight_sh.py"),
    ("where weights go when HF_HOME is unset",
     ("harness/env.py", "scripts/env.sh"), "#108",
     "tests/test_env_sh.py"),
    ("whether an inherited HF_HOME is respected",
     ("harness/env.py", "scripts/env.sh"), "#140",
     "tests/test_harness_env.py"),
    ("how a Windows path is spelled for native Python",
     ("harness/env.py", "scripts/env.sh"), "#223",
     "tests/test_env_sh.py"),
    ("what resolution the image lane runs at",
     ("harness/cli.py", "evals/cases/image"), "#157",
     "tests/test_cli_resolution.py"),
    ("which gateway alias each text lane defaults to",
     ("harness/cli.py", "gateway/config.yaml"), "#160",
     "tests/test_duplicated_policy.py"),
]


@pytest.mark.parametrize("policy,where,issue,guard", REGISTRY,
                         ids=[r[2] for r in REGISTRY])
def test_every_duplicated_policy_still_has_a_guard(policy, where, issue, guard):
    """Deleting the guard must fail here rather than going unnoticed.

    A registry that merely lists the instances is a document. One that asserts
    each row still has a live test is a check.
    """
    for f in where:
        assert (REPO / f).exists(), f"{policy}: {f} moved, and {issue} was about it"
    assert (REPO / guard).exists(), f"{policy}: the guard named for {issue} is gone"


# ---- the sixth instance, which had no guard until now ---------------------

def _lane_defaults():
    """Read from source rather than imported, so this test does not depend on
    harness's dependencies being installed -- same shape as _players_from_python
    in tests/test_preflight_sh.py."""
    tree = ast.parse((REPO / "harness" / "cli.py").read_text(encoding="utf-8"))
    out = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                name = getattr(t, "id", "")
                if re.fullmatch(r"DEFAULT_(SVG|WEB|CODE|EXTRACT)_MODEL", name):
                    out[name] = ast.literal_eval(node.value)
    assert out, "harness/cli.py no longer names its lane defaults this way"
    return out


@pytest.mark.parametrize("config", GATEWAYS)
def test_every_lane_default_is_an_alias_the_gateway_serves(config):
    """Rename an alias in either gateway config and every text lane breaks at
    runtime, on a machine, with nothing in the suite noticing. The CLI names
    aliases; the config defines them; neither imports the other."""
    served = {m["model_name"] for m in
              yaml.safe_load((REPO / config).read_text(encoding="utf-8"))["model_list"]}
    for name, alias in sorted(_lane_defaults().items()):
        assert alias in served, (
            f"{name} = {alias!r}, which {config} does not serve. "
            f"The lane would fail on a real machine and pass here")


def test_both_gateways_serve_the_same_alias_names():
    """The runtime differs; the VOCABULARY must not. `lh svg` names one alias
    and must reach an implementation whichever machine it runs on."""
    names = [ {m["model_name"] for m in
               yaml.safe_load((REPO / c).read_text(encoding="utf-8"))["model_list"]}
              for c in GATEWAYS ]
    for name in _lane_defaults().values():
        assert all(name in n for n in names), \
            f"{name} is served by one gateway config and not the other"
