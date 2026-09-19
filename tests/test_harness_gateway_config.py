"""One question, one answer: where is the gateway config, and how is it read.

Gauntlet entry #2, and the instance was added while fixing another instance of
the same class. Five modules resolved the path independently with the same
expression copied five times. They agreed; nothing made them.
"""
import re
from pathlib import Path

import pytest

from harness import gateway

HARNESS = Path(__file__).resolve().parents[1] / "harness"
CONFIGS = [Path(__file__).resolve().parents[1] / "gateway" / n
           for n in ("config.yaml", "config.cuda.yaml")]

#: Modules that still carry their own copy, with nothing that would notice them
#: drifting. An INVENTORY, not a gate: these predate gateway.py and converting
#: them is not this change's job. The count may only go DOWN.
UNCONVERTED = {"coverage.py", "discover.py", "rank.py"}


def _own_resolvers() -> set[str]:
    """Modules naming GATEWAY_CONFIG without asking harness.gateway."""
    out = set()
    for f in sorted(HARNESS.glob("*.py")):
        if f.name in ("gateway.py",):
            continue
        text = f.read_text(encoding="utf-8")
        if "GATEWAY_CONFIG" in text and "gateway.config_path" not in text \
                and "gateway.load" not in text:
            out.add(f.name)
    return out


def test_the_copies_are_an_inventory_that_only_shrinks():
    """Red-proofed by construction: add a sixth copy and this fails."""
    assert _own_resolvers() <= UNCONVERTED, (
        f"new module(s) resolving GATEWAY_CONFIG by hand: "
        f"{sorted(_own_resolvers() - UNCONVERTED)}. Ask harness.gateway.")


def test_the_converted_modules_stay_converted():
    """screen.py was converted; if it regrows a copy, say so."""
    assert "screen.py" not in _own_resolvers()


def test_every_copy_resolves_to_the_same_path():
    """The reason the copies were tolerable, asserted rather than assumed."""
    pattern = re.compile(r'"gateway"\s*/\s*"config\.yaml"')
    for name in sorted(UNCONVERTED):
        text = (HARNESS / name).read_text(encoding="utf-8")
        assert pattern.search(text), (
            f"{name} resolves GATEWAY_CONFIG to something other than "
            f"gateway/config.yaml; it and harness.gateway.DEFAULT disagree")


def test_the_default_is_a_file_a_clone_gets():
    """Gauntlet #6. A config only the author has would make every gateway
    test pass here and nowhere else."""
    import subprocess

    tracked = subprocess.run(
        ["git", "ls-files", "gateway"], cwd=HARNESS.parent,
        capture_output=True, text=True, check=True).stdout.split()
    assert "gateway/config.yaml" in tracked


# --- gauntlet #11: external state with no assertion on it ------------------

def test_every_provider_prefix_in_the_tracked_configs_is_one_we_strip():
    """LOAD-BEARING FOR: resolving a lane's incumbent alias to the repo id
    mlx_lm.server can serve. An unrecognised prefix does not raise -- it
    yields `someprovider/org/model`, which the server treats as a repo id,
    fails to find, and reports as the candidate's failure.

    Either add the provider to gateway.PROVIDER_PREFIXES, or the config should
    not name it.
    """
    import yaml

    seen = set()
    for cfg in CONFIGS:
        if not cfg.exists():
            continue
        data = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
        for entry in data.get("model_list") or []:
            model = str((entry.get("litellm_params") or {}).get("model", ""))
            if "/" in model:
                seen.add(model.split("/", 1)[0] + "/")
    unknown = seen - set(gateway.PROVIDER_PREFIXES)
    assert not unknown, (
        f"{sorted(unknown)} appears in a tracked gateway config and "
        f"gateway.strip_provider does not know it, so an alias using it "
        f"resolves to a model id nothing can serve")


def test_strip_provider_leaves_a_bare_repo_id_alone():
    """The negative control. Stripping a repo id would rename a candidate."""
    assert gateway.strip_provider("mlx-community/Qwen3-4B") == \
        "mlx-community/Qwen3-4B"
    assert gateway.strip_provider("") == ""


@pytest.mark.parametrize("prefix", gateway.PROVIDER_PREFIXES)
def test_each_known_prefix_is_actually_stripped(prefix):
    assert gateway.strip_provider(f"{prefix}org/model") == "org/model"


def test_an_unreadable_config_means_nothing_is_served(tmp_path):
    """Fails OPEN, and the test says so: a fresh clone with no config must not
    take down every tier that asks what this machine serves."""
    assert gateway.load(tmp_path / "absent.yaml") == {}
