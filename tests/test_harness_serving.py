"""Issue #190: which server produced the tokens is part of the exam."""
import re
from pathlib import Path

from evals.core import Receipt, comparable
from harness import serving

REPO = Path(__file__).parent.parent


def receipt(**kw):
    base = dict(modality="code", case_ids=["a"], tier="measure",
                repeat=1, sampling={}, gateway="http://gw",
                instruments={"serving": "mlx_lm.server"})
    base.update(kw)
    return Receipt(**base)


def test_the_default_is_what_the_launcher_actually_starts():
    """RULE #237, sixth instance. The engine name lived only in a shell line;
    a second copy in Python that nothing compares is how that pair drifts."""
    text = (REPO / "scripts" / "serve-mlx.sh").read_text(encoding="utf-8")
    started = re.search(r"exec\s+uv\s+run\s+(\S+)", text)
    assert started, "serve-mlx.sh no longer execs a recognisable server"
    assert started.group(1) == serving.DEFAULT


def test_an_unset_variable_gives_the_default():
    assert serving.text_engine({}) == serving.DEFAULT


def test_an_empty_variable_gives_the_default():
    """An exported-but-empty variable is how a shell passes 'unset'."""
    assert serving.text_engine({serving.ENV_VAR: ""}) == serving.DEFAULT
    assert serving.text_engine({serving.ENV_VAR: "   "}) == serving.DEFAULT


def test_a_config_line_names_a_different_engine():
    """The whole claim under test: a new method costs a config line."""
    assert serving.text_engine({serving.ENV_VAR: "vllm-mlx"}) == "vllm-mlx"


def test_two_engines_are_not_one_table():
    """The point of recording it. Before this, swapping the engine left the
    receipt byte-identical and comparable() said 'same exam'."""
    ok, why = comparable(receipt(),
                         receipt(instruments={"serving": "vllm-mlx"}))
    assert not ok
    assert "serving" in why or "vllm-mlx" in why


def test_the_same_engine_still_compares():
    """The negative half. An axis that refuses everything ranks nothing."""
    ok, _ = comparable(receipt(), receipt())
    assert ok


def test_a_receipt_without_the_key_does_not_block_an_old_one():
    """Receipts written before #190 carry no `serving`. comparable() compares
    only keys both runs have, so history stays rankable rather than being
    invalidated by a field that did not exist when it was recorded."""
    ok, _ = comparable(receipt(instruments={}), receipt())
    assert ok
