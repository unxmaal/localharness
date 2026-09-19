"""A GGUF needs llama.cpp, and nothing probed for it. Issue #228.

30 proposals in the store name GGUF. On a machine serving the text lane through
mlx_lm.server they cannot load at all, and until this probe existed the ladder
had no way to say so: a GGUF fetched, screened against a server that cannot
read it, and was recorded `broken` -- our gap, written down as the candidate's.

THE MACHINE IS PINNED IN EVERY VERDICT TEST. decide() asks the real machine
through machine.detect(), so a fixture that does not pin one tests the runner.
RULE #249, and RULE #267 for the same reason one layer up.
"""
import pytest

from harness import inspect as ins
from harness import machine as mach
from harness.memory import Accelerator

APPLE = mach.Machine(frozenset({"mlx", "cpu"}),
                     Accelerator("unified", 32.0, 25.0, "Mac16,1"))
WITH_LLAMACPP = mach.Machine(frozenset({"mlx", "cpu", "llamacpp"}),
                             Accelerator("unified", 32.0, 25.0, "Mac16,1"))


@pytest.mark.parametrize("model_id,data", [
    ("cturan/Olmo-3-7B-Instruct-Q1_0", {"siblings": [{"rfilename": "m.gguf"}]}),
    ("prism-ml/Bonsai-8B-gguf", {}),
    ("IFM/K2-Horizon-7B-GGUF", {}),
    ("org/x", {"tags": ["gguf"]}),
    ("org/x", {"library_name": "gguf"}),
])
def test_the_card_or_the_name_says_it_is_gguf(model_id, data):
    assert ins.is_gguf(model_id, data)


@pytest.mark.parametrize("model_id,data", [
    ("org/ggufmaker-tools", {}),
    ("mlx-community/Qwen3-4B-Instruct-2507-4bit", {"tags": ["mlx"]}),
    ("org/regguffing", {}),
    ("", {}),
])
def test_a_name_merely_containing_the_letters_is_not_gguf(model_id, data):
    """The negative control. A substring match calls `ggufmaker-tools` a GGUF
    repo and refuses a perfectly runnable candidate."""
    assert not ins.is_gguf(model_id, data)


def test_a_gguf_is_refused_by_name_on_a_machine_without_llamacpp():
    fit = ins.Fit(repo="org/x", gguf=True)
    got = ins.decide(fit, machine=APPLE)
    assert got.verdict == "needs-llamacpp"
    assert "GGUF" in got.why


def test_the_same_gguf_is_not_refused_where_llamacpp_exists():
    """The verdict names the RUNTIME, not the format: the Linux machine builds
    llama.cpp and serves GGUF, so the same candidate is runnable there."""
    fit = ins.Fit(repo="org/x", gguf=True)
    assert ins.decide(fit, machine=WITH_LLAMACPP).verdict != "needs-llamacpp"


def test_an_mlx_model_is_untouched_by_any_of_this():
    """The negative control that matters: this must not start refusing the
    candidates the machine actually runs."""
    fit = ins.Fit(repo="org/x", mlx=True)
    assert ins.decide(fit, machine=APPLE).verdict != "needs-llamacpp"


def test_needs_llamacpp_is_a_verdict_the_store_will_accept():
    """verdicts() derives the needs-* half from the probe table, so adding a
    probe must extend it rather than produce an unknown verdict."""
    assert "needs-llamacpp" in ins.verdicts()


def test_the_probe_reads_the_variable_the_runbook_sets(monkeypatch, tmp_path):
    binary = tmp_path / "llama-server"
    binary.write_text("", encoding="utf-8")
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setenv("LLAMACPP_BIN", str(binary))
    assert mach._has_llamacpp()


def test_the_probe_says_no_when_nothing_is_installed(monkeypatch):
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.delenv("LLAMACPP_BIN", raising=False)
    assert not mach._has_llamacpp()


def test_a_named_binary_that_does_not_exist_is_not_a_runtime(monkeypatch):
    """A stale LLAMACPP_BIN from another machine must not claim the runtime."""
    monkeypatch.setattr("shutil.which", lambda n: None)
    monkeypatch.setenv("LLAMACPP_BIN", "/nowhere/llama-server")
    assert not mach._has_llamacpp()


@pytest.mark.parametrize("model_id", [
    "cturan/Olmo-3-7B-Instruct-Q1_0",
    "bartowski/Something-Q4_K_M",
    "org/model-IQ2_XS",
])
def test_a_llamacpp_quantisation_label_says_gguf(model_id):
    """These name no format and are GGUF. Across the whole store the pattern
    matches exactly one repo, and that repo says so no other way."""
    assert ins.is_gguf(model_id, {})


@pytest.mark.parametrize("model_id", [
    "mlx-community/Qwen3-4B-Instruct-2507-4bit",
    "mlx-community/Kokoro-82M-bf16",
    "Qwen/Qwen3-8B",
    "org/q4-experiments",
])
def test_an_mlx_quantisation_is_not_a_llamacpp_one(model_id):
    """The negative control. MLX spells its quantisations without the
    underscore, and refusing them would empty this machine's own queue."""
    assert not ins.is_gguf(model_id, {})


def test_prose_mentioning_gguf_does_not_make_a_tool_a_gguf():
    """unslothai/unsloth SUPPORTS GGUF export. Refusing a tool for the formats
    it can write is the opposite of the question being asked."""
    card = {"description": "Local UI to run and train LLMs. Supports GGUF."}
    assert not ins.is_gguf("unslothai/unsloth", card)
