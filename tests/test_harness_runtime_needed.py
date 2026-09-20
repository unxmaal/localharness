"""Can THIS machine run these weights? Issue #245.

`rank.unrunnable` knew exactly one format. Four candidates in the live queue
were tagged `cuda`, `gemlite`, `nvfp4` and `modelopt`, ranked, and would have
been fetched and handed to a runner that cannot load them -- with the screen
then recording a verdict about the CANDIDATE. That is the eighth instance of
a harness-side gap settling a real model.

The answer is a RUNTIME NAME, never a verdict, because it is a fact about the
machine asking. The same gemlite weights are perfectly runnable on a box with
a card, and the store is shared between machines.
"""
import pytest

from harness import inspect as ins
from harness import machine as _machine
from harness import rank
from harness.memory import Accelerator

#: Real cards, copied from the store rather than invented.
GEMLITE = ("task text-to-image; served by diffusers; tagged 1-bit, gemlite, "
           "hqq, cuda, text-to-image, diffusion; built from "
           "prism-ml/bonsai-image-binary-4B-unpack")
NVFP4 = ("task text-generation; served by transformers; tagged gemma4, "
         "text-generation, nvfp4, modelopt, vllm; built from google/gemma-4-31B-it")
VLLM = ("task automatic-speech-recognition; served by vllm; tagged vllm, "
        "voxtral_realtime, mistral-common, fr, es; 16.5 GiB of weights")
MLX = ("task automatic-speech-recognition; served by mlx; tagged mlx, "
       "mistral-common, automatic-speech-recognition, fr, es, de")
TRANSFORMERS = ("task text-generation; served by transformers; tagged qwen2, "
                "text-generation, math, code, reasoning; built from "
                "Qwen/Qwen2.5-Coder-3B")


def mac():
    return _machine.Machine(frozenset({"mlx", "cpu"}),
                            Accelerator("unified", 32.0, 25.0, "Mac14,12"))


def card():
    return _machine.Machine(frozenset({"cuda", "vllm", "cpu"}),
                            Accelerator("discrete", 12.0, 11.0, "RTX 4070"))


@pytest.mark.parametrize("desc,want", [
    (GEMLITE, "cuda"), (NVFP4, "cuda"), (VLLM, "vllm"),
])
def test_a_card_that_names_its_runtime_is_read(desc, want):
    assert ins.runtime_needed(desc) == want


@pytest.mark.parametrize("desc", [MLX, TRANSFORMERS, "", "tagged qwen3, moe"])
def test_a_card_that_does_not_name_a_foreign_runtime_is_left_alone(desc):
    """THE NEGATIVE CONTROL, and the half that decides whether this ships.

    A filter that fires on everything empties the queue and reads exactly like
    a queue that ran out. `transformers` is deliberately not refused: torch
    runs here, and what a conversion costs is a different question.
    """
    assert ins.runtime_needed(desc) == ""


def test_the_answer_is_a_fact_about_the_machine_asking():
    """The store is shared. The same weights are unrunnable here and ordinary
    on the box with the card, so this must never be written down as a verdict
    about the candidate."""
    row = {"name": "org/m", "lane": "image", "description": GEMLITE}
    assert rank.unrunnable(row, mac()) == "needs-cuda"
    assert rank.unrunnable(row, card()) == ""


def test_vllm_is_probed_rather_than_assumed_absent():
    """Without a probe, refuses("vllm") answers needs-vllm on the box that HAS
    one, which is the mirror of the mistake #228 made about GGUF."""
    assert "vllm" in _machine._RUNTIME_PROBES
    row = {"name": "org/m", "lane": "stt", "description": VLLM}
    assert rank.unrunnable(row, card()) == ""


def test_awq_and_gptq_are_deliberately_not_refused():
    """They are quantisation formats with CUDA kernels in practice and
    implementations elsewhere. Refusing them would be PREDICTING a failure
    rather than reading one, which is how a filter starts settling candidates
    on our guesses."""
    for fmt in ("awq", "gptq"):
        assert ins.runtime_needed(f"tagged qwen3, {fmt}, text-generation") == ""


def test_an_mlx_card_survives_on_the_mac():
    row = {"name": "mlx-community/x", "lane": "stt", "description": MLX}
    assert rank.unrunnable(row, mac()) == ""
