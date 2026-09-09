"""Ranking a proposal by how much it looks like it runs HERE.

relevance() scored +2 for Apple terms and -1 for cuda, nvidia, rtx, vram, 4090.
On a Mac that is right. On the box with the card it scores a CUDA technique
NEGATIVE, sorts it last and the limit then cuts it, so the proposals most worth
that machine's time are the ones it never sees.

Techniques matter more here than models. A model can be found in a registry; a
LoRA that makes an existing job five times faster is only ever mentioned in
prose, and prose about a technique names the hardware it was measured on. That
is exactly the text this function reads.
"""
import pytest

from harness import feeds
from harness import machine as mach
from harness.memory import Accelerator

APPLE = mach.Machine(frozenset({"mlx", "cpu"}),
                     Accelerator("unified", 32.0, 25.0, "Mac16,1"))
CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                    Accelerator("discrete", 12.0, 10.5, "RTX 4070"))
LINUX_CARD = mach.Machine(frozenset({"cuda", "cpu"}),
                          Accelerator("discrete", 12.0, 10.5, "RTX 4070"))

CUDA_TECHNIQUE = ("a LoRA that makes SDXL five times faster on a 4090, "
                  "tested at 12GB VRAM with CUDA 12")
MLX_TECHNIQUE = ("an MLX quantisation that halves memory on Apple Silicon, "
                 "measured on an M2 Pro with unified memory")
NEUTRAL = "a scheduler change that needs no new weights"


def test_a_cuda_technique_ranks_positively_on_a_card():
    """THE POINT OF THIS FILE. This scored negative and sorted last on the
    machine it was written for."""
    assert feeds.relevance(CUDA_TECHNIQUE, CARD) > 0


def test_an_mlx_technique_still_ranks_positively_on_a_mac():
    assert feeds.relevance(MLX_TECHNIQUE, APPLE) > 0


def test_each_machine_ranks_the_other_one_down():
    assert feeds.relevance(MLX_TECHNIQUE, CARD) < 0
    assert feeds.relevance(CUDA_TECHNIQUE, APPLE) < 0


def test_text_naming_no_hardware_is_neutral_everywhere():
    """Zero is the honest default for text that says neither, and a technique
    that names no hardware is not thereby a worse technique."""
    for m in (APPLE, CARD):
        assert feeds.relevance(NEUTRAL, m) == 0


def test_linux_and_windows_with_the_same_card_rank_identically():
    for text in (CUDA_TECHNIQUE, MLX_TECHNIQUE, NEUTRAL):
        assert feeds.relevance(text, CARD) == feeds.relevance(text, LINUX_CARD)


def test_vram_is_not_a_foreign_word_on_a_machine_with_vram():
    """It was in the list of terms meaning "will not run here", which is a
    word this machine uses about itself."""
    assert feeds.relevance("needs 10GB VRAM", CARD) >= 0


def test_it_still_answers_without_being_given_a_machine():
    """Callers that predate this keep working, against the local machine."""
    assert isinstance(feeds.relevance(NEUTRAL), int)


@pytest.mark.parametrize("runtime", sorted(mach._RUNTIME_PROBES))
def test_every_runtime_has_terms_to_recognise_it(runtime):
    """A runtime with no vocabulary scores every mention of itself at zero, so
    a machine detecting it would rank its own techniques as though they named
    no hardware at all."""
    assert feeds.RUNTIME_TERMS.get(runtime) is not None
