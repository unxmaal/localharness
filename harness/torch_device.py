"""Which accelerator torch should use, asked once.

Both diffusers generators asked `torch.cuda.is_available()` and nothing else,
so "no card" and "no Apple GPU" were the same answer and the image lane had
exactly one engine on Apple Silicon for months. Fixing that in one file and
not the other is how this repo has collected four separate copies of "which
engines exist"; RULE #237 says the fix is a read, not more care.

Separate from harness/engines.py because these run inside the diffusers venv,
which has torch and has no need of the rest of this package.
"""
from __future__ import annotations


def accelerator(torch) -> str:
    """The device to generate on, or "" when there is no accelerator.

    Passed the module rather than importing it, so a test can pin the answer:
    `torch.cuda.is_available()` reads the machine, and a test that reads the
    machine tests the machine (RULE #249).
    """
    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return ""


#: Accelerators that should load half precision. Metal supports float16 and
#: the weights are published in it; fp32 on a 32 GB unified machine doubles
#: the working set for nothing.
HALF = ("cuda", "mps")
