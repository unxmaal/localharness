"""Shared machinery for the local media harness.

Everything here is used by BOTH the CLI (harness.cli) and the eval suite
(evals/), which is the reason it is a package rather than living inside either.
The eval suite grew first, so for a while the only code that knew how to invoke
an image generator lived in the test harness. That is backwards: the product is
the CLI and the evals measure it.
"""
