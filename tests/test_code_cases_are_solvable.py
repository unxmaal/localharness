"""Every shipped code case must be passable by a correct implementation.

A case whose checks no correct answer can satisfy penalises every model
identically and looks exactly like all of them being bad at it. The reference
solutions in cases/code/reference/ are the red-proof: if one of them fails, the
CASE is wrong, not the model.

They also pin down what the prompt actually means. Writing the reference is
where an ambiguous prompt gets found -- twice here, once for whether touching
intervals merge and once for what an empty duration string should do.
"""
from pathlib import Path

import pytest

from evals.core import load_cases
from harness.checks import code

CASES = Path(__file__).resolve().parents[1] / "evals" / "cases"
REFERENCE = CASES / "code" / "reference"

code_cases = [c for c in load_cases(CASES) if c.modality == "code"]


@pytest.mark.parametrize("case", code_cases, ids=lambda c: c.id)
def test_a_reference_solution_exists(case):
    assert (REFERENCE / f"{case.id}.py").exists(), (
        f"no reference solution for {case.id}; a case nobody has solved is a "
        f"case nobody has checked")


@pytest.mark.parametrize("case", code_cases, ids=lambda c: c.id)
def test_the_reference_solution_passes_every_check(case):
    source = (REFERENCE / f"{case.id}.py").read_text()
    r = code.check(source, case.assertions["checks"])
    assert r.ok, f"{case.id}: {r.reason}"


@pytest.mark.parametrize("case", code_cases, ids=lambda c: c.id)
def test_the_case_has_enough_checks_to_be_discriminating(case):
    """One assertion cannot tell a correct answer from a lucky one."""
    assert len(case.assertions["checks"]) >= 3, case.id
