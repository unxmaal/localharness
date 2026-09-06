"""Does the generated code work?

Every other check in this suite inspects an artifact. This one RUNS it, because
a syntax check is nearly worthless for code: the interesting failures all parse.
An off-by-one, a reversed comparison and a forgotten edge case are all valid
Python.

THIS EXECUTES MODEL-GENERATED CODE. It is bounded by a subprocess, a timeout,
and a scratch working directory, and that is the whole of the sandbox. It is
appropriate for code your own local models wrote against your own cases; it is
not appropriate for running someone else's eval suite.

The result is graded rather than binary: three assertions passing out of four
is a different model from zero out of four.
"""
import pytest

from harness.checks import code

GOOD = '''
def slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
'''
OFF_BY_ONE = '''
def slugify(text):
    import re
    return re.sub(r"[^a-z0-9]+", "-", text.lower())
'''
SYNTAX_ERROR = "def slugify(text)\n    return text\n"
HANGS = "def slugify(text):\n    while True:\n        pass\n"
CHECKS = [
    "slugify('Hello, World!') == 'hello-world'",
    "slugify('  A  B  ') == 'a-b'",
    "slugify('already-fine') == 'already-fine'",
]


def test_correct_code_passes_every_check():
    r = code.check(GOOD, CHECKS)
    assert r.ok
    assert r.passed == 3 and r.total == 3
    assert r.metrics["code_pass"] == 1.0


def test_partially_correct_code_is_graded_not_just_failed():
    """The reason this is a fraction: a model that nearly works is a different
    model from one that does not work at all."""
    r = code.check(OFF_BY_ONE, CHECKS)
    assert not r.ok
    assert 0 < r.passed < 3
    assert 0 < r.metrics["code_pass"] < 1.0


def test_a_failing_check_is_named_so_you_can_see_which_case_broke():
    r = code.check(OFF_BY_ONE, CHECKS)
    assert "hello-world" in r.reason or "A  B" in r.reason


def test_code_that_does_not_parse_fails_as_a_syntax_error():
    r = code.check(SYNTAX_ERROR, CHECKS)
    assert not r.ok
    assert "syntax" in r.reason.lower()
    assert r.metrics["code_pass"] == 0.0


def test_code_that_hangs_is_killed():
    """A while True in generated code must not stop a fifty-case run."""
    r = code.check(HANGS, CHECKS, timeout=2.0)
    assert not r.ok
    assert "timed out" in r.reason.lower()


def test_code_is_recovered_from_a_markdown_fence():
    fenced = f"Sure, here you go:\n```python\n{GOOD}\n```\nHope that helps!"
    assert code.check(fenced, CHECKS).ok


def test_prose_with_no_code_fails_clearly():
    r = code.check("I would rather not write that function.", CHECKS)
    assert not r.ok
    assert "no code" in r.reason.lower() or "syntax" in r.reason.lower()


def test_the_checks_run_in_a_scratch_directory(tmp_path, monkeypatch):
    """Generated code that writes a file must not write it into the repo."""
    monkeypatch.chdir(tmp_path)
    r = code.check(GOOD + "\nopen('side-effect.txt', 'w').write('x')\n", CHECKS)
    assert r.ok
    assert not (tmp_path / "side-effect.txt").exists()


def test_an_import_the_model_forgot_is_a_normal_failure_not_a_crash():
    r = code.check("def slugify(text):\n    return json.dumps(text)\n", CHECKS)
    assert not r.ok
    assert r.metrics["code_pass"] == 0.0


def test_no_checks_is_a_configuration_error_not_a_free_pass():
    """A code case with nothing to run would otherwise pass every model."""
    with pytest.raises(ValueError):
        code.check(GOOD, [])


def test_stdout_from_the_code_does_not_confuse_the_result():
    noisy = GOOD + "\nprint('PASS 999')\n"
    r = code.check(noisy, CHECKS)
    assert r.ok and r.passed == 3


def test_raises_is_available_to_checks():
    """"It raises ValueError on bad input" is a normal thing to assert, and
    there is no readable way to write it as a bare expression."""
    src = ("def f(x):\n"
           "    if x < 0: raise ValueError('negative')\n"
           "    return x\n")
    r = code.check(src, ["raises(f, -1, exc=ValueError)", "f(3) == 3"])
    assert r.ok, r.reason


def test_raises_is_false_when_nothing_is_raised():
    r = code.check("def f(x):\n    return x\n", ["raises(f, -1, exc=ValueError)"])
    assert not r.ok


def test_raises_does_not_swallow_the_wrong_exception():
    """Catching everything would let a NameError count as correct handling."""
    src = "def f(x):\n    return undefined_name\n"
    r = code.check(src, ["raises(f, 1, exc=ValueError)"])
    assert not r.ok
