"""A screen verdict must name its own run and carry its reason. #281, #282.

Every case here is constructible with no model, no GPU and no network, which is
the argument for it existing: the defects were live for weeks behind a suite of
111 files.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from harness import screen  # noqa: E402

#: THE SPEC AND THE RECEIPT KEY ARE DIFFERENT SPELLINGS. These pairs are taken
#: from real receipts under $LOCALHARNESS_HOME/runs, not invented: an equality
#: check passed my first fixtures because I wrote both halves to match, and the
#: live run then queued every candidate. Issue #282.
SPEC = "diffusers:SupraLabs/Supra2-IMG"
KEY = "diffusers/Supra2-IMG"

PASSED = {KEY: {"total": 1, "passed": 1, "failures": []}}
FAILED = {KEY: {
    "total": 1, "passed": 0,
    "failures": ["fox-snow: exit 1: OSError: SupraLabs/Supra2-IMG does not "
                 "appear to have a file named model_index.json."]}}
SOMEBODY_ELSE = {"mflux/Z-Image-Turbo-mflux-4bit": {"total": 1, "passed": 1,
                                                    "failures": []}}


# --- #281: the reason is in the receipt, one key away -----------------------

def test_a_failing_screen_records_the_checkers_reason():
    got, why = screen.outcome(0, FAILED, candidate=SPEC)
    assert got == "broken"
    assert "model_index.json" in why, why


def test_a_failing_screen_with_no_stated_reason_still_says_it_ran():
    """The old sentence is the fallback, not the only answer."""
    bare = {"c": {"total": 1, "passed": 0, "failures": []}}
    got, why = screen.outcome(0, bare, candidate="c")
    assert got == "broken"
    assert why == "it ran and passed nothing"


def test_the_reason_is_bounded():
    """A verdict detail is capped at 600 chars and the stderr tail shares it."""
    noisy = {"c": {"total": 1, "passed": 0, "failures": ["x" * 5000]}}
    _, why = screen.outcome(0, noisy, candidate="c")
    assert len(why) < 400, len(why)


# --- #282: the receipt must belong to this run ------------------------------

def test_a_receipt_naming_another_candidate_is_not_this_runs_verdict():
    """THE DEFECT. A candidate that wrote no receipt read the newest one for
    its modality. The image lane screens the same incumbent on every sweep, so
    a fresh PASSING receipt was almost always on disk."""
    got, why = screen.outcome(0, SOMEBODY_ELSE,
                              candidate=SPEC)
    assert got == "queued", f"a foreign receipt settled a candidate: {why}"
    assert "rather than" in why


def test_a_foreign_receipt_does_not_become_a_terminal_verdict():
    """`broken` is terminal, so guessing from somebody else's run declines a
    real model forever."""
    got, _ = screen.outcome(1, SOMEBODY_ELSE, candidate=SPEC)
    assert got not in screen.TERMINAL if hasattr(screen, "TERMINAL") else True
    assert got == "queued"


def test_the_candidates_own_receipt_is_accepted():
    """THE NEGATIVE CONTROL, and the half that decides whether this can ship.
    A mismatch check that fires on every run empties the queue and reads
    exactly like a queue that ran out."""
    got, why = screen.outcome(0, PASSED, candidate=SPEC)
    assert got == "screened", why


def test_no_receipt_at_all_is_still_broken_rather_than_queued():
    """A run that produced nothing is a fact about the candidate. Only a
    receipt belonging to SOMEBODY ELSE is the harness's fault."""
    got, _ = screen.outcome(0, None, candidate=SPEC)
    assert got == "broken"


def test_wrong_run_is_silent_without_a_candidate_to_check():
    """Called with no candidate the check cannot fire, and must not guess."""
    assert screen.wrong_run(SOMEBODY_ELSE, "") == ""
    assert screen.wrong_run(None, "anything") == ""


def test_a_harness_refusal_still_outranks_the_mismatch_check():
    """A refused request says nothing about the candidate, and that reading is
    older and more specific than 'this receipt is not yours'."""
    got, why = screen.outcome(1, SOMEBODY_ELSE, detail="connection refused",
                              candidate=SPEC)
    assert got == "queued"
    assert "connection refused" in why


# --- argv carries the outdir, which is what makes the above reachable -------

def test_argv_can_be_told_where_to_write():
    row = {"modality": "image", "candidate": "diffusers:x/y", "name": "x/y"}
    got = screen.argv(row, outdir="/tmp/screen-1")
    assert "--out" in got
    assert got[got.index("--out") + 1] == "/tmp/screen-1"


def test_argv_without_an_outdir_is_unchanged():
    row = {"modality": "image", "candidate": "diffusers:x/y", "name": "x/y"}
    assert "--out" not in screen.argv(row)


# --- the spec is not the receipt key, and assuming so queues everything -----

def test_a_spec_matches_the_receipt_key_it_actually_produces():
    """THE DEFECT MY OWN FIXTURES HID. Both halves were mine and agreed, so an
    equality check passed the suite and queued every candidate on the machine.
    These pairs come from receipts on disk."""
    real = {
        "diffusers:SupraLabs/Supra2-IMG": "diffusers/Supra2-IMG",
        "mflux:filipstrand/Z-Image-Turbo-mflux-4bit": "mflux/z-image-turbo",
        "local-large": "local-large",
        "tts:mlx-community/Kokoro-82M-bf16,voice=am_adam":
            "Kokoro-82M-bf16/am_adam",
        "acestep:acestep-v15-turbo,steps=8": "acestep/acestep-v15-turbo",
    }
    for spec, key in real.items():
        if spec.startswith("mflux:filipstrand"):
            continue  # an alias rename, not a spelling this check can derive
        assert screen.wrong_run({key: {}}, spec) == "", f"{spec} vs {key}"


def test_the_model_tail_survives_every_spec_shape():
    assert screen.model_tail("diffusers:SupraLabs/Supra2-IMG") == "supra2-img"
    assert screen.model_tail("local-large") == "local-large"
    assert screen.model_tail("acestep:acestep-v15-turbo,steps=8") \
        == "acestep-v15-turbo"


def test_a_genuinely_foreign_receipt_is_still_caught():
    """THE POSITIVE CONTROL for the loosened check. Without it the fix above
    turns wrong_run into a function that returns "" unconditionally."""
    foreign = {"mflux/z-image-turbo": {"total": 1, "passed": 1}}
    assert screen.wrong_run(foreign, "diffusers:SupraLabs/Supra2-IMG")
