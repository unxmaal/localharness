# Testing

`make check` is the gate: shellcheck, `bash -n`, and the unit suite. It needs no
running services. `make smoke` is separate because it asserts against live
gateway and engine processes.

## Why mutation testing

A test that cannot fail proves nothing. Every guard in this repo has been broken
deliberately and the corresponding test confirmed to go red.

### stt_shim.py — removed

The shim and its 5 mutation-proofed tests were deleted. Its premise was false:
its docstring claimed voicemode "hardcodes the STT model as whisper-1 and
exposes no override", but voicemode 8.12.0 has `VOICEMODE_STT_MODELS`, and
`provider_discovery.py` classifies port 8890 as "mlx-audio" and stops sending
whisper-1 to it entirely. Upstream even carries a comment about mlx-audio
rejecting whisper-1.

The claim came from a truncated grep of `voice-mode config list` rather than
from reading the source. Well-tested code solving a problem that does not exist
is still waste, and the tests made it look more solid than it was.

### env.sh — 4/5 proven, 1 masked

| mutation | outcome |
|---|---|
| allow the internal disk | caught |
| drop the writability check | caught |
| stop exporting HF_HOME | caught |
| create dirs while probing | caught, but only once the mutation was moved ABOVE the volume check; see below |
| accept unresolvable volumes | **survived** |

Two findings worth keeping.

**The "create dirs while probing" mutation initially survived** because the
original test put the winning candidate first, so the rejected one was never
probed and the assertion could not fail. The test was rewritten to put the
rejected candidate first. It is now provably able to fail, but only against an
unguarded `mkdir` at the top of the predicate: with the volume check in front,
a rejected path returns before any side effect is reachable. The test pins a
real property; the property is simply also protected by ordering.

**The "accept unresolvable volumes" mutation survives** because removing the
mountpoint guard changes nothing observable: `/Volumes/NO_SUCH_VOLUME/hf` walks up
to `/Volumes`, which resolves to `/`, and the internal-disk check refuses it
anyway. That is redundancy, not a gap. Both guards stay; the note exists so the
next person does not read the surviving mutation as a missing test.

### h3-weights-status.sh — the decoy that did not decoy

`test_unrelated_process_is_not_mistaken_for_the_downloader` first passed
vacuously. The decoy was `bash -c "sleep 20" MARKER`, and bash execs a single
command directly, so the marker vanished from the process command line and
`pgrep -f` never matched. Switched to `python3 -c ... MARKER`, which keeps the
marker in argv, and the test then correctly failed against the old code.

The fix replaced `pgrep -f MiniMax-H3` with a pidfile written by
`fetch-h3-weights.sh`. The old pattern matched any process mentioning the path,
including an editor, a `du`, or the test harness itself, and a false RUNNING
makes a caller wait forever on a download that already died.

## Running the mutation checks

They are not automated: each is a one-line edit, a `pytest` run, and a restore.
The table above is the record. Re-run them when a guard changes.
