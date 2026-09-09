# Assumptions

A blind-spot sweep. Phase 1 is this enumeration; phase 2 validates each entry.

Derived by walking every module-level constant and default in `harness/` and
`evals/` (142 of them), every numeric claim in the README, and the decisions
recorded while building. Not curated from memory: the constant list is
mechanical, so it can be regenerated.

States: **measured** (a number exists, taken here) · **tuned** (fitted to our
own data, so a setting rather than a finding) · **assumed** (nothing tested it).

---

## A. The instrument itself

| id | assumption | where | state |
|---|---|---|---|
| A1 | `MIN_INK = 0.005` — marking 0.5% of the canvas means "not blank" | `vector.py` | tuned |
| A2 | STT and TTS are **deterministic**, so n=1 per case is enough | `STOCHASTIC_MODALITIES` omits them | assumed |
| A3 | `DEFAULT_TEMPERATURE = 0.2` is the right sampling for every text lane | `completion.py` | assumed |
| A4 | 3000 bootstrap resamples is enough for a stable interval | `compare.py` | assumed |
| A5 | corpus WER, not mean per-utterance WER | RULE #186 | **measured** |
| A6 | paired-over-cases is the right resampling unit | `compare.py` | reasoned, not tested |
| A7 | `SECONDS_PER_WORD_CEILING = 1.0`, `RUNAWAY_MIN_WORDS = 6` detect runaway speech | `audio.py` | tuned on one defect |
| A8 | `MIN_AUDIO_BYTES = 8000` means "audio actually produced" | `audio.py` | assumed |
| A9 | ink / OCR / round-trip-WER are valid proxies for "good" | `CHECKERS` | catches failures; does not prove quality |
| A10 | `comparable()` guards cross-run comparison | `evals/core.py` | **written, tested, called by nothing** (#1) |
| A11 | median latency is the honest latency | throughout | warm-state only; cold is 5-10x |
| A12 | `FPS = 24`, `H3_MIN_FRAMES = 22` describe the video product | `core.py`, `engines.py` | inherited from the engine |

## B. The corpus

| id | assumption | state |
|---|---|---|
| B1 | LibriSpeech test-clean predicts speech generally | assumed — clean, read, American English, small speaker set |
| B2 | 300 clips is enough | measured that 40 is not; **no stopping rule**, and FluidAudio was *not separable* at 300 |
| B3 | the seed-1 sample of 300/2620 is representative | assumed — never varied the seed |
| B4 | the reference transcripts are correct | assumed |
| B5 | punctuation normalisation makes the score about the model | measured (RULE #183) |

## C. Hardware and memory

| id | assumption | where | state |
|---|---|---|---|
| C1 | `GPU_FRACTION = 0.75` of unified memory is available to Metal | `memory.py` | vendor-undocumented, version-dependent |
| C2 | `DEFAULT_RESERVE_GB = 6.0` is enough headroom for macOS | `memory.py` | tuned after crashing the machine once |
| C3 | `MEMORY_CEILING = 22 GiB` of weights is the real limit | `inspect.py` | derived from C1, so it inherits C1's uncertainty |
| C4 | one working set at a time | queue and jobs design | assumed — never tested by running two |
| C5 | disk is not a constraint | 611 GiB free | true today, false silently as the queue fetches |
| C6 | `DISK_FLOOR = 50 GiB`, `MAX_DOWNLOAD = 60 GiB` | `fetching.py` | round numbers |
| C7 | the M5 Ultra arrives ~Nov 2026 with 96 GB | several deferrals parked on it | assumed |
| C8 | README timings (~19s image, ~40 min video) | M2 Pro, warm, quiet | will not survive the hardware change; unstated |

## D. Discovery: the crowd

| id | assumption | state |
|---|---|---|
| D1 | the four `DEFAULT_SEEDS` are the right root | assumed — chosen *because they are installed*, the same failure the model audit found in every lane |
| D2 | `POPULATION = 5e6` | **tuned against its own control set**; a pass proves no regression, not correctness |
| D3 | the ranking is scale-invariant | **measured false** — 250 passes, 450 fails at every POPULATION |
| D4 | `MIN_SHARED = 3` | tuned |
| D5 | `HALF_LIFE_DAYS = 365` | assumed |
| D6 | `POPULAR_STARS = 100_000` marks "everyone stars it" | tuned |
| D7 | `min_degree = 2` earns a second hop | assumed |
| D8 | two hops beats one at equal size | **measured** — 12 new repos, control still passes |
| D9 | contributors ≈ the people worth following | assumed |

## E. Discovery: judging and inspecting

| id | assumption | state |
|---|---|---|
| E1 | the novelty rubric encodes what is worth measuring | one person's notion, scored by a 4B model |
| E2 | a 5-item control validates the judge | measured that it separates; five items is a very small exam |
| E3 | q3-4b is adequate for triage | assumed — never compared against a larger judge |
| E4 | `MAX_MENTIONS = 12` per comment | assumed |
| E5 | the longest 8 comments carry the signal | assumed |
| E6 | smallest named weight decides feasibility | reasoned; a tokenizer makes the floor optimistic |
| E7 | README + repetition + name overlap identify the headline weight | tuned |
| E8 | `SIZE_LIMIT = 12` weights sized per repo | tuned for cost |
| E9 | `DEAD_DAYS = 730` means abandoned | assumed |
| E10 | `CLONE_KB_CAP = 250_000` means "weights in git" | assumed |
| E11 | a CUDA marker in a dependency file disqualifies | measured false twice, then narrowed |
| E12 | the registry declares a task | measured absent on 3 of 6; silence held back a genuine STT model |
| E13 | `NOT_WEIGHTS`, `SKIP_DIRS`, `READ_SUFFIXES` scope the scan correctly | assumed |

## F. Discovery: sources and coverage

| id | assumption | state |
|---|---|---|
| F1 | Reddit + release feeds + crowd + comments cover the field | assumed — no arXiv, Discord, vendor blogs, or non-English sources |
| F2 | extraction precision 0.45 is a stable baseline | measured twice (0.44, 0.45); assumes the recap format holds |
| F3 | a 7-day interval matches how fast the field moves | assumed |
| F4 | cache TTLs (2-30 days) match how fast each endpoint moves | assumed; only the ordering was reasoned |
| F5 | recurrence separates lasting from trending | partial answer, stated as such |
| F6 | popularity is anti-correlated with novelty | stated limit, unquantified |

## G. Services outside our control

| id | assumption | evidence |
|---|---|---|
| G1 | Reddit keeps answering a spoofed browser UA | already refuses JSON; 429s under load |
| G2 | `<permalink>.rss` keeps serving comments | measured working, undocumented |
| G3 | `gh` stays authenticated at 5000/hr | measured |
| G4 | repo→stargazers is readable | **already false** — 404 with token, 401 without |
| G5 | HF returns sizes and pipeline tags | 429s under a sweep |
| G6 | mflux / mlx-audio / mlx-lm keep their shapes | version-pinned; bumps have broken behaviour |
| G7 | `H3_DEFAULT_BIN` exists at a hardcoded path in `~/projects/github/antirez` | assumed |
| G8 | LibriSpeech sits at `/Volumes/Models/corpora` | assumed |

## H. Defaults nobody revisited

| id | assumption | state |
|---|---|---|
| H1 | `bm_george` is the best voice | chosen from **3 of 54** |
| H2 | `KNOWN_VOICES` — 5 cached of 54 available | a cache accident became the candidate set |
| H3 | `CHATTERBOX_REPETITION_PENALTY = 1.2` | the model author's value, restored after a server default overrode it |
| H4 | the text lanes' default models | Qwen2.5 entered as a smoke test and became production |

## I. The headline claims

| id | claim | reality |
|---|---|---|
| I1 | "Nothing leaves the machine" | **true of generation, false of discovery** — Reddit, GitHub and HF are all called, and weights are downloaded. Never stated |
| I2 | "Every model earned its place by winning a run" | true of what was *compared*; the candidate sets were mostly what was installed |
| I3 | "It does not fall behind" | nothing is scheduled; every sweep is typed by hand |
| I4 | models are "4 GB to 31 GB" | true of what is installed |

## J. Process

| id | assumption | state |
|---|---|---|
| J1 | one reviewer, so batch into one PR | stated by Eric |
| J2 | never stack PRs | measured the hard way, twice |
| J3 | verify a merge by file content, never PR status | measured the hard way, twice |
| J4 | a check gates an action | **measured false once** — the check ran in the same `&&` chain as the action it should have prevented |
| J5 | the loop stays manual until it is stable | "stable" is undefined |

## K. The largest one

| id | assumption | state |
|---|---|---|
| K1 | this generalises beyond media | the engine only runs where verification is cheap and objective. Most interesting questions have no checker. Bounded in the KB; the boundary has never been tested |


---

# Phase 2: validation results

Run 2026-09-08, unattended. Verdicts: **HELD** · **REFUTED** · **NARROWED**
(true but smaller than stated) · **UNTESTED** (with the reason).

## The five that changed something

### E1/E2/E3 — REFUTED. The judge is the instrument, and only one pairing works

Running the same rubric under a larger judge:

    q3-4b        won_min 10,5   lost_max 3,3,3    gap +2   separates
    local-large  won_min 10,10  lost_max 10,3,3   gap +0   DOES NOT separate

`local-large` scores `upscale-seedvr2` — a known failure, 0/3 crashes — at
**10/10**, calling it "a composition of existing tools", which is a fair reading
of its description and is the rubric's own top criterion. The rubric does not
encode novelty; it encodes something a 4B model happens to read the way we
intended.

Worse: **the control is not stable across runs.** q3-4b gave gap +4 in one run
and +2 in the next, on identical inputs, because the judge samples and nothing
pins a seed. "separates=True" near the boundary is a coin flip.

Consequence: every judge score in the store was produced by an instrument whose
validation is noisy and whose criterion is ambiguous.

### A6 — HELD, and it is carrying the result

    paired    CI [+0.0005, +0.0086]  width 0.0081  separates
    unpaired  CI [-0.0009, +0.0101]  width 0.0110  NOT separable

Pairing is what makes the Qwen3-ASR finding a finding. Unpaired, it spans zero.
The resampling unit was reasoned and never tested, and it was deciding the
answer.

### B3 — NARROWED. The ranking survives a new sample; the number does not

    seed 1:  parakeet 0.0162   challenger +0.0045 [+0.0005, +0.0089]  worse
    seed 7:  parakeet 0.0190   challenger +0.0060 [+0.0019, +0.0105]  worse

Same verdict both times. But the headline WER moved 17% on nothing but the
sample. "parakeet-v2 scores 0.0162" is a property of those 300 clips, not of the
model. It has been quoted as the model's, including in the roadmap.

### D5 — REFUTED. The recency decay does nothing

`HALF_LIFE_DAYS` at 90, 365, 1095 and effectively infinite all produce an
identical top ten. What removed the archived 2020 physics course was the
**archived filter**; the decay was shipped alongside it and credited with half
the fix. It is inert on this data.

### A1 — NARROWED. `MIN_INK` never binds

63 real ink values: minimum **0.0391** against a **0.005** threshold. Blank
renders give exactly 0.0 and real ones give ≥0.039, so the threshold sits in a
wide empty gap. Arbitrary, and no result depends on where in the gap it sits.

## Held

| id | verdict | evidence |
|---|---|---|
| A2 | **HELD** | same clip 3x through parakeet → byte-identical. n=1 is sound for STT |
| A4 | **HELD** | 500 / 3000 / 20000 resamples agree to 3 decimals; 3000 is generous |
| A5 | HELD | RULE #186 |
| A11 | **HELD** | first case is 1.5x–8.4x the median, one run at 8.4x. Cold is real and unreported |
| C1 | HELD | `sysctl iogpu.wired_limit_mb` = 0 (system default); 0.75 × 32 GiB = 24 GiB, and `MEMORY_CEILING` is 22, so ~2 GiB implicit reserve |
| D4 | **HELD with a band** | `MIN_SHARED` passes at 2/3/5, fails at 8. Default sits inside, not on an edge |
| D6 | **HELD with a band** | `POPULAR_STARS` fails at 20k, passes at 100k and 500k |
| D8 | HELD | measured earlier: 12 new repos, control still passes |
| E12 | HELD | `pipeline_tag` absent on 3 of 6 |
| G3 | HELD | 5000/hr, full |
| G4 | HELD (already false) | stargazers still refused |
| G7/G8 | HELD | h3 binary, model dir and LibriSpeech all present |
| H1/H2 | HELD | 5 voice packs on disk, default chosen from 3 of 54 |
| I3 | HELD | 4 launchd agents, all services (tts, mlx, mcp, gateway). Nothing schedules discovery. 0 cron |
| J2/J3/J4 | HELD | each demonstrated by failing at it during this session |

## Corrected

| id | correction |
|---|---|
| A10 | **overstated.** `comparable()` *is* called, from `compare_runs()`. What it does not guard is prose: quoting one run's number against another in a sentence. A human gap, not a code gap |
| I1 | **confirmed exactly.** Network calls live in `cli.py`, `feeds.py`, `fetching.py`, `inspect.py`. Generation is local; discovery is not |
| G2 | **fragile, demonstrated.** The comment feed returned **429** during this sweep. It worked an hour earlier |

## Untested, with reasons

| id | why not |
|---|---|
| A3 | "temperature 0.2 is right" needs a lane-wide sweep at several temperatures; hours of GPU |
| A7/A8 | the runaway thresholds allow a 6-word line 6.0s when Kokoro typically takes ~0.3s/word, so they are loose by ~20x. Tightening needs a corpus of real failures, and one defect is not a corpus |
| A9 | "passing a checker means good" is not decidable by a checker. Needs a human looking at artifacts |
| A12, C6, E4–E11, E13, F1, F3–F6 | no cheap experiment; each needs either a second corpus, a second judge, or time to pass |
| B1 | needs a second speech corpus that is not clean read English. None on disk |
| B4 | reference transcripts assumed correct; verifying means re-transcribing by ear |
| C4 | "one working set at a time" would be tested by running two. This machine has already been taken down once that way (RULE #193) |
| C7 | hardware that has not arrived |
| K1 | needs a second domain with a cheap checker. The claim is bounded and the boundary is the experiment |

## What this sweep says overall

Three assumptions were doing work nobody had checked: **pairing** (carries the
STT result), **the judge/rubric pairing** (only one works, and its control is
noisy), and **the absolute WER** (sample-dependent, quoted as a model property).

Two were inert: the **recency decay** and the **ink threshold**. Both are
harmless and neither earns its place as an explanation.

The rest either held or could not be tested cheaply, and the untested list is
now explicit rather than invisible, which was the point of the sweep.


---

# Phase 2 corrections

Two entries in the results above were wrong, both from the same cause.

## D5 was not refuted. My test could not see the effect

I varied `HALF_LIFE_DAYS` and compared `hits`, `leaks` and `top[0]`. **A count
over a set cannot see ordering within that set**, and a ranking change is
exactly a reordering. The statistics I chose were blind to the effect I was
testing for, and their stability read as absence of it. I then wrote "identical
top ten" without ever printing the top ten.

Comparing the actual orderings: **10 of the top 15 positions differ** with decay
on versus off, and the top-15 set is different. Across 439 candidates the oldest
push is 2018 and the factors are large -- `neural-style` x0.003,
`arxiv-sanity-lite` x0.107. The decay does what it was built for. Issue #87
closed as invalid.

## D4/D6 were measured with the same blind statistics

Redone as rank comparisons:

    MIN_SHARED   2 -> top-15 identical to the default
                 3 -> default
                 5 -> 4 positions differ, set differs
                 8 -> 11 positions differ, set differs

So the default has margin below it and the constant genuinely bites above it.
"Passes at 2/3/5" was not wrong, it was uninformative.

`POPULAR_STARS` turned out not to be a ranking parameter at all: the ranking
function never reads it. It is the control's leak detector. Varying it was
testing the control, not the ordering, and the earlier result should be read
that way.

## The general lesson

**Before a sensitivity test, ask what the statistic cannot see.** A count cannot
see order. A mean cannot see spread. A top-1 cannot see anything below it. If
the statistic is blind to the effect under test, a null result is uninformative
rather than negative.

---

# Phase 3: what was fixed

| id | issue | fix |
|---|---|---|
| E1/E2 | #85 | `control_repeated()` runs the control N times and reports the spread; separates only if EVERY run separates |
| E1/E2 | #86 | the ambiguous criterion now distinguishes composing what is already installed from adding a new dependency. Rubric v4 |
| A11 | #89 | `first_s` sits beside the median in the summary, with `first_is_cold` saying which candidate it is actually cold for |
| B3 | #88 | `evals.compare` prints the absolute with its corpus fingerprint and n, and says to quote the difference instead |
| D5 | #87 | closed as invalid; the test was wrong, not the code |

Rubric v4 verified under both judges, three runs each:

    q3-4b        gaps [3,3,3]  spread 0  separates 3/3
    local-large  gaps [7,7,7]  spread 0  separates 3/3

Sharpening the criterion fixed the cross-judge disagreement AND the instability:
spread went from 2 to 0. q3-4b still scores `upscale-seedvr2` 7/10, so the
criterion is better rather than perfect.
