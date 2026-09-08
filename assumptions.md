# Assumptions

Everything this project believes without having proved it, or proved only once
under conditions that may not hold again. Written by traversing the README and
the working history.

Three states are used throughout:

- **measured** — a number exists, taken here, on this hardware
- **tuned** — a value chosen by fitting it to our own data, so it is a setting rather than a discovery
- **assumed** — nothing has tested it

---

## 1. What "nothing leaves the machine" actually means

**GENERATION is local. DISCOVERY IS NOT.** The README's headline claim is true of
every prompt, log and voice clip, and false as a description of the whole tool.
`lh discover` makes outbound requests to Reddit, GitHub and HuggingFace, and
`lh fetch` downloads weights. What is never uploaded is *your content*; what is
sent is *queries about other people's repos*.

Nobody has been misled by this yet because the same person wrote and reads it,
but a stranger reading the first line would not expect the network calls.

- `assumed`: that the distinction is obvious. It is not stated anywhere.

## 2. Hardware and environment

| assumption | state | note |
|---|---|---|
| 22 GiB is the weight ceiling on 32 GB unified | `tuned` | derived from "macOS gives the GPU roughly 70-75%", a figure that is undocumented and version-dependent |
| disk is not a constraint | `measured` | 611 GiB free, but this becomes false quietly as the queue fetches |
| one working set at a time | `assumed` | the whole queue and job design rests on it; never tested by running two |
| the M5 Ultra arrives ~Nov 2026 with 96 GB | `assumed` | several deferrals are parked on it |
| `HF_HUB_OFFLINE=1` keeps evals from phoning home | `measured` | and fetching had to lift it deliberately, at the constant, not the env var |
| Apple Silicon only, forever | stated intent | not an assumption so much as a decision |

Absolute numbers in the README (`~19s` an image, `~40 min` a video, models
`4 GB to 31 GB`) are all **M2 Pro, warm, quiet machine**. None will survive the
hardware change, and the README does not say so.

## 3. The measurement instrument

This is where the assumptions are load-bearing and least visible.

**The corpus stands in for the task.** LibriSpeech test-clean is clean, read,
American English by a small speaker set. Every STT number here is a measurement
of *that*, and is quoted as though it were "speech".
- `assumed`: that it predicts conversational, accented or noisy audio. It almost
  certainly does not, and no second corpus exists to check.

**300 clips is enough.** n=40 was demonstrated insufficient — it got the winner
right and both effect sizes wrong. 300 separates the differences seen so far.
- `assumed`: that 300 is sufficient in general. There is no stopping rule, and
  the FluidAudio comparison came back *not separable* at n=300, which is exactly
  the case where more clips would change the answer.

**The seeded sample is representative.** 300 of 2620 utterances, seed 1, fixed
so runs are comparable.
- `assumed`: that this particular 300 is not systematically easier or harder.
  Never checked against a different seed.

**The checkers are valid proxies.**
- ink coverage stands in for "is it a picture rather than blank markup"
- OCR stands in for "the text rendered legibly"
- round-trip WER stands in for "the speech is intelligible"
- `measured` that each catches real failures; `assumed` that passing one means
  the artifact is *good*. A picture can mark the canvas and still be wrong.

**Median latency is the honest latency.** Cold starts are 5-10x warm ones, so
medians are quoted.
- `assumed`: that a caller experiences the warm number. First use of anything
  is the cold number, and nothing reports it.

**Comparability is unguarded.** `comparable()` is written, tested, and **called
by nothing** (#1). Every cross-run comparison in this repo rests on the operator
having remembered that two runs were run the same way.

## 4. The discovery loop

**The crowd is the right crowd.** It starts from contributors to four repos we
happen to run.
- `assumed`: that these four are the right root. Chosen because they are what is
  installed, which is the same "candidate set chosen by what was at hand"
  problem the model audit identified in every lane.

**`POPULATION = 5e6`** is `tuned` — fitted against our own control set, so the
control now proves *no regression*, not *correctness*. Stated in `control()`.

**The ranking is not scale-invariant.** `measured`: 250 people passes, 450 fails
at every POPULATION tried. Widening is therefore only safe by depth.

**Constants nobody has justified**, all `tuned` or `assumed`:

| constant | value | basis |
|---|---|---|
| `MIN_SHARED` | 3 | "below this it rests on a couple of people" |
| `SIZE_LIMIT` | 12 | bounded so the tier stays cheap |
| `HALF_LIFE_DAYS` | 365 | a year feels like the right decay |
| `POPULAR_STARS` | 100_000 | above this, everyone stars it |
| `min_degree` (hop 2+) | 2 | consensus of two |
| `mention_comments` | 8 | one model call per comment is too many |
| `DISK_FLOOR` / `MAX_DOWNLOAD` | 50 / 60 GiB | round numbers |
| cache TTLs | 2-30 days | ordered by how fast each endpoint moves, not measured |

Each is defensible and none is derived. They were set once and have never been
varied to see whether the output is sensitive to them.

**The judge knows what novelty is.** The rubric encodes one person's notion of
what is worth measuring, scored by a 4B model.
- `measured`: it separates known-good from known-bad on a five-item control
- `assumed`: that the control set generalises, and that a bigger judge would
  agree. Five items is a very small exam.

**The registry declares a task.** Lane inference reads `pipeline_tag`, then tags.
- `measured`: absent on 3 of 6 real models. Voxtral Realtime is genuinely STT and
  is held back because its entry says nothing.
- `assumed`: that silence means "we cannot tell", not "not measurable". Held back
  and visible is the safe reading, and it costs real candidates.

**Popularity is anti-correlated with novelty.** Recorded as a standing limit: a
genuine innovation is rare and under-discussed exactly when it matters most.
Recurrence is a partial answer and nobody should pretend it is a full one.

**The sources cover the field.** Three Reddit feeds, four release feeds, the
GitHub crowd, and post comments.
- `assumed`: no arXiv, no Discord, no Chinese-language sources, no vendor blogs.
  The measured extraction precision (0.45 from the recap) says nothing about
  what is being missed entirely.

## 5. Services outside our control

Every one of these can withdraw without notice, and several already changed
under us:

| dependency | assumption | evidence |
|---|---|---|
| Reddit RSS | keeps answering a spoofed browser UA | already refuses the JSON API, rate-limits to 429 |
| `<permalink>.rss` | keeps serving comments | `measured` working; undocumented |
| `gh` API | stays authenticated, 5000/hr | `measured` |
| repo stargazers | **already gone** | 404 with a token, 401 without |
| HF API | keeps returning sizes and pipeline tags | rate-limits to 429 under a sweep |
| mflux / mlx-audio / mlx-lm | keep their CLI and server shapes | version-pinned; a bump has broken behaviour before |

The standing answer is to cache every fetched fact permanently, so a service
disappearing costs future answers rather than past ones.

## 6. Working process

- Eric is the only reviewer, so PRs are batched rather than split. `stated`
- Every PR targets main; never stack. `measured` the hard way, twice
- Verify a merge by file content, never by PR status. `measured` the hard way, twice
- `assumed`: that the manual loop stays manual. Nothing is scheduled, and the
  reason is that manual runs are what found most of the defects. That reason
  expires the moment the loop is stable, and nobody has defined stable.

## 7. The largest assumption

That this generalises beyond media. The stated reframe is that discovery is the
product and media is the first domain, chosen because verification there is
cheap and objective.

- `measured`: the loop turns without a human in each cycle, in these lanes
- `assumed`: that a domain with cheap ground truth can be found for anything
  worth discovering. The engine cannot run where there is no checker, and most
  interesting questions have no checker. The claim is bounded by that and the
  boundary has never been tested.
