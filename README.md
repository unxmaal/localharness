# SoHoT

[![check](https://github.com/unxmaal/SoHoT/actions/workflows/ci.yml/badge.svg)](https://github.com/unxmaal/SoHoT/actions/workflows/ci.yml)

**Make pictures, video, speech and code on your own machine. Nothing leaves it.**

One command, `soh`, generates an image, a short video, an SVG icon, a web page,
some code, or speech in a voice you chose. It also transcribes what you say.
No account, no API key, no per-token bill, no rate limit, and no model quietly
retired out from under you.

SoHoT is the Self-optimizing Harness of Theseus. Like the ship, every plank,
engine and model in it gets replaced over time while the harness stays itself,
and it does the replacing on its own: the discovery loop finds a challenger,
measures it paired against the incumbent, and adopts it only on a significant
win.

`lh` still runs the same command as `soh` and is deprecated; it will be removed
after a transition period.

The part that makes it more than a pile of scripts: **it goes looking for better
ways to do its own job, and then proves whether they are better.**

This field moves weekly. A model or a technique that was best when this was
written is probably not best now. So `soh discover` reads the model registries,
the places practitioners talk, and what the people who build your tools are
starring on GitHub. It sorts what it finds cheapest-check-first, throws out
what cannot run on your machine before downloading anything, and hands you the
command that would test the rest against what you already use. You run it, and
the numbers decide.

Nothing here was adopted because it was popular. Every model and method earned
its place by winning a run, on this hardware, against real cases. A few things
people are confident about lost, including one upgrade that turned out to be a
downgrade: the newer version of the speech model in use is measurably worse
than the one it would have replaced.

## What it does

| | |
|---|---|
| `soh image "a red fox in falling snow"` | an image, tens of seconds |
| `soh video "a fox running" --seconds 2` | a video with sound, ~40 min |
| `soh svg "a settings gear icon"` | a real vector icon |
| `soh web "a landing page for a coffee roaster"` | a self-contained HTML page |

Every timing in this file was taken on an M2 Pro with 32 GB unless it says
otherwise, and image and video default to 512x512. Trust your own machine
instead: each generation prints its wall time, its peak memory and the
resolution that produced them, because a number without its configuration
compares to nothing.
| `soh code "parse an ISO timestamp"` | code, to stdout, from the code lane's adopted model |
| `soh extract --file build.log "which tests failed?"` | ask a question about a file |
| `soh decide "is this urgent?" --schema schema.json -f ticket.txt` | answers with a probability per choice |
| `soh say "the tests all passed"` | speak it aloud |
| `soh hear --seconds 5` | record and transcribe |
| `soh hear --worst stt -n 10` | the stored clips with the highest WER (stt: median across candidates; tts: per clip), with reference, hypothesis and a play command; read-only |
| `soh hear --rates` | seconds per word of every stored tts clip, passing and failing, against the runaway ceiling; read-only |
| `soh discover` | what this machine can do that nobody has measured |
| `soh discover --neighbors` | what the people who build your tools are starring |
| `soh fetch` | what is queued for download, and nothing more until you say so |

Everything lands in `~/localharness/out/`.

## Why bother

**It is yours.** The prompts, the logs you pipe into it, the voice clips: none
of it is uploaded anywhere. For anything touching work, health or family that is
the whole argument.

**It costs nothing to run.** After the download, generating a thousand images
costs electricity.

**It does not rot.** A hosted model changes under you or is deprecated. Weights
on your disk keep behaving the same way in a year.

**It tells you which option is best.** Rather than trusting anyone's opinion,
it runs the options against the same test jobs and scores the results. Some of
what that turned up:

- for vector icons the language models lose to a different method entirely.
  Drawing a picture and tracing it scored 4 out of 4, against 2 out of 6 for the
  best model, on identical requests.
- letting a model check its own work and try again took code from 20 right out
  of 27 to 24, and icons from 6 out of 9 to 9, at one extra attempt on average
  and no extra download.
- of two image models, quality came out a statistical tie, so the decision fell
  to 19 seconds against 43 and 11.4 GiB against 13.7, both at 512x512 on an M2
  Pro. Knowing it was a tie is the useful part.
- a quality score that looked useless turned out to work. The experiment
  measuring it had compared two things that were never comparable.

**It does not fall behind.** It reads the model registries and the places
practitioners talk, on a schedule, and tells you what is new. See below.

## Try it

```bash
uv tool install --python 3.12 --editable .
```

Then start the engine your machine has. On Apple Silicon:

```bash
./scripts/serve-mlx.sh &        # inference engine
./scripts/serve-gateway.sh &    # the address clients use
```

On a machine with an NVIDIA card, one command starts all three:

```bash
./scripts/services.sh start     # gateway, text, audio
```

Either way:

```bash
soh svg "a settings gear icon"
```

That prints a path. Open it. For speech on Apple Silicon, also start
`./scripts/serve-tts.sh`; with an NVIDIA card it is already up. Then
`soh say "hello"`.

## What it is not

Before you invest an afternoon:

- **Every lane runs on every machine, with a different tool per runtime.** What
  is missing away from Apple Silicon is voice cloning. See "Running on more than
  one machine" below for which tool serves which lane.
- **The lanes are built for two runtimes so far, `mlx` and `cuda`.** macOS,
  Windows and Linux are each tested on every push, but on a machine with
  neither runtime most candidates are refused for want of an engine rather than
  run slowly. `rocm` is probed and has no implementations behind it yet.
- **It needs disk.** Measured on 2026-09-12: the two image engines are 15 and
  31 GiB, the quality scorers 2 to 4 GiB each, and 4-bit text models run from
  under a gigabyte to about 17. Nothing here is fetched until a lane asks for
  it, so the bill arrives one model at a time.
- **Video takes about 40 minutes a generation** on an M2 Pro with 32 GB. It
  works; it is not something you will use casually.
- **It is a workshop, not a product.** There is no GUI, and some lanes are better
  than others.

## Running on more than one machine

**Every lane is meant to run on any machine, with a different tool per runtime.**
What draws an image on Apple Silicon is not what will draw one on an NVIDIA
card, and it is not meant to be. What has to match is that the lane has an
implementation for each runtime, that the implementation is tested there, and
that the result says which one produced it. A lane that exists on one machine
and not another is unfinished.

Nothing here branches on the operating system. `harness/machine.py` asks which
runtimes are present -- `mlx`, `cuda`, `rocm`, `cpu` -- so adding a machine is
adding a row to a probe table, not forking a lane. That is why nothing below
counts them.

The checks work that way now. Text in a generated image is read by Apple's
Vision on macOS, by Windows.Media.Ocr on Windows, and by RapidOCR on Linux,
which ships no OCR engine of its own. HTML is rasterised by Chrome, or by Edge,
which is Chromium and is already on every Windows install. None of them is
named by the caller: each is found by asking the platform and the install, and
a machine with none warns and withholds the metric instead of failing the
candidate.

Which one ran is part of the result. The engines do not agree -- on this
project's own generated images Vision and RapidOCR read the sign 18 times out
of 18 and tesseract 6 -- so the engine goes into the run's receipt and two runs
graded by different ones are refused a shared table. The same holds for peak
memory, which is a job object on Windows, a phys_footprint on macOS and an
`ru_maxrss` on Linux. The card alone does not identify the instrument: the same
RTX 4070 under Windows and under Linux is one accelerator and two rigs.

Vision fails with an `e5rtError` when the Neural Engine is busy with another
client, which the check names `OcrTransient`. A test that reads text with the
real engine asks for the `real_ocr` fixture, which retries that one class up to
three times and then fails naming it; any other OCR error fails the test at once.

The generators do too, each through whatever tool suits the machine:

| lane | Apple Silicon | NVIDIA, on Windows or Linux |
|---|---|---|
| web, code, extract | mlx_lm.server | llama.cpp's router |
| image | mflux | diffusers |
| video | h3 | diffusers |
| tts | mlx-audio | Kokoro through onnxruntime |
| stt | mlx-audio, mlx-whisper | faster-whisper |
| svg | traced from an image, or a text model | follows image and text |
| ocr check | Apple Vision | Windows.Media.Ocr, or RapidOCR |
| html render | Chrome | Chrome, Edge or Chromium |
| svg rasterise | rsvg-convert | rsvg-convert |

Neither generator column names a model, which is on purpose. `mflux:z-image-turbo`
and `diffusers:stabilityai/sdxl-turbo` are both specs an eval reads, so which
model a lane should run on either machine is a row in a results table rather
than a line in this file.

Each machine has its own launchers and its own gateway config, and one
`serve-gateway.sh` reads whichever it is given:

```bash
GATEWAY_CONFIG=gateway/config.cuda.yaml ./scripts/serve-gateway.sh &
./scripts/serve-llamacpp.sh &      # the text lanes
./scripts/serve-audio-cuda.sh &    # tts and stt
```

The image and video generators need no server. `harness/engines.py` turns a
spec into a command line and the two scripts build their own environment on
first use, which keeps a 2.5 GB CUDA torch out of the venv the test suite
creates.

**A result records what produced it.** `hw_model` identifies the GPU on Apple
Silicon because it is the same part. On a PC it says nothing about it, so
results.json carries an `accelerator` field with the kind, the name and the
memory. Without it a run on unified memory and a run on a discrete card are the
same row to a score sheet, which is the one thing this suite exists to tell
apart.

**The memory ceiling is read off the card.** Unified memory hands the GPU a
fraction of system RAM. A discrete card is a wall, and the 61.6 GB of RAM behind
a 12 GB 4070 is not budget: computing it that way produced 46 GB that does not
exist. Qwen3-30B-A3B at 4-bit is too big for that card and fits a 24 GB one, the
same candidate and two answers. Without a card the harness still runs and
reports system RAM as its budget.

**Whether a machine comes back serving after a reboot is a decision about that
machine's job.** On one that exists to serve, `scripts/launchd.sh` installs
launchd agents with RunAtLoad and KeepAlive. On one that is borrowed -- a
workstation, a shared box, anything with a day job -- `scripts/services.sh`
registers nothing at all, so there is no scheduled task, no Run key, no startup
shortcut and no systemd unit to find later. One file covers Windows and Linux
both: the only differences between them are how a process is launched detached,
how it is asked whether it is alive, and how its tree is ended.

```bash
./scripts/services.sh start      # gateway, text, audio
./scripts/services.sh stop       # and the card is free
./scripts/services.sh status     # what is up, and what the card holds
./scripts/services.sh restart llamacpp  # the router sees a newly fetched GGUF; a fetch does this itself
```

The text server also gives the card back on its own after
`$LLAMACPP_SLEEP_IDLE` seconds, which matters more than the stop verb: a
session left running overnight is otherwise noticed as a frame rate rather than
as a log line. Measured on the 4070, a request takes VRAM from 2462 to 3296 MiB
and fifteen idle seconds return it to 2473.

Voice cloning is the one lane still missing there. On Apple Silicon a voice is
cloned from a reference clip by Chatterbox; Kokoro, which serves the card, has
a fixed table of 54 and no cloning, so a request carrying a reference is
refused rather than answered in a substitute voice.

### Every machine's report on one public page

Each machine keeps its own receipts, so a GitHub runner has nothing to measure.
Instead each machine publishes its own slice and a workflow renders the set at
https://unxmaal.github.io/SoHoT/ as three pages:

- the front page: what SoHoT is, the loop, the lanes and a quick start, with
  stat cards and a ticker computed from the exports at render time. Each
  number names the machines and dates it came from, and the lane list is the
  harness's own lanes plus any an export carries;
- `reports/`: a section per machine, a lane table across machines, and when
  each machine last published, so a quiet machine reads as stale rather than
  current;
- `benchmarks/`: one table per exam per lane on each machine that published
  one (candidate, pass, median, first, peak, metrics, run date). An exam is
  every measure run on that machine that `evals.core.comparable()` accepts
  as the same: case ids and digest, repeat, split, this lane's sampling,
  accelerator, place, devices, launch and instruments. Each candidate shows its
  latest result across those runs, so a two-candidate adopt run adds its two
  rows to the wider comparison instead of replacing it. The `serving`
  instrument is left out of that check because it is the union of the
  candidates' own engines, and a row always comes whole from one run. A changed
  case set starts a new table; tables run newest first, each labelled with its
  case count, repeats, how many runs it merges and its newest date. Reference
  rows (`claude-code:*`) sort last, are tagged `reference, not adoptable`, and
  are never highlighted as serving; the serving candidate's row is. Exports
  are `export_version` 2; the page still renders a version 1 export as its
  latest run's table. A candidate whose every case was
  refused by the harness or the gateway (a failure class whose reason is
  `harness`) shows as `not run (<class>)` with no score, never as 0/N.

The pages are self-contained HTML and CSS with fonts from Google Fonts, work at
phone width and in dark mode, and switch their animation off under
`prefers-reduced-motion`. To look at them locally:

```bash
python -m harness.publish render --data <dir of machine JSON> --out /tmp/site
python -m harness.privacy --files /tmp/site      # a directory means every file under it
```

The front page's "How it works" diagram is drawn from
`harness/how-it-works.workflow.json`, a workflow model in the format of
[Archify](https://github.com/tt-a1i/archify). Every box cites the code it stands
for, pinned to one commit, and links there. After changing the loop, edit the
model (or regenerate it with the Archify skill), set `meta.repository.revision`
to a commit on main, fix any line ranges that moved, and check it against the
repository with a local clone of Archify (no install; Node 18 or later):

```bash
make diagram ARCHIFY=<path to an archify clone>
```

`tests/test_site_diagram.py` fails if a cited file or line range no longer
exists, if any box links outside this repository, or if a box goes undrawn.

```bash
soh report --export              # $LOCALHARNESS_HOME/reports/<machine>.json, checked
soh report --publish             # write it to the reports branch and rebuild the page
soh report --auto-publish on     # publish at the end of every discovery loop here
soh report --auto-publish off
```

The JSON is keyed by a hardware label such as `Mac Studio M5 Ultra 96 GB` or
`RTX 4070 (Windows)`, never a hostname. It carries what each lane serves, that
candidate's pass rate, median and metrics, the latest run's full comparison
table per lane, adoptions, the schema and when it was generated. Paths are cut
to their last component, receipts' environment, artifacts and failure detail
are not exported, and the result goes through `harness/privacy.py` with this
machine's username and hostname added; any finding refuses the export.

Publishing writes `machines/<slug>.json` on the `reports` branch through the
GitHub contents API (one file per machine, never a force-push) using `gh`, then
dispatches `.github/workflows/pages.yml` on main. The workflow renders the site,
fails if the privacy scanner matches anything in any file under `site/`, and
deploys to Pages.

Two one-time steps: in the repository's Settings, Pages, set Source to "GitHub
Actions"; then on each machine that should appear, run `soh report
--auto-publish on` (it is off on a fresh machine).

---

# Using it

## Voices

`soh voices` lists them. The default is **`fr-male`**, a French-accented English
voice, cloned from a reference clip rather than picked from a table.

That works because Chatterbox clones *across* languages: the reference clip
speaks French, the output speaks English, and the accent comes along with the
voice. No accented-English corpus was needed.

It costs roughly 2s of fixed overhead per call plus about 1.5x the length of
the audio, so a line is the wrong unit: four words took 4.4s for 1.6s of
speech, twenty-eight took 11.3s for 6.2s. Measured 2026-09-12 on an M2 Pro that
was already swapping, so read them as an upper bound.

```bash
soh say "the tests all passed"                     # cloned, 4.4s
soh say "the tests all passed" --voice bm_george   # Kokoro, 3.5s
```

Use `bm_george` when a line needs to come back immediately.

## Piping

`code` and `extract` print to stdout, because you pipe or read them rather than
open them in a viewer. `extract` reads stdin when given no `--file`:

```bash
make test 2>&1 | soh extract "which test failed, and why?"
```

That is the lane for handing a cheap question to a small model instead of
spending a large one's context on a log.

## Keeping up with a field that moves weekly

This is the unusual part, so here it is in full.

Any tool like this is out of date the moment it ships. New models appear
constantly, and so do new *techniques*: a way of chaining two steps, a 200 MB
add-on file that makes a 20 GB model five times faster, a trick for running
something that should not fit in memory. Left alone, you keep using
whatever was good the week you set it up and never find out.

So `soh` looks, on your behalf.

**1. It works out what you have and what you have never tried.**

```bash
soh discover           # everything available here, and whether it has been tested
soh discover --gap     # just the untested things
```

It reads your installed tools, your downloaded models and the results of every
past test run. Nothing is hand-maintained, so the answer is right whenever you
ask rather than as of whenever someone last updated a list.

**2. It goes out and finds what exists now.**

```bash
soh discover --external --lane image   # ask the model registries
soh discover --sweep                   # every source family below, in one pass
soh discover --feeds                   # read where practitioners talk
soh discover --neighbors               # read what the people who build your tools star
```

Use `--sweep` rather than `--feeds` unless you mean only the feeds. `--feeds`
refreshes the Atom sources and leaves the star graph untouched, which left the
best-measured source nine days stale while the sweep reported success.

`--external` queries the HuggingFace registry. Good for "what models exist",
useless for anything that is not a single model.

`--feeds` reads community aggregation posts, because a registry can tell you a
model exists but not that everyone has moved to a small add-on file that made
generation five times faster. That is what it found on its first real run: a
*MiniMax-H3-Turbo* LoRA claiming a 5x speedup, against a video lane that takes
40 minutes a generation.

`--neighbors` is the newest and the highest signal. People who maintain the
tools you already run follow and star each other, and what they star is a
curated list rather than a popularity poll. `soh` starts from whoever contributes
to the tools installed here, follows their network outward, and ranks what that
crowd stars by how *concentrated* it is in them:

```
score = shared * log( (shared / crowd) / (stars / population) )
```

Both halves are needed and this was measured rather than assumed. Ranking on
the raw count returns whatever giant everyone stars. Ranking on the ratio alone
returns 33-star repos that four people happen to share. Six scoring functions
were tried against a control; one passed.

The first sweep found zero overlap with anything the feeds had ever produced,
so it is a second axis rather than a better version of the same one.

The crowd grows toward **consensus** rather than outward. Everyone considered
carries a count of how many people already in the crowd follow them, and a
second hop needs several of them to agree before someone joins. That matters:
two hops at the same crowd size found twelve repos one hop never surfaced,
while simply making the crowd bigger broke the ranking outright at every
setting tried.

**Comments, not just posts.** The comparative judgements are in the replies. A
post title says "what do you use for local image generation"; one reply names
four tools across four lanes and says which is best at what. Reddit serves a
thread's comments as a feed, so the same reader handles them:

```bash
soh discover --feeds --comments 5            # replies on the 5 newest posts
soh discover --feeds --comments 5 --judge    # and read names out of the prose
```

The judge is what reads the prose, because the linked repos are the easy half:
"Krea 2 being replaced with Anima" is a claim no registry can produce and no
pattern-matcher was going to find.

**3. It sorts proposals cheapest-first, so measurement is spent where it counts.**

A sweep produces far more proposals than this machine can run. Each tier is
more expensive than the last, and each one only sees what survived the one
before:

| tier | cost | what it answers |
|---|---|---|
| inspect | seconds, a source clone | can it run on this machine at all |
| rank | arithmetic over the store | what would a screen teach that is not already known |
| screen | one minimal run | does it run |
| measure | the full suite | is it better |

Inspect runs first because it costs seconds and produces facts. Ranking by
eventual quality was tried and does not work: a triage tier sits above the
tiers that produce quality, so it is asked to know what does not exist yet.
What it can order is the value of the information a screen would buy.

```bash
soh discover --sweep               # read every source family, not only the feeds
soh discover --inspect             # clone the source and check it fits
soh discover --queue               # what a screen would teach, best first
soh fetch                          # what is queued for download
soh fetch --run                    # download it, one at a time
```

**Inspect** is the one that saves the most. It clones a candidate's source,
which is single-digit megabytes, and reads what the description could not say:
which weights it names and how big they are, which RUNTIME they need as a
*declared dependency* rather than merely mentioned somewhere, whether there is
anything to call, and when it was really last touched. Nothing is executed and
nothing is downloaded.

The verdict names the runtime, not the operating system. `needs-cuda` on an
Apple Silicon machine and `needs-mlx` on a machine with an NVIDIA card are one
rule read from two directions, which is why each refuses a candidate the other
queues and neither is wrong. A repo offering both paths runs wherever one of
them lands.

The declared-versus-mentioned distinction is load-bearing. An Apple on-device
repo mentions `torch.cuda` in one export recipe; treating that as a CUDA
requirement threw away the best candidate in a sweep. So only a declared
dependency disqualifies.

Anything that cannot run here is recorded as answered, permanently, and never
proposed again -- here meaning this machine, so the two keep different books.
Anything that can is queued for download, and `soh fetch --run` takes them one
at a time with a disk floor, because either machine holds one working set.

**A weight is only queued if something here can measure it.** A model needs a
lane -- a case, a runner and a metric -- and this project has eight. A voice
activity detector and a speech enhancer both downloaded cleanly once and
neither could be scored by anything, so they sat on disk. Those are listed
now, and not fetched. It is not a verdict on the model: sometimes building the
lane is the work.

**4. You measure, and the result decides.**

Discovery never concludes anything. It hands you proposals, each with a source
URL, a date, and the command that would test it:

```
  inclusionAI/LLaDA-Image
    linked from: LLaDA-Image: a unified 6B image/edit model has been released
    -> uv run python -m evals.run --modality image --candidates ...
```

Run that, and it competes against what you already use on identical cases. That
is the whole loop: **it looks, it proposes with evidence, you measure, the
numbers decide.**

**Dev, holdout and power.** Each lane's cases are split by content digest into
`dev` (about 70%) and `holdout` (about 30%, at least 3). Anything that adapts to
failures, the screen included, runs on dev only (`evals.run --split dev`); `soh
adopt` and the discover loop decide on holdout and print dev beside it. A lane
with fewer than 6 cases is too small to split, decides on all of them, and the
verdict says so. Before measuring, `soh adopt` works out the repeat count the
paired test needs to see a gain of `--effect` (default 0.2 per case) from the
incumbent's stored pass rates, capped at 16. A loss with too little power is
recorded as `underpowered`, not as "no better", unless the challenger is
significantly worse on holdout, which is an ordinary loss whatever the power to
see a gain. `soh report` marks a lane
`saturated` when its incumbent passes at least 95% of holdout: those cases can
no longer separate candidates.

**The reply budget is part of the exam.** Every text request in an eval runs at
one token budget, the lane's from `completion.BUDGET` unless the run names
another with `evals.run --max-tokens N`, and the receipt records it. The code
lane's default is sized for a reasoning model, the others keep the old one.
`--compare` refuses to rank two runs at different budgets, and `--compare a b
--across max_tokens` reports what a bigger budget changed case by case. A reply
the budget cut off (no answer after reasoning, or `finish_reason` length on an
answer that then fails its checks) is recorded as `token_budget_exhausted` with
the limit it hit, and the comparison table counts it under `budget`, apart from
`wrong`.

**A budget ladder retries a cut-off reply at a larger budget.** `evals.run
--budget-ladder 4000,16000,32000,65536` asks each case at the first budget and,
only when the reply ends `token_budget_exhausted`, asks it again at the next; a
pass or any other failure stops the climb. A bare `--budget-ladder` takes the
lane's from `completion.LADDER`. Every candidate climbs the same ladder, with
rungs past its served context per slot (less the prompt) cut to that room. Each
row records its `attempts` (budget, tokens, seconds, outcome) and its `seconds`
is their sum. The receipt records `budget_ladder`, `max_tokens` is its top rung,
and `--compare` refuses a laddered run against a single-budget run or another
ladder. The report adds pass by rung: how many cases reached each budget, how
many passed there, and the seconds spent there.

**A run whose model the router evicted is not ranked.** For a candidate served
by llama-server's router (directly or through a gateway alias), the eval reads
the router's `/models` before and after each case. If the model was held and
then is not (unloaded, or reloading), the receipt's `router_swaps` records the
case, whether it was seen at the start or the end, and which models were
resident instead. `comparable()` refuses such a run against anything, itself
included, `--compare` and `--across` refuse it, and the benchmarks page leaves
it out. Two snapshots per case cannot see an evict-and-reload that finishes
inside one case on a one-model router; that case still shows up in its timing.

**Repeats of one case are not independent cases.** A model at low temperature
answers a case the same way most of the time, so the case, not the repeat, is
the unit. The power calculation and the adopt gate both measure the within-case
correlation `rho` (the intraclass correlation of the incumbent's stored draws,
and of both candidates' draws in the run) and shrink the cell count by the
design effect `1 + (repeat - 1) * rho`. At `rho` 1 a case repeated sixteen
times counts once; at 0 every repeat counts, as before. A lane with no repeated
case assumes `rho` 1 until a run measures it. The negative control is pinned: a
challenger that wins two cases and loses one, the same way on every draw, is
never adopted at any repeat (the old cell count adopted it from repeat 14).

`soh adopt --power` prints, per lane, the repeat that reaches power 0.8 to
detect `--effect` at alpha 0.05, the holdout size needed when no repeat up to
16 can (and how many more holdout and lane cases that is), and the projected
wall time from the incumbent's median seconds per case over every lane case,
both candidates, at that repeat. `--lane` narrows it; `--receipt
<results.json>` (repeatable) reads the draws from receipts instead of the store.
A holdout case the incumbent has never run takes a rate drawn from the
incumbent's measured per-case rates (a deterministic stratified draw over every
measured case in the lane), not their pooled mean: measured rates are bimodal,
and a pooled 0.77 gave every new case room to gain that a case at 1.0 does not
have (#608). A lane measured only at 1.0 therefore projects no power at all.
Whenever any holdout case is unmeasured the line says how many, and gives the
power over the measured cases alone beside the projection (`unmeasured`,
`measured` and `power_measured` under `--json`). Over the three 2026-10-07
adopt-code receipts, with 119 of 136 code holdout cases unmeasured, the
projection is power 0.80 at repeat 3 (it was 1.00 at repeat 1 with the pooled
rate) and the 17 measured cases alone give 0.00.
The measure tier spends the powered repeat when its projected time fits
`--power-budget-min` (default 120, on `soh adopt` and `soh discover --loop`);
otherwise it runs repeat 3, or less if that does not fit either, records the
null as `underpowered`, and the verdict says what the powered repeat would have
cost. An explicit `--repeat` is always honoured. More holdout cases come from
new case files; `soh discover --benchmarks` proposes the datasets to draw them
from.

It has already returned a result nobody asked for. `parakeet-tdt-0.6b-v3`, the
newer version of the speech model in use, is measurably *worse* than the v2 it
would have replaced. A version number is a hypothesis, not an upgrade.

Two rules keep it useful. A feed measures popularity, and popularity is not
quality. And a name someone typed in a sentence might be a typo or might not
exist at all, so every one is checked against the registry, GitHub repositories
included, before it reaches you.

**It remembers.** Every proposal, every sighting and every verdict goes into a
small database, so a thing already answered is not offered again, a thing seen
in three places over two months is ranked above a thing mentioned once, and
"we tried this and it lost" survives long enough to be worth something.

```bash
soh discover --recurrence     # what keeps coming back, and how each source is doing
```

Discovery goes stale, which defeats the point, so `soh` tracks when it last
looked and tells you in ordinary `soh discover` output when it has been too long.
Default is 7 days, set `$LOCALHARNESS_DISCOVERY_DAYS` to change it. Each kind
of answer is cached for less time than that, or a more frequent sweep would
just re-read what it read last time and report success.

```bash
soh discover --sources        # which places it reads, and when it last looked
```

It also watches for **new places to read**, since the site everyone uses
in a year may not be the one they use now. When a feed keeps pointing somewhere
`soh` does not read, it probes whether that address actually serves a feed and
tells you. Never automatic: a web address suggested by a stranger should need a
human to agree before this thing starts fetching it on a timer. Places it reads
live in `~/localharness/discovery-sources.json`; edit that file freely.

## Letting another computer use this one

If you use an AI coding assistant on a laptop, it can hand work to this machine
instead of doing it itself. The laptop asks for an image, whichever machine is
serving makes it.
That is done over MCP, a small standard for letting an assistant call outside
tools.

```bash
./scripts/serve-mcp.sh      # listen on 0.0.0.0:8899
claude mcp add --transport http soh http://<host>.local:8899/mcp
```

> **There is no authentication.** Anyone who can reach port 8899 can use this
> machine's GPU. That is a deliberate choice for a home network. On any network
> you do not control, bind to localhost instead: `MCP_HOST=127.0.0.1`.

That exposes `svg`, `web`, `code`, `image` and `video` to the assistant. Each of those shells
out to `soh`, so the CLI, the eval suite and the MCP server run identical
commands, and what gets measured is what ships.

`image` and `video` go on the work queue (see "The work queue"): they return a
job id, `job_status` carries the position and what it is waiting for, and
`job_result` returns the file. Copies stay here, in `~/localharness/out/mcp/`.
Speech is not exposed over MCP; that was ruled out.

Two more tools let the assistant hand a text subtask to the model a lane has
adopted here, through the same route `soh code` uses:

- `local_complete(prompt, lane="code", system=None, max_tokens=2048, temperature=None, thinking=None)`
  returns `text`, the `model` that answered, `ttft_s`, `seconds` and token counts.
  Lanes: code, web, svg, extract, decide. `system` replaces the lane's own system prompt.
  `thinking=false` sends `chat_template_kwargs: {enable_thinking: false}`, so a
  hybrid thinking model answers without reasoning first; left out, the lane is
  served as it was measured. `max_tokens` may go up to the context the model is
  served at less the prompt (estimated at 3 characters a token), read from the
  llama-server router's launch args or the stored context; when that is unknown
  the cap is 8192. A reply that spends the whole budget reasoning is an error
  that says how many tokens went to reasoning and suggests `thinking=false`.
- `local_decide(question, schema, context="")` is `soh decide`: the same flat
  schema, answers plus a probability per choice, from the decide lane's model.
  It is not streamed, so `ttft_s` is empty: through the gateway a streamed
  reply loses the logprobs the probabilities come from.

Both answer directly rather than queueing. Prompts are capped at 100,000
characters and calls time out after 300 s. When something holds the machine lock
(an eval, a ramp, an image or video generation) they still answer if the lane's
model is already loaded and memory pressure is normal, since that loads nothing.
Otherwise they refuse at once rather than wait, naming the holder, the queued job
running and how long earlier runs of that job took: a run holds the lock for
hours, and loading a lane's model beside it risks the swap that took a 32 GB
machine down, and skews the run's timings. Loaded means the llama-server router
reports the model's status as `loaded` (not `sleeping`) on `GET /models`, the
mlx_lm.server wrapper names it on `GET /sohot/loaded`, or a vLLM server lists it
on `GET /v1/models`; a gateway alias is followed to its upstream first, and a
server that cannot say counts as not loaded.

A snippet to paste into a project's CLAUDE.md:

```markdown
## Local model (SoHoT MCP)
Delegate to `local_complete` / `local_decide` when the work is cheap and checkable:
boilerplate, test scaffolding, docstrings, summaries of material you pass in,
bulk rewrites, and classifying many items against a fixed schema.
Do it yourself when it needs judgment, touches security-sensitive code (auth,
crypto, input handling, secrets), or needs repo-wide context you cannot pass in
the prompt. Review what comes back before using it; the local model sees only
the prompt. If the tool says the machine is busy, do the work yourself.
```

DNS-rebinding protection stays on, with an allowlist in `MCP_ALLOW`. It guards a
different thing than the missing authentication does: rebinding needs only that
someone here opens a web page, not that the port is reachable from outside.

## Use the adopted models from opencode / Claude Code

Every lane command uses what its lane has adopted on this machine, else the
typed default, and sends it to the server that serves it: a gateway alias to
the gateway, an MLX repo id to mlx_lm.server, `llamacpp:<stem>` to llama-server,
`vllm:<repo id>` to the vLLM server on `VLLM_PORT`, `ds4:<stem>` to ds4-server
on `DS4_PORT`.
`-m` overrides; `--gateway` sends the request somewhere verbatim.

An adoption serves the machine it was made on, whether measured (`soh adopt`)
or by hand (`soh judge`). `soh judge --all-machines` serves the winner on every
machine instead, and each machine still refuses it if its stored size or
measured peak is over that machine's memory ceiling. A by-hand adoption needs
10 votes (`--min-votes`) or `--force`, and records the votes, how far they
agreed, whether it was forced, and its median latency and peak against the
previous adoption on that machine. `soh report` prints each lane's adoption
with its scope and how it was made, and any adoption refused here.

```bash
soh code "Write is_palindrome(s)."          # the code lane's adopted model
soh code "..." -m mlx-community/Qwen3-4B-Instruct-2507-4bit   # a specific one
```

For other programs the gateway serves one stable alias per text lane:
`sohot-code`, `sohot-web`, `sohot-svg`, `sohot-extract`, `sohot-decide`, `sohot-agent`.
`scripts/serve-gateway.sh` writes `gateway/config.served.yaml` at start (the
base config plus those aliases, from the adoptions table), and an adoption in a
text lane switches the alias to the winner without anyone editing
`gateway/config.yaml`. An adopted GGUF is also served under its own stem.
llama-server runs with `--jinja`, so tool calls pass through.

### Model names

Every gateway entry is named by the id it sends upstream: a repo id for MLX
(`mlx-community/Qwen3-4B-Instruct-2507-4bit`), the GGUF stem for llama-server
(`Qwen2.5-7B-Instruct-Q4_K_M`). Only two kinds of alias remain: the lane aliases
`sohot-<lane>`, which adoption repoints, and `cloud-opus`. Reports, the
benchmarks page and `soh` output name the model a row measured, with the alias
beside it as "served as"; a run records what each alias resolved to in its
receipt (`resolved`), and an older row labelled only by a lane alias reads
"(model unknown)". Each benchmark row also says what the model is (base family,
parameter count, quantisation, publisher, source link), from the store's
proposals, lineage and downloads, or "unknown". Every gateway reply carries the headers `x-sohot-model` (the real
id) and `x-sohot-served-as` (the alias asked for, if any) from
`gateway/served_model.py`, streamed or not, so a client records what answered
without a lookup. The body's `model` stays the name asked for: LiteLLM 1.100.0
restamps it after every callback.

The old nicknames were renamed in #670 and stay one release as deprecated
entries (`deprecated_for` in the config), so queued jobs and clients keep
working; `soh` and `evals.run` print a warning naming the replacement:

| old | new (Mac) |
|---|---|
| local-small | mlx-community/Qwen2.5-0.5B-Instruct-4bit |
| local-mid | mlx-community/Qwen2.5-1.5B-Instruct-4bit |
| local-large | mlx-community/Qwen2.5-7B-Instruct-4bit |
| q3-1.7b | mlx-community/Qwen3-1.7B-4bit |
| q3-4b | mlx-community/Qwen3-4B-Instruct-2507-4bit |
| q3-8b | mlx-community/Qwen3-8B-4bit |
| q3-14b | mlx-community/Qwen3-14B-4bit |
| q3-30b | mlx-community/Qwen3-30B-A3B-Instruct-2507-4bit |
| q3-coder | mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit |
| eval-4b | Qwen3-4B-Instruct-2507-Q4_K_M |
| eval-imajev-4b | imajev-4b-Q8_0 |
| eval-7b | Qwen2.5-7B-Instruct-Q4_K_M |
| eval-12b | google_gemma-3-12b-it-Q4_K_M |

`gateway/config.cuda.yaml` names its GGUF builds by stem
(`qwen2.5-7b-instruct-q4_k_m`) and marks each with `in_place_of`, the Mac id it
stands in for, so a typed default reaches this machine's build. A stored run
of an old nickname still counts as the lane's measured row until the renamed id
has one of its own. Which server answers an entry is read from its `api_base`
(a GGUF entry carries `source_file` and goes to `:8082`), never from its name.

An alias switch does not drop a session. LiteLLM 1.100.0 can add or change a
model on a running gateway only with a Postgres database behind it
(`/model/new` and `/model/update` refuse without one), so the switch is a
restart, deferred: after any run holding the machine lock finishes, a detached
waiter polls the gateway's `/health/backlog` in-flight counter and restarts only
once nothing has been in flight for 60 s, which covers a streamed reply until
its last byte. If the gateway is never quiet for 20 minutes it restarts anyway
and logs that to `logs/gateway-switch.log`. A client sees at most a refused
connection between turns, which opencode and Claude Code retry. Each switch is
a `gateway_switches` row: lane, old spec, new spec, `idle` or `forced`, when it
was asked for and when it happened, so a client log can be lined up with it:

```bash
sqlite3 "$LOCALHARNESS_HOME/discovery.db" 'SELECT * FROM gateway_switches'
```

opencode, as an OpenAI-compatible provider (`opencode.json`):

```json
{
  "provider": {
    "sohot": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://<host>:4000/v1", "apiKey": "{env:SOHOT_GATEWAY_KEY}" },
      "models": { "sohot-code": {} }
    }
  }
}
```

Claude Code, through the gateway's Anthropic `/v1/messages` route:

```bash
ANTHROPIC_BASE_URL=http://<host>:4000 ANTHROPIC_AUTH_TOKEN="$SOHOT_GATEWAY_KEY" \
  claude --model sohot-code
```

## Gateway auth

The gateway listens on every interface so other machines on the LAN can use
this one, and it demands a key for every request (#482). LiteLLM with no master
key serves anyone who can reach port 4000, which now includes tool calling.

- **The key.** One per machine, generated on first use and never in the repo:
  the login Keychain on the Mac (service `localharness-gateway`), a 0600 file at
  `$LOCALHARNESS_HOME/gateway.key` elsewhere. `soh gateway key` prints it,
  creating it if missing; `soh gateway key --rotate` replaces it, after which
  the gateway must restart and every client needs the new one.
- **The gateway.** `scripts/serve-gateway.sh` exports it as
  `LITELLM_MASTER_KEY`, which the configs read as
  `general_settings.master_key`. With no key it refuses to start rather than
  serve unauthenticated.
- **In-repo clients** (`soh` lane commands, the MCP tools, the eval runners,
  throughput, rubric evals, the alias switch's backlog probe, `scripts/smoke.sh`) send it as
  `Authorization: Bearer`. `SOHOT_GATEWAY_KEY` overrides the stored key: set it
  to the serving machine's key when using another machine's gateway. A refused
  request says to run `soh gateway key`.
- **External clients.** `export SOHOT_GATEWAY_KEY="$(soh gateway key)"` and use
  the opencode and Claude Code settings above. LiteLLM takes the key as
  `Authorization: Bearer` (Claude Code's `ANTHROPIC_AUTH_TOKEN`, opencode's
  `apiKey`) or as `x-api-key` (`ANTHROPIC_API_KEY`).
- **The engines** (`mlx_lm.server` and llama-server, `:8081` and `:8082`) take
  no key, so they bind `127.0.0.1` and other machines reach them through the
  gateway. `MLX_HOST` and `LLAMACPP_HOST` override.
- **The error.** A missing or wrong key gets HTTP 401 naming `soh gateway key`
  (`gateway/key_auth.py`, #501), not LiteLLM's own 500 or 400 "No connected db".
- **The cluster judge** reads it from the Secret `localharness-gateway`, key
  `key`: `soh gateway key | tr -d '\n' | kubectl create secret generic
  localharness-gateway -n lh --from-file=key=/dev/stdin`.

`scripts/services.sh status` probes the gateway at `/health/liveliness`, which
LiteLLM serves without a key.

## What real use looks like

The gateway records every request it serves as a `gateway_requests` row, so the
loop can see the work done through it and not only the case sets. A row holds
when, the alias asked for (and its lane for `sohot-<lane>`), the upstream model
and the adopted spec it served, the client (the key's alias, else the first
token of the user agent, such as `opencode/1.2` or `claude-cli/2.0`), prompt and
completion tokens, time to first token (streamed requests only) and total time,
whether it streamed, how many tool calls it made and how many of those parsed
to a JSON object naming a tool the request offered, the finish reason, and the
error class of a failure.

No prompt or completion text is kept by default. To collect samples for
building cases later, turn it on; each prompt and completion is cut to 8000
characters and only the newest 2000 are kept:

```bash
soh usage --text on      # off again with --text off
```

`gateway/usage_log.py` is a LiteLLM callback loaded from the gateway config.
It only queues a row; a thread writes them in batches, and a failure to write
drops rows rather than failing a request. Only a gateway run from the deploy
checkout writes the live store; one started from another checkout drops its
rows. On an instant fake upstream it added 0.2 to 0.5 ms to the median request.

```bash
soh usage                          # per alias and model, the last 7 days
soh usage --lane code --since 24h
```

prints requests, tokens, TTFT p50 and p95, error rate and invalid tool-call
rate per alias and served model, then each alias switch in the window with the
same figures before and after it. When the error rate or the invalid tool-call
rate rose after a switch by more than three standard errors, with at least 20
requests (or tool calls) on each side, it prints a WARNING naming the switch.
Nothing is reverted automatically. `soh report` and the published page carry the
per-lane figures under "Real use", without clients.

---

# Setup and upkeep

## Install

```bash
uv tool install --python 3.12 --editable .
```

`soh` lands on PATH via `~/.local/bin`, in its own venv. It never touches the
system Python.

**`--python 3.12` is not optional.** `uv tool install` picks an interpreter
silently, and mflux once installed against 3.9 where every entry point died on
`int | None`. `--editable` keeps the checkout as the source of truth.

### Where the weights go

`./hf_root` in the checkout, unless you say otherwise:

```sh
export HF_ROOT=/Volumes/FAST/hf     # an external drive on a Mac
export HF_ROOT=D:/hf                  # a Windows box with a fast drive
export HF_ROOT=/data/hf               # a Linux box, wherever the room is
```

One location, checked for room and writability, and fatal if it is not usable.
There is no candidate list and no search: a search is how the wrong disk gets
chosen quietly, and the old list -- `/Volumes/FAST/hf`, `/Volumes/PORTABLE/hf`,
`~/.cache/huggingface` -- was one machine written into the repo, forked once for
Windows and again in shell.

Both `soh` and the service scripts resolve it the same way. `soh` has to do it
itself: on PATH it runs with nothing sourced, and an unset `HF_HOME` sends
huggingface_hub off to re-download what is already on the drive.

MiniMax-H3's 134 GiB checkpoint is read from `$H3_MODEL_DIR`, else
`~/localharness/MiniMax-H3` when it exists, else beside the weights root. The
video lane streams it from disk, so its speed is the disk's: on a Mac Studio an
external USB volume read 376 MB/s with the GPU idle, against 5.27 GB/s from the
internal SSD. A copy or symlink at the home path moves it without moving the
cache.

`HF_MIN_FREE_GB` is 20, which is enough for a normal model and deliberately not
enough for MiniMax-H3. `fetch-h3-weights.sh` demands its own 160GB and
`setup-omnisvg.sh` its own 40, where the size is actually known. A global floor
cannot know what is about to be fetched, and set to 160 it refused every
ordinary machine.

### What the weights cache holds

Every download is a row in the store's `downloads` table: repo, kind (hub
or gguf), path, bytes, whether it was complete (config or weights present, not
just a card), when it started, finished and was removed, and on which machine.
The fetch tier and `scripts/fetch-gguf.sh` write it; `have()`, the screen's
readiness check, the memory guard and the GGUF route read it, never the hub
directory names.

`soh disk` lists every hub repo and GGUF file with its size, grouped as
`keep` (a gateway alias, a lane default, an adopted winner, a measured verdict,
or the tooling list in `harness/disk.py`), `queued` (discovery still wants it),
`rejected` (latest verdict broken or declined after a fetch) and `unknown`.
It also reports drift: rows whose path is gone, and paths no row explains.
A path with no row is never deleted; `soh disk --record` gives each one a row
and stamps gone rows removed.
`soh disk --delete rejected` or `--delete unknown` removes only what the shared
`safe_to_delete` rule allows, asks first, and needs `--yes` under `--json`.
Each removal stamps `removed_at` on the path's download row.

Rejected weights stay for 24 hours, so a rejection that turns out to be a
harness fault can be re-screened without a download. `soh discover --loop --run`
then removes them itself at the start of each run, along with partial
downloads (`*.incomplete`) of a recorded download no fetch is still writing. Nothing outside the hub and
GGUF directories is ever deleted, and if any keeper cannot be read, nothing is.
An mflux preset such as `mflux:dev` counts as the repo it loads
(black-forest-labs/FLUX.1-dev), per `MFLUX_REPOS` in `harness/disk.py`. A lane
default or adopted winner that still names no weights keeps every candidate of
its lane, and every candidate with no recorded lane, and cleanup goes ahead for
the rest. A gateway alias that names no weights still stops all deletion.

A candidate rejected at the screen or measure tier is retested up to three
times, each about a week after its latest rejection, because many rejections
have turned out to be harness faults. At the start of each `--loop --run`,
before the disk cleanup, each due retest is reopened to `queued` as a named
`retest` and goes through fetch and screen again, within the same budget and
backpressure. A rejection waiting on a stated condition (too big, needs a
runtime, dead upstream) is not retested; it reopens when that condition is
met. `soh discover --queue` prints how many retests are due, scheduled and
final, and how many retests then passed (recovered false negatives).

**Re-verification.** A served model is measured once, when it is adopted, so
each `--loop --run` (after the retests) also re-checks every wanted lane's
served model. It queues a low-priority job, never an inline run, when a runtime
that lane uses (mlx-lm, llama.cpp, mflux, LiteLLM, macOS and so on) has a
different version than the model's last passing run recorded, when the lane
has no passing run here within its threshold (7 days by default; `soh reverify
--lane code --days 3` changes it), or when `soh usage` saw real use regress
after the lane's alias last moved. A lane gets one queued re-run at a time. The
next check reads the run that job stored: no passes, or more cases lost than
gained against the last pass at the adopt tier's significance, is recorded with
its reason and failure class and flagged in `soh report`. The served model is
never swapped; that is left to a person. `soh reverify --dry-run` says what
would be queued.

**Exit status.** `soh discover --loop` exits nonzero when any step did, and
its last stderr line names them, for example `loop exit 1: sweep reported a
failure`. A lane with no benchmark queries (ocr, retrieval, pii and others) has
no benchmark source to read, which is not a failure.

**Lanes wanted.** `soh discover --loop` ends with the candidates no lane can
test, because adding a lane is a decision for a person. Adapters are left out,
as the ranking leaves them out, and a GitHub repo with no model task is listed
separately under `engines/tools wanted`. The rest are grouped by task (the
card's task, an OCR tag, or a task named in the card's tags), and each group
shows its distinct models, distinct publishers, total sightings, whether any
model fits this machine's memory ceiling (`unknown` when none has a measured
size), and whether the task has a reference-based metric (translation, OCR and
image-to-text, text ranking and retrieval, token classification; `unknown` for
anything else). Whether anything here would use the answer cannot be computed,
so each group ends with that as a question. The report decides nothing.

A candidate with no task on its card that was built from a known text model
(Qwen, Llama, Mistral, Phi, Gemma, DeepSeek, not their vision, speech or
embedding variants) is filed under code; an any-to-any card whose tags name
exactly one narrower task takes that task's lane; a text-to-audio card whose
tags mention music goes to music. Schema 52 applies these to stored rows that
have no lane, and only those; it records no verdict.

The install carries no torch on Apple Silicon. `mlx-whisper` needs it
unconditionally, so the multilingual ear lives in the `whisper` dependency
group: 370MB installed rather than 1.1GB. Dependencies are marked by platform
in `pyproject.toml`, so a Windows install pulls neither mlx nor pyobjc and a
Mac pulls no winsdk.

## Bringing up a Linux machine

Written against Ubuntu 24.04 with an NVIDIA card. Any distribution will run
this; 24.04 is the one under test, because `check-linux` runs on GitHub's
`ubuntu-24.04` runner, so a red `check-linux` is a real failure rather than a
runner-only one.
On anything else the package names in step 2 are the part that changes.

Nothing here is a lane. Linux needed instruments and launchers only, because
`harness/machine.py` asks which runtimes are present rather than which
operating system it is on, so every CUDA implementation came over unchanged.
That also makes this the cheapest machine to add if you already run Windows on
the card: the two can share a disk, and neither registers anything with its
operating system, so dual booting costs nothing here.

### 1. The driver, before anything else

```bash
sudo ubuntu-drivers install
sudo reboot
nvidia-smi                      # name, total and free memory
```

`nvidia-smi` ships with the driver, and it is the whole cuda probe: no toolkit
and no Python binding is needed for the machine to identify itself. With Secure
Boot on, the install queues a MOK enrolment and the driver does not load until
you complete the blue screen on the next boot. A machine that reboots straight
back to a working desktop with no `nvidia-smi` is usually that.

### 2. Packages

```bash
sudo apt install -y git curl make shellcheck librsvg2-bin ffmpeg time \
                    fonts-dejavu-core zsh sox
```

Each earns its place: `librsvg2-bin` for the ink lane, which silently skips
without it; `ffmpeg` for the video checks; `time` because `harness/proc.py`
measures peak memory with `/usr/bin/time -v` and **raises rather than reporting
a zero** when it is missing, and the `time` most shells have is a builtin that
reports no memory at all; `fonts-dejavu-core` because the OCR fixture renders
real type and PIL's bitmap fallback would measure the fixture; `zsh` because the
`env.sh` tests run in three shells and a missing one is a silent skip; `make`
because step 7 runs `make check`; `sox` for `rec`, which the stt lane records
with and `soh discover` reports missing without.

### 3. A browser, and specifically Chrome

```bash
curl -fsSLo /tmp/chrome.deb \
  https://dl.google.com/linux/direct/google-chrome-stable_current_amd64.deb
sudo apt install -y /tmp/chrome.deb
```

**Not the snap.** `chromium` on Ubuntu is a snap, which is confined and cannot
read a page written to a temporary directory, so `render.py` skips any candidate
resolving under `/snap`. And measured on a runner carrying both: the Chromium
build there writes no screenshot at all when handed its own `--user-data-dir`,
while Chrome at the same version is fine either way. The code retries without
the profile and remembers, so a Chromium-only machine still works; it pays a
timeout on its first render.

`LH_CHROME` names a browser explicitly and `LH_CHROME_PROFILE=0` drops the
isolated profile, for a machine that has opinions about both.

### 4. The repo, uv, then soh

```bash
git clone https://github.com/unxmaal/SoHoT.git
cd SoHoT
curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --python 3.12 --editable .
export PATH="$HOME/.local/bin:$PATH"
export HF_ROOT=/data/hf            # wherever the room is
```

`--editable .` is the checkout, so the clone comes first. Only `HF_ROOT` needs
setting: `scripts/env.sh` exports `HF_HOME` from it, and the two name the same
directory everywhere below.

### 5. The text lane

`serve-llamacpp.sh` cannot install its own server, and there is no winget here.
There is also no CUDA release binary to take: every `-cuda-` asset `ggml-org`
publishes is Windows, and `llama-*-bin-ubuntu-x64.tar.gz` is CPU-only. On Linux,
CUDA means building it.

```bash
sudo apt install -y cmake build-essential gcc-12 g++-12 nvidia-cuda-toolkit
cmake -B build -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=89 \
               -DCMAKE_CUDA_HOST_COMPILER=/usr/bin/g++-12
cmake --build build --config Release -j
export LLAMACPP_BIN=/path/to/llama-server     # or put it on PATH
```

Neither flag is optional. Noble's `nvidia-cuda-toolkit` is CUDA 12.0, whose
`nvcc` refuses any host compiler newer than gcc-12 while 24.04 defaults to
gcc-13; pointing CUDA at `g++-12` fixes that without moving the system compiler.
Without `CMAKE_CUDA_ARCHITECTURES` cmake builds every architecture rather than
the one card present -- 89 is Ada, and `nvidia-smi --query-gpu=compute_cap`
names yours.

Then check the backend, not the exit status: `llama-cli --list-devices` has to
name the card. A CPU-only build also exits 0 and also serves, with the card
idle, which is the whole failure being guarded against here. An apt
`llama.cpp`, where a distribution has one, is usually that. The prebuilt
`-vulkan-x64` asset does drive the card and needs no toolchain, but a Vulkan
backend and a CUDA one are two instruments, and `comparable()` will refuse them
a shared table. Record whichever build you end up with in `scripts/versions.sh`,
beside the versions the other machines report.

GGUF weights go in `$HF_HOME/gguf`, beside the cache rather than inside it:
huggingface_hub owns `$HF_HOME/hub` and nothing else belongs in there.

### 6. The other lanes install themselves

`scripts/diffusers-venv.sh` builds the image and video environment on first use,
in a venv of its own so the test suite never drags a CUDA torch in.
`serve-audio-cuda.sh` brings its own wheels through `uv run --with`, including
the two nvidia ones that carry the CUDA runtime CTranslate2 needs, so no system
CUDA install appears anywhere.

### 7. Run it

```bash
GATEWAY_CONFIG=gateway/config.cuda.yaml ./scripts/services.sh start
./scripts/services.sh status        # what is up, and what the card holds
./scripts/smoke.sh                  # the seams, not the models
uv sync && make check               # the suite
soh discover                         # what this machine can do
```

`services.sh` registers nothing with Linux, exactly as it registers nothing with
Windows. Same file, same verbs; the only differences are how a process is
launched detached, how it is asked whether it is alive, and how its tree is
ended. There is no systemd unit and that is a decision, not a gap: a machine
with a day job should start these when asked and not before. If this is the
machine you want serving on boot, write the unit yourself -- nothing in the
harness will fight you.

### What is different here, and worth knowing before it surprises you

`soh say --play` uses `paplay`, `aplay` or `ffplay`, whichever is present. Text in
a generated image is read by RapidOCR rather than by an engine the operating
system ships, so it installs with the project through a `sys_platform` marker.
Peak memory is `ru_maxrss` rather than a footprint, which means it cannot see
pages that were swapped out, and a run measured here is refused a shared table
with one measured on Windows for exactly that reason -- the same card under two
operating systems is one accelerator and two rigs.

## Keeping the services up

The answer depends on what else the machine is for. A machine that exists to
serve gets launchd agents and comes back after a reboot. A machine with a day
job gets nothing registered at all: no scheduled task, no Run key, no startup
shortcut, no systemd unit. One `scripts/services.sh` covers that case on
Windows and Linux alike.

### On Apple Silicon

Installed as launchd agents, so the machine comes back serving after a reboot:

```bash
./scripts/launchd.sh probe       # can an agent read the weights volume?
./scripts/launchd.sh install
./scripts/launchd.sh status
```

The agents do not run the checkout you work in. `install` puts `origin/main` in
a separate worktree, `$LOCALHARNESS_HOME/deploy`, and points every agent there,
so checking out a branch here changes nothing that is serving. After a merge,
run `install` again to deploy it; `status` names the commit the agents are
running and the current `origin/main`. `deploy` refuses to overwrite local
edits made inside the deploy worktree.

**Before deploying to a machine that has not deployed in a while, run
`soh store dry-run` there from a checkout of the commit you are about to deploy.**
It migrates a read-only copy of that machine's store with this checkout's code,
checks the invariants CI checks on the golden stores in `tests/golden/`, and
exits 1 if any fails; the live store is never opened for writing.

After a deploy, `install` runs `soh audit` from the deploy checkout, and the
`audit` agent runs it again every night at 03:00. It opens the live store read-only
(SQLite `mode=ro`, so it can neither migrate nor write), runs the same
invariants, and adds four about serving: every adoption has a passing run on
this machine, no lane serves a reference model, each `sohot-<lane>` alias in
the served config is the adoption, and no alias's requests through the gateway
fail more than the threshold in `harness/live_audit.py`. A fifth,
`sweep_on_schedule`, fails when the last discovery loop started more than 25
hours ago. A failure exits 1 and names the rows.

The `discover` agent sweeps once a day at 21:00, out of the way of daytime work
(`StartCalendarInterval`; override with `DISCOVER_HOURS`). It used
`StartInterval`, which counts from when the agent was loaded, so a day of
`install`s postponed the sweep all day (#625). Neither periodic agent runs at
load (#328).

**Run `probe` first, always.** macOS TCC denies `/Volumes` to launchd jobs, and
the failure is horrible unprepared: the volume stats fine, reports free space and
appears in `/Volumes`, so nothing looks wrong until mlx_lm hangs forever inside
`os.listdir`, accepting connections, answering none, logging nothing, at 0% CPU.

The fix is granting Full Disk Access to **`/bin/bash`** (System Settings →
Privacy & Security → Full Disk Access, then Cmd+Shift+G to reach `/bin`). TCC
propagates the grant to child processes. It is broad, and that is the trade: a
dedicated binary keys its grant to a code hash and needs re-granting on every
rebuild, and `uv` is a symlink into `Cellar/uv/<version>/` so a grant to it dies
on the next upgrade. `/bin/bash` is SIP-protected and never moves.

Restart one after editing it:

```bash
launchctl kickstart -k gui/$UID/com.unxmaal.localharness.mcp
```

Or run them by hand, from a terminal that already has the access:

```bash
./scripts/serve-mlx.sh       # inference engine on :8081
./scripts/serve-gateway.sh   # gateway on :4000, the only address clients use
./scripts/serve-tts.sh       # Kokoro TTS + Parakeet STT on :8890
./scripts/smoke.sh           # check the whole chain still works end to end
```

### Swapping the disk the weights live on

```bash
soh volume status        # mounted?, which services and processes hold files there
soh volume stop          # then eject, swap, remount
soh volume start
soh disk speed           # before and after, on the same volume
```

`stop` pauses the queue and waits for a running job, boots out the model
servers, the discover agent and any other agent with a file open on the volume
(found with `lsof`), and records what it stopped in `volume-stopped.json` under
the home. It exits 1 and names every process that still has a file open there,
so do not eject until it says it is safe. `start` refuses while the volume is
not mounted, loads exactly what `stop` stopped, and resumes the queue only if
it was running before. Both default to the volume holding `HF_HOME`; `--path`
names another. `soh volume` is macOS only, because it drives launchd and
`lsof`, and refuses on Linux and Windows; `soh disk speed` runs everywhere.

`soh disk speed [path]` writes a scratch file on the volume and reads it back
sequentially with the page cache off (`F_NOCACHE` on macOS), so no `purge` or
remount is needed. Each run appends one row to `disk-speed.jsonl` under the
home: volume, device and bus, bytes, block size, method, read and write rates.
`--mib` sets the size.

### On a machine with an NVIDIA card

```bash
./scripts/services.sh start      # gateway on :4000, llama.cpp, audio
./scripts/services.sh status     # what is up, and what the card holds
./scripts/services.sh stop       # and the card is free
./scripts/smoke.sh               # the same end-to-end check
```

Nothing is registered to start on its own. There is no `probe` step because
there is no TCC: the equivalent trap on Windows is a path, not a permission --
Git Bash reports `/d/a/...` where native Python needs `D:\a\...`, and
`env.sh` converts before exporting `HF_HOME` for exactly that reason.

## Running it in a cluster

`deploy/localharness` is a Helm chart for the discovery tiers: the weekly
sweep as a CronJob, inspect as an Indexed Job whose workers each take a slice
of the candidates, and a measure Job that asks for a card. The store becomes
Postgres, because several pods write to it and SQLite's locking is only as good
as the filesystem under it. `soh` on a laptop keeps its SQLite file and needs no
cluster at all.

```bash
make image                                   # build, and import it where the kubelet looks
helm install lh deploy/localharness -f deploy/localharness/values-local.yaml
```

Three profiles, and what has actually happened to each:

| profile | what it is | status |
|---|---|---|
| `values-local.yaml` | Docker Desktop's Kubernetes, no card, the in-chart Postgres | RUN: a real sweep, three sharded inspect workers, verdicts written back |
| `values-gpu-host.yaml` | a node whose card is driven by its host | RENDERED, never run |
| `values-eks.yaml` | a managed node pool, GPU Operator driving the card | RENDERED, never run |

Rendered is not a synonym for working. Every assertion in
`tests/test_deploy_chart.py` reads the rendered YAML, which catches a job that
forgets to ask for a GPU and cannot catch anything about a cluster nobody here
has.

### The tiers, and what each one costs

The ladder is sweep, inspect, judge, screen, measure, cheapest first. Three of
those run in the cluster today:

| tier | what it does | where |
|---|---|---|
| sweep | reads the feeds, writes proposals | `in-pod`, on a CronJob |
| inspect | reads a candidate's source or its model card, with nothing downloaded | `in-pod`, sharded across an Indexed Job |
| judge | scores what inspect queued, against a rubric | the pod asks, a model outside the cluster answers |

A lane's winner is read rather than typed:

```bash
soh discover --winners                     # the receipts against the constants
python -m evals.run --modality image --from-winners
```

A deployment built around "the current winner" has to read it, or it is built
around whatever was true the last time somebody edited a source file. With no
receipt for that lane the run refuses instead of falling back to a constant,
because a silent fallback is the constant again with a flag on it that makes
the claim look checked.

The judge refuses to score at all unless its control separates known-good from
known-bad on every run, and the control has to be shaped like the data that
tier will actually judge. A rubric that does not discriminate produces numbers
rather than a ranking, and a tier scoring a queue unattended has nobody present
to doubt it.

That refusal fires today. Asked to rank model candidates the judge floors every
one of them, because the rubric rewards a measured claim and a registry card
has never carried one. So the queue is ordered by arithmetic instead:

```bash
soh discover --queue        # what a screen would teach us, best first
```

It ranks by the value of what a screen would find out, not by a guess at which
candidate wins: a lane with no run receipt, a thing seen repeatedly, something
that is not a requantised copy of a model already being served, small enough to
try cheaply. Which candidate is better is what the tiers below it are for.

The screen is the tier that answers it:

```bash
soh discover --screen         # what it would run, and what is in the way
soh discover --screen --run   # actually run one
```

It asks one question -- did it run, did it emit anything -- and writes the
answer back, because that is how things have actually failed here rather than
by scoring slightly worse. **It never downloads.** Fetching is its own step with
its own disk budget, and a tier that pulls gigabytes because something ranked
well is how a laptop fills up overnight; a candidate whose weights are absent
is reported as waiting, not screened and not failed.

Each candidate is screened on a case its method can take: a case that declares
`methods:` is only offered to those methods, so OmniSVG screens `icon-gear`
rather than `chart-bars`. When no case in the lane fits a candidate's method,
it is not screened and not recorded as a harness error; it is reported under
runners wanted, with the cases and the methods they take.

### What discovery never saw

Extraction precision measures the quality of what gets caught. Reach is a
different number and nothing measured it:

```bash
soh discover --coverage
```

It takes what this machine actually runs -- the gateway's upstream ids, the
candidates in run receipts, the typed lane defaults -- and asks which
configured source ever surfaced each. Anything adopted that no source produced
is a coverage hole with a name, and the names are the useful part.

### Methods are candidates

A method is a named transform over a base candidate, registered in
`harness/methods.py` with its spec grammar, the lanes it applies to and the
case methods it can take:

```
trace:<image engine>        svg     draw a raster, then vectorize it
trace-icon:<image engine>   svg     the same with the icon preset
best-of:<n>:<base>          code, web, svg      sample n at the lane's temperature,
                                    keep the one the lane's own checks score highest
plan:<base>                 code, web, svg, extract   a plan, then the answer: two calls
```

The base is any spec its runner already takes (a gateway model id, `llamacpp:<stem>`, a
repo id, `mflux:flux2-klein-4b`), and the method runs over that runner and its
route, so it needs no server of its own. A method spec is a candidate
everywhere: `evals.run --candidates <id>,best-of:3:<id>`, the screen and the
paired adopt gate. Its rows carry `method_calls` (and for best-of
`method_samples` and `method_chosen`), reported in the table and never ranked
on; its seconds are the whole method. The receipt records each method and its
base, and the adopt verdict stores the cost beside the incumbent's (calls,
median latency and their ratio). `soh discover --loop` crosses each method with
each lane's incumbent (trace with the image lane's) and queues the combination
under `methods`, so a newly registered method is measured without anyone
typing a spec. A crossing downloads nothing, so it never holds back a fetch. In
the tests, a method that returns its base unchanged measures as a tie and one
that corrupts the output loses.

A method that wins is adopted like a model, with an adoptions row, and served
over its base's route:

- `soh code`, `soh web`, `soh svg`, `soh extract` and `soh prompt` run the
  adopted method (plan's two calls, best-of's n samples kept by the lane
  command's own check, since a one-off prompt has no case assertions). An svg
  lane that adopted `trace:<engine>` draws with that engine. `--json` adds
  `method` (method, base, calls, and for best-of samples and chosen); without
  it the same line goes to stderr.
- `local_complete` (and `delegate.complete`) run it too and return the same
  `method` field, with tokens summed over every call. `thinking` and the
  context-sized `max_tokens` cap apply to each call a method makes, and a
  budget refused or spent on reasoning names the call (`call 2 of
  plan:<id> (the answer)`). A lane serving a trace
  method refuses delegation and says to run `soh svg`.
- The gateway's `sohot-<lane>` alias keeps serving the method's base model. A
  LiteLLM alias is a routing entry and cannot run Python, so a LAN client that
  asks the gateway directly gets one call to the base; only the lane commands
  and the MCP run the method. A trace method's base is an image engine, so the
  svg alias is absent while one is adopted.

### Benchmark discovery

Lanes saturate and leak into training data, so the yardsticks get discovered
too. `soh discover --sweep` (and so the loop) reads HuggingFace datasets and
GitHub per lane once per discovery interval, newest and most liked, into a
`benchmarks` table: license, gating, dates, size and task format.
`soh benchmarks --training REPO=ALIAS --lane L` reads a candidate's cutoff
(stated on its card, else its release date, a quantization's parent's) and the
datasets its card declares. `soh benchmarks [--lane L]` then marks each source
`declared` (a candidate trained on it), `predates` (older than the newest
cutoff, or a copy of something that is), `after-cutoff` or `unknown`, shows
how many cases came from it, and shows each candidate's lane score with and
without the cases it trained on. `soh benchmarks --import SOURCE --n N` writes
cases with provenance (source, item, revision, license) through a converter
for that task format; the #479 digest split places them. `--probe MODEL` asks
a model through the gateway to continue each case's source text from a
prefix and records how much it reproduced verbatim. A miss is not proof the
model never saw the text.

### Importing cases

The adopt gate counts cases, not repeats, so a lane needs about 130 holdout
cases to decide (#595, #603). Those come from importers, never by hand.
`soh cases import SOURCE` re-runs one and `soh cases control SOURCE` runs its
negative control over every case it wrote. An importer is a module
`evals/importers/<source>.py`, listed by one line in `REGISTRY` in
`evals/importers/__init__.py`, that defines:

- `LANE`, `SOURCE` (`hf:<id>`), `DATASET`, `URL`, `LICENSE` (one of
  `ALLOWED_LICENSES`) and `REVISION`, the pinned 40-hex dataset revision;
- `fetch()`, the rows at `REVISION`, read over the network only here;
- `convert(row)`, an `Imported(case, reference, files)` or `None` to skip the
  row, with `case["attribution"] = provenance(module, item, transform, date)`;
  `files` are (path, bytes) written beside the case, such as an ocr image;
- `RESPONDERS` (name to a function from a loaded case to an answer) and
  `passes(case, answer)`, the negative control: `reference` must pass, a
  constant must not.

`importers.run(name, rows=...)` converts and writes through `write()`, which
replaces `evals/cases/<lane>/<source>/` in id order, references in its
`reference/`, so the same rows give the same files and case digests. Every
file starts with an `Imported by evals/importers` header, which keeps the
prose scanners off dataset text. Tests feed a recorded slice from
`tests/fixtures/importers/` and run a slice of the shipped cases; the full set
is `soh cases control`.

`leetcode` (code lane) imports LeetCodeDataset (Apache-2.0) at a pinned
revision: its whole temporal test split, items dated 2024-08 to 2025-03, plus
train items dated 2024 or later, read from the file's tail by a byte range.
Each case is the problem text and `class Solution` starter, the first ten
short `check(candidate)` asserts as checks, and the dataset's completion as
reference. Rows needing `ListNode` or `TreeNode`, checks naming a helper, and
items whose checks all expect one value (a constant could pass) are skipped,
as is any row whose reference fails its own checks. Each case records its
item date, so a receipt can be read against a candidate's cutoff.

The dataset's `prompt` field is LeetCode's own preamble (`from typing import *`,
collections, heapq, bisect, math, `inf`, the list and tree helpers). It is
written once as `preamble.py` beside the cases, and the code check runs it
before the reference and before every answer alike, so `nums: List[int]`
needs no import of its own (#657). Any code case directory may carry one; it
enters the case digest, so receipts from before it are not comparable with
receipts after. Rows whose prompt imports more than the preamble (24 import
`sortedcontainers`) are skipped. The negative control over every shipped case
is pinned in `tests/test_importers.py`.

`squad2_check` (decide), `squad2_extract` and `squad2_retrieval` read SQuAD 2.0
dev (CC BY-SA 4.0) at a pinned commit of rajpurkar/SQuAD-explorer, checked by
sha256. Decide asks five questions per paragraph, mixed answerable and not;
extract wants one human answer exactly (`equals` takes a list of
alternatives); retrieval ranks one article's paragraphs, a corpus every case
of that article carries in `Imported.files` (cases may share a path only with
the same bytes), and passes only when the gold one is
first (`k: 1`, because bm25 passes 96% at recall@5 and the holdout would be
saturated). `librispeech` (stt, CC BY 4.0) keeps a committed manifest of 480
test-clean clips pinned by sha256 and fetches the audio (42 MB measured on
2026-10-07) into `corpora/librispeech-rows` beside the weights root, or
`$LIBRISPEECH_CACHE`; its cases are gitignored like `evals.corpora`'s, pass
only on every word right (`max_wer: 0`), and replace a generated case for the
same clip. `python -m evals.importers.librispeech` rebuilds the manifest.
Neither source has per-item dates, so these cases carry a `date_note` bound
instead; `trained_on` names the measured candidates whose cards list the
source (the parakeet cards list LibriSpeech's train splits).

### Local-only cases: the claims lane

The `claims` lane measures infovore's claim extraction: a system prompt, a
numbered chat transcript as the user turn, and a reply held to a JSON schema
(`{"c": [[user, claim, [refs]]]}`, claim at most 220 characters) at
temperature 0 and 400 tokens. It is a schema lane, so it is served and adopted
only on llama-server; the typed default is `Qwen2.5-7B-Instruct-Q4_K_M`.

Its cases are verbatim chat text, so they never enter this repo. They live
under `$LOCALHARNESS_HOME/cases/claims/`, written by:

    uv run python -m evals.claims_import [EXPORT] [--negatives PATH] [--out DIR]

`EXPORT` defaults to `$LOCALHARNESS_HOME/import/claims.jsonl` (infovore's
`claims export-cases`), the negatives to `claims-negatives.jsonl` beside it
(absent is fine). A negatives line has `"reviews": []`, `"expect_empty": true`
and a `basis` (such as `human_irrelevant`) and optional `origin`, carried onto
the case. The importer replaces the directory on every run and refuses a
destination inside a git work tree.

The reply schema has one source, infovore's export. The importer refuses an
export (with its negatives) whose cases disagree on the schema, and writes that
one schema to `schema.json` beside the cases. `delegate.complete("claims")`
(and so `soh` delegation to the lane) serves that file and refuses until an
import has written it; the lane keeps no copy of its own. The schema under
`tests/fixtures/claims/` is the synthetic fixture only.

`evals.run`, the holdout split and the screen read the shipped tree plus
`$LOCALHARNESS_HOME/cases` (`--local-cases` names another), and mark those
cases private. `case_digest`, the split and `comparable()` treat them as any
other case. Run the lane with:

    uv run python -m evals.run --modality claims --candidates Qwen2.5-7B-Instruct-Q4_K_M

Each review carries the `interface` its reviewer judged it in: `cited-only`
(only the cited lines) or `conversation` (the whole conversation). These are
different labelling functions, so a case is scored once per interface and
the two are never pooled. A review without `interface` (an older export)
reads as `legacy`; any other value is refused at import and load.

Scoring: the reply must validate against the case's schema, else the case
fails. Per interface, each emitted claim is matched one to one to a reviewed
claim with the same user, overlapping refs and a similarity of at least
`MATCH_THRESHOLD` (0.5). The similarity (matcher version 2, #662) is Dice
over content words, each weighted by its smoothed IDF over the case's own
reviewed claims, so a word every review shares (the part under discussion)
weighs least. Stopwords, reported-speech framing ("they", "user", "says",
an anonymised handle) are dropped, number words read as digits, negations
read as "not", and plural and tense endings are stripped. An interface passes when no matched claim was
reviewed `made_up` or `wrong` and at least `MIN_RECALL` (0.5) of its `good`
claims were recovered, and the case passes when every interface does; the
reason names the interface that failed. Metrics are named
`claims_<metric>_<interface>` (`claims_recall_conversation`,
`claims_precision_cited_only`, `claims_pass_legacy`, ...); unmatched claims are
counted (`claims_unreviewed_<interface>`) and not judged, and
`claims_verbatim_*` and `claims_near_miss_*` count matches with identical
words and good claims missed by under 0.15 of the threshold. An
`expect_empty` case passes only on an empty list and reports
`claims_empty_rate` apart. Both
thresholds are knobs on the receipt, and a failed case records which one
decided it (`claims_min_recall`, or `claims_match_threshold` when an
unmatched claim fell just under the threshold). No judge scores unmatched
claims yet (`claims_judge_calibrated` is 0): one must first separate the
reviewed negatives from good claims in a pinned test.

    uv run python -m evals.claims_report [RESULTS.JSON | --run ID] [--json] [--rescore] [--cases DIR]

reads a claims run (the newest stored one by default) and prints, per
candidate, one reviewed row per interface and the expect_empty cases per
`origin` with a Wilson 95% interval, since some origins are small. A row
scored before interfaces existed reads as `legacy`. `--rescore` re-checks each
row's stored reply against the cases (this machine's, or `--cases DIR`, such
as a fresh import written outside the cases tree) without loading a model.

Every reviewed claim so far was extracted by Qwen2.5-7B-Instruct-Q4_K_M (then
called eval-7b), so recall against the reviews is circular and favours its
output; the report says so on every run until reviews of another model's claims
exist. The `conversation` reviews came from that model under the current prompt
at temperature 0, which makes its `conversation` row (under either name) a
control for the matcher: the report prints its recall, verbatim matches and
near misses, and blames the matcher
only when recall is under 0.9 and at least half the misses sat just under
the threshold. Otherwise the model did not reproduce its own reviewed claims.
`tests/fixtures/claims/paraphrase.json` pins the matcher on synthetic text:
36 triples of a reviewed claim, a rewording and a different claim about the
same part, in three cases of 12. All 36 rewordings match and no different
claim does, and the gap between the classes (lowest rewording 0.75, highest
different claim 0.47) exceeds the larger class's standard deviation (0.11).
Plain word Dice, matcher version 1, matched 12 of the 36 different claims
with a gap of 0.03, because shared subject words carried the overlap. A case
holding a single reviewed claim has no evidence of what its subject is, so
there the weights are uniform and 3 of 36 different claims still match.

Changing the matcher changes every stored claims score, so each claims row
carries `claims_matcher_version` and a claims run's `cases_digest` includes
it: a run scored under version 1 is not comparable with one scored under
version 2 (the digest differs, so `comparable()` refuses), and the report
says so when its rows mix versions. A row with no version was scored under 1.
`claims_report --rescore` re-checks an old run under the current matcher.

A private case's text never leaves the machine: its scoring reasons quote
nothing, `publish` refuses an export carrying any of its lines or reviewed
claims, and a test fails if any file under `evals/cases/` or `tests/` holds a
private case's text or digest. Tests use the synthetic export in
`tests/fixtures/claims/`.

### Technique discovery

A lane can be beaten by a new method as well as by new weights: svg's
`trace:` (draw a raster, then vectorize it) is one. Every proposal carries a
`category`: `model` (a registry card or its weights), `tool` (a GitHub repo
with no model task) or `technique` (a paper, or a repo whose description or
topics say it implements one). Inspect sets it from the card or repo; schema
56 fills stored rows from what the store already holds and leaves the rest
unknown. `soh discover --papers` (and so `--sweep` and the loop) reads
HuggingFace daily papers, one request per day not yet read, at most seven, and
records each paper as a technique `arxiv:<id>` with its title, abstract and
link, recorded in the `sources` table like the feeds. A paper is laned by the
same prose routing as a model, title first and then title and abstract
together; one that names no lane, or several, stays laneless. A technique is
never queued for fetch, screen or measure: `soh discover --loop` lists them
under `techniques wanted`, per lane, with title, link and sightings.
Implementing a method is a job for a person or an agent, so the report decides
nothing.

### Refusals that stop being true

A refusal records the condition that would end it, as a predicate rather than
a sentence: `runtime:llamacpp`, `ceiling_gb:>52.7`, `version:mlx-lm>0.31.3`,
`commit_after:<date>`. `runtime:cuda|rocm` is met by either. When the tier
does not pass one, it is read from the refusal's own `needs-*` or `too-big`
wording.

```bash
soh discover --revisit             # refusals this machine now satisfies
soh discover --revisit --requeue   # send them back to inspect
```

A screen runs one case once, so text candidates are screened at
temperature 0 unless the spec sets one. A single sampled draw made
Qwen3-0.6B pass on one screen and fail on the next, and a failure is
terminal (#308). The receipt's `sampling` records the 0.

A runtime that cannot build a model is a fact about the runtime, so the
screen declines it until a newer one: `version:mlx-lm>X` for mlx_lm.server's
"Model type ... not supported", and `version:llama.cpp>BUILD` when
llama-server's router answers "failed to load". The router keeps the reason,
such as an unknown architecture, in `logs/eval.log`. A chat template that has
no system role (Gemma 2, Mistral v0.3) is neither: the request is retried
once with the system prompt at the top of the user turn.

This counts refusals made on this machine too. A machine that installs
llama.cpp after declining 41 GGUF repos for lacking it has changed, and those
refusals should reopen. Only terminal verdicts are listed, since a queued row
is already waiting.

### GGUF candidates in the text lanes

A text-lane repo that ships only GGUF files (no safetensors) is run by
llama-server rather than mlx_lm.server. Other lanes keep the whole-repo
download, since their runners load the repo:

- **Inspect** sizes it by the one file it would fetch, not the sum of every
  quant. That file is Q4_K_M when it fits the ceiling, then Q4_K_S, IQ4_XS,
  Q4_0, Q5_K_M, Q5_K_S, then the largest that fits. Split files, imatrix
  calibration data, `mmproj`
  projectors and files in subdirectories are never chosen, because the router
  cannot serve them alone.
- **Fetch** downloads that one file into `$LLAMACPP_MODELS_DIR` (default
  `$HF_HOME/gguf`) and records it as a `downloads` row. A GGUF-only repo
  already in `$HF_HOME/hub` from an older whole-repo fetch is not downloaded
  again. The fetch tier symlinks its chosen file into the router's directory
  and records the link, for text-lane and decide-lane repos only. Other lanes
  never trigger this, because their runners load the repo. Until it is linked,
  a GGUF-only repo id still routes to llama-server by the stem it would be
  linked under, never to mlx_lm.server, which cannot load a GGUF or enforce
  the decide lane's schema. That holds when `evals.run` is given a `--gateway`
  too (it always is, by default), the same as for a `llamacpp:` spec.
- **Screen and measure** spell it `llamacpp:<file stem>`, and the run sends it
  straight to the eval server at `127.0.0.1:8082` (`scripts/serve-eval.sh`).
  LiteLLM would refuse a name it has no alias for. A measure pits it against
  the lane's incumbent through the gateway, so the two halves of one run can
  come from different engines.

Each GGUF gets its own context (#498). At startup `scripts/serve-llamacpp.sh`
reads every recorded GGUF's header and serves it at the smaller of its trained
context and the most whose f16 KV cache fits beside the weights, in steps of
1024. The KV room is the machine's ceiling less its measured reserve, the
weights and the largest text model mlx_lm.server serves beside it, and never
more than `LLAMACPP_KV_MAX_GIB` (8), because llama-server allocates the whole
cache at load. KV bytes per
token come from the header (KV heads, key and value widths, and only the
attention layers of a hybrid; sliding-window layers are costed as full). A
model under 8192 tokens is refused, and `serving.route` says why. The result
is stored on the `downloads` row and passed per model through a
`--models-preset` file, with one slot per model unless `LLAMACPP_PARALLEL`
asks for more (more slots share one pool, so each request can still use the
whole context). If the preset cannot be written, every model gets
`LLAMACPP_CTX` (16384).

`Qwen2.5-7B-Instruct-Q4_K_M` (once eval-7b) is the exception: it is served at a
fixed number of slots and a context per slot, the knobs `eval7b_slots` and
`eval7b_slot_ctx` (`harness/context.py`), so infovore's bulk claims run can
keep several requests in flight. Its pool is slots times the slot context, one
request may still use all of it up to the trained 32768, and a slot context
under 3072 tokens (`MIN_SLOT_CTX`: a 6000-character window plus a 400-token
reply) is refused. The default is 8 slots of 4096 (#665).

Each receipt has an `engines` map from candidate to server, and
`instruments.serving` names every engine in the run, such as
`llama-server+mlx_lm.server`. `comparable()` therefore refuses to pool a
mixed run with a single-engine one. Adoption only records a GGUF winner. A
lane command does not use one until the gateway has an alias for it.

### Writing a prompt for whatever is installed

```bash
soh prompt image "a fox in falling snow"   # writes one
soh prompt video                            # or just shows the engine's guide
```

A caller should not have to know which engine serves a lane. This asks the same
resolver the generating command asks, so the two cannot disagree, and the
per-engine knowledge lives in `harness/prompting/` as versioned assets rather
than in prose somewhere. Where nothing has been established about an engine the
guide says so, because a prompt guide that invents a confident tone is worse
than none.

### Where a lane's work executes

Not a GPU toggle, because there are three answers rather than two:

| `where` | what runs it |
|---|---|
| `in-pod` | the container itself: sweep, inspect, the judge's caller |
| `gpu-node` | a node with a card the scheduler can see |
| `host` | a machine outside the cluster entirely |

The third is not an exotic case. Metal is a macOS userspace API and MLX links
against it directly, so every MLX lane belongs there: the Linux VM behind
Docker Desktop has no `/dev/dri` and no nvidia device, and there is no flag
that adds one. A pod can still hold such a lane's identity and ask `soh` on the
host to do the work, which is how the judge tier already reaches the gateway.

It reaches the receipt as `LOCALHARNESS_WHERE` and `evals.core.comparable()`
refuses to rank two runs across it, because a pod that dispatched to a host
measured a host. The dispatch half is not built, and the chart refuses to
render `measure.where=host` rather than shipping a pod that measures itself and
labels the result someone else's machine.

## Where things land

One root, `~/localharness`, overridable with `$LOCALHARNESS_HOME`:

```
out/                        artifacts: images, audio, svg, pages
out/mcp/                    artifacts asked for over MCP
runs/<stamp>-<modality>/    one eval run: artifacts and a results.json export;
                            the run itself is the store's runs/results rows
logs/                       service stdout and stderr
```

Each results.json row carries `output` (the text a text lane returned) and `artifact_path` (the file written for it); the old `artifact` key is deprecated and will be removed in the next release.

There were four places before this, and one was relative. `soh` runs from
anywhere, so a relative `out/` scattered artifacts into whatever directory the
caller was standing in, and a generation you cannot find is a generation you did
not make.

## Measuring things yourself

Three words show up throughout:

- a **lane** is one kind of job: image, video, svg, web, code, extract, speech,
  transcription. Each has its own test cases and its own way of being scored.
- a **candidate** is one contender in a lane: usually a model, sometimes a
  *method*. Candidates in a lane compete on identical cases.
- **measured** means a candidate has been run here and has a score.
  Untested is the default, which is what `soh discover` exists to make visible
  rather than letting it be assumed.

To run two candidates against each other:

```bash
uv run python -m evals.run --modality image --out .logs/img \
  --candidates mflux:flux2-klein-4b,mflux:z-image-turbo
```

A candidate is written as a short spec: a model id for text
(`mlx-community/Qwen2.5-7B-Instruct-4bit`), `mflux:<model>` for anything that runs as a separate program, or
`tts:<model>,voice=<name>` for speech.

A candidate can also be a **method**, a way of working rather than a model. A
method can beat a better model:

```bash
--candidates trace:mflux:flux2-klein-4b        # draw a picture, then trace it to vector
--candidates trace-icon:mflux:flux2-klein-4b   # same, tuned small: 3.2x fewer bytes
--candidates omnisvg:4B                        # a model that emits SVG draw commands as tokens
--candidates repair:mlx-community/Qwen3-4B-Instruct-2507-4bit   # generate, check it, fix it, repeat
--candidates claude-code:claude-opus-5-5       # a frontier reference, through headless Claude Code
```

`claude-code:` runs each text case through `claude -p` under the Claude Code
login on this machine, with the lane's own system prompt and no tools, so a
subscription covers it. It is a ceiling to measure the local models against,
not a lane default. On the code lane it scored 9/9 at a 5.7 s median against
mlx-community/Qwen3-4B-Instruct-2507-4bit's 7/9 (then q3-4b; Mac Studio, 2026-10-05).

`omnisvg` needs `./scripts/setup-omnisvg.sh` first: it has its own checkout, its
own venv and 16 GiB of weights.

The music lane (`acestep:`) runs an ACE-Step checkout through that checkout's
own `.venv`. It looks at `$ACESTEP_ROOT`, then at `~/localharness/acestep`, so
a checkout kept elsewhere needs only a symlink there for the services and
`soh verify` to find it.

The decide lane asks for typed decisions: a context plus a flat schema of
enum and boolean fields, scored on accuracy per field and on the calibration
of the probabilities (Brier, and ECE once there are 100 decisions to bin).
Its 24 cases are generated from human-labelled public data by
`uv run python -m evals.decide_corpus`; each names its dataset and license.
A text model is asked for one letter per field, with the answer tokens'
logprobs where the server returns them; `claude-code:` gives answers only and
is scored one-hot, as uncalibrated. Every decide request carries the reply as
`response_format` json_schema, which only llama-server enforces, so the lane's
default is `imajev-4b-Q8_0` (mindchain/imajev-4b-GGUF, Q8_0, best measured on the
lane in #311) on llama-server, never an mlx_lm.server model, which the gateway
refuses for a schema. Its GGUF must be in the router's models dir
(`scripts/fetch-gguf.sh mindchain/imajev-4b-GGUF imajev-4b-Q8_0.gguf`), and
`scripts/smoke.sh` asks `sohot-decide` for an answer under a schema.
The adopt tier will not adopt anything for a schema lane (`gateway.SCHEMA_LANES`)
that llama-server does not serve: the win is recorded as `queued` with reason
`harness` and class `refused_by_gateway`, never as a loss. The eval refuses a
decide case aimed straight at mlx_lm.server (the same class) instead of scoring a
reply whose schema was dropped.
`nimble:<repo>` runs Bespoke-Nimble-9B
through nimble's own MLX ParallelScorer. On first use `scripts/nimble-venv.sh`
clones nimble at its pinned commit into `~/localharness/nimble/checkout` and
builds a Python 3.12 venv beside it (mlx, mlx-lm, transformers, torch, peft).
The first case then downloads the base its adapter pins (Qwen/Qwen3.5-9B,
about 18 GB in bf16) and merges the adapter once on the CPU into
`~/localharness/nimble/models`; that needs the disk for both and well over
18 GB of free memory. Apple Silicon only.

`decider:<repo>` runs a strands-decider checkpoint such as
StrandsAgents/strands-decider-2B-hobson-v21. `scripts/decider-venv.sh` clones
strands-decider at its pinned commit into `~/localharness/decider/checkout` and
installs it into a Python 3.12 venv beside it, with the `mlx` extra on Apple
Silicon (`device=mlx`, the default there; `device=cuda|mps|cpu` elsewhere). An
enum field is asked as a choice question over its choices, a boolean as a
yes/no (`noul`) question whose P(true) is the field's probability. The first
case downloads the checkpoint and its base (Qwen/Qwen3.5-2B-Base, about 4.5 GB),
so run it once with `HF_HUB_OFFLINE=0`.

Which engine an adapter in the queue is spelled for comes from its card, not
its lane: a repo id, base model, library or tag naming `strands-decider` goes
to `decider:`, one naming `nimble` to `nimble:`. An adapter naming neither is
reported under runners wanted rather than guessed at.

`needle:needle3[,layers=N]` runs Cactus-Compute/needle3 through its own CLI (the card's platform binary and `needle3.cact` at a pinned revision, fetched into the HF cache on first use with `HF_HUB_OFFLINE=0`) in the decide lane, one call per field with a whole-call confidence rather than per-option probabilities, and in the agent lane over `--serve`.

    uv run python -m evals.run --modality decide --candidates \
      mlx-community/Qwen3-4B-Instruct-2507-4bit,claude-code:claude-opus-5-5,nimble:bespokelabs/Bespoke-Nimble-9B,decider:StrandsAgents/strands-decider-2B-hobson-v21

The agent lane measures a model the way opencode and Claude Code use it: a
tool loop over a small repo. Each case under `evals/cases/agent/` names a bundle
(`repos/<name>/repo`, a `hidden/` test the model never sees, and a reference
`solution/` the tests apply to prove the hidden test is satisfiable), a goal,
and the tools it pins: `read_file`, `write_file`, `list_dir`, `run_tests`. The
categories are reading a repo to answer a question, a single-file edit, a
multi-file refactor, a bug fix guided by a failing test, and edit and fix cases
with 16k and 32k tokens of generated repo pasted into the first message.

The runner speaks OpenAI tool calls, streaming, to the route the candidate's
spec resolves to (a gateway alias such as `sohot-agent`, or llama-server for a
`llamacpp:` stem). Each case runs in a tmp copy of its repo: a path that is
absolute, contains `..` or passes through a symlink is refused, `run_tests`
builds its argv from an allowlist with no shell, and the test process installs
an audit hook first that refuses writes outside the copy, process spawns,
sockets and ctypes. A case passes when its hidden test passes (or its final
answer matches) and the model made at least one valid tool call. Per step it
records time to first token (content or tool call), step seconds, tokens and
whether each call parsed, fit its schema and named a real tool. Completion
decides adoption, paired like the other lanes; valid-call rate, steps, model
seconds and summed TTFT are reported beside it. `--modality all` leaves the
lane out.

`claude-code:<model>` is the Opus baseline. There is no API key, so it runs
`claude -p` with every built-in tool disabled and one stdio MCP server
(`evals/agent_mcp.py`) serving the same four tools over the same sandbox. Its
steps and first tokens come from stream-json and include the CLI's start and
the network; Claude Code validates a call before the sandbox sees it, so a
malformed call counts as invalid only when it shows up as a tool_use that
never reached the server.

    uv run python -m evals.run --modality agent --candidates \
      sohot-code,mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit,claude-code:claude-opus-5-5

The ocr lane reads text out of an image. Its eight cases under
`evals/cases/ocr/` are PNGs rendered by `uv run python -m evals.ocr_corpus`
(Pillow's bundled font, fixed sizes, one seeded noisy background, one tilted
label), each with the exact text it was drawn from. A case scores the
character error rate of the transcription against that text, case-sensitive,
with line breaks and runs of spaces counted as one space; the lane's figure is
errors over reference characters across all cases, so a short sign weighs less
than a long line. A case passes at a CER of 0.1 or less. The negative control
is pinned in `tests/test_ocr_lane.py`: a reader handed each case's neighbour's
text fails every case at a CER over 0.5, and the OS reader reads the committed
images under 0.1, so a miss is the candidate's and not the fixture's.

`osocr:auto` is the incumbent: the reader the image lane's text check already
uses (Apple Vision here, Windows.Media.Ocr or RapidOCR elsewhere).
`hf-ocr:<repo>[,prompt=...,max_new_tokens=...,device=...]` runs a
transformers image-text-to-text or image-to-text model through
`scripts/hf-task.sh`, which builds its own venv under
`~/localharness/venvs/hf-task` (torch, torchvision, transformers, accelerate,
safetensors, pillow, sentencepiece at the pins in `scripts/versions.sh`) on
first use and rebuilds it when a pin moves. A package that venv lacks exits 4
with `hf-task venv lacks a package: ...` and is queued as a harness fault,
never recorded broken; a failed run keeps the last 40 lines of the runner's
stderr beside its artifact as `<artifact>.stderr.txt`. An abort inside the
MPS backend (`LLVM ERROR` on an `mps.*` op, an MPSGraph assertion) is a
runtime fault, declined until the runtime moves rather than broken: hf-ocr
runs each attempt as its own process and retries such an abort with
`attn_implementation="eager"`, then on the CPU (PaddleOCR-VL-1.6's
grouped-query attention aborts torch 2.6.0's MPS matmul and answers with eager
attention). The device and attention that answered are written beside the
artifact as `<artifact>.runtime.json` and land in the receipt's `devices`,
which `comparable()` refuses to mix. An MLX conversion
(card tagged `mlx`, such as mlx-community/DeepSeek-OCR-bf16) has no hf-task
runner and is refused before download. It never passes `trust_remote_code`: a repo that ships its own
modelling code, such as baidu/Unlimited-OCR or hayai-ocr-v2, is reported under
runners wanted before anything is downloaded, for every hf-task engine
(hf-ocr, rerank, embed, hf-pii). Inspect keeps the card's `config.model_type`
and the code files its `config.auto_map` names; the gap holds when the auto_map
is present and the model type is not in `harness/transformers_model_types.txt`,
the types the pinned transformers ships. A custom_code tag or a `.py` file
alone decides nothing. After `TRANSFORMERS_PIN` moves, run
`scripts/transformers-types.sh` to rewrite that list; a test fails until it is.
PaddleOCR-VL loads through transformers' own class and wants `prompt=OCR:`.
Discovery files a card under ocr when its task is image-to-text or
image-text-to-text and a tag names OCR; a captioner without that tag stays
laneless.

    uv run python -m evals.run --modality ocr --candidates \
      osocr:auto,hf-ocr:PaddlePaddle/PaddleOCR-VL-1.6,prompt=OCR:

The retrieval lane ranks a corpus for a query. `evals/cases/retrieval/` holds
one corpus of thirty short operations notes (`corpus.jsonl`) and fourteen
queries, each labelled with the one or two documents that answer it; most
queries are paraphrased so that word overlap alone does not find the answer.
A case passes when every labelled document is in the top five (recall@5 of
1). The lane reports recall@5 and binary-gain nDCG@10, each a mean over
queries. The negative controls in `tests/test_retrieval_lane.py`: a ranker
that puts the labelled documents last scores recall 0 and passes nothing, a
seeded shuffle stays under 0.5, and BM25 has to land between chance and
perfect.

`bm25` is the incumbent: Okapi BM25 in this checkout's own interpreter, with no
weights. `rerank:<repo>` scores every (query, document) pair with a
sentence-transformers CrossEncoder; `embed:<repo>` ranks by the cosine of
normalised SentenceTransformer embeddings, using the model's own query and
document prompts when it names them. Both run in the hf-task venv and take
`revision=` and `device=`. Discovery files text-ranking, text-retrieval and
visual-document-retrieval cards under retrieval, and a sentence-similarity or
feature-extraction card only when a tag names retrieval or reranking; a
general embedder stays laneless and a text-classification reranker stays in
decide. The ladder spells a card's model `embed:` when its task is
sentence-similarity or feature-extraction and `rerank:` otherwise. A card whose
library is not sentence-transformers or transformers (Contrastive-LM's own
`contrastive-lm`, the colpali-engine EVIE models), or whose task is
visual-document-retrieval (it embeds page images, and these cases are text),
is reported under runners wanted before anything is downloaded.

    uv run python -m evals.run --modality retrieval --candidates \
      bm25,embed:BAAI/bge-small-en-v1.5,rerank:cross-encoder/ettin-reranker-1b-v1

The pii lane marks the personal data in a sentence. Its sixteen cases under
`evals/cases/pii/` are sentences with the exact strings that are personal
(names, emails, phone numbers, a street address, an SSN, a card number, an
IBAN, an IP address, an API key, a date of birth, a passport number, a
private URL), plus four with none. A label must occur exactly once in its
sentence, or the case fails to load. Scoring is per token, a token being a
run of letters and digits: a token is personal when it overlaps a labelled
string and marked when it overlaps a predicted span. Label names are ignored,
because every model names its classes differently. The lane's `pii_f1` is
pooled over all tokens, with `pii_precision` and `pii_recall` beside it. A
case passes at a token F1 of 0.8, and a sentence with nothing personal passes
only when nothing is marked. The negative controls in `tests/test_pii_lane.py`:
marking nothing scores 0, marking every token passes no case and stays under
0.6, and the pattern baseline lands between them.

`pii-regex` is the incumbent: patterns for things with a shape (emails, URLs,
IPv4, IBANs, card numbers, SSNs, phone numbers, API keys), with no weights. It
cannot see a name. `hf-pii:<repo>[,revision=,device=]` runs a transformers
token-classification model in the hf-task venv and counts every entity group
it returns as personal. Discovery files a token-classification card under pii
only when a tag names personal data (`pii`, `privacy`, `anonymization`,
`deidentification`, `redaction`, read word by word, so
`openai_privacy_filter` counts). A general named-entity tagger stays laneless.
mistralai/Shieldstral-1.0-3B is a generative guard with no task on its card:
lineage files it under code, and its vllm library is not one hf-pii loads.

    uv run python -m evals.run --modality pii --candidates \
      pii-regex,hf-pii:openai/privacy-filter,hf-pii:LH-Tech-AI/Shield-82M

The ocr, pii and tts lanes also carry generated cases, so no candidate can
have trained on them: the sources `ocr_synth`, `pii_synth` and `tts_synth`
write `evals/cases/<lane>/<source>/` through the same importer framework.
`fetch(seed, count)` draws the rows and `convert` renders each one, so the
same seed gives the same bytes. With no upstream commit, `SOURCE` names the
generator version and seed, `REVISION` is the SHA-1 of `SOURCE`, the license
is `generated`, and the attribution adds version and seed. There
are 440 ocr, 540 pii and 460 tts cases, giving the lanes 140, 166 and 135
holdout cases. The
ocr images use only Pillow's bundled font, since no other renders alike on
every platform, and vary size, weight, slant, width, contrast and noise; most
also get one degradation (low resolution, blur, faded ink, speckle or a steep
rotation), because the clean ones were read by the OS reader 436 times in 440,
which leaves the adopt gate nothing to detect. With the degradations it reads
315 of 440. The pii sentences fill templates with fake values only (example.com
addresses, 555-01xx numbers, published example IBANs and test card numbers,
TEST-NET addresses, `sk-test-` keys), and about one in five has no personal
data; `pii-regex` passes 74% of the holdout. The tts sentences draw two-, three- and
four-digit numbers equally often (2 to 9999, generator v2). The English scorer
reduces both the reference and the transcript to one spoken form before
counting: digits become cardinal words with no "and" (105 is "one hundred
five", and a transcript's "one hundred and five" or "a hundred" reduces the
same way), "21st" becomes "twenty first", "3.5" becomes "three point five", and
"1,250" loses its comma (#606). A digit is always read as a count, so "6319
crates" heard year-style as "sixty three nineteen" is still two errors. With
v1 (numbers 2 to 99, because the scorer then charged "one hundred five" a
word) Kokoro passed all 131 English holdout cases at a corpus WER of 0.013,
which left the metric no room above the incumbent. Receipts: every v2 case is
new content, so `cases_digest` keeps a v1 tts run from being ranked against a
v2 one; no hand-written tts or LibriSpeech reference normalizes differently.
`soh cases control <source>` runs the negative controls that
`tests/test_generated_cases.py` pins: another case's text on ocr, marking
nothing or everything on pii, silence or one fixed sentence on tts.

`repair` costs nothing extra, so it is the one to understand. Everything already
checks its own output. It runs the code it wrote, draws the SVG to see whether
anything is visible, opens the web page in a browser. All of that was
being used to *score*, and never to *improve*. `repair` simply asks the same
model again with the complaint attached: "this failed, here is why, try again."
It uses the same model and downloads nothing, and the scores above are what it
bought.

**How many cases is enough?** More than feels necessary. At 40 speech clips this
project got the winner right and both effect sizes wrong in the same run: it
called a 1.43x gap "twice as bad", and called a significant loss a tie. Nothing
was separable at that size, so whichever way the noise fell got read as signal.

So a difference between two candidates gets an interval, not an eyeball:

```bash
uv run python -m evals.compare <run-dir> --baseline <candidate>
```

```
  parakeet-ctc-0.6b       0.0225  +0.0063  95% CI [+0.0032, +0.0097]  worse
  parakeet-tdt-0.6b-v3    0.0232  +0.0070  95% CI [+0.0037, +0.0106]  worse
```

It resamples the *cases*, not the runs, because case difficulty is the largest
source of variance: every speech model here shares the same worst clip. An
interval spanning zero prints `not separable` instead of a ranking.

**Is that constant doing anything?** Fourteen numbers decide what gets
proposed, screened and fetched. `soh sensitivity` varies each one against cached
data and says whether the output moved at all -- inert, inside a band, or on an
edge where the value next door behaves differently.

```bash
soh sensitivity                 # all eleven probes, about four minutes, no network
soh sensitivity --list          # and the constants nothing covers, with reasons
soh sensitivity --inventory     # gating constants the knob registry does not cover yet
```

Every constant that gates an outcome is meant to be a knob in `harness/knobs.py`:
the values worth trying, the lanes it shapes, and the limit a result records
when it binds. The suite fails on a new gating constant that is not registered.
Knobs that change the exam (the request and load timeouts, the TTS runaway
cutoff) are recorded on the receipt beside the reply budget, `--compare` refuses
runs at different settings, and `--across knobs.request_timeout` reads a sweep.

A knob whose limit shows up on enough of a lane's recent rows is binding: the
loop and `soh audit` say so, and the loop queues a sweep of the lane's served
model across the knob's range (`soh sensitivity --sweeps` shows what it would
queue and what earlier sweeps decided). A sweep records the cheapest value the
paired test cannot tell from the best, and what the sweep cost. It runs again
when the model or the machine changes. Changing the constant stays a person's
decision.

It compares rankings position by position rather than by pass rate, because a
count over a set cannot see a reordering. `HALF_LIFE_DAYS` was recorded inert
on exactly that mistake and in fact moves 22 of 25 positions.

To put two finished runs side by side:

```bash
uv run python -m evals.run --compare a/results.json b/results.json
```

It will refuse if the two runs are not fairly comparable, and tell you which
difference disqualified them. Comparing a run from before a settings change
against one from after compares two different exams, so it stops you.

### The work queue

Long jobs from anyone go on one queue: an agent on another machine calling the
MCP `image` or `video` tool, or someone here typing

```bash
soh jobs add --title "30B coder on code" -- uv run python -m evals.run --modality code --candidates mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit,mlx-community/Qwen3-4B-Instruct-2507-4bit
soh jobs list      # each job's state, and whether the next one may start
soh jobs pause     # hold everything; a running job still finishes
soh jobs resume
soh jobs cancel 0003
soh jobs priority 0005 10   # higher runs first; ties run in the order added
```

A queued job runs in the deploy checkout, so it runs the deployed code. Queued
from a git checkout at a different commit, `soh jobs add` refuses and names both
commits, since `uv run` there would run that checkout's code instead; pass
`--cwd DIR` to run somewhere on purpose. Without a deploy checkout the job runs
where it was queued.

`soh jobs cancel` drops a pending job. On a running one it signals the job's
process group (each job is started as its own), waits for it to exit, kills
what is left after a grace period, and records the job `cancelled` with a note.
It works on the machine running the job.

While `soh discover --loop` runs it writes a heartbeat to
`$LOCALHARNESS_HOME/loop-heartbeat.json`: its lane, current tier, candidate,
step N of M, start time and last progress time, marked finished (with the exit
code, or the error) however the loop ends. `soh jobs list`, `soh report` and the
local report page print it. A heartbeat with no progress for three hours, or
whose process is gone, reads as stalled rather than running.

The `worker` service runs them in order, one at a time. It does not hold the
machine lock below for a whole job: each command takes it for the part that
loads a model, so a job that is downloading does not block a GPU run. It starts
the next one whenever the queue is not paused and macOS memory pressure is
normal, whether or not someone is at the machine; under pressure it waits,
and an unreadable pressure level does not block it. Jobs are rows in the
store's `jobs` table, so they survive a restart, and output goes to
`~/localharness/logs/jobs/<id>.log`. A failed job is recorded and the next
starts. A job cut off by a reboot is marked failed rather than rerun. An eval
run a job produced is linked to it (`runs.job_id`).

Over MCP, `image` and `video` are queued at priority 10, ahead of batch work
at 0, so someone waiting on a picture is not behind a night of benchmarks. They
return a job id at once, `job_status` says what
it is waiting for, and `job_result` returns the file itself, so a caller on
another machine gets the picture rather than a path it cannot open.

### One model-loading run at a time

Image and video generation, eval runs (screen, measure, `soh rubric run`) and
`soh memory ramp` hold one machine-wide lock,
`$LOCALHARNESS_HOME/queue/generation.lock`. A second run waits and says what
it is waiting behind. Text lane prompts do not take it, because each server
already queues its own requests.

Other projects on the same machine take the same lock around anything that
loads a model:

```bash
~/localharness/deploy/scripts/with-gpu-lock <command> [args...]
```

macOS's own `lockf -k ~/localharness/queue/generation.lock <command>` takes
the same lock. Checked both ways on 2026-10-04: each refuses while the other
holds it. The helper adds two things: it records who is holding the lock, so a
waiter can say what it is waiting behind, and it marks nested runs so they
don't deadlock. It sets `LH_GPU_LOCK_HELD=1` for the command, so a `soh` run inside
it does not wait on its own parent. `scripts/launchd.sh install` runs under
the lock too: restarting mlx_lm.server in the middle of another project's run
killed 32 of its 50 requests on 2026-10-04.

### How many requests a server can take at once

```bash
soh throughput --model Qwen2.5-7B-Instruct-Q4_K_M --texts convos.jsonl --levels 1,2,4
soh throughput --model Qwen2.5-7B-Instruct-Q4_K_M --claims claims.jsonl --levels 1,4,8 --max-tokens 400
```

The command sends every line's `text` to a gateway alias with 1, 2 and 4
requests in flight, holding the machine lock, and reports requests per hour,
the speedup over one at a time, p50/p95 latency and errors. llama-server runs
four slots that batch together, so a client sending one request at a time
leaves most of that unused. One untimed request goes first and its time is
printed: a model that is not resident loads on it, and timed into the first
level that load made a 1.9x speedup read as 6.8x (#333).

Measured 2026-10-04 on the M2 Pro: eval-7b (Qwen2.5-7B Q4_K_M), 16
conversations of median 1,130 characters, 300-token budget (about 70 tokens
used), warmed:

| in flight | per hour | speedup | p50 |
|---|---|---|---|
| 1 | 985 | 1.0x | 4.3 s |
| 2 | 1,959 | 2.0x | 3.6 s |
| 4 | 2,465 | 2.5x | 5.8 s |

Measured 2026-10-05 on the M5 Ultra (96 GB), same alias, texts and budget,
warm, with Photos analysis paused:

| in flight | per hour | speedup | p50 |
|---|---|---|---|
| 1 | 7,713 | 1.0x | 0.45 s |
| 2 | 10,522 | 1.4x | 0.68 s |
| 4 | 14,320 | 1.9x | 0.98 s |
| 8 | 14,626 | 1.9x | 1.45 s |

One request at a time is 7.8x faster than the M2 Pro, and concurrency buys
less: a single request already keeps more of this GPU busy. Past four in
flight nothing is gained, because the server has four slots.

`--claims FILE` (repeatable) sends each case of a claims export as the
claims lane sends it: its system prompt, its transcript and its schema as
`response_format`. Every level then also reports how many replies validate
against their schema, the prompt tokens, and the load average at its start and
end. Requests per hour counts answered requests only, so a server that refuses
everything reads 0 (#666).

Measured 2026-10-08 on the M5 Ultra (96 GB), llama.cpp build 11146, eval-7b
Q4_K_M with 4096 tokens per slot on a unified f16 pool, the 48 reviewed cases of
a claims export sent 4 times (about 627 prompt and 102 output tokens each),
max_tokens 400, temperature 0: 1 slot 4,700 requests an hour (p95 1.6 s), 4 slots
8,229 (p95 4.0 s), 8 slots 9,570 (p95 6.6 s, 3.7 GiB peak), 16 slots 7,697, 32 slots
9,409 (p95 24.5 s, 9.0 GiB) on the M5 Ultra, and 32 slots with 8 in flight 9,639.
Eight slots is the knee on the M5 Ultra. 188 of 192 replies matched the schema at
every level there, so batching does not break the grammar (#665).

`--model` takes any text spec, so `llamacpp:<stem>` and `vllm:<repo id>` go
straight to their server. `--server-pid` samples that server's peak memory
(phys_footprint of it and its children) during each level, and every level
reports total tokens per second and why any request failed. `--serve
vllm-mlx` or `--serve vllm-metal` starts that engine from
`$LOCALHARNESS_HOME/venvs/<engine>` on `VLLM_PORT` (default 8086), sweeps, and
stops it. vllm-metal always gets `--gpu-memory-utilization 0.2 --max-model-len
16384`: without them, on the M5 Ultra on 2026-10-06, it reserved 68 GiB of KV
for a 4B model. A `vllm:` spec
works anywhere a text spec does (evals, lane commands, an adoption the
generated gateway config fronts); `VLLM_ENGINE` names which implementation it
is on the receipt's `engines` map.

Measured 2026-10-07 on the M5 Ultra, Qwen3-4B-Instruct-2507 (MLX 4-bit; Q4_K_M
GGUF for llama-server with 16 slots), 32 texts, 300-token budget, total tokens
per second (#310):

| in flight | mlx_lm.server 0.32.0 | vllm-mlx 0.5.0 | vllm-metal 0.30.0 | llama-server b11146 |
|---|---|---|---|---|
| 1 | 194 | 234 | 172 | 185 |
| 4 | 418 | 533 | 487 | 325 |
| 16 | 746 | 829 | 929 | 387 |

mlx_lm.server batches by default since 0.32.0, but its HTTP server's listen
backlog is 5, so at 16 in flight some connections are reset. The code lane
scores the four alike (20 to 22 of 42 at repeat 3).

### antirez/ds4 (DwarfStar)

[ds4](https://github.com/antirez/ds4) is a Metal-first engine for a few large
models (DeepSeek V4 Flash, Qwen3.8 Flash Next, GLM 5.x). It is not a general GGUF
runner: it loads only the GGUFs its own project publishes. A `ds4:` spec names one
of those files by stem and the launch that serves it:

    ds4:Qwen3.8-Flash-Next-Q2
    ds4:DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731,ssd_streaming=on,expert_cache=32GB

Options: `ssd_streaming=on|off` (default off), `expert_cache=auto|<slots>|<N>GB`
(only with streaming), `ctx=<tokens>` (default `ds4.DEFAULT_CTX`), and the usual
sampling keys. ds4 has SSD expert streaming for DeepSeek and GLM, not for Qwen3.8
Flash Next at the pinned commit, so `ssd_streaming=on` on a Qwen file is refused.
ds4-server has no `response_format` and returns no logprobs, so it serves the
code, agent and other text lanes and not decide. With thinking on (its default)
ds4-server ignores the client's sampling fields, so a pinned temperature on a
ds4 receipt is what was asked, not what ran.

**Build.** `scripts/ds4-build.sh` clones `DS4_REPO_URL` at `DS4_REV`
(`scripts/versions.sh`) into `$LOCALHARNESS_HOME/ds4/checkout` and runs
`make ds4-server ds4`. The checkout lives under the localharness home, not
beside the weights, so moving the models volume does not touch it. The build
makes no network calls; ds4-server makes none either unless started with its
distributed or tensor-parallel flags. `machine` records the checkout's commit
date as the `ds4` version, which reverify and `version:ds4>` reopen on.

**Weights.** Never fetched by the loop: the fetch tier leaves a ds4 repo queued
with the command to run by hand,
`uv run python -m harness.ds4 fetch <repo> <file>`, which downloads into
`$DS4_MODELS_DIR` (default `ds4/` beside the HF cache) and records the file in
the downloads table. ds4's own `download_model.sh` is not used.

**Memory.** Inspect sizes a ds4 file by what it keeps resident: the file less
the tables ds4 reads from disk on demand. Per ds4's model guide at `DS4_REV`
(read 2026-10-07), Qwen3.8 Flash Next Q2 is a 137.10 GiB file of which 41.73 GiB
is main and MTP weights and 95.37 GiB is BF16 n-grams that never load, so on the
M5 Ultra (96 GB, 66 GiB ceiling) it fits where a file-size check refused it.
A file whose resident weights
are over the ceiling and which ds4 can stream fits with `ssd_streaming=on`. The
screen's memory check uses the same number, and with streaming it counts the
expert cache budget when one is given; with `expert_cache=auto` ds4 sizes the
cache itself and the check says so. A general GGUF of the same model (for
example llama.cpp's own conversion) is still a llama-server candidate.

**Serving.** One ds4-server holds one launch on `DS4_PORT` (default 8087).
`scripts/serve-ds4.sh` serves `$DS4_SPEC`, else the ds4 spec a text lane has
adopted here, else exits 0 saying there is nothing to serve; its argv comes from
`python -m harness.ds4 argv`, the same code a measurement uses. It is the `ds4`
launchd unit (restarted on a crash, not after a clean exit) and an optional
`scripts/services.sh start ds4`, never started by `services.sh start` alone.
After adopting a ds4 spec, `scripts/launchd.sh restart ds4` loads it; the
generated gateway config then fronts it as `sohot-<lane>`.

An eval run with a `ds4:` candidate reuses the server when it holds exactly
that launch, starts one for the run and stops it after when the port is free,
and refuses before measuring when the port holds a different launch or a server
the harness did not start. The receipt's `launch` map records, per receipt key,
`ssd_streaming`, `expert_cache`, `ctx` and the ds4 version of the server that
answered, and `comparable()` refuses two runs whose launch differs for the same
key. `evals.run --compare a b --across launch` reports streaming on against off
case by case.

### How much memory a run can take

```bash
soh memory ramp    # allocate 1 GB at a time on the GPU until macOS first warns
soh memory show    # what was measured, per machine
```

The ramp fills memory with random data, so the compressor cannot shrink it.
It samples `kern.memorystatus_vm_pressure_level` after every step and stops
at the first warning, at compressor swap-outs, or at a floor of 10% free. It
frees everything at the end and records the run as a `memory_limits` row in
the store, against this machine's `machines` row. A machine is
hw_model/OS family/arch, never the Python build, so two venvs on one Mac are
one machine (#415). A `memory-limits.json` from before schema 30 is imported
once and then ignored.

Each run records a **margin**: how far short of the guard's own "available"
figure (vm_stat free + inactive) macOS warned. The headroom guard reserves
the largest margin measured on this machine. Before any measurement it
reserves 6 GB, and it never trusts a margin under 1 GB.

Measured 2026-10-03 on the M2 Pro (32 GB), with the gateway, mlx_lm.server and
llama-server idle:

| run | available at start | warned at | margin |
|---|---|---|---|
| 1 | 14.2 GB | +13 GB | 2.2 GB |
| 2 | 17.8 GB | +15 GB | 3.8 GB |

The kernel's own free percentage read 72-77% at the start of both runs
(about 24 GB) and stayed there through the first 10 GB. It is not the
figure to budget against. Wired pages did not move either: MLX buffers are
not counted as wired.

### Judging text against a rubric

For a job where a model reads some text and answers in a fixed shape (is this
a fact, which category, how severe). The answer is JSON that must match a
schema, and the only reference is a person's labels.

`mlx_lm.server` ignores `response_format`, so on the Mac the gateway refuses
it for an MLX model with a 400 naming the models that enforce it
(`gateway/schema_guard.py`). Before that, the schema was silently dropped and
the model wrote whatever it liked. `llama-server` enforces it. Each alias's
real upstream id is at `GET :4000/model/info`.
`scripts/serve-eval.sh` runs it beside the MLX server on `:8082`, and the
gateway's GGUF entries (those with a `source_file`) point there. It unloads its model after five idle
minutes. Weights are GGUF files in `$HF_HOME/gguf`:

```bash
./scripts/fetch-gguf.sh unsloth/Qwen3-4B-Instruct-2507-GGUF Qwen3-4B-Instruct-2507-Q4_K_M.gguf
```

An eval set is a directory under `$LOCALHARNESS_HOME/evalsets/<name>/`, kept
out of the repo because the items are usually other people's words:

| file | written by | holds |
|---|---|---|
| `rubric.yaml` | you | `name`, `version`, `instructions`, `schema`, and `label`, the schema field whose `enum` is the answer |
| `items.jsonl` | you | one `{"id", "text"}` per line |
| `labels.jsonl` | the labelling page only | append-only answers, each with the rubric stamp and the page's interface version |

```bash
soh rubric label <name>                     # a page, one item at a time
soh rubric status <name>                    # how many are labelled, and your repeat agreement
soh rubric run <name> --candidates Qwen3-4B-Instruct-2507-Q4_K_M  # score against your labels
```

About one item in five is shown again, looking like any other, so your
agreement with yourself is measured. That is the ceiling: no model can be shown
to agree with you more often than you agree with yourself. The floor is
always answering with the most common label. A run prints both beside every
model's agreement, says when a model does not clear the floor, and keeps every
raw reply in its receipt.

### Reading a run's memory conditions

A wall-clock number taken while the machine was busy is not the same number as
one taken on an idle machine, so every receipt records what the machine was
doing. Two fields do that, and they are not interchangeable.

`pressure` is current, and is the one to read:

| field | source | meaning |
|---|---|---|
| `free_pct` | `kern.memorystatus_level` | percent free, the figure jetsam acts on |
| `level` | `kern.memorystatus_vm_pressure_level` | 1 normal, 2 warning, 4 critical |
| `swapouts` | `vm.compressor.segment.swapout_regular` | a counter; a rate only when sampled twice |
| `wired_gb` | `vm_stat` pages wired down | memory that cannot be paged out |

`swap_used_mb` is a high-water mark. macOS grows the swap file and does not
shrink it when pages are freed, so on a 32 GB machine it can read 8 GB while
63% of memory is free, and it stays there until reboot. It is kept as a
description of the machine. Do not read it as a measure of pressure during the
run, and do not conclude anything about a candidate from it: 14 candidates
passed their screen with it above 2 GB, one of them at 7.9 GB.

Receipts written before 2026-09-25 have `swap_used_mb` only, so their timings
carry no usable pressure reading. Their pass/fail verdicts and peak memory are
unaffected; only wall-clock comparisons across them are suspect.

`swaps`, in `/usr/bin/time -l` output, is unmaintained on macOS and always
reads 0. Nothing should branch on it.

Whether a run is refused for lack of memory is a separate question, answered by
`memory.check_model` against free memory minus what is already resident, not by
either field above.

## Developing

```bash
make check     # shellcheck + bash -n + unit tests, no services needed
make test      # unit tests only
make smoke     # end-to-end, REQUIRES the services running
make mutation  # nightly mutation testing of the core modules, minutes per module
```

CI runs `make check` on an Apple Silicon runner for every push and pull
request. Linux and Windows run nightly on main (06:00 UTC), from the Actions
tab on demand, or on a pull request labelled `full-ci`; a failing night opens
or updates the issue "Nightly CI is failing on main". Each runner reports
which tests it skipped and why. A
skipped test otherwise reports green for something it never checked -- and
that is not hypothetical here: 29 tests skipped on both runners for a week
because they needed a volume only one machine has, while ten of them were
failing on that machine. The first run that could execute them found a bug in
the product, not the tests.

`make mutation` runs only the test files `tests/gauntlet/mutation.py` lists for
each module. A change to a mutated module re-pins its survivors with
`make mutation MUTATION_ARGS="--module harness/adopt.py --pin"`, and a new test
file for that module goes into its list first, or the nightly reports code it
tests as surviving.

Every safety check here has been broken on purpose to confirm its test then
fails. A test that passes against known-broken code is testing nothing, and the
only way to know the difference is to try it.

### Where the code lives

- `harness/cli.py` is the argparse wiring and `main`. Each group of verbs is a
  module in `harness/commands/`: `lanes` (image, video, svg, web, code, say,
  hear ...), `discover`, `loop`, `screen` (and fetch), `measure` (and verify,
  throughput), `adopt`, `judge` (and rubric), `report`, `jobs`, `store` (and
  memory, disk), `gateway`, `usage`. `common` holds `err`, `say`, `note`,
  `emit` and the `--json` state they read.
- `harness/memory_store/` is the store. `connection` opens it, `schema` holds
  the DDL and the shared vocabularies, and the accessors are `proposals`,
  `transitions` (the one writer of proposal state, #409), `retests`,
  `revisit`, `machines`, `cards` and `sources`. `harness.memory_store`
  re-exports every name, so `ms.decide(...)` still works.
- `harness/memory_store/migrations/` is the migration chain: one `vNN.py` per
  schema version, helpers beside them.

To add schema step N: create `migrations/vNN.py` with `VERSION = N` and any of
`columns(conn)` (every migration: add this schema's columns to an older store),
`early(conn)` (before the older steps' data), `data(conn)` and
`indexes(conn)`; set `FRESH = True` if it must also run on a new store; then
bump `SCHEMA_VERSION` in `schema.py`. `tests/test_module_layout.py` fails if the
steps are not exactly 1..SCHEMA_VERSION, and if a golden store under
`tests/golden/` no longer migrates to the content recorded in
`tests/golden/migrated/` (re-record an intended change with
`LH_RECORD_MIGRATED=1 uv run pytest tests/test_module_layout.py`). The same file
fails when a test patches a name where no code reads it, and when a module
under `harness/` passes 1500 lines.

Work is tracked in [issues](https://github.com/unxmaal/SoHoT/issues);
[#16](https://github.com/unxmaal/SoHoT/issues/16) is the roadmap.
`PLAN.md` holds the reasoning, the issues hold the state, and the issues win when
they disagree. `docs/validation-log.md` holds the evidence behind every claim,
including conclusions that turned out wrong and how they were caught.
`docs/testing.md` records which mutations were run and which two survived.

### The gauntlet

Every missed defect is traced to the logical failing behind it, that failing
becomes a generic class with a test shape, and later code is tested against
every class that has bitten (#492). The classes are defined once, in the
gauntlet skill (`SKILL.md` and `general-corpus.md`). This repo binds its
defects to them in `tests/gauntlet/classes.py`: `INDEX` maps a class id to its
instances here (issues and KB rules) and any scanner tests, `PENDING` holds
proposed classes that have no skill entry yet, and `UNCLASSIFIED` gives a
one-line reason for a defect that fits nothing.

When you fix a defect: label the issue `defect`, add `{"issue": N}` to the
class it belongs to (or a `PENDING` entry, then add the class to the skill and
move it into `INDEX`), and refresh the snapshots:

```bash
uv run python -m tests.gauntlet snapshot-defects   # closed `defect` issues, via gh
uv run python -m tests.gauntlet snapshot-skill     # class ids from the skill
uv run python -m tests.gauntlet audit              # gaps, classes per tier, kill rate
```

`audit` lists closed defects no class names and closing references to them in
`main`'s history and merged PR bodies; `--offline` uses the committed
snapshot. CI checks the registry against the snapshot without network.

Tier-2 classes with an honest trigger detector (`tests/gauntlet/detectors.py`,
named by the `detector` field in `INDEX`) find the sites where the trigger
fires, such as a fixed table looked up with a fallback or an environment
variable read with no test of who sets it. Every hit needs a test bound to it
with `@pytest.mark.gauntlet("<class id>", site="<site the detector names>")`,
or a waiver in the class's `waivers` with a one-line reason, so new code that
trips a detector fails CI until it gets its own test. The hits that predate the
detectors sit in `tests/gauntlet/backlog.json`, which may only shrink. The
classes no detector can find carry a `review` regex instead, and
`soh gauntlet review [range]` (default `origin/main...HEAD`) prints each one
whose heuristic fires on the added lines, as its trigger question and test
shape from the skill; `soh gauntlet audit` is the audit above.
`soh chaos --yes-break-things` runs three fault scenarios on a real machine
and reports pass or fail for each: a scratch model server killed mid-screen, a
scratch directory filled below the fetch floor, and the HF client pointed at a
socket that resets. Each runs on a scratch `LOCALHARNESS_HOME` it creates and
removes, under the machine lock. It refuses without the flag, under pytest or
CI, or while another run holds the lock, and nothing schedules it.

## Two traps this repo exists to remember

`mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages` there
by default. The opt-out is in both `gateway/config.yaml` and
`scripts/serve-gateway.sh`.

LiteLLM's `/health/readiness` returns 200 before the proxy can serve. Never
conclude anything from a request made in that window, and when restarting, wait
for port 4000 to free before probing or you will test the dying process.
