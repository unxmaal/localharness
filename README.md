# localharness

[![check](https://github.com/unxmaal/localharness/actions/workflows/ci.yml/badge.svg)](https://github.com/unxmaal/localharness/actions/workflows/ci.yml)

**Make pictures, video, speech and code on your own Mac. Nothing leaves the machine.**

One command, `lh`, generates an image, a short video, an SVG icon, a web page,
some code, or speech in a voice you chose. It also transcribes what you say.
No account, no API key, no per-token bill, no rate limit, and no model quietly
retired out from under you.

The part that makes it more than a pile of scripts: **it goes looking for better
ways to do its own job, and then proves whether they are better.**

This field moves weekly. A model or a technique that was best when this was
written is probably not best now. So `lh discover` reads the model registries,
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
| `lh image "a red fox in falling snow"` | an image, ~19s |
| `lh video "a fox running" --seconds 2` | a video with sound, ~40 min |
| `lh svg "a settings gear icon"` | a real vector icon |
| `lh web "a landing page for a coffee roaster"` | a self-contained HTML page |
| `lh code "parse an ISO timestamp"` | code, to stdout |
| `lh extract --file build.log "which tests failed?"` | ask a question about a file |
| `lh say "the tests all passed"` | speak it aloud |
| `lh hear --seconds 5` | record and transcribe |
| `lh discover` | what this machine can do that nobody has measured |
| `lh discover --neighbors` | what the people who build your tools are starring |
| `lh fetch` | what is queued for download, and nothing more until you say so |

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
  to 19 seconds against 43 and 11.4 GiB against 13.7. Knowing it was a tie is
  the useful part.
- a quality score that looked useless turned out to work. The experiment
  measuring it had compared two things that were never comparable.

**It does not fall behind.** It reads the model registries and the places
practitioners talk, on a schedule, and tells you what is new. See below.

## Try it

```bash
uv tool install --python 3.12 --editable .
./scripts/serve-mlx.sh &        # inference engine
./scripts/serve-gateway.sh &    # the address clients use
lh svg "a settings gear icon"
```

That prints a path. Open it. For speech, also start `./scripts/serve-tts.sh` and
run `lh say "hello"`.

Those two servers are MLX. On Windows the install and the suite work, and the
text lanes need an OpenAI-compatible server of your own in `gateway/config.yaml`
in place of `serve-mlx.sh`.

## What it is not

Before you invest an afternoon:

- **Half the lanes have no tool on Windows yet.** The harness, the checks and
  the eval runner all run there. The generators do not: every one that ships is
  MLX or Metal. Those gaps are being filled. See "A second
  machine" below for which lane is where.
- **There is no Linux path.**
- **It needs disk.** The models this uses run 4 GB to 31 GB each.
- **Video takes about 40 minutes a generation** on an M2 Pro with 32 GB. It
  works; it is not something you will use casually.
- **It is a workshop, not a product.** There is no GUI, and some lanes are better
  than others.

## A second machine

**Every lane is meant to run on either machine, with a different tool on each.**
What draws an image on Apple Silicon is not what will draw one on an NVIDIA
card, and it is not meant to be. What has to match is that the lane has an
implementation on both, that the implementation is tested there, and that the
result says which one produced it. A lane that exists on one machine and not the
other is unfinished.

The checks work that way now. Text in a generated image is read by Apple's
Vision on macOS and by Windows.Media.Ocr on Windows. HTML is rasterised by
Chrome, or by Edge, which is Chromium and is already on every Windows install.
Neither is named by the caller: each is found by asking the platform and the
import, and a machine with neither warns and withholds the metric instead of
failing the candidate.

The generators do not yet. This is where each lane stands:

| lane | Apple Silicon | Windows and NVIDIA |
|---|---|---|
| web, code, extract | mlx_lm.server behind the gateway | any OpenAI-compatible server, no script yet |
| image | mflux | not built |
| video | h3 | not built |
| tts | mlx-audio | not built |
| stt | mlx-audio, mlx-whisper | not built |
| svg | traced from an image, or a text model | follows image and text |
| ocr check | Apple Vision | Windows.Media.Ocr |
| html render | Chrome | Chrome or Edge |
| svg rasterise | rsvg-convert | rsvg-convert |

The gateway is LiteLLM, so an OpenAI-compatible server on the Windows box
answers the text lanes today by editing `gateway/config.yaml`. There is no
`serve-*.sh` for it yet, and `./scripts/serve-mlx.sh` is Apple Silicon and does
not start there.

**A result records what produced it.** `hw_model` identifies the GPU on Apple
Silicon because it is the same part. On a PC it says nothing about it, so
results.json carries an `accelerator` field with the kind, the name and the
memory. Without it a run from the mini and a run from the 4070 are the same row
to a score sheet, which is the one thing this suite exists to tell apart.

**The memory ceiling is read off the card.** Unified memory hands the GPU a
fraction of system RAM. A discrete card is a wall, and the 61.6 GB of RAM behind
a 12 GB 4070 is not budget: computing it that way produced 46 GB that does not
exist. Qwen3-30B-A3B at 4-bit is too big for that card and fits a 24 GB one, the
same candidate and two answers. Without a card the harness still runs and
reports system RAM as its budget.

Service supervision is the one thing that is not a lane and has no Windows
counterpart: launchd is macOS, and nothing replaces it there yet.

---

# Using it

## Voices

`lh voices` lists them. The default is **`fr-male`**, a French-accented English
voice, cloned from a reference clip rather than picked from a table.

That works because Chatterbox clones *across* languages: the reference clip
speaks French, the output speaks English, and the accent comes along with the
voice. No accented-English corpus was needed. It costs about 2s a line.

```bash
lh say "the tests all passed"                     # cloned, ~2s
lh say "the tests all passed" --voice bm_george   # Kokoro, sub-second
```

Use `bm_george` when a line needs to come back immediately.

## Piping

`code` and `extract` print to stdout, because you pipe or read them rather than
open them in a viewer. `extract` reads stdin when given no `--file`:

```bash
make test 2>&1 | lh extract "which test failed, and why?"
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

So `lh` looks, on your behalf.

**1. It works out what you have and what you have never tried.**

```bash
lh discover           # everything available here, and whether it has been tested
lh discover --gap     # just the untested things
```

It reads your installed tools, your downloaded models and the results of every
past test run. Nothing is hand-maintained, so the answer is right whenever you
ask rather than as of whenever someone last updated a list.

**2. It goes out and finds what exists now.**

```bash
lh discover --external --lane image   # ask the model registries
lh discover --feeds                   # read where practitioners talk
lh discover --neighbors               # read what the people who build your tools star
```

`--external` queries the HuggingFace registry. Good for "what models exist",
useless for anything that is not a single model.

`--feeds` reads community aggregation posts, because a registry can tell you a
model exists but not that everyone has moved to a small add-on file that made
generation five times faster. That is what it found on its first real run: a
*MiniMax-H3-Turbo* LoRA claiming a 5x speedup, against a video lane that takes
40 minutes a generation.

`--neighbors` is the newest and the highest signal. People who maintain the
tools you already run follow and star each other, and what they star is a
curated list rather than a popularity poll. `lh` starts from whoever contributes
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
lh discover --feeds --comments 5            # replies on the 5 newest posts
lh discover --feeds --comments 5 --judge    # and read names out of the prose
```

The judge is what reads the prose, because the linked repos are the easy half:
"Krea 2 being replaced with Anima" is a claim no registry can produce and no
pattern-matcher was going to find.

**3. It sorts proposals cheapest-first, so measurement is spent where it counts.**

A sweep produces far more proposals than this machine can run. Four tiers, each
more expensive than the last, and each one only sees what survived the one
before:

| tier | cost | what it answers |
|---|---|---|
| judge | about a second, no GPU | is this worth looking at |
| inspect | seconds, a source clone | can it run on this machine at all |
| screen | one minimal run | does it run |
| measure | the full suite | is it better |

```bash
lh discover --neighbors --judge   # score proposals 1-10 against a rubric
lh discover --inspect             # clone the source and check it fits
lh fetch                          # what is queued for download
lh fetch --run                    # download it, one at a time
```

**Inspect** is the one that saves the most. It clones a candidate's source,
which is single-digit megabytes, and reads what the description could not say:
which weights it names and how big they are, whether CUDA is a *declared
dependency* rather than merely mentioned somewhere, whether it is MLX-native or
torch with an MPS fallback, whether there is anything to call, and when it was
really last touched. Nothing is executed and nothing is downloaded.

That last distinction is load-bearing. An Apple on-device repo mentions
`torch.cuda` in one export recipe; treating that as a CUDA requirement threw
away the best candidate in the sweep. So only a declared dependency
disqualifies.

Anything that cannot run here is recorded as answered, permanently, and never
proposed again. Anything that can is queued for download, and `lh fetch --run`
takes them one at a time with a disk floor, because this machine holds one
working set.

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
lh discover --recurrence     # what keeps coming back, and how each source is doing
```

Discovery goes stale, which defeats the point, so `lh` tracks when it last
looked and tells you in ordinary `lh discover` output when it has been too long.
Default is 7 days, set `$LOCALHARNESS_DISCOVERY_DAYS` to change it. Each kind
of answer is cached for less time than that, or a more frequent sweep would
just re-read what it read last time and report success.

```bash
lh discover --sources        # which places it reads, and when it last looked
```

It also watches for **new places to read**, since the site everyone uses
in a year may not be the one they use now. When a feed keeps pointing somewhere
`lh` does not read, it probes whether that address actually serves a feed and
tells you. Never automatic: a web address suggested by a stranger should need a
human to agree before this thing starts fetching it on a timer. Places it reads
live in `~/localharness/discovery-sources.json`; edit that file freely.

## Letting another computer use this one

If you use an AI coding assistant on a laptop, it can hand work to this machine
instead of doing it itself. The laptop asks for an image, this Mac makes it.
That is done over MCP, a small standard for letting an assistant call outside
tools.

```bash
./scripts/serve-mcp.sh      # listen on 0.0.0.0:8899
claude mcp add --transport http localharness http://styx.local:8899/mcp
```

> **There is no authentication.** Anyone who can reach port 8899 can use this
> machine's GPU. That is a deliberate choice for a home network. On any network
> you do not control, bind to localhost instead: `TTS_HOST=127.0.0.1`.

That exposes `svg`, `web`, `code` and `image` to the assistant. Every tool shells
out to `lh`, so the CLI, the eval suite and the MCP server run identical
commands, and what gets measured is what ships.

`image` is queued: it holds 11.4 GiB and the inference server swaps models
through a single queue, so it returns a job id and `job_status` carries the queue
position and the artifact path. Artifacts stay here, in `~/localharness/out/mcp/`.

Video and speech are not exposed over MCP. Video needs more than a queue to be
usable remotely, and speech was ruled out; both stay available locally.

DNS-rebinding protection stays on, with an allowlist in `MCP_ALLOW`. It guards a
different thing than the missing authentication does: rebinding needs only that
someone here opens a web page, not that the port is reachable from outside.

---

# Setup and upkeep

## Install

```bash
uv tool install --python 3.12 --editable .
```

`lh` lands on PATH via `~/.local/bin`, in its own venv. It never touches the
system Python.

**`--python 3.12` is not optional.** `uv tool install` picks an interpreter
silently, and mflux once installed against 3.9 where every entry point died on
`int | None`. `--editable` keeps the checkout as the source of truth.

`lh` finds the weights on its own, by the same free-space rule the service
scripts use. It has to: on PATH it runs with nothing sourced, and an unset
`HF_HOME` sends huggingface_hub off to re-download what is already on the volume.

The install carries no torch. `mlx-whisper` needs it unconditionally, so the
multilingual ear lives in the `whisper` dependency group: 370MB installed
rather than 1.1GB.

## Keeping the services up

Installed as launchd agents, so the machine comes back serving after a reboot:

```bash
./scripts/launchd.sh probe       # can an agent read the weights volume?
./scripts/launchd.sh install
./scripts/launchd.sh status
```

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

## Where things land

One root, `~/localharness`, overridable with `$LOCALHARNESS_HOME`:

```
out/                        artifacts: images, audio, svg, pages
out/mcp/                    artifacts asked for over MCP
runs/<stamp>-<modality>/    one eval run: artifacts and results.json
logs/                       service stdout and stderr
```

There were four places before this, and one was relative. `lh` runs from
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
  Untested is the default, which is what `lh discover` exists to make visible
  rather than letting it be assumed.

To run two candidates against each other:

```bash
uv run python -m evals.run --modality image --out .logs/img \
  --candidates mflux:flux2-klein-4b,mflux:z-image-turbo
```

A candidate is written as a short spec: a model nickname for text
(`local-large`), `mflux:<model>` for anything that runs as a separate program, or
`tts:<model>,voice=<name>` for speech.

A candidate can also be a **method**, a way of working rather than a model. A
method can beat a better model:

```bash
--candidates trace:mflux:flux2-klein-4b        # draw a picture, then trace it to vector
--candidates trace-icon:mflux:flux2-klein-4b   # same, tuned small: 3.2x fewer bytes
--candidates omnisvg:4B                        # a model that emits SVG draw commands as tokens
--candidates repair:q3-4b                      # generate, check it, fix it, repeat
```

`omnisvg` needs `./scripts/setup-omnisvg.sh` first: it has its own checkout, its
own venv and 16 GiB of weights.

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
proposed, screened and fetched. `lh sensitivity` varies each one against cached
data and says whether the output moved at all -- inert, inside a band, or on an
edge where the value next door behaves differently.

```bash
lh sensitivity                 # all eleven probes, about four minutes, no network
lh sensitivity --list          # and the constants nothing covers, with reasons
```

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

## Developing

```bash
make check     # shellcheck + bash -n + unit tests, no services needed
make test      # unit tests only
make smoke     # end-to-end, REQUIRES the services running
```

CI runs `make check` on an Apple Silicon runner for every push and pull request,
and reports which tests it skipped and why. A skipped test otherwise reports
green for something it never checked.

Every safety check here has been broken on purpose to confirm its test then
fails. A test that passes against known-broken code is testing nothing, and the
only way to know the difference is to try it.

Work is tracked in [issues](https://github.com/unxmaal/localharness/issues);
[#16](https://github.com/unxmaal/localharness/issues/16) is the roadmap.
`PLAN.md` holds the reasoning, the issues hold the state, and the issues win when
they disagree. `docs/validation-log.md` holds the evidence behind every claim,
including conclusions that turned out wrong and how they were caught.
`docs/testing.md` records which mutations were run and which two survived.

## Two traps this repo exists to remember

`mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages` there
by default. The opt-out is in both `gateway/config.yaml` and
`scripts/serve-gateway.sh`.

LiteLLM's `/health/readiness` returns 200 before the proxy can serve. Never
conclude anything from a request made in that window, and when restarting, wait
for port 4000 to free before probing or you will test the dying process.
