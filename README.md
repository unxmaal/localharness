# localharness

[![check](https://github.com/unxmaal/localharness/actions/workflows/ci.yml/badge.svg)](https://github.com/unxmaal/localharness/actions/workflows/ci.yml)

**Make pictures, video, speech and code on your own machine. Nothing leaves it.**

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
| `lh image "a red fox in falling snow"` | an image, tens of seconds |
| `lh video "a fox running" --seconds 2` | a video with sound, ~40 min |
| `lh svg "a settings gear icon"` | a real vector icon |
| `lh web "a landing page for a coffee roaster"` | a self-contained HTML page |

Every timing in this file was taken on an M2 Pro with 32 GB unless it says
otherwise, and image and video default to 512x512. Trust your own machine
instead: each generation prints its wall time, its peak memory and the
resolution that produced them, because a number without its configuration
compares to nothing.
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
lh svg "a settings gear icon"
```

That prints a path. Open it. For speech on Apple Silicon, also start
`./scripts/serve-tts.sh`; with an NVIDIA card it is already up. Then
`lh say "hello"`.

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

---

# Using it

## Voices

`lh voices` lists them. The default is **`fr-male`**, a French-accented English
voice, cloned from a reference clip rather than picked from a table.

That works because Chatterbox clones *across* languages: the reference clip
speaks French, the output speaks English, and the accent comes along with the
voice. No accented-English corpus was needed.

It costs roughly 2s of fixed overhead per call plus about 1.5x the length of
the audio, so a line is the wrong unit: four words took 4.4s for 1.6s of
speech, twenty-eight took 11.3s for 6.2s. Measured 2026-09-12 on an M2 Pro that
was already swapping, so read them as an upper bound.

```bash
lh say "the tests all passed"                     # cloned, 4.4s
lh say "the tests all passed" --voice bm_george   # Kokoro, 3.5s
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
Anything that can is queued for download, and `lh fetch --run` takes them one
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
instead of doing it itself. The laptop asks for an image, whichever machine is
serving makes it.
That is done over MCP, a small standard for letting an assistant call outside
tools.

```bash
./scripts/serve-mcp.sh      # listen on 0.0.0.0:8899
claude mcp add --transport http localharness http://<host>.local:8899/mcp
```

> **There is no authentication.** Anyone who can reach port 8899 can use this
> machine's GPU. That is a deliberate choice for a home network. On any network
> you do not control, bind to localhost instead: `TTS_HOST=127.0.0.1`.

That exposes `svg`, `web`, `code` and `image` to the assistant. Every tool shells
out to `lh`, so the CLI, the eval suite and the MCP server run identical
commands, and what gets measured is what ships.

`image` is queued: it holds 11.4 GiB at the default 512x512, against 23.9 at
1024, and the inference server swaps models
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

Both `lh` and the service scripts resolve it the same way. `lh` has to do it
itself: on PATH it runs with nothing sourced, and an unset `HF_HOME` sends
huggingface_hub off to re-download what is already on the drive.

`HF_MIN_FREE_GB` is 20, which is enough for a normal model and deliberately not
enough for MiniMax-H3. `fetch-h3-weights.sh` demands its own 160GB and
`setup-omnisvg.sh` its own 40, where the size is actually known. A global floor
cannot know what is about to be fetched, and set to 160 it refused every
ordinary machine.

The install carries no torch on Apple Silicon. `mlx-whisper` needs it
unconditionally, so the multilingual ear lives in the `whisper` dependency
group: 370MB installed rather than 1.1GB. Dependencies are marked by platform
in `pyproject.toml`, so a Windows install pulls neither mlx nor pyobjc and a
Mac pulls no winsdk.

## Bringing up a Linux machine

Written against Ubuntu 24.04 with an NVIDIA card. Any distribution will run
this; 24.04 is the one under test, because GitHub's `ubuntu-latest` **is**
24.04, so a red `check-linux` is a real failure rather than a runner-only one.
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
with and `lh discover` reports missing without.

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

### 4. The repo, uv, then lh

```bash
git clone https://github.com/unxmaal/localharness.git
cd localharness
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
lh discover                         # what this machine can do
```

`services.sh` registers nothing with Linux, exactly as it registers nothing with
Windows. Same file, same verbs; the only differences are how a process is
launched detached, how it is asked whether it is alive, and how its tree is
ended. There is no systemd unit and that is a decision, not a gap: a machine
with a day job should start these when asked and not before. If this is the
machine you want serving on boot, write the unit yourself -- nothing in the
harness will fight you.

### What is different here, and worth knowing before it surprises you

`lh say --play` uses `paplay`, `aplay` or `ffplay`, whichever is present. Text in
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

CI runs `make check` on an Apple Silicon runner AND a Windows one for every
push and pull request, and each reports which tests it skipped and why. A
skipped test otherwise reports green for something it never checked -- and
that is not hypothetical here: 29 tests skipped on both runners for a week
because they needed a volume only one machine has, while ten of them were
failing on that machine. The first run that could execute them found a bug in
the product, not the tests.

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
