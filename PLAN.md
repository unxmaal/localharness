# localharness

Local media generation and voice, on Apple Silicon, in a command line.

Last rewritten 2026-09-07, against a working tree. Numbers here were measured
on this machine; `docs/validation-log.md` holds the evidence, including
conclusions that were wrong and how they were caught.

## 1. What this is for

Generate images, video, SVG and web pages, and talk to the machine, entirely
locally. That is the whole goal, and `lh` is the product:

    lh image "a red fox in falling snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh code  "a python function that parses an ISO timestamp"
    lh extract --file build.log "how many tests failed?"
    lh say   "the tests all passed"          # cloned, French accent
    lh voices
    lh hear  --seconds 5

Installed with `uv tool install --python 3.12 --editable .`, which puts `lh` on
PATH in its own venv and touches nothing the system Python can see.

**The primary caller is an agent, not a person.** Eric asks Claude Code for an
SVG; Claude Code runs `lh`. That is not a fallback for the CLI, it is the point
of it, and it is why the verbs print paths and exit non-zero on a bad artifact
rather than being chatty. A second machine's agent reaches the same commands
over MCP (section 3).

An earlier version of this plan opened by saying the product was intelligent
routing between local and cloud models. It is not, and the phases that followed
from that premise put cloud routing, opencode integration and a ComfyUI lane
ahead of media. None of them were built and none of them should be.

Two things follow from "local only" that shape everything below:

- **Nothing is a wheel worth reinventing.** Find the mature project first. This
  repo has twice built something that already existed and was better; the voice
  front end was 400 lines superseded by `voicemode`, and an STT proxy was 275
  lines solving a problem `voicemode` had already solved with a config key.
- **Every claim gets measured.** "The command exited 0" is not evidence. A
  broken diffusion pipeline emits a uniform grey square at the right resolution
  with a clean exit status.

## 2. The machine

M2 Pro Mac mini, 32 GB, macOS 26 (Darwin 25.5.0). Ten CPU cores, sixteen GPU
cores. It has been through a lightning strike and one HDMI port is dead; it is
otherwise sound.

An **M5 Ultra Mac Studio, 96 GB, arrives around November 2026** and becomes the
serving machine. Section 7 lists what has to change for it.

### Storage

Model weights live on `/Volumes/Models`, and nowhere else. One statement, since
this document previously contradicted itself four times over whether it was the
T7:

| Volume | Interface | Write | Cold read |
|---|---|---|---|
| `/Volumes/Models` | DockCase C1P, direct to a host controller, `ioreg` Speed=4 (10 Gbps) | 1013 MB/s | **959 MB/s** |
| `/Volumes/T7` | Samsung T7 behind a VIA Labs USB 3.0 hub, `ioreg` Speed=3 (5 Gbps) | 422 MB/s | 432 MB/s |

The T7 is itself a 10 Gbps device; the hub halves it. Cold reads were forced by
unmount and remount so the page cache could not flatter them.

This matters more than it looks. `mlx_lm.server` hot-swaps models per request,
so load time is paid on every switch, and h3's `--ssd-streaming` reads a DiT
block off disk per step. On the T7 that feature would cost roughly twice as
much.

`scripts/env.sh` enforces it, and refuses to run rather than let
`huggingface_hub` silently recreate its cache under `$HOME`. Its rule is free
space, not "is this external": 160 GB, because MiniMax-H3's checkpoint alone is
134 GiB. That refuses this mini's internal disk, which sits at 91%, for the
reason that actually matters, and it will not refuse the Studio's.

## 3. Architecture

```
   lh (harness/cli.py)      evals/run.py      harness/mcp_server.py :8899
          |                       |                    |
          +-----------+-----------+--------------------+
                      |                          (shells out to `lh`)
                   harness/
  engines.py  proc.py  completion.py  audio.py  checks/  jobs.py  memory.py  env.py
       |         |           |            |                 |        |         |
   mflux, h3  subprocess  gateway     mlx_audio      one-at-a-time  will it   where the
              + peak mem   :4000        :8890           queue        fit?     weights are
                             |
                         mlx_lm :8081
```

`jobs.py`, `memory.py` and `env.py` are all consequences of things that went
wrong; each is described in section 6.

The rule is that **the CLI and the eval suite run identical commands**. They
import the same engine specs, the same system prompts and the same process
runner. For a while the only code that knew how to invoke a generator lived
inside the eval suite, which meant the harness could measure something the
product did not ship.

### The gateway is a seam, not a feature

LiteLLM on `127.0.0.1:4000` fronts `mlx_lm.server` on `:8081`. Its value is that
swapping what is under test costs a string in a config file rather than a code
change, and that it speaks both `/v1/chat/completions` and Anthropic's
`/v1/messages`. It is bound to loopback and nothing else.

### Serving the other machine

`scripts/serve-mcp.sh` exposes `svg`, `web`, `code` and `image` over MCP on
`0.0.0.0:8899`, so another person's Claude Code on another machine can use this
one's GPU. `claude mcp add --transport http localharness http://styx.local:8899/mcp`
is the whole client setup.

Every tool SHELLS OUT TO `lh`. The CLI, the eval suite and the MCP server run
identical commands, which is the same rule as everywhere else in this repo and
the reason a third caller cannot quietly drift from the product.

`image` is queued and returns a job id; the rest answer in seconds. Video and
the speech verbs are deliberately not exposed: video needs job semantics past a
queue, and speech over the LAN was ruled out. Both stay reachable locally.

Scoped from a real requirement rather than guessed: svg, image and some code
generation are what the second user actually wants, and none of it is needed
until after the Studio migration.

### Engines are data

A candidate is a spec string:

    mflux:flux2-klein-4b               image, 8-bit (default)
    mflux:z-image-turbo,quantize=4     image, 4-bit
    h3                                 video

Adding one is a string. Quantization is part of the candidate name, because
`z-image-turbo` at q4 and at q8 have different speed and memory and sharing a
row makes the comparison meaningless.

**mflux has one entry point per model family**, and this is not optional.
`mflux-generate --help` lists every built-in model under `--model` because all
the binaries share one parser, then rejects the wrong ones at runtime:
`FLUX.2 Klein is not supported by mflux-generate. Use mflux-generate-flux2
instead.` The mapping in `harness/engines.py` is read off mflux's
`entry_points.txt`, and a test asserts every binary it names exists.

### The eval suite

Cases are YAML: a prompt, `params:` for the engine, `assert:` for the checks.
Unknown keys in either block are rejected at load, so `widht: 512` costs nothing
rather than generating at the default size and passing.

It measures two different things and keeps them separate:

- **A competence gate.** Does it decode, is it the size requested, is it more
  than one colour, does the SVG parse, is the HTML self-contained. This
  separates working from broken and cannot rank two candidates that both work.
- **A quality axis.** Numbers a checker emits alongside its verdict: word error
  rate for speech, character error rate for text rendered into an image. This is
  what puts two working candidates in an order.

Aggregation differs on purpose. **Latency takes a median** so one cold model
load cannot decide the winner. **A quality metric takes a mean**, because most
cases score a clean zero and the entire signal is in the few that do not — the
first voice comparison medianed 0.000 for two voices that were not equal.
`metrics_worst` keeps the outlier visible.

## 4. What is built and measured

All measured 2026-09-05 to 09-07. Versions: uv 0.12.5, mlx 0.32.2,
mlx-lm 0.31.3, mflux 0.19.1, mcp 2.1.1.

| Lane | State |
|---|---|
| Text (svg, web) | mlx_lm behind the gateway. Working, and weak: see the SVG note below. |
| Code | mlx_lm behind the gateway. The eval RUNS the generated code. |
| Extract | The delegate-to-a-small-model lane: a log in, one fact out. |
| Image | mflux. `flux2-klein-4b` at ~19s per 512² warm, ~54s cold. Working. |
| Video | antirez/h3.c. 512×512×22 frames in 40.5 min, 9.48 GiB peak. Working, and Studio-gated by time not memory. |
| Speech out | Kokoro-82M via mlx-audio on :8890, plus Chatterbox for cloned voices. |
| Speech in | Parakeet (English, fast) or mlx-whisper in-process (multilingual). |
| Voice UI | `voicemode` (MCP, 1349★) is the front end. Not ours, deliberately. |
| Serving | Four launchd agents; survives reboot and crash. MCP on :8899 for the second machine. |

### The SVG lane is the wrong tool, and this is settled

`lh svg` asks a general chat model to write bezier coordinates it cannot see.
Asked for "a cartoon frog holding a coffee mug", Qwen2.5-7B emitted eighty
near-identical `<path>` elements and hit the token ceiling mid-attribute; with
sampling fixed it produced a complete, valid document of coloured blobs. Asked
for two concentric gears it drew two offset squares. The default at the time,
Qwen2.5-1.5B, produced valid SVG in which every path was `M256 256 L256 256` --
a zero-length line, `ink=0.0000`, structurally perfect and visually empty.

**Generate a raster and vectorize it instead.** `lh image` produces a good
cartoon frog in 54s; `vtracer` turns it into real vector paths in **0.05s**,
`ink=0.2605`. That is a complete answer to a question an LLM cannot do at any
size available here, and no amount of a better chat model changes it.

Not yet wired into `lh` as a pipeline. When it is, the dedicated text-to-SVG
models (OmniSVG, StarVector) are the other candidate worth measuring against
it; both are torch on MPS rather than MLX.

**Video fits in 32 GB, which this plan once said it would not.** Only the DiT
streams from SSD, and prompt encoding and the two VAEs run in separate phases,
so peak is the largest phase rather than the sum. Measured peak 9.48 GiB against
a 134 GiB checkpoint.

`tools/h3probe.c` calls h3's standalone `h3_metal_probe()`, needing no weights:

    Apple M2 Pro (applegpu_g14s)
      physical memory       32.0 GiB
      recommended GPU set   25.0 GiB
      max Metal buffer      18.7 GiB      <- a hard per-allocation ceiling
      Apple GPU family      8

`--ssd-streaming` is therefore mandatory here: 25.0 GiB does not fit the 36.5
GiB full-residency DiT, and does fit the 2.0 GiB streamed one.

**int8 is unavailable here, and Metal 4 does not imply otherwise.** h3 gates
TensorOps on a literal substring match of the device name in `h3_gpu.m`:
`[gpu.device.name rangeOfString:@"M5"]`. The probe's `metal4` field reads yes on
this M2 Pro, because macOS 26 extends the Metal 4 API family well beyond
hardware with tensor units, but h3 never consults it. "Apple M2 Pro" contains no
"M5", so `--use-int8-row-fc2` is inert. It will light up by itself on the Studio.

h3 builds clean under its own `-Wall -Wextra -Wpedantic -Wconversion` and passes
1768 checks, including native Metal primitives matching host references to
~1e-7. Other constraints from its documentation: canvas dimensions multiples of
32 with the product under 768×1344; audio references 2-15 seconds; FFmpeg and
FFprobe required.

### Decisions taken

**The text models were never chosen, and now they have been.** Qwen2.5-0.5B
entered this repo as a smoke test proving `mlx_lm.generate` ran at all. The
family then served four lanes for a month while the words "SOTA", "surveyed"
and "leaderboard" appeared nowhere in 671 lines of validation log. Measured
2026-09-07 against Qwen3, five candidates that fit in 32 GB, `--repeat 3` on the
stochastic lanes:

| lane | winner | runner-up | previous default |
|---|---|---|---|
| svg | local-large 7/9 @ 4.7s | q3-14b 7/9 @ 10.0s | local-mid **2/9** |
| web | local-large 6/6 @ 21s | q3-14b 6/6 @ 61s | local-mid **3/6** |
| code | **q3-4b** 6/9 @ 2.9s | q3-8b 6/9 @ 130s | local-large 5/9 |
| extract | local-large 9/10 @ 0.8s | q3-14b 9/10 @ 13.4s | local-large |

Only `code` moved. **Qwen2.5-7B keeps svg, web and extract on merit**: equal
accuracy to Qwen3-14B and two to seventeen times faster. A generation behind is
not the same as wrong for the job.

What disqualifies Qwen3-8B and Qwen3-14B is not their scores, which look fine.
They are hybrid THINKING models, and Qwen3-4B-Instruct-2507 is not. Asked to
"reply with exactly: OK" they spend 140-152 completion tokens against 2. mlx_lm
puts the reasoning in a separate `reasoning_content` field so nothing leaks into
the artifact -- which is why it is easy to miss -- and what it does instead is
eat the token budget. On an SVG the whole 4000 goes to reasoning and `content`
returns NULL. `lh svg` timed out twice at 180s before anyone looked at the
response shape, and q3-8b scored 0/9 on svg, every run a timeout.

**A pass rate says how often something fails, never how.** q3-14b's svg row read
7/9 at a 9.96s median and looked like a winner. Check the response shape.

The 30B candidates (Qwen3-30B-A3B and Qwen3-Coder-30B-A3B, 16 GB each) are
unmeasured. 16 GB of weights against a 24 GB Metal working set is what took the
machine down; see section 6.

**The default voice is `fr-male`, a cloned French accent.** Chatterbox clones
ACROSS languages: the reference clip speaks French, the output speaks English,
and the accent comes with the voice. That is why no accented-English corpus was
needed. About 2.4s a line against Kokoro's 0.3s, which is the price of the
default being the voice that was wanted; `--voice bm_george` is the fast one.
Two reference clips ship in `harness/voices/` (google/fleurs, CC-BY-4.0).
The reference clip is a first-class variable: three clips from the same corpus,
same language, same gender, scored 0.122, 0.144 and 0.578 corpus WER.

**French TTS, scored by whisper pinned to fr:** Kokoro `ff_siwis` 5/5 at 0.34s,
wer 0.044, beating every Chatterbox clone. But ff_siwis is the ONLY French voice
Kokoro has and it is female, so the table's winner is not the whole decision.
Chatterbox's dominant failure is invented trailing speech, not mispronunciation.

**Multilingual STT is mlx-whisper called IN-PROCESS**, bypassing mlx_audio's
server, which cannot load it: the mlx whisper repos ship weights.npz and
config.json without the `preprocessor_config.json` the server demands, and
mlx_whisper wants exactly what they do ship. English LibriSpeech 40/40, corpus
wer 0.022 at 1.14s median, against parakeet-1.1b's 0.011 at 0.18s. Whisper is
the multilingual ear; parakeet stays the English one.

**`flux2-klein-4b` is the image engine,** and the suite now supports that choice
rather than deferring to taste. Over 4 seeds x 3 cases with PickScore:

| | pass | median | peak | adherence | cer |
|---|---|---|---|---|---|
| flux2-klein-4b-q8 | 12/12 | **19.37s** | **11.4 GiB** | 23.975 | 0.000 |
| z-image-turbo-q8 | 12/12 | 42.64s | 13.7 GiB | 24.121 | 0.000 |

**Prompt adherence does not separate them.** Paired by case and seed the
difference is +0.146 in Z-Image's favour with a standard deviation of 0.490
across 12 pairs — t = 1.03, a 95% interval of -0.14 to +0.43, and a 7-5 split of
individual pairs. Between-case variance (a fox at ~22.5, a shop sign at ~26)
dwarfs it. Two backends agree: HPSv2 put Z-Image ahead by a similarly
meaningless 0.6.

Neither does text rendering, once the checker stopped being wrong about it. The
one apparent difference was Z-Image writing `OPEN.` with a period on one seed,
scored as one character in four. That is a good sign, not a bad render; the OCR
check now strips punctuation at the edges of a string but never inside it, and
both models are 4/4.

What actually separates them is **2.2x on speed and 2.3 GiB on memory**, both to
flux2. There is no tradeoff to weigh, which is why this agrees with the choice
originally made by eye.

**`bm_george` on Kokoro-82M is the default voice**, and the field has now been
checked rather than assumed. Male, as asked for, and the best-scoring male
voice: 0.000 against `am_adam`'s 0.031 over five cases.

| tts candidate | pass | median | wer |
|---|---|---|---|
| Kokoro-82M-8bit / bm_george | 5/5 | 0.31s | 0.000 |
| Kokoro-82M-bf16 / bm_george | 5/5 | 0.32s | 0.000 |
| Qwen3-TTS-12Hz-0.6B | 5/5 | 2.75s | 0.022 |

Qwen3-TTS is 9x slower and less intelligible; there is nothing to trade for.
The 8-bit Kokoro is indistinguishable from bf16 on both axes and 3MB smaller,
which is not a reason to switch either way. **Chatterbox could not be
measured**: it is a voice-cloning model and refuses to speak without either a
reference clip or a `conds.safetensors` the mlx-community repo does not ship.
That is the same blocker as the deferred French-voice thread, not a separate
one.

**STT is now ranked on its own**, against human transcripts rather than jointly
with a TTS:

| stt candidate | pass | median | corpus wer |
|---|---|---|---|
| parakeet-tdt-1.1b | 40/40 | 0.18s | **0.011** |
| parakeet-tdt-0.6b-v2 | 40/40 | **0.14s** | 0.013 |
| whisper-large-v3-turbo | 0/40 | - | - |

40 seeded utterances of LibriSpeech test-clean. Both parakeets land where their
published numbers say they should, which is the check that the instrument is
sound. The 1.1b is 15% more accurate and 29% slower — a real tradeoff, and the
0.6b stays the default because voice latency is felt and 0.2% is not.
whisper-large-v3-turbo is a packaging failure, not a model result: mlx_audio
wants a HuggingFace processor the mlx-community repo does not ship.

**Ollama is not the serving path.** It wraps llama.cpp, lags upstream, and hides
tuning flags. Still installed, with no models. llama.cpp remains a second lane
for GGUF-only architectures MLX has not ported; not installed.

**ComfyUI is not a lane.** `ComfyUI_MiniMax_H3_Extender` has a Motion Context
feature the hosted API does not expose, carrying a clip's sampled latent forward
rather than a last-frame still. It is a ComfyUI plugin and does not compose with
h3.c. Fallback only.

## 5. What is next, in order

### The bar

**Best effort, proven with real tests.** Two conditions, and they pull against
each other on purpose:

- **Every lane must have at least two models or methods actually measured.** One
  candidate is not a comparison, and a lane with one candidate has never been
  asked whether it should exist in its current form.
- **Do not spend hours on a job this machine makes slow.** A 32 GB ceiling and a
  40-minute video run are facts, not obstacles to push through. Where a
  candidate is refused for size or time, the refusal is recorded with the number
  that caused it, so it reads as a measurement rather than an omission.

Above both: **the codebase should be defect free.** A ranking produced by a
broken instrument is worse than no ranking, and this repo has now shipped three
of those.

### Where each lane stands against "at least two"

| lane | measured | verdict |
|---|---|---|
| web, code, extract | 5 models each | met |
| stt | parakeet 0.6b, 1.1b, whisper | met on models, NOT on methods -- all three run through MLX; see item 5 |
| tts | Kokoro (3 quantisations + 2 voices), Qwen3-TTS, Chatterbox (3 refs) | met |
| image | flux2-klein, z-image-turbo | met, though both are the SPEED picks |
| **svg** | 5 models but ONE METHOD | **not met** -- see item 3 |
| **video** | h3.c alone | **not met**, and blocked: see below |

### The work, in order

1. **Fix the mcm-engine PreToolUse hook.**

   **(a) DONE 2026-09-07, mcm-engine 80a43e0.** The hook was registered, ran in
   0.33s against a 2s timeout, and counted correctly -- and wrote every nudge to
   stderr with **exit 0**, which Claude Code shows the model only on exit 2. An
   agent ran several hundred built-in tool calls in one session and received
   none of them. Fixed by also emitting JSON on stdout as
   `hookSpecificOutput/additionalContext`, the channel `hooks/session_start.py`
   already uses. Verified live: the next Bash call came back with the nudge
   attached. Also fixed: the nudge named `mcp__knowledge__search`, which does
   not exist in this deployment.

   **(b) DONE 2026-09-07, mcm-engine ba706ed.** It had never blocked
   anything. RULE #61 records the reason as fail-open safety: a block would
   "dead-lock the agent exactly when the knowledge backend is unreachable, the
   one moment it cannot call a reset tool". **That reason does not hold**, tested
   directly: the counter resets in `_decide` on the PreToolUse *attempt*, before
   the tool runs and regardless of whether it succeeds. 7 edits -> `mutators 7`;
   one `search` attempt -> `mutators 0`. A down backend traps nothing, and the
   hook needs no network in its default path.

   Fail-open on the hook's OWN errors is separate and worth keeping: a malformed
   payload and a corrupted state file both exit 0. Verified.

   What actually argues against blocking is different from what is recorded:

   - **Restricted-tool subagents deadlock permanently.** `statusline-setup` has
     `Tools: Read, Edit` and no MCP tools at all. Six edits and it is blocked
     with no reset available to it, forever. This is the real trap and it is not
     the one on record.
   - **The thresholds are far below what the contract claims.** Actual
     `WARN_THRESHOLD = 3`, `BLOCK_THRESHOLD = 6`; CLAUDE.md advertises 8 and 20.
     Six edits is nothing during a refactor, and warn-at-3 is already noisy
     enough to fire twice inside one turn of read-only investigation.

   Shipped: thresholds raised to the advertised warn 8 / block 20, and a gap now
   emits `permissionDecision: "ask"` so the human decides. The hook still never
   denies, so a restricted-tool subagent cannot be stranded, and fail-open on the
   hook's own errors is untouched (malformed payload and corrupt state both still
   exit 0). Verified live: additionalContext at 7 edits, ask at 20. CLAUDE.md's
   wording was corrected to match -- it had claimed a hard block that never
   existed.

2. **DONE 2026-09-07. Make the eval report HOW a candidate fails, and say
   when two rows may not be compared at all.** Two gaps in the same instrument.

   **(a) How, not just how often.** A pass rate hid Qwen3-14B returning null
   content behind a 7/9 score, and hid whisper scoring a perfect 0.0 by failing
   every case. Separate "wrong answer" from "no answer".

   **(b) Comparability.** Borrowed from EnviousWispr
   (`scripts/eval/model_registry.py::comparable()`), which is further along than
   this suite: an explicit authority on whether two evaluations may be ranked
   TOGETHER. It enumerates every axis off the run receipt -- corpus and case
   count, rubric identity, judge identity, grading system, blinding, prompt
   variant -- and ARGUES each field it excludes, so an exclusion can be
   challenged rather than discovered later. Its docstring carries a RETRACTED
   argument, left visible, after review falsified it. Their registry also states
   "Numbers are read from score receipts, never typed."

   This suite has no such notion and the absence has already cost us. Rows have
   been ranked across runs with different sampling, different candidate sets and
   different warm/cold conditions; a warm eval median was quoted as CLI latency;
   and `local-mid`'s svg `ink` of 0.564 was computed over the two cases it passed
   and printed beside numbers computed over nine. A run should carry a receipt,
   and the report should refuse to put two incomparable rows in one table.

   Also worth stealing: their `TailBenchmarkHarness` runs every candidate from
   ONE frozen checkpoint so the comparison is PAIRED, where `--repeat` here
   averages unpaired samples.

   Shipped: `failure_kind()` splits every failure into wrong / empty / error
   and the report prints the breakdown; `metric_n` records how many rows each
   metric was computed over and the report flags any PARTIAL one; a `Receipt`
   goes into results.json and `comparable()` says whether two runs may share a
   table, arguing each excluded axis so it can be challenged.

   It caught a live bug on its first run. The "pass rate and ink DISAGREE" note
   announced that local-mid scored better on ink, 0.239 against 0.195 -- its
   0.239 computed over the ONE case it passed, the rival's over two. The note
   was itself making the mistake this item exists to prevent, and is now silent
   when any candidate's metric is partial.

3. **DONE 2026-09-07. Give the svg lane its second method.** Wire image-then-vectorize into
   `lh svg` and measure it against the LLM path on the same cases. Proven at
   0.05s producing a recognisable frog, where five language models produced
   coloured blobs.

   Shipped as `lh svg --method trace` and the eval candidate
   `trace:mflux:flux2-klein-4b`, scored by the same checker on the same cases.
   MEASURED, two cases at --repeat 2:

   | candidate | pass | median | ink |
   |---|---|---|---|
   | trace/flux2-klein-4b | **4/4 100%** | 52.7s | **0.222** |
   | local-large (LLM) | 2/6 33% | 5.3s | 0.057 |

   Tracing wins the lane outright on quality and loses by 10x on time. `llm`
   stays the CLI default because seconds against a minute is sometimes the right
   trade for a two-shape icon; `trace` is the one that draws the picture.

   THREE DEFECTS THIS TURNED UP, all fixed:
   - `failure_kind` scored a Metal GPU timeout as a WRONG DRAWING. Metal says
     "GPU Timeout Error", never "timed out", so a driver failure counted against
     the model.
   - `chart-bars` asserts `must_contain: ["text"]`, which a vectorizer cannot
     satisfy at any quality -- tracing turns glyphs into outlines. The case was
     written when an LLM was the only method and encodes that assumption. Cases
     can now declare `methods:` and it is `[llm]`.
   - Two candidates in ONE run then sat different exams (4 rows against 6), and
     4/4 beside 2/6 reads as a pass-rate comparison. The report now says so and
     names the cases only one of them ran.

   STILL OPEN: a traced icon is ~28KB of paths where a hand-authored one is a
   few hundred bytes. Right for illustration, wrong for a 24x24 UI glyph.
   OmniSVG and StarVector remain unmeasured -- both are torch on MPS.

4. **DONE 2026-09-07. Widen `web` from two cases.** At two cases the best
   candidate scored 6/6 and the lane discriminated between nothing. Three cases
   added, each aimed at where a small model actually stops short: interactive
   form validation (behaviour that must agree between markup and script),
   accessible navigation (aria-expanded, landmarks, a skip link -- the
   attributes a model omits while emitting the visible furniture), and a
   persisted dark-mode toggle (a small amount of real state).

   IT CHANGED THE ANSWER, which is the point of widening a saturated lane:

   | candidate | pass | median |
   |---|---|---|
   | **q3-4b** | **5/5** | 21.5s |
   | local-large | 3/5 | 19.1s |
   | local-mid | 1/5 | 4.5s |

   At two cases local-large and q3-4b were indistinguishable and speed decided.
   At five, q3-4b wins outright at the same speed -- and the OTHER way from svg,
   where local-large still leads. So `DEFAULT_TEXT_MODEL` split into
   `DEFAULT_SVG_MODEL` and `DEFAULT_WEB_MODEL`: one default could only ever have
   been wrong for one of the two lanes.

   `video` stays at one case, refused on time: see the table below.

5. **DONE 2026-09-07. STT: a second RUNTIME, not just more models.** The lane compared parakeet
   0.6b, parakeet 1.1b and whisper -- but all three run through MLX, so it has
   measured models and never measured a runtime.
   `~/projects/github/EnviousWispr` (Swift, 1767 commits, shipping on-device
   dictation, GPLv3) runs the same two model families on entirely different
   stacks:

   | family | EnviousWispr | localharness |
   |---|---|---|
   | Whisper | **WhisperKit** (CoreML) | mlx-whisper (MLX) |
   | Parakeet | **FluidAudio** (Swift) | mlx_audio (MLX) |

   Worth saying before the work rather than after: its `ASRManager.swift` sets
   `activeBackendType = .parakeet`, so it defaults to Parakeet with WhisperKit as
   the alternative -- independently the same conclusion this repo reached from 40
   clips. That is real corroboration, and it also lowers the expected value of
   re-running the comparison.

   It also ships `LanguageDetector.swift` / `LIDObservation.swift`: automatic
   language ID. This repo treats pinning the language as a CONSTRAINT
   (`ear=whisper:fr`); they treat it as a solved problem. Closing that would
   remove the need for a caller to know what language it is about to hear.

   MEASURED, same 40 LibriSpeech clips:

   | candidate | runtime | corpus wer | median |
   |---|---|---|---|
   | parakeet-tdt-0.6b-v2 | MLX | **0.013** | **0.14s** |
   | large-v3 (WhisperKit) | **CoreML** | 0.017 | 4.60s |
   | whisper-large-v3-mlx | MLX | 0.022 | 1.14s |

   THE RUNTIME MATTERS, which is the thing the lane had never asked. Same model
   family, and CoreML reads it more accurately than MLX does: 0.017 against
   0.022. Parakeet still wins the lane outright on both axes and stays the
   default, so the ranking did not change -- but "whisper scores 0.022 here" was
   a statement about mlx-whisper, not about whisper, and nothing in the suite
   could have told the difference.

   WhisperKit arrives as `brew install whisperkit-cli`, no Swift build. Its
   4.60s is NOT comparable as latency: the CLI reloads the model on every clip
   where EnviousWispr holds it in process. That is exactly why `comparable()`
   excludes timing from the axes it compares.

   parakeet-tdt-0.6b-v3 was measured and is WORSE, at 0.026 against v2's 0.013,
   twice the error for the same speed. It is the multilingual release and that
   appears to be what English accuracy paid for. A newer version number is not
   a better model.

   NOT DONE: FluidAudio (Swift Parakeet) has no CLI and would need a Swift
   target written against it. Automatic language ID, which EnviousWispr ships
   and this repo makes the caller supply, is also still open.

6. **DONE 2026-09-07. Measure the aliases with zero runs.** A defined alias
   nobody has run is a claim nobody has checked. Measured across four lanes:

   | lane | local-small (Qwen2.5-0.5B) | q3-1.7b | q3-4b |
   |---|---|---|---|
   | extract | 4/10 @ 0.19s | **7/10 @ 0.39s** | 7/10 @ 0.65s |
   | code | 3/9 | 1/9 | **7/9** |
   | svg | 1/3 | 2/3 | 2/3 |
   | web | - | - | **5/5** |

   `q3-1.7b` matches `q3-4b` on extract at 0.39s against 0.65s, and nearly
   doubles `local-small`'s 4/10. That is the "small and fast, watches a log,
   answers one question" lane Eric asked for originally, and it had been sitting
   defined and unmeasured while a 0.5B answered four questions in ten.

   NOT a clean supersede, which is why it stays a table rather than a swap:
   `local-small` beats `q3-1.7b` on code, 3/9 to 1/9. A 1.7B that reads well
   does not necessarily write.

   `local-large` still holds extract at 9/10, so no default changed. The value
   here is knowing what the cheap end actually costs rather than assuming the
   smallest alias is the cheap one.

   `q3-coder` remains refused at 16 GB; see the table below.

7. **DONE 2026-09-07. `--json` on every verb.** The primary caller is an agent
   parsing stdout, not a person reading it.

       $ lh extract "how many tests passed?" -f run.log --json
       {"ok": true, "verb": "extract", "body": "671"}
       $ lh extract "how many?" -f missing.log --json          # exit 1
       {"ok": false, "verb": "extract", "error": "no such file: missing.log"}

   FAILURES ARE DATA TOO, on stdout, not a line on stderr. An agent that has to
   read stderr to discover something went wrong will not read stderr. The exit
   code is unchanged either way.

   Added centrally to every subparser rather than verb by verb: a flag only
   some verbs accept is worse than no flag, because the caller cannot rely on it
   without first knowing which.

8. **ATTEMPTED AND RETRACTED 2026-09-07. Verify a cloned voice resembles its
   reference.**
   WER measures intelligibility and says nothing about identity, so an
   intelligible clone in a completely different voice scores a perfect 0.000.
   Every French voice number in this repo was an intelligibility number.

   `harness/checks/similarity.py` embeds two clips with resemblyzer (~17MB,
   CPU, sub-second) and reports the cosine as `speaker_similarity`, wired into
   the tts lane whenever a candidate was given a `ref_audio`, with a declared
   metric direction.

   AND IT DOES NOT WORK ON THIS DATA. The control that should have been run
   first: **two genuinely different men, both real recordings, score 0.827
   against each other**, while a clone against its own reference scores
   0.855-0.904. A gap of ~0.05 between "same person" and "different person" is
   inside the spread of everything else -- one wrong-speaker pair scored 0.881,
   above a correct pair at 0.742.

   So the earlier reading of these numbers (clone-vs-own 0.878, clone-vs-other
   0.781, "separation but overlapping") is RETRACTED. Without knowing what two
   different people score, 0.878 said nothing at all.

   THE LESSON, and it cost a set of confident numbers: **a similarity metric is
   meaningless without a negative control.** Run the same-versus-different check
   BEFORE quoting any figure from it.

   The code stays because the QUESTION is right. A working answer needs a
   stronger embedding -- a WavLM x-vector or ECAPA-TDNN rather than
   resemblyzer's small old encoder -- validated against that control first. The
   failing control is pinned in the module docstring and in a test, so it cannot
   quietly stop being true.

   THE HUMAN EAR HAS NOW ANSWERED IT, and the answer is that the cloning works.
   Eric listened to three clones of three different men saying one English
   sentence: "they all sound different". So the reference clip DOES transfer
   speaker identity, `fr-male` and `fr-male-2` are a real distinction, and it
   was only the metric that failed. That is the finding; the numbers above are
   still withdrawn.

   UNTIL A BETTER METRIC EXISTS THE ONLY EVIDENCE IS A HUMAN EAR. `./scripts/audition.sh` plays
   reference and clone back to back. Note also that French clips are the wrong
   thing to audition for this project's actual use case, which is French-ACCENTED
   ENGLISH: `.logs/fr-accent/` holds three clones of three different men saying
   one English sentence, which is the comparison worth making.

9. **A defect sweep of the codebase**, standing rather than one-off.

   FIRST FINDING, Eric's: outputs had FOUR homes and one was relative --
   `out/` (relative to the caller's cwd), `~/localharness-out/` (MCP),
   `.logs/` (eval runs mixed with service logs) and `/tmp/` (whatever I was
   doing). `lh` installs onto PATH, so the relative one scattered artifacts
   into every directory anyone happened to be standing in.

   Fixed: `harness/paths.py` is the single authority, everything resolves
   through it, and a test fails if any module reintroduces a relative `out/`.
   Existing scattered outputs were migrated rather than abandoned.

       $LOCALHARNESS_HOME            default ~/localharness
         out/                        artifacts
         out/mcp/                    artifacts asked for over MCP
         runs/<stamp>-<modality>/    one eval run
         logs/                       service stdout and stderr

   `--out` is no longer required for an eval: every run names its own
   directory, which is why the old ones were all called things like
   `.logs/img`, `.logs/voices` and `.logs/ev-svg-fair`.

   Two more defects came out of the same pass: the MCP tests shared one
   module-global job queue with a single worker, so a test that passed alone
   failed in a full run with its job still queued; and the queue's `ahead`
   counted only QUEUED jobs, telling a caller waiting behind a 54-second image
   that nothing was ahead of it.

### Refused, with the number that refused it

Not deferred, not forgotten -- measured against this machine and found not worth
the hours. Revisit every one on the Studio.

| item | number |
|---|---|
| Qwen-Image-2512-4bit (image quality leader) | **24.1 GB** vs an 18 GB safe budget |
| Qwen3-30B-A3B, Qwen3-Coder-30B-A3B | **16 GB** each; attempting one crashed the machine |
| A second video engine | **40 min** per generation here; the lane costs hours per candidate |
| The other 49 Kokoro voices | Eric's call: the lane already has enough to rank |
| Canary-Qwen-2.5B | NeMo format, no MLX port |
| MCP from a real second machine | Needs the second machine |

## 6. Traps this repo exists to remember

- `mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages`
  there by default. The opt-out is in both `gateway/config.yaml` and
  `scripts/serve-gateway.sh`.
- LiteLLM's `/health/readiness` returns 200 **before** the proxy can serve.
  Never conclude anything from a request in that window. A whole diagnosis was
  once built on one.
- `mlx_lm.server` serializes through a single queue and swaps models per
  request, so any concurrent client inserts a full model load into your timings.
  `harness/jobs.py` now enforces one generation at a time; before it, that
  isolation was assumed.
- **An abandoned request stays queued.** A readiness loop that fires every
  second and gives up after N seconds does not retry, it enqueues. 120 of them
  once wedged all three services. Two of them were enough to make `lh svg` look
  like a broken model when the server was simply working through my own
  timeouts. After any tool timeout, check for survivors before concluding
  anything: `lsof -nP -iTCP:8081 | grep -c ESTABLISHED`.
- **A background process started in one shell is not `%1` in the next.** Twice
  I "restarted" a server, tested, and concluded a code change had not worked --
  while the original process still held the port. Kill by match or by port and
  verify the port is clear before rebinding.
- `uv tool install` picks an interpreter silently. mflux installed against
  Python 3.9 and every entry point died on `int | None`. Use `--python 3.12`;
  the tell was 2 executables instead of 37.
- Peak memory must come from `/usr/bin/time -l`'s "peak memory footprint".
  `ru_maxrss` for `RUSAGE_CHILDREN` is a monotone high-water mark and reads 0
  for every child after the largest.
- `env.sh` is **sourced**, so it runs under zsh. zsh does not word-split
  unquoted parameters, and a bare `local name` prints the parameter once it has
  a value. Test sourced scripts under bash, zsh **and** sh.
- A GUI-spawned wezterm hands children `PATH=/usr/bin:/bin:/usr/sbin:/sbin`.
  `rec`, `jq` and `wezterm` are all in `/opt/homebrew/bin`.
- Kokoro's own default voice, `af_heart`, is not in this machine's cache. An
  absent voice fails as a mid-stream close, which reads exactly like a server
  that is down.
- **32 GB is a hard ceiling and there is no warning before it.** An eval sweep
  reached a 16 GB model while a 7.8 GB one was still resident and a Docker VM
  held 7.2 GB. The machine stopped: no panic report, no jetsam entry, just a log
  ending mid-line. `harness/memory.py` budgets against the Metal working set
  (24 GB of 32, per `tools/h3probe`), assumes the hot-swapping server still
  holds the outgoing model, and reserves 6 GB for everything else. An unknown
  size WARNS rather than reading as zero, because zero means "fits easily".
  Sweeping several >12 GB models through one server is unsafe even with it.
- **macOS TCC denies launchd agents access to `/Volumes`.** The volume stats
  fine and appears in `/Volumes`; `mlx_lm` then hangs forever inside
  `os.listdir`, serving nothing and logging nothing at 0% CPU. Fixed by granting
  Full Disk Access to `/bin/bash`, which TCC propagates to children. Verify with
  `./scripts/launchd.sh probe`, which bootstraps a real throwaway agent --
  checking from your shell proves nothing, because the shell has consent the
  agent can never be prompted for.
- **`env.sh` used to relocate the cache silently.** A reboot came back without
  the weights drive, so it walked its candidate list, found the T7 with room,
  and started every service against an empty cache. They listened, served
  nothing, and said nothing. `HF_HUB_OFFLINE=1` is the only reason that was a
  confusing hour rather than a 93 GB re-download onto the wrong disk. It now
  remembers where it landed and refuses to move; when the old root is absent it
  says so and points at the cable, because an unplugged drive is a cable problem.
- **A detached drive once failed 600 tests that never touch a disk**, because a
  module about `code` cases loaded the whole case tree and the generated `stt`
  cases reference audio on the volume. Scope a test's data load to its own lane.
- **An eval median is a WARM number.** The suite sequences by candidate so the
  model load amortises across cases; a one-shot `lh` command pays a cold load
  every time the resident model differs. Do not quote one as CLI latency.
- **MCP SDK 2.x moved everything.** `FastMCP` is `MCPServer`, host and port are
  `run()` kwargs rather than settings, the client names went snake_case, and
  DNS-rebinding protection defaults ON with an EMPTY allowlist -- so binding
  `0.0.0.0` is not enough and `Host: styx.local` is refused before it reaches a
  tool. That guard stays on with an allowlist: "no LAN auth" is about who can
  reach the port, and rebinding only needs someone here to open a web page.

## 7. Studio prep

- `env.sh` is ready: the free-space rule replaced the categorical refusal of
  internal disks, and the candidate list now ends at `~/.cache/huggingface`.
- Topology is settled in shape and parameterized in code. Every service binds
  `${*_HOST:-0.0.0.0}`, so Studio-serves/mini-is-a-client is a config change
  rather than a rewrite. There is deliberately NO LAN auth: this is a house
  network, the models are local, and the point of the machine is that other
  machines on it can use the GPU. (DNS-rebinding protection on the MCP server is
  a different threat and stays on; see section 6.)
- The launchd units port as-is, but the Full Disk Access grant is per-machine
  and has to be redone. `./scripts/launchd.sh probe` before anything else.
- **The 30B models are the reason to want the Studio.** Qwen3-30B-A3B and
  Qwen3-Coder-30B-A3B are 16 GB each against a 24 GB Metal working set here, and
  attempting one is what took this mini down. On 96 GB they are comfortable, and
  they are the strongest local coding models available. Re-run the section 4
  comparison there before assuming they win.
- Re-run every lane. The whole point of the eval suite is that a hardware change
  invalidates a ranking, and several current answers turn on speed and memory
  rather than quality -- exactly the axes that move.
- h3's int8 path switches on itself. Expect roughly 36.30s → 25.80s on the
  operations where it applies, and `--ssd-streaming` becomes optional rather
  than mandatory.
- Video is gated on time, not memory: about 40 minutes per second of output
  here. That is the number the Studio has to move.
