# localharness

Local media generation and voice, on Apple Silicon, in a command line.

Last rewritten 2026-09-05, against a working tree. Numbers here were measured
on this machine; `docs/validation-log.md` holds the evidence, including
conclusions that were wrong and how they were caught.

## 1. What this is for

Generate images, video, SVG and web pages, and talk to the machine, entirely
locally. That is the whole goal, and `lh` is the product:

    lh image "a red fox in falling snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh say   "the tests all passed"
    lh hear  --seconds 5

Two lanes exist in the eval suite but not yet as CLI verbs: `code`, which runs
the generated code against assertions, and `extract`, which hands a log or grep
output to a small fast model instead of spending a large one's context on it.

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
        lh (harness/cli.py)                 evals/run.py
              |                                   |
              +---------------+-------------------+
                              |
                        harness/
     engines.py    proc.py    completion.py    audio.py    checks/
          |           |             |             |
      mflux, h3   subprocess    gateway :4000   mlx_audio :8890
                  + peak mem         |
                                 mlx_lm :8081
```

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

All measured 2026-09-05 unless noted. Versions: uv 0.12.5, mlx 0.32.2,
mlx-lm 0.31.3, mflux 0.19.1.

| Lane | State |
|---|---|
| Text (svg, web) | mlx_lm behind the gateway. Working; needs 7B+ to be much good. |
| Code | mlx_lm behind the gateway. The eval RUNS the generated code. |
| Extract | The delegate-to-a-small-model lane: a log in, one fact out. |
| Image | mflux. `flux2-klein-4b` at ~19s per 512² warm. Working. |
| Video | antirez/h3.c. 512×512×22 frames in 40.5 min, 9.48 GiB peak, zero swap. Working, and Studio-gated by time not memory. |
| Speech out | Kokoro-82M via mlx-audio on :8890. 0.4s generation, 12-13× realtime. |
| Speech in | Parakeet TDT 0.6b v2, same server. 0.1s. English only. |
| Voice UI | `voicemode` (MCP, 1349★) is the front end. Not ours, deliberately. |

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

1. **Prompt adherence, and it needs a decision.** This is the one axis that
   would let the suite rank two image models. Neither `ink` nor OCR separated
   FLUX.2 klein from Z-Image Turbo — both scored a clean 0.000 — so "does the
   picture match the words" is what is missing. PickScore and HPSv2 both do it
   well, and both are torch models with a ~4 GB checkpoint, in a repo that is
   otherwise MLX-only and on a machine with 32 GB. **Not taken unilaterally:**
   it roughly doubles the dependency footprint to add one metric. The
   alternatives are an MLX CLIP port (Apple's `mlx-examples/clip` is a
   reference implementation, not a package, so this means carrying ~400 lines
   of someone else's model code) or a blind contact sheet, which is a person
   rather than a metric and is genuinely fine for a two-way choice.
2. **Rank STT on its own.** The speech metric is joint: it scores a TTS model
   and the STT model reading it together and cannot separate them. Holding one
   side fixed still orders the other, which is what the tts eval does, but
   ranking STT itself needs reference audio with a human transcript — ~200
   utterances of LibriSpeech test-clean. Candidates, all served by mlx_audio:
   Canary-Qwen 2.5B leads Open ASR on accuracy, Parakeet on speed at ~30×.
3. **More voices than the five cached.** Chatterbox and Qwen3-TTS against
   Kokoro. Deferred and not to be restarted unprompted: a male French-accented
   voice, which needs a Chatterbox clone and a reference clip.
4. **Rasterize HTML.** SVG is done with rsvg-convert, which was already
   installed. HTML needs a browser engine — Playwright is the obvious one and a
   much larger dependency — and the checks that matter most for a page
   (self-containment, no external URLs) are structural anyway. Low priority.
5. **launchd units.** The services are pinned now (`scripts/versions.sh`) but
   still started by hand.

## 6. Traps this repo exists to remember

- `mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages`
  there by default. The opt-out is in both `gateway/config.yaml` and
  `scripts/serve-gateway.sh`.
- LiteLLM's `/health/readiness` returns 200 **before** the proxy can serve.
  Never conclude anything from a request in that window. A whole diagnosis was
  once built on one.
- `mlx_lm.server` serializes through a single queue and swaps models per
  request, so any concurrent client inserts a full model load into your timings.
  Measurement isolation is assumed, not enforced.
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

## 7. Studio prep

- `env.sh` is ready: the free-space rule replaced the categorical refusal of
  internal disks, and the candidate list now ends at `~/.cache/huggingface`.
- Topology is undecided. `127.0.0.1` and fixed ports are hardcoded in
  `gateway/config.yaml`, `voicemode.env` and every script, all assuming one host
  forever. The likely shape is Studio serves, mini is a client, which needs LAN
  auth on the gateway that does not exist.
- h3's int8 path switches on itself. Expect roughly 36.30s → 25.80s on the
  operations where it applies, and `--ssd-streaming` becomes optional rather
  than mandatory.
- Video is gated on time, not memory: about 40 minutes per second of output
  here. That is the number the Studio has to move.
