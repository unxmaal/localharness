# evals

Answers one question cheaply: given several candidates for a job, which should
this machine use? That is the antidote to re-architecting every time a better
method turns up on GitHub.

    make evals                                       # default candidates
    make evals MODALITY=svg CANDIDATES=local-mid,local-large
    uv run python -m evals.run --modality image --repeat 3 --out .logs/img \
      --candidates mflux:flux2-klein-4b,mflux:z-image-turbo

## What a candidate is

| form | example | runner |
|---|---|---|
| gateway alias | `local-large` | HTTP to the gateway |
| engine spec | `mflux:z-image-turbo,quantize=4` | subprocess, measured |
| speech | `tts:mlx-community/Kokoro-82M-bf16,voice=bm_george` | HTTP to mlx-audio |

A candidate is only handed cases of a modality it can run. Giving mflux an SVG
case produces a failure row that says nothing about mflux.

Quantization and voice are part of the candidate NAME. `z-image-turbo` at q4 and
at q8 have different speed and memory, and sharing a row makes the comparison
meaningless.

## Two different measurements, kept apart

**The competence gate** separates working from broken, and it is objective. No
human squints at anything.

| modality | what it checks |
|---|---|
| svg | parses, has an xmlns, draws N shapes, **and rasterizes to something visible** |
| web | parses, has a body, is self-contained, no external URLs |
| image | decodes, is the size requested, is more than one colour, **OCR of rendered text** |
| video | demuxes, right size and length, frames are not uniform, **something moves** |
| tts | the audio is real, and transcribes back to what was asked for |
| stt | *not yet: ranking it needs reference audio with a human transcript* |

The rasterization and motion checks exist because the structural ones are
blind to a whole class of failure. Well-formed SVG can draw nothing — white on
white, a shape outside the viewBox — and it passed every check here until
rsvg-convert was added. A valid MP4 of the right length in which nothing moves
plays perfectly and is not a video.

**The quality axis** is what orders two candidates that both pass. Checkers
publish numbers alongside the verdict:

| metric | modality | direction | meaning |
|---|---|---|---|
| `wer` | tts | lower | word error rate, speaking then transcribing |
| `cer` | image | lower | character error rate of text OCR'd out of the picture |
| `ink` | svg | higher | fraction of the canvas actually marked |
| `motion` | video | higher | mean change between consecutive frames |
| `adherence` | image | higher | how well the picture matches the prompt |

**Direction is declared, not assumed.** Every metric started out as an error
rate, so "lower is better" got baked into both the ranking and the worst-case
column — and `ink` and `motion` silently inverted both the moment they arrived.
`METRIC_DIRECTION` in `evals/core.py` is the registry; a metric that does not
appear there warns rather than guessing quietly.

### Prompt adherence

    uv sync --group metrics
    uv run --group metrics python -m evals.run --modality image \
      --adherence pickscore --repeat 4 --candidates a,b

Opt-in, because it loads a multi-GB preference model. Two backends, since the
literature disagrees about which is better:

- `pickscore` — `yuvalkirstain/PickScore_v1`, a CLIP-H fine-tune on 500k human
  preference pairs from Pick-a-Pic.
- `hpsv2` — `xswu/HPSv2`, a CLIP-H fine-tune on HPD v2.

**Scores are comparable between candidates on one backend, never between
backends**: different heads, different scales.

Treat the number as evidence, not as a verdict. These are models of *aggregate*
human preference, with known biases — toward saturation and contrast among
others — and they can disagree with the person whose images they are.

## How the numbers are aggregated

**Latency takes a median**, so one cold model load cannot decide which candidate
looks fastest. Cold starts here are not subtle: z-image-turbo's first case ran
239s against a 43s warm steady state.

**A quality metric takes a mean**, and the difference is not cosmetic. Most
cases score a clean 0.0 and the entire signal is in the few that do not. The
first voice comparison medianed 0.000 for two voices that were not equal — one
had four cases at zero and one at 0.154. `metrics_worst` keeps that outlier
visible.

Candidates are ranked on pass rate, then metrics, then latency. Sorting on
latency first is how a faster-but-worse candidate reaches the top line — and
that is a live risk here, not a theoretical one: a 0.5B model that never closes
its tags runs to the token limit every time, so it is both the worst and,
without a pass rate beside it, the slowest-looking.

## --repeat

One sample per prompt ranks noise. Diffusion varies enormously with the seed and
a language model at temperature 0.2 is not deterministic either. `--repeat 3`
runs each stochastic case with three seeds and reports them as separate rows;
tts is skipped, since Kokoro at a fixed voice and speed is deterministic.

It earned itself on its first run: a 1.5B passed `icon-gear` on one seed and
failed it on two others.

## Adding a case

Drop a YAML file under `cases/<modality>/`. `params:` are generation knobs handed
to the engine; `assert:` are checks. Unknown keys in either block are rejected at
load, so `widht: 512` costs nothing rather than generating at the default size
and passing.

    id: text-render
    modality: image
    prompt: A wooden shop sign with the word "OPEN" carved into it.
    params:
      width: 512
      height: 512
      seed: 42
    assert:
      text: OPEN

Assertions are deliberately blunt. Anything subtler is a judgement call, and
judgement calls belong to you rather than to a regex.

## Sequencing

Runs are grouped by candidate, never interleaved. `mlx_lm.server` hot-swaps
models per request, so alternating aliases pays a model load on every case and
measures disk throughput instead of the model. It also serializes through one
queue, so any concurrent client — a stray curl, a voice request — inserts a full
model load into your timings. Measurement isolation is assumed, not enforced.

## Fenced output is recovered before judging

Punishing a model for wrapping its SVG in a code fence measures prompt
compliance, not SVG ability.
