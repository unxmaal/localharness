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
| tts | `tts:mlx-community/Kokoro-82M-bf16,voice=bm_george` | HTTP to mlx-audio |
| tts, cloned | `tts:litmudoc/Chatterbox-Multilingual-MLX-v2-Q8,lang_code=fr,ref_audio=<clip>,ear=whisper:fr` | HTTP to mlx-audio |
| stt | `stt:mlx-community/whisper-large-v3-mlx,backend=whisper,language=fr` | in-process mlx-whisper |

A candidate is only handed cases of a modality it can run. Giving mflux an SVG
case produces a failure row that says nothing about mflux.

Quantization, voice and reference clip are part of the candidate NAME.
`z-image-turbo` at q4 and at q8 have different speed and memory, and sharing a
row makes the comparison meaningless. The reference clip turned out to matter
just as much: three clips from the same corpus, same language, same gender,
gave 0.122, 0.144 and 0.578 corpus word error rate.

### Language is part of selection

A case declares its `language:` (default `en`), and an English-only candidate is
not handed French cases — the same rule as not handing mflux an SVG case, for
the same reason. Parakeet does not speak French, so scoring French through it
returns a word error rate near 1.0 for every candidate and ranks them all as
equally broken.

For a **tts** candidate the language comes from the EAR, not from the model:
`ear=whisper:fr` says both which transcriber reads the audio back and what
language it is scored in. Those are one constraint rather than two settings
that happen to agree — a French sentence can only be scored by a transcriber
that speaks French.

For an **stt** candidate it is the candidate's own `language=` option.

### Two ears

| backend | model | speaks | median |
|---|---|---|---|
| `server` (default) | Parakeet, on mlx-audio :8890 | English only | 0.18s |
| `whisper` | mlx-whisper, in this process | multilingual | 1.14s |

Whisper cannot run on the audio server at all: mlx_audio demands a HuggingFace
processor and the mlx whisper repos ship `weights.npz` and `config.json`
without one. Calling `mlx_whisper` directly wants exactly what those repos
hold. It costs a model load per process rather than per request, which is the
right trade for an eval and the wrong one for a chat loop.

## Two different measurements, kept apart

**The competence gate** separates working from broken, and it is objective. No
human squints at anything.

| modality | what it checks |
|---|---|
| svg | parses, has an xmlns, draws N shapes, **and rasterizes to something visible** |
| web | parses, has a body, is self-contained, no external URLs, **and renders something** |
| image | decodes, is the size requested, is more than one colour, **OCR of rendered text** |
| video | demuxes, right size and length, frames are not uniform, **something moves** |
| code | **the code is executed** and every assertion in the case must pass |
| extract | the answer contains, excludes, or exactly equals what it should |
| tts | the audio is real, and transcribes back to what was asked for |
| stt | *not yet: ranking it needs reference audio with a human transcript* |

The rasterization and motion checks exist because the structural ones are
blind to a whole class of failure. Well-formed SVG can draw nothing — white on
white, a shape outside the viewBox — and it passed every check here until
rsvg-convert was added. A page can parse, have a body, be perfectly
self-contained and paint a white rectangle. A valid MP4 of the right length in
which nothing moves plays perfectly and is not a video.

### code: this lane executes what the model wrote

A syntax check is nearly worthless for code — an off-by-one, a reversed
comparison and a forgotten edge case are all valid Python — so the generated
code is **run**, in a subprocess, with a timeout, in a scratch working
directory. That is the whole of the sandbox: no seccomp, no container, no
filesystem restriction beyond the cwd. Fine for your own local models against
your own cases; not fine for running an eval suite someone else wrote. Read
cases before you run them.

    id: slugify
    modality: code
    prompt: Write a Python function `slugify(text)` that ...
    assert:
      checks:
        - "slugify('Hello, World!') == 'hello-world'"
        - "raises(slugify, None, exc=TypeError)"

Each check is one expression evaluated against the generated module. `raises()`
is provided because "it raises ValueError on bad input" has no readable form as
a bare expression; it re-raises anything that is *not* the expected exception,
so a NameError in the generated code cannot pass as correct error handling.

### extract: the small, fast lane

Hand a log, a diff or grep output to a 0.5B instead of spending a large model's
context on it. The material lives in `context:` (or `context_file:`, resolved
beside the case) separately from the instruction, and it is sent **first**: a
small model that reads forty lines and then the question does better than one
that reads the question, works through the material, and has to remember what
it was looking for.

    id: exit-code
    modality: extract
    prompt: What exit status did the process end with? Reply with the number only.
    context_file: build-failure.log
    assert:
      equals: "137"

`equals` is the narrowest and most useful shape: an answer the caller can use
without parsing. A model that replies "The exit status was 137." has failed at
the thing this lane exists for.

**The quality axis** is what orders two candidates that both pass. Checkers
publish numbers alongside the verdict:

| metric | modality | direction | meaning |
|---|---|---|---|
| `wer` | tts | lower | word error rate, speaking then transcribing |
| `cer` | image | lower | character error rate of text OCR'd out of the picture |
| `ink` | svg | higher | fraction of the canvas actually marked |
| `motion` | video | higher | mean change between consecutive frames |
| `adherence` | image | higher | how well the picture matches the prompt |
| `code_pass` | code | higher | fraction of a case's assertions that ran green |

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

**When a graded metric and the pass rate disagree, the report says so.** On the
code lane a 7B wrote 75.6% correct code against a 1.5B's 52.8% while failing
*more* cases outright, because a case with six strict checks dies on a single
miss. Both numbers are true and neither is the answer alone.

## --repeat

One sample per prompt ranks noise. Diffusion varies enormously with the seed and
a language model at temperature 0.2 is not deterministic either. `--repeat 3`
runs each stochastic case with three seeds and reports them as separate rows.
`tts` is skipped because Kokoro at a fixed voice and speed is deterministic,
and `extract` because the answer is one token and the whole point of the lane
is that it is cheap — three samples of "137" buys nothing.

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
