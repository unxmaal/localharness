# evals

Answers one question cheaply: given several candidates for a job, which should
this machine use? That is the antidote to re-architecting every time a better
method turns up on GitHub.

    make evals                                    # default candidates
    make evals CANDIDATES=local-mid,local-large
    uv run python -m evals.run --modality svg --candidates a,b --out /tmp/x

## Why these metrics

Pass rate comes first, and it is **objective**: does the SVG parse, does it draw
anything, does the HTML render a body, is the page self-contained. A model that
emits prose instead of markup has failed regardless of taste, and no human has
to squint at anything to know it.

Latency is reported as a **median**, because one cold model load should not
decide which candidate looks fastest. Total time is reported alongside it, and
the gap between them is informative: a first run had a 1.5B beating a 7B on
median while losing badly on total, because its failures rambled to the token
limit.

Fenced and chatty output is **recovered before judging**. Punishing a model for
wrapping its SVG in ```svg measures prompt compliance, not SVG ability.

## Modality coverage

| modality | runner | candidates |
|---|---|---|
| svg | text, via the gateway | any gateway alias |
| web | text, via the gateway | any gateway alias |
| image | not built | mflux (2310 stars, MLX-native, FLUX.2 klein 4B) |
| video | not built | h3.c (already measured by hand: 3 clips, 9.5-11.6 GiB peak) |
| tts | not built | Kokoro on mlx-audio (measured: 12-13x realtime) |
| stt | not built | Parakeet on mlx-audio (measured: 0.1s), whisper.cpp |

The text runner exists because SVG and web generation are language-model jobs,
so the gateway alias IS the candidate and swapping it costs a string. The other
four need runners that measure a subprocess: wall time, peak RSS, and whether
the output file is a valid artifact.

## Adding a case

Drop a YAML file under `cases/<modality>/`:

    id: icon-gear
    modality: svg
    prompt: |
      Draw a settings gear icon...
    assert:
      min_shapes: 2
      must_not_contain: ["data:image"]

Assertions are deliberately blunt: `min_shapes`, `must_contain`,
`must_not_contain`. Anything subtler is a judgement call, and judgement calls
belong to you rather than to a regex.

## Sequencing

Runs are grouped by candidate, never interleaved. `mlx_lm.server` hot-swaps
models per request, so alternating aliases pays a model load on every case and
measures disk throughput instead of the model.
