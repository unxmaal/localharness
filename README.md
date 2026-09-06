# localharness

Local media generation and voice on Apple Silicon, in a command line.

## Install

    uv tool install --python 3.12 --editable .

That puts `lh` on PATH via `~/.local/bin`, in its own venv under
`~/.local/share/uv/tools/localharness`. It never touches the system Python:
this machine's `python3` is pyenv 3.9.4 and cannot import anything installed
here. `--python 3.12` is not optional -- `uv tool install` picks an interpreter
silently, and mflux once installed against 3.9 where every entry point died on
`int | None`. `--editable` so the checkout stays the source of truth.

`lh` finds the weights on its own (`harness/env.py`), by the same free-space
rule `scripts/env.sh` uses for the services. It has to: installed on PATH it
runs with nothing sourced, and an unset `HF_HOME` sends huggingface_hub to
`~/.cache/huggingface` to re-download what is already on the volume.

The install carries no torch. `mlx-whisper` requires it unconditionally, so the
multilingual ear lives in the `whisper` dependency group and only the eval suite
pulls it: 370MB installed rather than 1.1GB.

    lh image "a red fox in falling snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh code  "a python function that parses an ISO timestamp"
    lh extract --file build.log "how many tests failed?"
    lh say   "the tests all passed"            # cloned, French accent
    lh say   "the tests all passed" --voice bm_george   # kokoro, sub-second
    lh voices
    lh hear  --seconds 5

`lh voices` lists what can be spoken. **The default is `fr-male`**, which is
cloned rather than picked from a table: Kokoro has five fixed voices and no accented English
among them, while Chatterbox clones from a reference clip and clones ACROSS
LANGUAGES -- the clip speaks French, the output speaks English, and the accent
comes with the voice. That is why no accented-English corpus was needed, and
why a cloned voice is one name standing for three coupled settings (model,
clip, language code) that fail by naming each other when set separately.

The default costs about 2s a line against Kokoro's 0.3s, which is the price of
it being the voice that was wanted. `--voice bm_george` is there when a line
needs to come back immediately.

`code` and `extract` print to stdout, because both produce something you pipe
or read rather than an artifact you open in a viewer. `extract` reads stdin
when given no `--file`, so `make test 2>&1 | lh extract "which test failed?"`
works. It is the lane for handing a cheap question to a small model instead of
spending a large one's context on a log — though the eval had something to say
about which model is actually cheap: see `evals/README.md`.

Read `PLAN.md` for why it is shaped this way. `docs/validation-log.md` holds the
evidence behind every claim in it, including conclusions that were wrong and how
they were caught.

## Develop

    make check     # shellcheck + bash -n + unit tests, no services needed
    make test      # unit tests only
    make smoke     # end-to-end, REQUIRES the services running

Tests are red-proofed by mutation: every guard has been broken deliberately and
the corresponding test confirmed to fail. `docs/testing.md` records which
mutations were run and which two survived, and why.

## Run it

    ./scripts/serve-mlx.sh       # inference engine on :8081
    ./scripts/serve-gateway.sh   # gateway on :4000, the only address clients use
    ./scripts/serve-tts.sh       # Kokoro TTS + Parakeet STT on :8890
    ./scripts/smoke.sh           # assert the seam still holds

## Compare candidates

    uv run python -m evals.run --modality image --out .logs/img \
      --candidates mflux:flux2-klein-4b,mflux:z-image-turbo

A candidate is a gateway alias for text, or an engine spec for anything that
runs as a process, or `tts:<model>,voice=<name>` for speech. The suite reports a
pass rate, which separates working from broken, and quality metrics, which are
what actually rank two candidates that both work.

`smoke.sh` is the regression guard. Run it after any `mlx-lm` or `litellm` upgrade.
It exits non-zero if the architecture's assumptions broke.

## Two traps this repo exists to remember

`mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages` there by
default. The opt-out is in both `gateway/config.yaml` and `scripts/serve-gateway.sh`.

LiteLLM's `/health/readiness` returns 200 before the proxy can serve. Never conclude
anything from a request made in that window, and when restarting, wait for port 4000
to actually free before probing, or you will test the dying process.
