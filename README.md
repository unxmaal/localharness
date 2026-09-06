# localharness

Local media generation and voice on Apple Silicon, in a command line.

    lh image "a red fox in falling snow" --width 768
    lh video "a fox running" --seconds 2
    lh svg   "a settings gear icon"
    lh web   "a landing page for a coffee roaster"
    lh say   "the tests all passed"
    lh hear  --seconds 5

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
