# localharness

Local LLM serving and evaluation on Apple Silicon.

Read `PLAN.md` first. `docs/validation-log.md` holds the evidence behind every
claim in it, including one conclusion that was wrong and how it was caught.

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
    ./scripts/smoke.sh           # assert the seam still holds

`smoke.sh` is the regression guard. Run it after any `mlx-lm` or `litellm` upgrade.
It exits non-zero if the architecture's assumptions broke.

## Two traps this repo exists to remember

`mlx_lm.server` has no `/v1/responses`, and LiteLLM routes `/v1/messages` there by
default. The opt-out is in both `gateway/config.yaml` and `scripts/serve-gateway.sh`.

LiteLLM's `/health/readiness` returns 200 before the proxy can serve. Never conclude
anything from a request made in that window, and when restarting, wait for port 4000
to actually free before probing, or you will test the dying process.
