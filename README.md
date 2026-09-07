# localharness

Local media generation and voice on Apple Silicon, in a command line.

## Where things land

**One root**, `~/localharness`, overridable with `$LOCALHARNESS_HOME`:

    out/                        artifacts: images, audio, svg, pages
    out/mcp/                    artifacts asked for over MCP
    runs/<stamp>-<modality>/    one eval run: artifacts and results.json
    logs/                       service stdout and stderr

There were four places before this, and one of them was relative. `lh`
installs onto PATH and runs from anywhere, so a relative `out/` scattered
artifacts into whatever directory the caller happened to be standing in, and a
generation you cannot find is a generation you did not make. `out` and `logs`
are kept apart because a generated artifact and a server's stderr are different
things and only one of them is worth keeping.

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

`lh discover` answers "what can this machine do that nobody has measured", by
reading the receipts every eval run already writes. Built as a command rather
than done by hand because the answer changes whenever anything is installed or
any eval is run:

    lh discover --gap            # only what has never been run
    lh discover --lane image     # one modality
    lh discover --json           # for an agent

### Workflows are candidates too

A candidate can be a **workflow** rather than a model. Two exist:

    --candidates trace:mflux:flux2-klein-4b   # draw a raster, then vectorize it
    --candidates repair:q3-4b                 # generate, check, repair

`repair` closes a loop that was always available and never used: every checker
here is an automated verifier — the eval executes generated code, rasterizes
SVG, renders HTML in a browser — and none of them was ever fed back into
generation. Same model, asked again with the checker's own complaint attached.
Measured at `--repeat 3`:

| lane | single-shot | repair | mean attempts | tokens |
|---|---|---|---|---|
| code | 20/27 (74%) | **24/27 (89%)** | 1.33 | 115 vs 107 |
| svg | 6/9 (67%) | **9/9 (100%)** | 1.44 | 269 vs 226 |

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

## Serve it to another machine

    ./scripts/serve-mcp.sh          # MCP on 0.0.0.0:8899, house LAN, no auth

Exposes `svg`, `web`, `code` and `image` over MCP, so another machine's agent
can use this one's GPU. On the client:

    claude mcp add --transport http localharness http://styx.local:8899/mcp

Every tool shells out to `lh`, so `lh` must be installed. That is the point:
the CLI, the eval suite and the MCP server all run identical commands, which is
how the thing being measured stays the thing that ships.

`image` is **queued**, because it holds 11.4 GiB and mlx_lm swaps models per
request through a single queue. It returns a job id immediately; `job_status`
carries the queue position, what the machine is currently busy with, and the
artifact path once it is done. Artifacts stay here, under `~/localharness-out`.

`video` and the speech verbs are deliberately not exposed. Video needs job
semantics past a queue, and speech over the LAN was ruled out; both stay
reachable locally through `lh`.

DNS-rebinding protection is left ON with an allowlist rather than disabled. "No
LAN auth" is a decision about who can reach the port; rebinding does not need
the port reachable from outside, only for someone here to open a web page. Add
hosts with `MCP_ALLOW`.

## Develop

    make check     # shellcheck + bash -n + unit tests, no services needed
    make test      # unit tests only
    make smoke     # end-to-end, REQUIRES the services running

Tests are red-proofed by mutation: every guard has been broken deliberately and
the corresponding test confirmed to fail. `docs/testing.md` records which
mutations were run and which two survived, and why.

## Run it

Installed as launchd agents, so the machine comes back up serving after a
reboot or a crash:

    ./scripts/launchd.sh probe       # can an agent read the weights volume?
    ./scripts/launchd.sh install     # write the units and load them
    ./scripts/launchd.sh status      # what launchd thinks is running
    ./scripts/launchd.sh uninstall

`probe` first, always. It runs a throwaway agent that tries to list the weights
volume and reports the verdict. **macOS TCC denies /Volumes to launchd jobs**,
and the failure is horrible if you meet it unprepared: the volume stats fine,
reports free space and appears in /Volumes, so nothing looks wrong until
mlx_lm hangs forever inside os.listdir, accepting connections and answering
none, logging nothing, at 0% CPU.

The fix is granting Full Disk Access to **/bin/bash** (System Settings >
Privacy & Security > Full Disk Access, then Cmd+Shift+G to reach `/bin`). TCC
propagates the grant to child processes, so bash covers everything the units
start. It is broad, and that is the trade: a dedicated binary would key its
grant to a code hash and need re-granting on every rebuild, and `uv` is a
symlink into `Cellar/uv/<version>/` so a grant to it dies on the next upgrade.
`/bin/bash` is SIP-protected and never moves.

To restart one after editing it:

    launchctl kickstart -k gui/$UID/com.unxmaal.localharness.mcp

Or start them by hand instead, from a terminal that already has the access:

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
