# localharness: local LLM serving and evaluation on Apple Silicon

Status: plan. Written 2026-09-05. All claims marked VALIDATED were executed on this
machine on that date; the transcript of each check is summarized in
`docs/validation-log.md`.

## 1. Problem statement

The recurring failure is not a bad architecture. It is that the architecture has no
seam, so every new Apple Silicon inference method found on GitHub reads as a reason to
rebuild rather than a candidate to measure. Two things fix that permanently:

1. A stable interface boundary, so swapping the inference engine costs a config line.
2. A benchmark on real work, so a new method gets measured in twenty minutes instead
   of adopted on vibes.

Everything below serves those two goals. Model choice is deliberately deferred.

## 2. Hardware and constraints

### Current (VALIDATED 2026-09-05)

| Fact | Value | How checked |
|---|---|---|
| Model | Mac14,12 (M2 Pro Mac mini) | `sysctl hw.model` |
| Unified memory | 32 GiB (34359738368 bytes) | `sysctl hw.memsize` |
| macOS | 26.5.1, build 25F80 | `sw_vers` |
| GPU wired limit | `iogpu.wired_limit_mb: 0` (system default) | `sysctl iogpu.wired_limit_mb` |
| Free disk | 77 GiB, volume 92% full | `df -h /System/Volumes/Data` |
| MLX Metal backend | available, `Device(gpu, 0)` | `mx.metal.is_available()` |
| uv | 0.12.5 | `uv --version` |

### Disk is the binding constraint, not memory

77 GiB free. A single 70B at 4-bit is roughly 40 GiB. There is room for the tater
phase and not much else.

`~/.ollama` holds **44 GiB** of models, every one of them 9 to 20 months old
(`llava:7b`, `mistral-small3.1`, `llama3.2-vision`, `mistral`, `gemma3:12b`,
`llama3.1`, `nomic-embed-text`, `llama3.2`). Reclaiming that roughly doubles working
space. Decision deferred to the user; see Phase 0.

### There is no external storage attached

`/Volumes/Raid` and `/Volumes/External1` both report `/dev/disk3s5`, which is the
internal Data volume. They are stale mountpoints left behind by unmounted drives. The
`albums -> /Volumes/T7/Music/albums` symlink inside Raid points at a T7 that is also
absent. Any plan that assumed external model storage is wrong.

### Target (per user, arriving ~November 2026)

M5 Ultra Mac Studio, 96 GB unified memory. Planning math:

- Weight bytes is approximately `params x bits / 8`. A 70B at 4-bit is about 40 GB
  loaded, before KV cache.
- The macOS default GPU wired limit is approximately 70 to 75% of unified memory.
  Apple does not document the formula and it has changed across releases, so treat
  it as approximate and measure rather than compute.
- On 96 GB that default is roughly 67 GB, raisable with
  `sudo sysctl -w iogpu.wired_limit_mb=<mb>`.

Two cautions carried from experience on the current machine: the setting does not
survive a reboot, and setting it too aggressively will wedge the machine, because the
memory is taken from everything that is not the GPU. Leave the OS 8 GB minimum.

## 3. Architecture

```
clients        Claude Code        opencode        OpenExecutive        scripts
               (Anthropic shape)  (OpenAI shape)  (OpenAICompatible)   (either)
                      |               |                 |                |
                      +---------------+--------+--------+----------------+
                                               |
gateway                          LiteLLM proxy, 127.0.0.1:4000
                                 - the ONLY address any client knows
                                 - serves BOTH /v1/chat/completions and /v1/messages
                                 - model aliases: coder, chat, fast, embed
                                 - request logging, latency and token capture
                                               |
                      +------------------------+------------------------+
                      |                        |                        |
engines        mlx_lm.server            llama-server              next month's
               :8081                    :8082 (GGUF only)         discovery :808N
               PRIMARY                  FALLBACK                  CANDIDATE
                      |                        |                        |
weights                        $HF_HOME  (one directory, portable)
```

The rule that makes this work: **no client ever learns an engine name or port.**
Clients address aliases on :4000. Adopting a new engine means starting it on a free
port, pointing an alias at it, running the eval suite, and keeping it or killing it.

### Why LiteLLM at the gateway

It is the only piece that serves the Anthropic Messages shape and the OpenAI shape
from one process over the same backend. That matters specifically because Claude Code
speaks Anthropic and everything else speaks OpenAI, and without it the two would need
separate adapters.

### Why MLX as the primary engine

Apple's own framework, best memory efficiency and prompt-processing throughput on
M-series for models it supports, and the toolchain is already proven in this
household (Chatterbox MLX in the tsatd_video work, KNOWLEDGE #68).

### Why llama.cpp stays in the diagram

Broader model and quantization coverage. When a model exists only as GGUF, or MLX has
not ported an architecture, this lane carries it. It is not installed yet.

### Why not Ollama as the foundation

It wraps llama.cpp, lags upstream, and hides the flags that tuning requires. It is
installed and can stay for casual use. It is not the serving path.

## 4. Validated behaviour of the stack

All executed 2026-09-05. These are the facts the design leans on.

**MLX generation works.** `mlx_lm.generate` against
`mlx-community/Qwen2.5-0.5B-Instruct-4bit`: 25.1 tok/s prompt, 784 tok/s generation,
0.336 GB peak. Correct output.

**`mlx_lm.server` is OpenAI-compatible.** `GET /v1/models` and
`POST /v1/chat/completions` both return correctly shaped bodies. SSE streaming emits
proper `chat.completion.chunk` deltas terminated by `data: [DONE]` (11 data chunks
measured on a counting prompt).

**`mlx_lm.server` hot-swaps models per request.** This is significant and mostly
undocumented. The `model` field in the request body is treated as a live Hugging Face
`repo_id`. Serving with `--model A` and then requesting model B causes the server to
load B, and `GET /v1/models` afterwards lists both. Consequences:

- One server process can serve the whole local model library. Multiple ports per
  model are unnecessary.
- A typo in the model field triggers a **network fetch**, not a clean error. Sending
  `"model":"x"` produced an HTTP 404 from `huggingface.co/api/models/x`. The gateway
  must therefore be the only thing that ever supplies a model id.
- Model switching costs a load. Sequencing evals by model, not interleaved, matters.

**The gateway serves both API shapes over MLX.** Aliases resolve
(`local-small` and `local-mid` route to different upstream repo ids and return the
correct one in the response `model` field). `/v1/messages` returns a proper Anthropic
body.

**Tool calling round-trips on both shapes**, verified against a 1.5B model:
OpenAI returns `tool_calls[{function:{name:"get_weather",arguments:"{\"city\": \"Paris\"}"}}]`;
Anthropic returns `{"type":"tool_use","name":"get_weather","input":{"city":"Paris"}}`
with `stop_reason: "tool_use"`.

**One trap, now a KB rule.** LiteLLM routes `/v1/messages` for any
`custom_llm_provider: openai` deployment through its Responses API adapter, calling
`POST /v1/responses` upstream, which `mlx_lm.server` does not implement. Every
Anthropic-shaped request 404s. The opt-out is
`use_chat_completions_url_for_anthropic_messages`, available **both** as a
`litellm_settings:` YAML key and as the environment variable
`LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1`. Controlled A/B on
2026-09-05 confirmed each works on its own. `scripts/serve-gateway.sh` sets both,
which is harmless.

**A second trap that nearly poisoned this document.** LiteLLM's
`GET /health/readiness` returns `{"status":"healthy"}` and the log prints
`Application startup complete.` while the proxy still cannot serve a request. During
that window requests fail with empty bodies, `HTTP=000`, or misleading upstream
errors. This first-draft plan asserted that the YAML key was ignored and only the env
var worked, complete with a plausible source-level mechanism. That was wrong: it was
the readiness race, and the red-proof caught it. KB rule "LiteLLM /health/readiness
lies: it returns 200 before the proxy can serve requests" carries the detail. The
operational consequence is in `scripts/smoke.sh`, which polls a real completion on
the route under test before asserting anything.

### Not yet validated

- Claude Code driven end to end against the gateway via `ANTHROPIC_BASE_URL`. The
  endpoint answers correctly in isolation; the full agent loop has not been run.
- Any llama.cpp lane. Nothing installed.
- Throughput on any model above 1.5B on this hardware.
- Behaviour of the gateway under concurrent requests against a hot-swapping backend.
  Suspected to be bad: two clients asking for different models will thrash. Needs a
  test before any multi-client use.
- Prompt caching behaviour across the gateway. `cached_tokens` appeared in responses
  but was not exercised deliberately.

## 5. Phases

### Phase 0: reclaim disk (needs a user decision)

77 GiB free is not enough to work comfortably. `ollama rm` on the eight stale models
returns roughly 44 GiB. Nothing else in the plan proceeds comfortably without it.
Ask before deleting.

### Phase 1: pin the seam (largely done during validation)

- `gateway/config.yaml` with aliases, committed.
- Launch scripts for the MLX engine and the gateway, with the env var baked in so the
  Anthropic trap cannot recur.
- A `scripts/smoke.sh` that asserts all six validated behaviours above and exits
  non-zero on regression. This is the guard that catches a LiteLLM or mlx-lm upgrade
  breaking the seam.

### Phase 2: the eval harness (the actual deliverable)

This is the part that ends the churn. Without it, Phase 1 is just plumbing.

- `evals/cases/*.yaml`: 20 to 50 prompts drawn from real work already in these repos.
  Concrete sources available today: Swift diff review from xdripswift, Python from
  mcm-engine, the rules-adjudication classification job described in KNOWLEDGE #10
  (which is explicitly designed for a cheap local model and is the single best
  first customer for this harness), and shell/config generation.
- `evals/run.py`: runs every case against one or more gateway aliases, recording
  tokens/sec, time to first token, peak resident memory, wall time, and pass or fail
  against per-case assertions.
- Results land in the mcm KB via `add_knowledge`, so model comparisons persist across
  sessions rather than living in scrollback.
- Sequence by model, never interleaved, because of the hot-swap load cost.

Acceptance: running `evals/run.py --alias coder` on a newly discovered engine or
model produces a comparable score sheet in under twenty minutes with no code changes.

### Phase 3: wire the clients

- Claude Code: `ANTHROPIC_BASE_URL=http://127.0.0.1:4000`. Validate the full agent
  loop, not just a single completion. Expect friction; the agent loop exercises far
  more of the API surface than a curl does.
- opencode: OpenAI-compatible provider pointed at :4000.
- OpenExecutive: it already has `OpenAICompatibleProvider` at the
  `providers/provider.py` seam (KNOWLEDGE #79). Point it at :4000. Note the caveat
  from that entry: prompt caching is load-bearing there and local models will not
  reproduce it.

### Phase 4: model selection

Deliberately last. With Phases 1 and 2 built, this becomes an experiment rather than
a commitment. On the current 32 GB machine, work at 7B to 14B 4-bit. Do not tune
`iogpu.wired_limit_mb` here; the default already yields roughly 22 to 24 GB, and a
model needing more than that belongs to the Studio phase.

### Phase 5: migration to the Mac Studio

Should be close to a no-op if the earlier phases held.

- Set `HF_HOME` to a deliberate path **now**, before accumulating weights, so the
  cache is one movable directory. It is currently unset, and the default
  `~/.cache/huggingface` already holds 9.3 GiB of unrelated image models (FLUX,
  CLIP, Chatterbox).
- Re-run the eval suite on the Studio. The score sheet is the migration test.
- Only then revisit `iogpu.wired_limit_mb`, and via a LaunchDaemon if it needs to
  persist. `/etc/sysctl.conf` is not reliably honored on modern macOS.

## 6. Open questions for the user

1. **Phase 0**: delete the 44 GiB of stale Ollama models?
2. **Primary use case** drives model choice in Phase 4 and is not yet stated. Coding
   agent backend, general chat, batch document processing, and the creative pipeline
   have materially different context-length and latency requirements.
3. **Privacy posture**: is the point of local inference cost, latency, offline
   capability, or keeping data off third-party servers? If it is the last one, the
   gateway should have no cloud fallback configured at all, which changes the
   LiteLLM config.
4. Set `HF_HOME` where? Given no external storage exists, the honest answer today is
   somewhere on the internal disk, chosen so it can be copied to the Studio wholesale.
