# Validation log

Every claim in PLAN.md marked VALIDATED traces to a command here. Run on
2026-09-05, macOS 26.5.1 (25F80), Mac14,12 M2 Pro Mac mini, 32 GiB.

Versions at time of validation: uv 0.12.5, mlx 0.32.2, mlx-lm 0.31.3,
litellm 1.99.0, ollama 0.32.1, Homebrew 6.0.18.

## Hardware and OS

    $ sysctl hw.model hw.memsize
    hw.model: Mac14,12
    hw.memsize: 34359738368

    $ sysctl iogpu.wired_limit_mb
    iogpu.wired_limit_mb: 0          # 0 means "system default", not "no GPU memory"

    $ sw_vers
    ProductVersion: 26.5.1   BuildVersion: 25F80

sysctl is present at /usr/sbin/sysctl. Reading needs no privileges; writing needs
sudo and does not survive a reboot.

## Storage

    $ df -h /System/Volumes/Data
    /dev/disk3s5  926Gi  804Gi  77Gi  92%

    $ du -sh ~/.ollama
     44G

    $ du -sh ~/.cache/huggingface
    9.3G                              # FLUX, CLIP, Chatterbox. Not LLM weights.

/Volumes/Raid and /Volumes/External1 both resolve to /dev/disk3s5, the internal Data
volume. They are stale mountpoints, not attached drives. Confirmed by `df` reporting
the same filesystem and by `mount | grep -E 'Raid|External1'` returning nothing.

## External drives (measured after the user reconnected them)

    $ df -h /Volumes/T7 /Volumes/2TB
    /dev/disk53s1   931Gi  461Gi  470Gi  50%  /Volumes/T7
    /dev/disk51s1   1.8Ti  1.1Ti  780Gi  59%  /Volumes/2TB

Link speeds, ioreg "Device Speed" (3 = SuperSpeed 5Gbps, 4 = SuperSpeed+ 10Gbps):

    +-o AppleT8112USBXHCI@00000000
    | +-o DockCase SSD Enclosure C1P@00200000     <- /Volumes/2TB
    |       "Device Speed" = 4                     direct to host controller
    +-o AppleEmbeddedUSBXHCIASMedia3142@08000000
      +-o USB3.0 Hub@08100000  (VIA Labs)
      |     "Device Speed" = 3
      +-o PSSD T7@08200000                         <- /Volumes/T7
            "Device Speed" = 3                     behind the hub

`SPThunderboltDataType` reports "No device connected" on every port, so neither drive
is Thunderbolt.

Throughput, 1 GiB dd:

    T7,  cold read of pre-existing itunes_backup.tar.gz : 459,927,758 B/s (460 MB/s)
    T7,  same file at offset 400M (page cache)          : 1,117,559,080 B/s
    T7,  write                                          : 430,744,510 B/s
    internal, write                                     : 1,152,784,821 B/s
    2TB, read of a 12GB TM backup .img                  : 229,010,986 B/s
    2TB, same file at offset 2G                         : 118,718,271 B/s

The T7 numbers are trustworthy: 460 MB/s cold, reproduced across two different
pre-existing files, which is 5 Gbps saturation. The T7 is a 10 Gbps device, so the
VIA hub is halving it.

### Superseded: the 2TB reclaimed and re-measured

The user repartitioned the 2 TB NVMe into `Backups` and `Models`, two APFS volumes in
one container with 1 TB quotas each. `Models` is not a Time Machine target, so it
mounts `eric:staff` and writable.

Re-measured with 4 GiB dd and a cold read forced by `diskutil unmount` +
`diskutil mount`, so the page cache cannot inflate it:

    Models write     : 4294967296 bytes in 4.238520 secs (1,013,317,690 B/s)
    Models COLD read : 4294967296 bytes in 4.478884 secs (  958,936,935 B/s)
    Models warm read : 1073741824 bytes in 0.390211 secs (2,751,695,426 B/s)  <- cache, not the drive

    T7 write         : 4294967296 bytes in 10.168272 secs (422,389,104 B/s)
    T7 COLD read     : 4294967296 bytes in  9.932652 secs (432,408,917 B/s)

Models is 2.2x the T7 and effectively matches the internal SSD. This vindicates
discarding the earlier 229 MB/s reading rather than recording it: the clean volume is
over four times faster, confirming that number measured Time Machine's fragmented
backup structure and not the drive.

HF_HOME moved to /Volumes/Models/hf; the 1.1 GiB of existing weights were rsynced
from the T7 (82 files, 1,169,990,654 bytes) and all six smoke checks pass on the new
volume.

Guard re-verified by exit code after the switch: auto-pick 0, explicit Models 0,
explicit T7 0, explicit nonexistent 1.

### Original (superseded) reading

The 2TB numbers below are NOT trustworthy as a drive measurement. Every file on that volume
is Time Machine backup data stored with APFS clones and heavy fragmentation, and a
write test is impossible:

    $ dd if=/dev/zero of=/Volumes/2TB/.bench_tmp bs=1m count=1024
    dd: /Volumes/2TB/.bench_tmp: Permission denied

    $ ls -ld /Volumes/2TB
    drwxrwxr-x@ 8 root wheel /Volumes/2TB
    $ id -Gn eric | grep -x wheel   ->  eric NOT in wheel

Time Machine owns the volume root. Re-measure with a clean write/read test once the
user reclaims it.

## MLX backend

    $ uv run python -c "import mlx.core as mx; print(mx.default_device(), mx.metal.is_available())"
    Device(gpu, 0) True

    $ uv run mlx_lm.generate --model mlx-community/Qwen2.5-0.5B-Instruct-4bit \
        --prompt "Reply with exactly: OK" --max-tokens 20
    OK
    Prompt: 34 tokens, 25.120 tokens-per-sec
    Generation: 2 tokens, 784.455 tokens-per-sec
    Peak memory: 0.336 GB

## mlx_lm.server OpenAI surface

    $ curl -s localhost:8081/v1/models
    {"object":"list","data":[{"id":"mlx-community/Qwen2.5-0.5B-Instruct-4bit",...}]}

    $ curl -s localhost:8081/v1/chat/completions -d '{...,"max_tokens":10}'
    ...{"message":{"role":"assistant","content":"PONG!"}}...

Streaming, counting prompt, 30 max tokens:

    total lines: 26   data chunks: 11   keepalives: 2
    first: data: {..."object":"chat.completion.chunk"...,"delta":{"content":"1"}}
    last:  data: [DONE]

## Model hot-swap (undocumented, important)

Server launched with `--model mlx-community/Qwen2.5-0.5B-Instruct-4bit`. Request sent
with `"model":"mlx-community/Qwen2.5-1.5B-Instruct-4bit"` returned HTTP 200 answered
by the 1.5B, and afterwards:

    $ curl -s localhost:8081/v1/models
    {"data":[{"id":"...Qwen2.5-1.5B-Instruct-4bit"},{"id":"...Qwen2.5-0.5B-Instruct-4bit"}]}

The `model` field is treated as a live HF repo_id. Sending `"model":"x"` produced:

    {"error": "404 Client Error ... Repository Not Found for url:
     https://huggingface.co/api/models/x/revision/main"}

A bad model id is a network fetch, not a validation error.

## Gateway, both API shapes

    $ curl -s localhost:4000/v1/chat/completions -d '{"model":"local-small",...}'
    ..."content":"GATEWAY-OK"...   usage.prompt_tokens_details.cached_tokens: 28

    $ curl -s localhost:4000/v1/chat/completions -d '{"model":"local-mid",...}'
    model: local-mid | content: Hello! How can I help you today?

Anthropic shape, after the env-var fix (see below):

    $ curl -s localhost:4000/v1/messages -H 'anthropic-version: 2023-06-01' -d '{...}'
    {"type":"message","role":"assistant","content":[{"type":"text","text":"ANTHROPIC-OK"}],
     "stop_reason":"end_turn"}   HTTP=200

Upstream path confirmed in mlx-server.log as `"POST /v1/chat/completions"`, not
/v1/responses.

## Tool calling, both shapes, 1.5B model

OpenAI:

    tool_calls: [{"function":{"name":"get_weather","arguments":"{\"city\": \"Paris\"}"},
                  "id":"fa50aa26-...","type":"function"}]

Anthropic:

    content: [{"type":"tool_use","id":"c3a0f025-...","name":"get_weather",
               "input":{"city":"Paris"}}]
    stop_reason: tool_use

## The LiteLLM /v1/messages 404

Initial failure:

    litellm.NotFoundError: OpenAIException - Not Found. Received Model Group=local-small
    MaskedHTTPStatusError: Client error '404 Not Found' for url
      'http://127.0.0.1:8081/v1/responses'

Source read at
`litellm/llms/anthropic/experimental_pass_through/messages/handler.py`:

    _RESPONSES_API_PROVIDERS: Final = frozenset({"openai"})

    def _should_route_to_responses_api(...):
        if litellm.use_chat_completions_url_for_anthropic_messages:
            return False
        if custom_llm_provider in _RESPONSES_API_PROVIDERS:
            return True

and at `litellm/__init__.py:239`:

    use_chat_completions_url_for_anthropic_messages: bool = bool(
        os.getenv("LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES", False)
    )

### First conclusion, and why it was wrong

Attempt 1 put `use_chat_completions_url_for_anthropic_messages: true` under
`litellm_settings:` in config.yaml. Tested immediately after
`/health/readiness` returned healthy: appeared to FAIL with the identical 404.

Attempt 2 set `LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES=1` in the
launcher environment. Tested after a longer delay: SUCCEEDED, HTTP 200.

Conclusion drawn: "the YAML key is ignored, only the env var works." **This was
wrong.** It survived initial scrutiny because the source read supported it, which is
exactly what made it dangerous.

### The red-proof that falsified it

Restarting the gateway *without* the env var was expected to reproduce the 404. The
first smoke run after restart failed five of six checks including checks that had
nothing to do with Anthropic routing. Re-running the same script seconds later passed
all six. That inconsistency, not the original test, is what exposed the race.

Controlled A/B, each waiting on a real completion rather than `/health/readiness`:

CASE A, no YAML key and no env var:

    $ sed -n '/litellm_settings/,$p' /tmp/cfg_nokey.yaml
    litellm_settings:
      drop_params: true

    HTTP=404
    {"error":{"message":"litellm.NotFoundError: ... Received Model Group=local-small"}}
    caseA.log: for url 'http://127.0.0.1:8081/v1/responses'
    mlx-server.log: "POST /v1/responses HTTP/1.1" 404

CASE B, YAML key present, env var explicitly removed with
`env -u LITELLM_USE_CHAT_COMPLETIONS_URL_FOR_ANTHROPIC_MESSAGES`:

    HTTP=200
    {"type":"message","role":"assistant",
     "content":[{"type":"text","text":"OK, thank you for asking! ..."}],
     "stop_reason":"end_turn"}
    $ grep -c "v1/responses" .logs/caseB.log
    0

**Both mechanisms work.** The YAML key is honored. The env var is honored.

### Standing correction

Never gate a config A/B on `/health/readiness`. Poll a real request on the route
under test and require a successful body before concluding anything. Then run the
negative case and confirm the failure returns. See KB rule "LiteLLM
/health/readiness lies".

### The env var's bool() quirk (unchanged, still true)

`litellm/__init__.py:239` reads the flag as `bool(os.getenv(...))`, so any non-empty
value is true, including the string `false`. Set it to 1 or leave it unset.
