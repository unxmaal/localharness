#!/usr/bin/env bash
# Asserts the six behaviours the architecture depends on. Run after any
# mlx-lm or litellm upgrade. Non-zero exit means the seam broke.
set -uo pipefail
G="http://127.0.0.1:${GATEWAY_PORT:-4000}"
E="http://127.0.0.1:${MLX_PORT:-8081}"
FAIL=0
ok(){ printf 'PASS  %s\n' "$1"; }
no(){ printf 'FAIL  %s\n' "$1"; FAIL=1; }

# LiteLLM's /health/readiness returns 200 before the proxy can actually serve.
# Gate on a real completion, or these assertions produce false failures and, worse,
# false attributions of a fix. See docs/validation-log.md.
#
# The delay is load-bearing: curl fails INSTANTLY on connection-refused, so an
# unpaced loop burns all its attempts inside the startup window and reports a
# ready service as dead.
READY_TIMEOUT="${READY_TIMEOUT:-120}"
printf 'wait  gateway accepting real requests'
ready=0
for i in $(seq 1 "$READY_TIMEOUT"); do
  if curl -sf --max-time 10 "$G/v1/chat/completions" -H 'Content-Type: application/json' \
       -H 'Authorization: Bearer sk-x' \
       -d '{"model":"local-small","messages":[{"role":"user","content":"hi"}],"max_tokens":2}' \
       2>/dev/null | grep -q '"content"'; then
    printf ' ready after ~%ss\n' "$i"; ready=1; break
  fi
  printf '.'
  sleep 1
done
if [ "$ready" -ne 1 ]; then
  printf '\nFAIL  gateway not serving after %ss\n' "$READY_TIMEOUT"
  exit 1
fi

grep -q . <<<"$(curl -sf "$E/v1/models")" && ok "engine /v1/models" || no "engine /v1/models"

curl -sf "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-x' \
  -d '{"model":"local-small","messages":[{"role":"user","content":"Reply with exactly: OK"}],"max_tokens":10}' \
  | grep -q '"content"' && ok "gateway openai chat" || no "gateway openai chat"

curl -sfN "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-x' \
  -d '{"model":"local-small","messages":[{"role":"user","content":"count 1 2 3"}],"max_tokens":20,"stream":true}' \
  | grep -q 'data: \[DONE\]' && ok "gateway sse streaming" || no "gateway sse streaming"

# The regression this exists to catch.
curl -sf "$G/v1/messages" -H 'Content-Type: application/json' \
  -H 'x-api-key: sk-x' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"local-small","max_tokens":20,"messages":[{"role":"user","content":"Reply with exactly: OK"}]}' \
  | grep -q '"type":"message"' && ok "gateway anthropic /v1/messages" || no "gateway anthropic /v1/messages"

curl -sf "$G/v1/chat/completions" -H 'Content-Type: application/json' \
  -H 'Authorization: Bearer sk-x' \
  -d '{"model":"local-mid","messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],"tools":[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}}],"max_tokens":80}' \
  | grep -q '"tool_calls"' && ok "openai tool calling" || no "openai tool calling"

curl -sf "$G/v1/messages" -H 'Content-Type: application/json' \
  -H 'x-api-key: sk-x' -H 'anthropic-version: 2023-06-01' \
  -d '{"model":"local-mid","max_tokens":80,"messages":[{"role":"user","content":"Weather in Paris? Use the tool."}],"tools":[{"name":"get_weather","input_schema":{"type":"object","properties":{"city":{"type":"string"}},"required":["city"]}}]}' \
  | grep -q '"type": *"tool_use"' && ok "anthropic tool calling" || no "anthropic tool calling"

exit $FAIL
