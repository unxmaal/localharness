# voice

Speaking and hearing for Claude Code, local models only.

[voicemode](https://github.com/mbailey/voicemode) is the front end: an MCP
server with a `converse` tool, so Claude asks to speak rather than speaking on
every turn. This directory used to hold a competing Stop-hook front end and a
proxy; both are gone.

What remains from here is the backend choice, which is faster than what
voicemode ships by default.

## Serving it

    ./scripts/serve-tts.sh      # mlx_audio on 8890: Kokoro TTS + Parakeet STT

Port 8890 is not arbitrary. voicemode's `provider_discovery.py` classifies that
port as `mlx-audio` and stops sending it `whisper-1`, which mlx_audio rejects
because it wants a HuggingFace repo id.

## voicemode settings that matter

    VOICEMODE_TTS_BASE_URLS=http://127.0.0.1:8890/v1
    VOICEMODE_STT_BASE_URLS=http://127.0.0.1:8890/v1
    VOICEMODE_TTS_MODELS=mlx-community/Kokoro-82M-bf16
    VOICEMODE_STT_MODELS=mlx-community/parakeet-tdt-0.6b-v2
    VOICEMODE_VAD_AGGRESSIVENESS=1      # default 3 hears nothing on a Yeti
    OPENAI_API_KEY=<anything>           # required even with nothing cloud

Measured through this stack: `stt 0.1s`, `gen 0.4s`, `ttfa 0.4s`.

## Known gaps

- The `transcribe` CLI subcommand ignores `VOICEMODE_STT_BASE_URLS` and goes to
  api.openai.com. `OPENAI_BASE_URL` works there. The MCP `converse` path, which
  is the one in use, honours the config.
- No eval cases for TTS or STT yet, for the half of the goal in daily use.
- Parakeet is English-only; Canary-Qwen and Voxtral are the multilingual swaps
  and mlx-audio serves them too.
