# Shared voice settings. Sourced by speak.sh and listen.sh.

# TTS: mlx_audio.server speaks an OpenAI-compatible /v1/audio/speech, running
# Kokoro-82M on the GPU via MLX. No docker.
#
# Kept behind an HTTP API rather than an in-process library call for the same
# reason the LLM gateway is: a persistent server holds the model resident, so a
# spoken reply does not pay a cold model load every turn, and the engine behind
# the endpoint stays swappable.
TTS_HOST="${TTS_HOST:-http://127.0.0.1:8083}"
TTS_MODEL="${TTS_MODEL:-mlx-community/Kokoro-82M-bf16}"
TTS_VOICE="${TTS_VOICE:-am_adam}"
TTS_SPEED="${TTS_SPEED:-1.15}"

# Summarizer: the local gateway from this repo. Long responses are condensed to
# a couple of spoken sentences; reading a response with code blocks and tables
# aloud verbatim is unusable.
GATEWAY="${GATEWAY:-http://127.0.0.1:4000}"
SUMMARY_MODEL="${SUMMARY_MODEL:-local-summarize}"
SUMMARY_MAX_TOKENS="${SUMMARY_MAX_TOKENS:-90}"

# The summarizer prompt is load-bearing, not decoration. A small instruct model
# reads an assistant message as a turn to RESPOND to, not text to condense. The
# first version of this prompt made Qwen2.5-1.5B answer the closing question
# instead of relaying it, turning "Want me to wire up the voice layer next?" into
# "No, I don't need to wire up the voice layer next." Spoken aloud that is a
# fabricated answer in Claude's own voice, which is worse than saying nothing.
#
# The fixes that mattered: name the role as compressor rather than assistant,
# state outright that nothing is being asked, pin first person, and require the
# closing question be reproduced rather than resolved.
SUMMARY_SYSTEM="${SUMMARY_SYSTEM:-$(cat <<'PROMPT'
You are a text compressor, not a conversational partner. You will be given a transcript of something ANOTHER speaker already said. Rewrite it as at most two sentences to be read aloud, preserving their meaning and their point of view.

Rules:
- NEVER answer, agree, refuse, or react. You are not being asked anything.
- Keep first person: if the speaker said "I did X", you say "I did X".
- If the transcript ends with a question to the listener, reproduce that question as your final sentence, unchanged in meaning.
- Plain speech. No markdown, lists, code, or file paths.
- This will be READ ALOUD. Omit status codes, port numbers, version strings, and
  diagnostic output. Say the outcome in words: "everything came up" rather than
  "gateway 200, MLX 200, TTS 200".
- Avoid strings of letters that must be spelled out. Prefer "the local models"
  over "MLX", "the gateway" over ":4000".
- Output only the rewritten text.
PROMPT
)}"

# STT: served by the SAME mlx_audio.server as TTS, on /v1/audio/transcriptions.
# It accepts a Parakeet model id and holds it resident. Measured 0.26s to
# transcribe 6.5s of speech, against 0.79s in-process where each call pays a
# 0.53s model load.
PARAKEET_MODEL="${PARAKEET_MODEL:-mlx-community/parakeet-tdt-0.6b-v2}"

# Recording gate. VOICE_THRESHOLD is a percentage of full scale: sox starts
# capturing once the signal exceeds it and stops after VOICE_HANG seconds below
# it. 2% proved too high to trigger on quiet sources; 1% still sits well above
# a quiet room (measured RMS 59, or 0.18% of full scale, on a Yeti with nobody
# talking). Raise it if the gate self-triggers on background noise.
VOICE_THRESHOLD="${VOICE_THRESHOLD:-1%}"
VOICE_HANG="${VOICE_HANG:-1.8}"
VOICE_MAX_SECONDS="${VOICE_MAX_SECONDS:-30}"

# Hard stop on waiting for someone to start talking. Without it a gate that
# never triggers leaves sox holding the microphone open forever, which is what
# happened the first time this was tested.
VOICE_ONSET_TIMEOUT="${VOICE_ONSET_TIMEOUT:-12}"

# Kill switch. `touch` this file to go quiet without editing settings.json.
VOICE_MUTE_FLAG="${VOICE_MUTE_FLAG:-$HOME/.claude/voice-mute}"

VOICE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOICE_RUN="${VOICE_RUN:-${TMPDIR:-/tmp}/claude-voice}"
mkdir -p "$VOICE_RUN"
