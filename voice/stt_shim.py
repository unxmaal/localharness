"""Model-rewriting proxy for /v1/audio/transcriptions.

voicemode hardcodes the STT model as "whisper-1" and exposes no override: its
config has VOICEMODE_STT_BASE_URLS but no VOICEMODE_STT_MODELS. mlx_audio.server
takes a HuggingFace repo id, so it answers that request with

    404 {"detail": "Model not found: 'whisper-1' is not a known HuggingFace
         repo or local path"}

This sits in front, replaces whatever model was asked for with the Parakeet repo
id, and forwards. Keeps Parakeet's ~0.24s instead of falling back to whisper.cpp.

Deliberately dumb: it rewrites one form field and proxies bytes. Anything more
belongs in mlx_audio or voicemode, not in a shim.
"""
import os

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response

UPSTREAM = os.environ.get("STT_UPSTREAM", "http://127.0.0.1:8085")
MODEL = os.environ.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")

app = FastAPI()


@app.post("/v1/audio/speech")
async def speech(request: Request):
    # TTS passes straight through. The shim sits on the port voicemode already
    # knows about, so it has to carry every route that port served.
    body = await request.body()
    async with httpx.AsyncClient(timeout=180) as client:
        r = await client.post(f"{UPSTREAM}/v1/audio/speech", content=body,
                              headers={"content-type": "application/json"})
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


@app.get("/")
async def root():
    return {"status": "ok", "shim": "stt model rewrite", "upstream": UPSTREAM}


@app.get("/v1/models")
async def models():
    # voicemode probes this to build its provider registry.
    return {"object": "list", "data": [{"id": "whisper-1", "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    form = await request.form()
    upload = form.get("file")
    if upload is None:
        return JSONResponse({"error": "no file"}, status_code=400)

    files = {"file": (upload.filename, await upload.read(),
                      upload.content_type or "audio/wav")}
    # Every field except `model` is passed through untouched; `model` is the
    # whole point of this proxy.
    data = {k: v for k, v in form.items() if k not in ("file", "model")}
    data["model"] = MODEL

    async with httpx.AsyncClient(timeout=120) as client:
        r = await client.post(f"{UPSTREAM}/v1/audio/transcriptions",
                              files=files, data=data)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))
