"""Model-rewriting proxy for mlx_audio, sitting in front of voicemode.

voicemode hardcodes the STT model as "whisper-1" and exposes no override: its
config has VOICEMODE_STT_BASE_URLS but no VOICEMODE_STT_MODELS. mlx_audio.server
wants a HuggingFace repo id, so it answers that request with

    404 {"detail": "Model not found: 'whisper-1' is not a known HuggingFace
         repo or local path"}

This sits in front, replaces whatever model was asked for with the Parakeet repo
id, and forwards. Keeps Parakeet's ~0.1s instead of falling back to whisper.cpp.

It also carries /v1/audio/speech untouched, because it occupies the port
voicemode already has cached and therefore has to serve every route that port
served before.

Deliberately dumb about content: it rewrites one form field and proxies bytes.
It is NOT dumb about failure. It sits in a live audio path, so an unreachable or
slow upstream has to read as 502/504 rather than surfacing a stack trace.

Config:
    STT_UPSTREAM   base URL of mlx_audio.server   (default http://127.0.0.1:8085)
    STT_MODEL      repo id to rewrite `model` to  (default Parakeet TDT 0.6b v2)
    STT_TIMEOUT    upstream timeout in seconds    (default 120)
"""
import logging
import os
from contextlib import asynccontextmanager

import httpx
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response
from starlette.datastructures import UploadFile

UPSTREAM = os.environ.get("STT_UPSTREAM", "http://127.0.0.1:8085").rstrip("/")
MODEL = os.environ.get("STT_MODEL", "mlx-community/parakeet-tdt-0.6b-v2")
TIMEOUT = float(os.environ.get("STT_TIMEOUT", "120"))

log = logging.getLogger("stt_shim")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # One pooled client for the process. Building an AsyncClient per request
    # throws away connection reuse on a path that runs on every utterance.
    app.state.client = httpx.AsyncClient(timeout=TIMEOUT)
    try:
        yield
    finally:
        await app.state.client.aclose()


app = FastAPI(lifespan=lifespan)


async def _forward(request: Request, path: str, **kwargs) -> Response:
    """POST to the upstream, mapping transport failures to gateway statuses.

    Without this an unreachable mlx_audio raises out of the handler and the
    caller gets a 500 with a traceback, which reads as "the shim is broken"
    rather than "the thing behind it is down".
    """
    url = f"{UPSTREAM}{path}"
    try:
        r = await request.app.state.client.post(url, **kwargs)
    except httpx.TimeoutException as exc:
        log.warning("upstream timeout %s: %s", url, exc)
        return JSONResponse(
            {"detail": f"upstream timed out after {TIMEOUT}s: {url}"},
            status_code=504)
    except httpx.HTTPError as exc:
        log.warning("upstream unreachable %s: %s", url, exc)
        return JSONResponse(
            {"detail": f"upstream unreachable: {url}"}, status_code=502)
    return Response(content=r.content, status_code=r.status_code,
                    media_type=r.headers.get("content-type"))


@app.get("/")
async def root():
    return {"status": "ok", "shim": "stt model rewrite",
            "upstream": UPSTREAM, "model": MODEL}


@app.get("/v1/models")
async def models():
    # voicemode probes this to build its provider registry, and expects to see
    # the model id it is going to ask for.
    return {"object": "list", "data": [{"id": "whisper-1", "object": "model"}]}


@app.post("/v1/audio/transcriptions")
async def transcriptions(request: Request):
    form = await request.form()
    upload = form.get("file")

    # A plain string in `file` is a client error, not a reason to raise
    # AttributeError on .filename and answer a live audio path with a 500.
    if not isinstance(upload, UploadFile):
        return JSONResponse(
            {"detail": "field 'file' must be an uploaded file"},
            status_code=400)

    files = {"file": (upload.filename, await upload.read(),
                      upload.content_type or "audio/wav")}
    # Every field except `model` is passed through untouched; `model` is the
    # whole point of this proxy.
    data = {k: v for k, v in form.multi_items()
            if k not in ("file", "model") and isinstance(v, str)}
    data["model"] = MODEL

    return await _forward(request, "/v1/audio/transcriptions",
                          files=files, data=data)


@app.post("/v1/audio/speech")
async def speech(request: Request):
    body = await request.body()
    return await _forward(request, "/v1/audio/speech", content=body,
                          headers={"content-type": "application/json"})
