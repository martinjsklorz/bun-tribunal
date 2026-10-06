"""Bun Tribunal - LLM contender.

A general-purpose vision LLM behind any OpenAI-compatible API, wrapped in the same
/health + /classify contract as the CLEF-Flash and CNN servers (see ../CONTRACT.md).

Started by ./run.sh (port 8003). Configure with ./run.sh llm, which writes llm-server/.env
(see .env.example); real environment variables win over .env.
"""
from __future__ import annotations

import io
import logging
import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

HERE = Path(__file__).resolve().parent
MODEL_NAME = "llm"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
log = logging.getLogger("llm-server")



from llm_client import LLMClient, LLMConfig, UpstreamError, UpstreamTimeout, load_dotenv  # noqa: E402

load_dotenv(HERE / ".env")

CFG = LLMConfig.from_env()
HEALTH_TTL_S = 30.0

class State:
    client: LLMClient | None = None
    ready: bool = False
    detail: str | None = None
    checked_at: float = 0.0


state = State()


async def refresh_health(force: bool = False) -> None:
    if not force and time.monotonic() - state.checked_at < HEALTH_TTL_S:
        return
    ok, detail = await state.client.ping()
    state.ready, state.detail, state.checked_at = ok, detail, time.monotonic()


@asynccontextmanager
async def lifespan(_: FastAPI):
    state.client = LLMClient(CFG)
    await refresh_health(force=True)
    log.warning("LLM backend: model=%s base=%s ready=%s %s", CFG.model, CFG.base_url, state.ready, state.detail or "")
    yield
    if state.client:
        await state.client.aclose()


app = FastAPI(title="Bun Tribunal - LLM", version="1.0.0", lifespan=lifespan)
# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it (or, for the LLM, spend your API key).
UI_ORIGIN_RE = r"^https?://[^/]+:%s$" % os.environ.get("PORT_UI", "8080")
app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])


def decode(data: bytes) -> Image.Image:
    if not data:
        raise ValueError("empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
        img = ImageOps.exif_transpose(img)
        if img.mode in ("RGBA", "LA", "P"):
            img = img.convert("RGBA")
            bg = Image.new("RGB", img.size, (255, 255, 255))
            bg.paste(img, mask=img.split()[-1])
            return bg
        return img.convert("RGB")
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
        raise ValueError(f"could not decode image: {e}") from e


@app.get("/health")
async def health():
    await refresh_health()
    body = {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "api",
        "ready": state.ready,
        "model_id": CFG.model,
        "base_url_host": CFG.host,
    }
    if state.detail:
        body["detail"] = state.detail
    return body


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 20 MB)")
    try:
        img = await run_in_threadpool(decode, data)
    except ValueError as e:
        raise HTTPException(400, str(e))

    if not state.ready:
        await refresh_health(force=True)
        if not state.ready:
            raise HTTPException(503, state.detail or "model not ready")
    try:
        res = await state.client.classify(img)
    except UpstreamTimeout as e:
        raise HTTPException(504, str(e))
    except UpstreamError as e:
        raise HTTPException(502, str(e))

    probs = {k: round(float(v), 6) for k, v in res["probabilities"].items()}
    label = "hotdog" if probs["hotdog"] >= probs["not_hotdog"] else "not_hotdog"
    total_ms = (time.perf_counter() - t0) * 1000.0
    return {
        "model": MODEL_NAME,
        "label": label,
        "is_hotdog": label == "hotdog",
        "confidence": probs[label],
        "probabilities": probs,
        "latency_ms": round(res["latency_ms"], 1),
        "total_ms": round(total_ms, 1),
        "reason": res.get("reason"),
        "confidence_source": res["confidence_source"],
        "model_id": CFG.model,
        "attempts": res.get("attempts", 1),
    }
