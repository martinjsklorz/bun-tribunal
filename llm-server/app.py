"""Bun Tribunal - LLM contender.

A general-purpose vision LLM behind any OpenAI-compatible API, wrapped in the same
/health + /classify contract as the CLEF-Flash and CNN servers (see ../CONTRACT.md).

Started by ./run.sh (port 8003). Configure with ./run.sh llm, which writes llm-server/.env
(see .env.example); real environment variables win over .env.
"""

from __future__ import annotations

import asyncio
import io
import logging
import math
import os
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager, suppress
from pathlib import Path
from typing import Annotated, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

from llm_client import LLMClient, LLMConfig, UpstreamError, UpstreamTimeout, Verdict, load_dotenv

HERE = Path(__file__).resolve().parent
MODEL_NAME = "llm"
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
HEALTH_TTL_S = 30.0  # /health is polled by the UI; don't hit the provider every time
log = logging.getLogger("llm-server")

load_dotenv(HERE / ".env")
CFG = LLMConfig.from_env()


class Backend:
    """The shared API client plus a cached readiness check."""

    def __init__(self, client: LLMClient):
        self.client = client
        self.ready = False
        self.detail: str | None = None
        self._checked_at = -math.inf
        self._lock = asyncio.Lock()

    def _fresh(self) -> bool:
        return time.monotonic() - self._checked_at < HEALTH_TTL_S

    async def refresh(self, force: bool = False) -> None:
        if not force and self._fresh():
            return
        waiting_since = time.monotonic()
        async with self._lock:
            # Concurrent callers share one ping: skip ours if one finished while we waited for the lock.
            if self._checked_at >= waiting_since or (not force and self._fresh()):
                return
            self.ready, self.detail = await self.client.ping()
            self._checked_at = time.monotonic()

    async def require_ready(self) -> None:
        if not self.ready:
            await self.refresh(force=True)  # maybe the user fixed the settings or started Ollama
        if not self.ready:
            raise HTTPException(503, self.detail or "model not ready")


backend: Backend  # set by lifespan


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    global backend
    async with LLMClient(CFG) as client:
        backend = Backend(client)
        await backend.refresh(force=True)
        log.warning(
            "LLM backend: model=%s base=%s ready=%s %s", CFG.model, CFG.base_url, backend.ready, backend.detail or ""
        )
        yield


app = FastAPI(title="Bun Tribunal - LLM", version="1.0.0", lifespan=lifespan)
# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it (or spend your API key).
UI_ORIGIN_RE = rf"^https?://[^/]+:{os.environ.get('PORT_UI', '8080')}$"
app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])


# Everything Pillow raises for a bad upload, e.g. SyntaxError("broken PNG file"); same policy as
# clef-server/imaging.py.
_DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError)


def decode_image(data: bytes, max_side: int) -> Image.Image:
    """Decode an upload, upright. Big JPEGs are decoded at a reduced DCT scale (still >= max_side),
    which is much faster than a full decode that gets downscaled anyway."""
    if not data:
        raise ValueError("empty file")
    try:
        img = Image.open(io.BytesIO(data))
        img.draft("RGB", (max_side, max_side))  # no-op for non-JPEG
        img.load()
    except _DECODE_ERRORS as e:
        raise ValueError(f"could not decode image: {e}") from e
    with suppress(Exception):  # broken EXIF should not fail the request
        ImageOps.exif_transpose(img, in_place=True)
    return img


def classify_body(verdict: Verdict, total_ms: float) -> dict[str, Any]:
    probs = {k: round(v, 6) for k, v in verdict.probabilities.items()}
    label = verdict.label
    return {
        "model": MODEL_NAME,
        "label": label,
        "is_hotdog": label == "hotdog",
        "confidence": probs[label],
        "probabilities": probs,
        "latency_ms": round(verdict.latency_ms, 1),
        "total_ms": round(total_ms, 1),
        "reason": verdict.reason,
        "confidence_source": verdict.confidence_source,
        "model_id": CFG.model,
        "attempts": verdict.attempts,
    }


@app.get("/health")
async def health() -> dict[str, Any]:
    await backend.refresh()
    body: dict[str, Any] = {
        "status": "ok",
        "model": MODEL_NAME,
        "device": "api",
        "ready": backend.ready,
        "model_id": CFG.model,
        "base_url_host": CFG.host,
    }
    if backend.detail:
        body["detail"] = backend.detail
    return body


@app.post("/classify")
async def classify(file: Annotated[UploadFile, File()]) -> dict[str, Any]:
    t0 = time.perf_counter()
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "image too large (max 20 MB)")
    try:
        img = await run_in_threadpool(decode_image, data, CFG.max_image_side)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e
    del data  # the encoded upload can be large; the decoded image is all we need

    await backend.require_ready()
    try:
        verdict = await backend.client.classify(img)
    except UpstreamTimeout as e:
        raise HTTPException(504, str(e)) from e
    except UpstreamError as e:
        raise HTTPException(502, str(e)) from e
    return classify_body(verdict, (time.perf_counter() - t0) * 1000.0)
