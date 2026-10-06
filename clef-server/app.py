"""CLEF-Flash hotdog classifier — implements bun-tribunal CONTRACT.md (port 8001).

Started by ./run.sh (port 8001); weights come from ./run.sh download.
Env:  CLEF_MODEL (default Cloudflare/clef-flash, or local path), DEVICE, DTYPE (bf16|fp16|fp32),
      CLEF_SCHEMA, MAX_IMAGE_SIDE (1024), MAX_UPLOAD_MB (25), HF_HOME.
"""
from __future__ import annotations

import logging
import os
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

from imaging import BadImage, load_image
from prompt import load_prompt

# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it (or, for the LLM, spend your API key).
UI_ORIGIN_RE = r"^https?://[^/]+:%s$" % os.environ.get("PORT_UI", "8080")
MODEL_NAME = "clef-flash"
logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def make_backend():
    from clef_backend import ClefBackend  # torch imported lazily inside loader thread
    return ClefBackend(os.environ.get("CLEF_MODEL", "Cloudflare/clef-flash"), load_prompt())


def create_app() -> FastAPI:
    backend = make_backend()
    max_upload = int(float(os.environ.get("MAX_UPLOAD_MB", "25")) * 1024 * 1024)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backend.start()  # returns immediately; the model loads in a background thread
        yield

    app = FastAPI(title="CLEF-Flash hotdog classifier", lifespan=lifespan)
    app.state.backend = backend
    app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health():
        body = {
            "status": "ok",
            "model": MODEL_NAME,
            "device": backend.device,
            "ready": bool(backend.ready),
        }
        if backend.error:
            body["error"] = backend.error
            body["detail"] = backend.error  # same key the CNN / LLM servers use
        return body

    @app.post("/classify")
    async def classify(file: UploadFile = File(...)):
        t0 = time.perf_counter()
        if not backend.ready:
            raise HTTPException(503, backend.error or "model not ready (still loading weights)")
        raw = await file.read(max_upload + 1)
        if len(raw) > max_upload:
            raise HTTPException(413, f"file larger than {max_upload // (1024 * 1024)} MB")
        try:
            image = await run_in_threadpool(load_image, raw)
        except BadImage as e:
            raise HTTPException(400, str(e))

        probs, latency_ms = await run_in_threadpool(backend.classify, image)
        label = "hotdog" if probs["hotdog"] >= probs["not_hotdog"] else "not_hotdog"
        total_ms = (time.perf_counter() - t0) * 1000.0
        return {
            "model": MODEL_NAME,
            "label": label,
            "is_hotdog": label == "hotdog",
            "confidence": round(probs[label], 6),
            "probabilities": {k: round(v, 6) for k, v in probs.items()},
            "latency_ms": round(latency_ms, 1),
            "total_ms": round(total_ms, 1),
        }

    return app


app = create_app()
