"""CLEF-Flash hotdog classifier: implements Bun Tribunal's CONTRACT.md on port 8001.

Started by ./run.sh; the weights come from ./run.sh download.
Env:  CLEF_BACKEND (auto|mlx|torch; auto = 4-bit MLX on Apple Silicon, full BF16 PyTorch elsewhere),
      CLEF_MODEL (Hub repo id or local dir; default depends on the backend), CLEF_AUTO_DOWNLOAD (0),
      DEVICE, DTYPE (torch only), CLEF_SCHEMA, MAX_IMAGE_SIDE (1024), MAX_IMAGE_PIXELS, MAX_UPLOAD_MB (25),
      PORT_UI (8080), LOG_LEVEL, HF_HOME.
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from starlette.concurrency import run_in_threadpool

import clef_backend
from imaging import BadImage, load_image
from prompt import load_prompt

MODEL_NAME = "clef-flash"
MB = 1024 * 1024
# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it.
UI_ORIGIN_RE = rf"^https?://[^/]+:{os.environ.get('PORT_UI', '8080')}$"

logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))


def create_app() -> FastAPI:
    backend = clef_backend.make_backend(load_prompt())
    max_upload = int(float(os.environ.get("MAX_UPLOAD_MB", "25")) * MB)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        backend.start()  # returns immediately; the model loads in the background
        yield

    app = FastAPI(title="CLEF-Flash hotdog classifier", lifespan=lifespan)
    app.state.backend = backend
    app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])

    @app.get("/health")
    def health() -> dict:
        body = {
            "status": "ok",
            "model": MODEL_NAME,
            "device": backend.device,
            "ready": backend.ready,
            "backend": backend.kind,
            "weights": backend.weights,
            "model_id": backend.model_id,
        }
        if backend.error:
            body["detail"] = backend.error  # the key the other servers use
            body["error"] = backend.error  # kept for older UIs
        return body

    @app.post("/classify")
    async def classify(file: Annotated[UploadFile, File()]) -> dict:
        started = time.perf_counter()
        if not backend.ready:
            raise HTTPException(503, backend.error or "model not ready (still loading weights)")
        data = await _read_upload(file, max_upload)
        try:
            image = await run_in_threadpool(load_image, data)
        except BadImage as e:
            raise HTTPException(400, str(e)) from e
        probs, latency_ms = await run_in_threadpool(backend.classify, image)
        return _result(probs, latency_ms, total_ms=(time.perf_counter() - started) * 1000.0)

    return app


async def _read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """The upload's bytes, or 413. Checks the known size first so an oversized file is never read into memory."""
    if file.size is None or file.size <= max_bytes:
        data = await file.read(max_bytes + 1)
        if len(data) <= max_bytes:
            return data
    raise HTTPException(413, f"file larger than {max_bytes // MB} MB")


def _result(probs: dict[str, float], latency_ms: float, total_ms: float) -> dict:
    label = "hotdog" if probs["hotdog"] >= probs["not_hotdog"] else "not_hotdog"
    return {
        "model": MODEL_NAME,
        "label": label,
        "is_hotdog": label == "hotdog",
        "confidence": round(probs[label], 6),
        "probabilities": {name: round(p, 6) for name, p in probs.items()},
        "latency_ms": round(latency_ms, 1),
        "total_ms": round(total_ms, 1),
    }


app = create_app()
