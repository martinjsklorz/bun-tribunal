"""Bun Tribunal - CNN inference server (see ../../CONTRACT.md).

Started by ./run.sh (port 8002). Manually, from the project root:
      uvicorn --app-dir cnn/server app:app --port 8002

Env:
  CNN_ARTIFACTS  directory with model_meta.json + hotdog_cnn.pt / hotdog_cnn.ts.pt
                 (default cnn/artifacts; relative paths are relative to cnn/)
  DEVICE         force a torch device (cuda / mps / cpu); default auto cuda -> mps -> cpu
  MAX_UPLOAD_MB  largest accepted upload in MB (default 25; larger -> 413)
  PORT_UI        the UI port allowed by CORS (default 8080)
"""

from __future__ import annotations

import io
import json
import logging
import os
import statistics
import sys
import threading
import time
import warnings
from contextlib import suppress
from pathlib import Path

import torch
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError
from torch import nn
from torchvision import transforms

CNN_DIR = Path(__file__).resolve().parents[1]
if str(CNN_DIR) not in sys.path:  # hotdog_model.py lives in cnn/, next to the notebook
    sys.path.insert(0, str(CNN_DIR))
from hotdog_model import (  # noqa: E402  (needs the sys.path entry above)
    ARCHS,
    CLASSES,
    DEFAULT_ARCH,
    IMAGENET_MEAN,
    IMAGENET_STD,
    IMG_SIZE,
    RESIZE_SIZE,
    build_model,
    pick_device,
    sync_device,
)

MODEL_NAME = "cnn"
MB = 1024 * 1024
MAX_UPLOAD_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "25")) * MB)
NOT_TRAINED = "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"
RETRAIN = "retrain with ./run.sh train"
META_FILE, STATE_DICT_FILE, TORCHSCRIPT_FILE = "model_meta.json", "hotdog_cnn.pt", "hotdog_cnn.ts.pt"
# Everything Pillow raises for undecodable / truncated / fuzzed input (as in clef-server/imaging.py):
# e.g. ValueError("Truncated IHDR chunk"), SyntaxError("broken PNG file").
_DECODE_ERRORS = (UnidentifiedImageError, OSError, SyntaxError, ValueError, Image.DecompressionBombError)
log = logging.getLogger("cnn-server")


class BadImage(ValueError):
    """The upload is not a decodable image (-> 400)."""


# ---------------------------------------------------------------- loading
def load_weights(artifacts: Path, arch: str, num_classes: int) -> tuple[nn.Module, str]:
    """(model on CPU, "state_dict" | "torchscript"): the notebook's state_dict, else its TorchScript export."""
    sd_path, ts_path = artifacts / STATE_DICT_FILE, artifacts / TORCHSCRIPT_FILE
    errors = []
    if sd_path.exists() and arch in ARCHS:
        try:
            model = build_model(arch, num_classes=num_classes)
            model.load_state_dict(torch.load(sd_path, map_location="cpu", weights_only=True))
            return model, "state_dict"
        except Exception as e:
            errors.append(f"state_dict: {e}")
    if ts_path.exists():
        try:
            with warnings.catch_warnings():  # torch.jit is deprecated in recent torch but still works
                warnings.simplefilter("ignore", FutureWarning)
                return torch.jit.load(str(ts_path), map_location="cpu"), "torchscript"
        except Exception as e:
            errors.append(f"torchscript: {e}")
    if arch not in ARCHS:
        raise ValueError(f"artifacts are from an unsupported architecture '{arch}' — {RETRAIN}")
    raise RuntimeError(f"could not load weights from {artifacts} ({'; '.join(errors)}) — {RETRAIN}")


def _median_ms(model: nn.Module, x: torch.Tensor, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        model(x)
        sync_device(x.device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times) * 1000


def pick_memory_format(model: nn.Module, x: torch.Tensor) -> torch.memory_format:
    """Benchmark NCHW vs channels_last and keep the faster layout (also warms the model up).

    channels_last makes ConvNeXt ~3x faster on CPU but MobileNet slightly slower, so measure rather than guess.
    Not tried on MPS, where channels_last support is patchy.
    """
    candidates = [torch.contiguous_format]
    if x.device.type in ("cpu", "cuda"):
        candidates.append(torch.channels_last)
    timings = {}
    with torch.inference_mode():
        for fmt in candidates:
            model.to(memory_format=fmt)
            xf = x.contiguous(memory_format=fmt)
            model(xf)  # warm-up: first call allocates / selects kernels
            timings[fmt] = _median_ms(model, xf)
    best = min(timings, key=timings.get)
    model.to(memory_format=best)
    return best


class Classifier:
    """The model, its preprocessing and its readiness state."""

    def __init__(self, artifacts: Path) -> None:
        self.artifacts = artifacts
        self.ready = False
        self.detail: str | None = None  # why the model is not ready
        self.meta: dict = {}
        self.classes = list(CLASSES)
        self.device = torch.device("cpu")
        self.source: str | None = None  # "state_dict" | "torchscript"
        self.model: nn.Module | None = None
        self.transform: transforms.Compose | None = None
        self.temperature = 1.0
        self.resize = RESIZE_SIZE
        self.memory_format = torch.contiguous_format
        # One forward at a time: concurrent calls would just fight over the same cores / GPU queue
        # (and skew latency_ms); decoding and preprocessing still run in parallel.
        self._lock = threading.Lock()

    def load(self) -> None:
        """Load the artifacts; on failure stay up and report why via /health."""
        try:
            self._load()
        except Exception as e:
            self.ready, self.detail = False, str(e)
            log.error("model not loaded: %s", e)
        else:
            self.ready, self.detail = True, None

    def _load(self) -> None:
        meta_path = self.artifacts / META_FILE
        has_weights = any((self.artifacts / f).exists() for f in (STATE_DICT_FILE, TORCHSCRIPT_FILE))
        if not meta_path.exists() or not has_weights:
            raise FileNotFoundError(NOT_TRAINED)
        meta = json.loads(meta_path.read_text())
        classes = list(meta.get("classes", CLASSES))
        if sorted(classes) != sorted(CLASSES):
            raise ValueError(f"unexpected classes in meta: {classes}")
        model, source = load_weights(self.artifacts, meta.get("arch", DEFAULT_ARCH), len(classes))

        device = pick_device(os.environ.get("DEVICE") or None)
        img_size = int(meta.get("img_size", IMG_SIZE))
        self.resize = int(meta.get("resize_size", round(img_size * RESIZE_SIZE / IMG_SIZE)))
        self.transform = transforms.Compose(  # the notebook's eval preprocessing
            [
                transforms.Resize(self.resize),
                transforms.CenterCrop(img_size),
                transforms.ToTensor(),
                transforms.Normalize(meta.get("mean", IMAGENET_MEAN), meta.get("std", IMAGENET_STD)),
            ]
        )
        self.model = model.to(device).eval()
        self.memory_format = pick_memory_format(self.model, torch.zeros(1, 3, img_size, img_size, device=device))
        self.meta, self.classes, self.device, self.source = meta, classes, device, source
        # Temperature scaling from the notebook's calibration step (1.0 = uncalibrated)
        self.temperature = float(meta.get("temperature", 1.0)) or 1.0

    # ---------------------------------------------------------------- inference
    def decode(self, data: bytes) -> Image.Image:
        """Upright RGB image, or BadImage for anything PIL can't read."""
        if not data:
            raise BadImage("empty file")
        try:
            img = Image.open(io.BytesIO(data))
            # JPEG only: let libjpeg decode at 1/2../1/8 scale, still >= the resize target. A 12 MP photo
            # then decodes ~15x faster; the subsequent antialiased Resize sees practically the same pixels.
            img.draft("RGB", (self.resize, self.resize))
            img.load()
        except _DECODE_ERRORS as e:
            raise BadImage(f"could not decode image: {e}") from e
        with suppress(Exception):  # broken EXIF should not fail the request
            ImageOps.exif_transpose(img, in_place=True)
        return img if img.mode == "RGB" else img.convert("RGB")

    def predict(self, img: Image.Image) -> tuple[dict[str, float], float]:
        """({class: calibrated probability}, forward-pass latency in ms)."""
        x = self.transform(img).unsqueeze(0).to(self.device, memory_format=self.memory_format)
        with self._lock, torch.inference_mode():
            sync_device(self.device)
            t0 = time.perf_counter()
            logits = self.model(x)
            sync_device(self.device)
            latency = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(logits.float() / self.temperature, dim=1)[0].tolist()
        return dict(zip(self.classes, probs, strict=True)), latency

    def classify(self, data: bytes) -> tuple[dict[str, float], float]:
        return self.predict(self.decode(data))


def artifacts_dir() -> Path:
    # Relative paths are relative to cnn/, like in the notebook and run_training.py.
    return (CNN_DIR / os.environ.get("CNN_ARTIFACTS", "artifacts")).resolve()


clf = Classifier(artifacts_dir())
clf.load()  # at import time, so the model is ready before uvicorn accepts requests

app = FastAPI(title="Bun Tribunal - CNN", version="1.0.0")
# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it.
UI_ORIGIN_RE = rf"^https?://[^/]+:{os.environ.get('PORT_UI', '8080')}$"
app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health() -> dict:
    body = {"status": "ok", "model": MODEL_NAME, "device": clf.device.type, "ready": clf.ready}
    if clf.detail:
        body["detail"] = clf.detail
    if clf.ready:
        body |= {"arch": clf.meta.get("arch"), "weights": clf.source, "temperature": clf.temperature}
    return body


async def read_upload(file: UploadFile, max_bytes: int) -> bytes:
    """The upload's bytes, or 413. Checks the known size first so an oversized file is never read into memory."""
    if file.size is None or file.size <= max_bytes:
        data = await file.read(max_bytes + 1)
        if len(data) <= max_bytes:
            return data
    raise HTTPException(status_code=413, detail=f"file larger than {max_bytes // MB} MB")


@app.post("/classify")
async def classify(file: UploadFile = File(...)) -> dict:  # noqa: B008  (FastAPI's dependency idiom)
    t0 = time.perf_counter()
    if not clf.ready:
        raise HTTPException(status_code=503, detail=clf.detail or "model not ready")
    data = await read_upload(file, MAX_UPLOAD_BYTES)
    try:
        # Decode + preprocess + forward are CPU/GPU-bound: keep them off the event loop.
        probs, latency = await run_in_threadpool(clf.classify, data)
    except BadImage as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    label = max(probs, key=probs.get)
    return {
        "model": MODEL_NAME,
        "label": label,
        "is_hotdog": label == "hotdog",
        "confidence": round(probs[label], 4),
        "probabilities": {c: round(probs[c], 4) for c in CLASSES},
        "latency_ms": round(latency, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
