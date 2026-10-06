"""Bun Tribunal - CNN inference server (see ../../CONTRACT.md).

Started by ./run.sh (port 8002). Manually, from the project root:
      uvicorn --app-dir cnn/server app:app --port 8002

Env:
  CNN_ARTIFACTS  directory with model_meta.json + hotdog_cnn.pt / hotdog_cnn.ts.pt (default cnn/artifacts; relative paths are relative to cnn/)
  DEVICE         force a torch device (cuda / mps / cpu); default auto cuda -> mps -> cpu
"""
from __future__ import annotations

import io
import json
import logging
import os
import sys
import time
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image, ImageOps, UnidentifiedImageError

HERE = Path(__file__).resolve().parent
# hotdog_model.py lives in cnn/ (one level up)
if str(HERE.parent) not in sys.path:
    sys.path.insert(0, str(HERE.parent))

MODEL_NAME = "cnn"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024
NOT_TRAINED = "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"
CLASSES = ["hotdog", "not_hotdog"]
log = logging.getLogger("cnn-server")


class Classifier:
    """Holds the model and its preprocessing."""

    def __init__(self) -> None:
        # Relative paths are relative to cnn/, like in the notebook and run_training.py.
        self.artifacts = (HERE.parent / os.environ.get("CNN_ARTIFACTS", "artifacts")).resolve()
        self.ready = False
        self.detail: str | None = None
        self.model = None
        self.meta: dict = {}
        self.classes = list(CLASSES)
        self.device_name = "cpu"
        self.source: str | None = None
        self.temperature = 1.0

    # ---------- loading ----------
    def load(self) -> None:
        try:
            self._load_real()
            self.ready = True
            self.detail = None
        except Exception as e:  # keep serving /health with a useful message
            self.ready = False
            self.detail = str(e)
            log.error("model not loaded: %s", e)

    def _load_real(self) -> None:
        import torch
        from torchvision import transforms

        meta_path = self.artifacts / "model_meta.json"
        sd_path = self.artifacts / "hotdog_cnn.pt"
        ts_path = self.artifacts / "hotdog_cnn.ts.pt"
        if not meta_path.exists() or not (sd_path.exists() or ts_path.exists()):
            raise FileNotFoundError(NOT_TRAINED)
        self.meta = json.loads(meta_path.read_text())
        # Temperature scaling from the notebook's calibration step (1.0 = uncalibrated)
        self.temperature = float(self.meta.get("temperature", 1.0)) or 1.0
        self.classes = list(self.meta.get("classes", CLASSES))
        if sorted(self.classes) != sorted(CLASSES):
            raise ValueError(f"unexpected classes in meta: {self.classes}")

        from hotdog_model import ARCHS, DEFAULT_ARCH, build_model, pick_device

        device = pick_device(os.environ.get("DEVICE") or None)
        arch = self.meta.get("arch", DEFAULT_ARCH)
        model = None
        errors = []
        if sd_path.exists() and arch in ARCHS:
            try:
                model = build_model(arch, num_classes=len(self.classes))
                state = torch.load(sd_path, map_location="cpu", weights_only=True)
                model.load_state_dict(state)
                self.source = "state_dict"
            except Exception as e:
                errors.append(f"state_dict: {e}")
                model = None
        if model is None and ts_path.exists():
            try:
                import warnings

                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", FutureWarning)
                    model = torch.jit.load(str(ts_path), map_location="cpu")
                self.source = "torchscript"
            except Exception as e:
                errors.append(f"torchscript: {e}")
                model = None
        if model is None and arch not in ARCHS:
            raise ValueError(f"artifacts are from an unsupported architecture '{arch}' — retrain with ./run.sh train")
        if model is None:
            raise RuntimeError(
                f"could not load weights from {self.artifacts} ({'; '.join(errors)}) — retrain with ./run.sh train"
            )

        self.model = model.to(device).eval()
        self.device = device
        self.device_name = device.type
        img = int(self.meta.get("img_size", 224))
        resize = int(self.meta.get("resize_size", round(img * 256 / 224)))
        self.transform = transforms.Compose([
            transforms.Resize(resize),
            transforms.CenterCrop(img),
            transforms.ToTensor(),
            transforms.Normalize(self.meta.get("mean", [0.485, 0.456, 0.406]),
                                 self.meta.get("std", [0.229, 0.224, 0.225])),
        ])
        # warm-up so the first request's latency is representative
        with torch.inference_mode():
            self.model(torch.zeros(1, 3, img, img, device=device))

    # ---------- inference ----------
    @staticmethod
    def decode(data: bytes) -> Image.Image:
        if not data:
            raise ValueError("empty file")
        try:
            img = Image.open(io.BytesIO(data))
            img.load()
            img = ImageOps.exif_transpose(img)
            return img.convert("RGB")
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as e:
            raise ValueError(f"could not decode image: {e}") from e

    def predict(self, img: Image.Image) -> tuple[list[float], float]:
        import torch
        from hotdog_model import sync_device

        x = self.transform(img).unsqueeze(0).to(self.device)
        with torch.inference_mode():
            sync_device(self.device)
            t0 = time.perf_counter()
            logits = self.model(x)
            sync_device(self.device)
            latency = (time.perf_counter() - t0) * 1000
        probs = torch.softmax(logits.float() / self.temperature, dim=1)[0].cpu().tolist()
        return probs, latency


clf = Classifier()
clf.load()

app = FastAPI(title="Bun Tribunal - CNN", version="1.0.0")
# Only the Bun Tribunal UI (any host, on the UI port) may call this server from a browser;
# other websites you visit can't use it (or, for the LLM, spend your API key).
UI_ORIGIN_RE = r"^https?://[^/]+:%s$" % os.environ.get("PORT_UI", "8080")
app.add_middleware(CORSMiddleware, allow_origin_regex=UI_ORIGIN_RE, allow_methods=["*"], allow_headers=["*"])


@app.get("/health")
def health():
    body = {"status": "ok", "model": MODEL_NAME, "device": clf.device_name, "ready": clf.ready}
    if clf.detail:
        body["detail"] = clf.detail
    if clf.ready:
        body["arch"] = clf.meta.get("arch")
        body["weights"] = clf.source
        body["temperature"] = clf.temperature
    return body


@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    t0 = time.perf_counter()
    if not clf.ready:
        raise HTTPException(status_code=503, detail=clf.detail or "model not ready")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="image too large (max 25 MB)")
    try:
        img = await run_in_threadpool(clf.decode, data)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    probs, latency = await run_in_threadpool(clf.predict, img)

    prob_map = {c: float(p) for c, p in zip(clf.classes, probs)}
    label = max(prob_map, key=prob_map.get)
    return {
        "model": MODEL_NAME,
        "label": label,
        "is_hotdog": label == "hotdog",
        "confidence": round(prob_map[label], 4),
        "probabilities": {c: round(prob_map[c], 4) for c in CLASSES},
        "latency_ms": round(latency, 2),
        "total_ms": round((time.perf_counter() - t0) * 1000, 2),
    }
