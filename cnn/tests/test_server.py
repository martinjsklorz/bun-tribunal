"""Server tests: missing artifacts, and a real tiny model trained by the notebook in SMOKE mode.

Run from cnn/:  pytest -q tests
The real-model tests execute train_hotdog_cnn.ipynb with SMOKE=1 ARCH=mobilenet_v3_large into a temp dir
(about a minute on CPU).
Set CNN_TEST_ARTIFACTS=/path/to/artifacts to reuse existing artifacts instead.
"""

from __future__ import annotations

import importlib
import io
import json
import os
import shutil
import struct
import sys
import zlib
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from PIL import Image

CNN_DIR = Path(__file__).resolve().parents[1]
SERVER_DIR = CNN_DIR / "server"
sys.path.insert(0, str(SERVER_DIR))
sys.path.insert(0, str(CNN_DIR))


def make_client(monkeypatch, **env) -> TestClient:
    for k in ("CNN_ARTIFACTS", "DEVICE", "MAX_UPLOAD_MB"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    import app as app_module

    app_module = importlib.reload(app_module)  # model is loaded at import time
    return TestClient(app_module.app)


def png_bytes(color=(200, 60, 40), size=(320, 240), fmt="PNG") -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format=fmt)
    return buf.getvalue()


def png_chunk(kind: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", zlib.crc32(kind + payload))


PNG_SIG = b"\x89PNG\r\n\x1a\n"
IHDR_8X8_RGB = png_chunk(b"IHDR", struct.pack(">IIBBBBB", 8, 8, 8, 2, 0, 0, 0))
IDAT_HEAD = zlib.compress(b"\0" + b"\x80" * 24)[:6]
# Fuzzed PNGs: Pillow raises ValueError("Truncated IHDR chunk") / SyntaxError("broken PNG file") for these.
TRUNCATED_IHDR_PNG = PNG_SIG + png_chunk(b"IHDR", b"\0\0\0\x10\0\0")
BROKEN_PNG = PNG_SIG + IHDR_8X8_RGB + png_chunk(b"IDAT", IDAT_HEAD) + png_chunk(b"I\x01AT", b"junk")


def untrained_client_accepting_uploads(monkeypatch, tmp_path, **env) -> tuple[TestClient, object]:
    """Client without a model, but marked ready: enough for paths that fail before the forward pass."""
    c = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path / "nope", **env)
    import app as app_module

    monkeypatch.setattr(app_module.clf, "ready", True)
    return c, app_module


def check_classify_body(body: dict):
    assert body["model"] == "cnn"
    assert body["label"] in ("hotdog", "not_hotdog")
    assert body["is_hotdog"] == (body["label"] == "hotdog")
    assert set(body["probabilities"]) == {"hotdog", "not_hotdog"}
    assert abs(sum(body["probabilities"].values()) - 1) < 1e-3
    assert body["confidence"] == pytest.approx(body["probabilities"][body["label"]], abs=1e-4)
    assert body["confidence"] >= 0.5
    assert 0 <= body["latency_ms"] <= body["total_ms"]


# ---------------- missing artifacts ----------------
def test_missing_artifacts(monkeypatch, tmp_path):
    c = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path / "nope")
    b = c.get("/health").json()
    assert b["ready"] is False
    assert b["detail"] == "CNN not trained yet — run ./run.sh train (or open cnn/train_hotdog_cnn.ipynb)"
    assert "./run.sh train" in b["detail"]
    r = c.post("/classify", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 503 and r.json()["detail"] == b["detail"]


def test_cors_only_for_the_ui(monkeypatch, tmp_path):
    c = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path / "nope")
    pre = {"Access-Control-Request-Method": "POST"}
    ok = c.options("/classify", headers={"Origin": "http://localhost:8080", **pre})
    bad = c.options("/classify", headers={"Origin": "https://evil.example", **pre})
    assert ok.headers.get("access-control-allow-origin") == "http://localhost:8080"
    assert "access-control-allow-origin" not in bad.headers


def test_meta_without_weights_counts_as_untrained(monkeypatch, tmp_path):
    (tmp_path / "model_meta.json").write_text(
        json.dumps({"arch": "convnext_tiny", "classes": ["hotdog", "not_hotdog"]})
    )
    b = make_client(monkeypatch, CNN_ARTIFACTS=tmp_path).get("/health").json()
    assert b["ready"] is False and "./run.sh train" in b["detail"]


def test_decode_large_jpeg_upright_and_downscaled(monkeypatch, tmp_path):
    import app as app_module

    img = Image.new("RGB", (4000, 3000), (200, 60, 40))
    exif = img.getexif()
    exif[0x0112] = 6  # EXIF orientation: rotate 90° CW to display
    buf = io.BytesIO()
    img.save(buf, format="JPEG", exif=exif)
    clf = app_module.Classifier(tmp_path)
    out = clf.decode(buf.getvalue())
    assert out.mode == "RGB"
    w, h = out.size
    assert h > w  # portrait after the EXIF rotation
    assert clf.resize <= w < 4000 / 2  # decoded at reduced scale, still >= the resize target
    with pytest.raises(app_module.BadImage):
        clf.decode(b"not an image")


@pytest.mark.parametrize("data", [TRUNCATED_IHDR_PNG, BROKEN_PNG], ids=["truncated_ihdr", "broken_png"])
def test_fuzzed_png_is_400(monkeypatch, tmp_path, data):
    c, app_module = untrained_client_accepting_uploads(monkeypatch, tmp_path)
    with pytest.raises(app_module.BadImage):
        app_module.clf.decode(data)
    r = c.post("/classify", files={"file": ("x.png", data, "image/png")})
    assert r.status_code == 400, r.text
    assert r.json()["detail"].startswith("could not decode image")


def test_broken_exif_is_ignored(monkeypatch, tmp_path):
    import app as app_module

    def boom(*args, **kwargs):
        raise ValueError("broken EXIF")

    monkeypatch.setattr(app_module.ImageOps, "exif_transpose", boom)
    out = app_module.Classifier(tmp_path).decode(png_bytes(size=(40, 30)))
    assert out.mode == "RGB" and out.size == (40, 30)


def test_max_upload_mb(monkeypatch, tmp_path):
    c, app_module = untrained_client_accepting_uploads(monkeypatch, tmp_path, MAX_UPLOAD_MB="1")
    assert app_module.MAX_UPLOAD_BYTES == 1024 * 1024
    r = c.post("/classify", files={"file": ("big.jpg", b"\0" * (1024 * 1024 + 1), "image/jpeg")})
    assert r.status_code == 413 and r.json()["detail"] == "file larger than 1 MB"
    r = c.post("/classify", files={"file": ("ok.jpg", b"\0" * (1024 * 1024), "image/jpeg")})
    assert r.status_code == 400  # within the limit: rejected as undecodable, not as too large


def test_default_max_upload_is_25_mb(monkeypatch, tmp_path):
    _, app_module = untrained_client_accepting_uploads(monkeypatch, tmp_path)
    assert app_module.MAX_UPLOAD_BYTES == 25 * 1024 * 1024


# ---------------- real SMOKE-trained model ----------------
@pytest.fixture(scope="session")
def smoke_artifacts(tmp_path_factory) -> Path:
    existing = os.environ.get("CNN_TEST_ARTIFACTS")
    if existing:
        return Path(existing)
    nbformat = pytest.importorskip("nbformat")
    nbclient = pytest.importorskip("nbclient")
    out = tmp_path_factory.mktemp("artifacts")
    keys = ("SMOKE", "ARCH", "CNN_ARTIFACTS", "MPLBACKEND", "DEVICE")
    old = {k: os.environ.get(k) for k in keys}
    os.environ.update(SMOKE="1", ARCH="mobilenet_v3_large", CNN_ARTIFACTS=str(out), MPLBACKEND="Agg", DEVICE="cpu")
    try:
        nb = nbformat.read(CNN_DIR / "train_hotdog_cnn.ipynb", as_version=4)
        nbclient.NotebookClient(
            nb, timeout=900, kernel_name="python3", resources={"metadata": {"path": str(CNN_DIR)}}
        ).execute()
    finally:
        for k, v in old.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    assert (out / "model_meta.json").exists() and (out / "hotdog_cnn.pt").exists()
    return out


def test_real_health(monkeypatch, smoke_artifacts):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    b = c.get("/health").json()
    assert b == {**b, "status": "ok", "model": "cnn", "device": "cpu", "ready": True}
    assert b["weights"] == "state_dict"
    if not os.environ.get("CNN_TEST_ARTIFACTS"):  # our own SMOKE model; a user's model may be any arch
        assert b["arch"] == "mobilenet_v3_large"
    assert b["temperature"] > 0


@pytest.mark.parametrize("fmt,mime", [("PNG", "image/png"), ("JPEG", "image/jpeg"), ("WEBP", "image/webp")])
def test_real_classify(monkeypatch, smoke_artifacts, fmt, mime):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    r = c.post("/classify", files={"file": (f"a.{fmt.lower()}", png_bytes(size=(1024, 640), fmt=fmt), mime)})
    assert r.status_code == 200, r.text
    check_classify_body(r.json())


def test_real_bad_image(monkeypatch, smoke_artifacts):
    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    r = c.post("/classify", files={"file": ("x.jpg", b"\xff\xd8\xff garbage", "image/jpeg")})
    assert r.status_code == 400
    assert c.post("/classify", files={"file": ("e.png", b"", "image/png")}).status_code == 400


def test_real_too_large(monkeypatch, smoke_artifacts):
    import app as app_module

    c = make_client(monkeypatch, CNN_ARTIFACTS=smoke_artifacts, DEVICE="cpu")
    big = b"\0" * (app_module.MAX_UPLOAD_BYTES + 1)
    r = c.post("/classify", files={"file": ("big.jpg", big, "image/jpeg")})
    assert r.status_code == 413


def test_real_torchscript_fallback(monkeypatch, smoke_artifacts, tmp_path):
    ts_only = tmp_path / "ts_only"
    ts_only.mkdir()
    for f in ("model_meta.json", "hotdog_cnn.ts.pt"):
        shutil.copy(smoke_artifacts / f, ts_only / f)
    c = make_client(monkeypatch, CNN_ARTIFACTS=ts_only, DEVICE="cpu")
    assert c.get("/health").json()["weights"] == "torchscript"
    r = c.post("/classify", files={"file": ("a.png", png_bytes(), "image/png")})
    assert r.status_code == 200
    check_classify_body(r.json())


def test_unsupported_arch_falls_back_to_torchscript(monkeypatch, smoke_artifacts, tmp_path):
    old = tmp_path / "old"
    shutil.copytree(smoke_artifacts, old)
    meta = json.loads((old / "model_meta.json").read_text())
    (old / "model_meta.json").write_text(json.dumps({**meta, "arch": "scratch"}))
    c = make_client(monkeypatch, CNN_ARTIFACTS=old, DEVICE="cpu")
    b = c.get("/health").json()
    assert b["ready"] is True and b["weights"] == "torchscript"


def test_unsupported_arch_without_torchscript(monkeypatch, smoke_artifacts, tmp_path):
    old = tmp_path / "old"
    old.mkdir()
    shutil.copy(smoke_artifacts / "hotdog_cnn.pt", old / "hotdog_cnn.pt")
    meta = json.loads((smoke_artifacts / "model_meta.json").read_text())
    (old / "model_meta.json").write_text(json.dumps({**meta, "arch": "scratch"}))
    b = make_client(monkeypatch, CNN_ARTIFACTS=old, DEVICE="cpu").get("/health").json()
    assert b["ready"] is False
    assert b["detail"] == "artifacts are from an unsupported architecture 'scratch' — retrain with ./run.sh train"
